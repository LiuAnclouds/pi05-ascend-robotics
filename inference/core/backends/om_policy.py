"""Run the paired OM policy with explicit model, noise, and normalization inputs.

The worker owns one ACL context and borrows KV buffers on device. Runtime code
does not import the quantization pipeline or historical experiments.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

try:
    import fcntl
except ModuleNotFoundError:
    fcntl = None

from include.project_paths import PROJECT_ROOT
from inference.core.backends.acl_executor import BufferedACL, DTYPES, check, resident_model_pair
from inference.include.definitions import (
    ACTION_DIM,
    ACTION_HORIZON,
    DEFAULT_DENOISING_STEPS,
    DEFAULT_SEED,
    TEXT_SPECIALIZATION_SLOTS,
)


class OMPolicy:
    """执行视觉语言与动作 OM，串行模式支持设备上的 FP16 Euler 更新。"""

    def __init__(
        self,
        root=None,
        pair_directory=None,
        action_q01=None,
        action_q99=None,
        *,
        part1_om=None,
        part2_om=None,
        seed=DEFAULT_SEED,
        noise_file=None,
        device_euler=False,
    ):
        """Load the pair and source normalization without any robot/camera I/O.

        Args:
            root: Project/runtime asset root, used only for the exclusive device lock.
            pair_directory: Optional directory containing part1.om and part2.om.
            action_q01, action_q99: Source model's seven physical action quantiles.
            part1_om, part2_om: Explicit OM files instead of a pair directory.
            seed: Deterministic initial-noise seed when noise_file is absent.
            noise_file: Optional reference NPZ whose noise is reused exactly.
            device_euler: Run the serial Euler updates and denoising steps on one NPU stream.
        Returns:
            None. This instance owns its model/context/lock until close().
        """
        self.context = self.lock = None
        self.device_euler = device_euler
        self.device_euler_stream = None
        self.device_euler_executors = []
        self.device_euler_pending = False
        if fcntl is None:
            raise RuntimeError("ACL OM inference requires the Linux deployment environment")
        self.root = Path(root) if root is not None else PROJECT_ROOT
        if pair_directory is not None:
            pair_directory = Path(pair_directory)
            part1_om = part1_om or pair_directory / "part1.om"
            part2_om = part2_om or pair_directory / "part2.om"
        if part1_om is None or part2_om is None:
            raise ValueError("Provide both OM paths or a model-pair directory")
        self.low, self.high = np.asarray(action_q01, np.float32), np.asarray(action_q99, np.float32)
        high = self.high
        if (
            self.low.shape != (7,)
            or high.shape != (7,)
            or not np.isfinite([self.low, high]).all()
            or np.any(high <= self.low)
        ):
            raise ValueError("Expected seven finite nondegenerate action quantiles")
        self.span = (high - self.low + np.float32(1e-6)) / 2
        if noise_file is None:
            self.noise = np.random.default_rng(seed).standard_normal(
                (1, ACTION_HORIZON, ACTION_DIM), dtype=np.float32
            )
        else:
            with np.load(noise_file) as archive:
                self.noise = archive["noise"].astype(np.float32)
        if self.noise.shape != (1, ACTION_HORIZON, ACTION_DIM) or not np.isfinite(self.noise).all():
            raise ValueError("Expected a finite complete initial noise tensor")
        self.metadata = {
            "om_pair": [str(part1_om), str(part2_om)],
            "steps": DEFAULT_DENOISING_STEPS,
            "precision": "Defined by loaded OM artifacts; no runtime re-quantization",
            "euler_dtype": "float32",
            "allocation": {},
            "seed": seed,
            "noise_file": None if noise_file is None else str(noise_file),
        }
        lock_path = self.root / "outputs/.device_benchmark.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.lock = lock_path.open("a")
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            context = resident_model_pair(
                Path(part1_om), Path(part2_om), allocation="normal", telemetry=self.metadata["allocation"]
            )
            pair = context.__enter__()
            self.context = context
            self.prefix, self.suffix = pair[-1]
            if self.device_euler:
                self._prepare_device_euler()
            self.metadata["inputs"] = [executor.io["inputs"] for executor in (self.prefix, self.suffix)]
        except BaseException:
            self.close()
            raise

    def _prepare_device_euler(self):
        """建立十步 Part2 输入输出，逐步复用设备上的 FP16 动作状态。

        Args:
            无；串行 OM 同时输出速度和下一步动作状态。
        Returns:
            无；中间状态始终留在 NPU 内存。
        """
        inputs, outputs = self.suffix.io["inputs"], self.suffix.io["outputs"]
        if (
            len(outputs) != 2
            or inputs[2]["dtype_id"] != 1
            or outputs[0]["dtype_id"] != 1
            or outputs[1]["dtype_id"] != 1
            or inputs[2]["shape"] != outputs[1]["shape"]
        ):
            raise ValueError("Serial Euler OM must expose FP16 velocity and a matching FP16 next state")
        self.device_euler_executors.append(self.suffix)
        self.device_euler_stream, code = self.prefix.acl.rt.create_stream()
        check(code, "serial Euler create stream")
        for _ in range(1, DEFAULT_DENOISING_STEPS):
            previous = self.device_euler_executors[-1]
            borrowed = {
                0: (self.prefix.addresses["outputs"][0], self.prefix.io["outputs"][0]),
                1: (self.prefix.addresses["outputs"][1], self.prefix.io["outputs"][1]),
                2: (previous.addresses["outputs"][1], previous.io["outputs"][1]),
            }
            executor = BufferedACL(
                self.prefix.acl,
                self.suffix.model_id,
                self.suffix.io,
                borrowed=borrowed,
                malloc_policy=2,
            )
            self.device_euler_executors.append(executor)
        for step, executor in enumerate(self.device_euler_executors):
            timestep = (
                np.asarray([step], np.int64)
                if executor.io["inputs"][3]["dtype_id"] == 9
                else np.asarray([1 - step / DEFAULT_DENOISING_STEPS], np.float32)
            )
            executor.feed({3: timestep})
        self.metadata["euler_dtype"] = "FP16 on NPU"
        self.metadata["denoising_execution"] = "10 Part2 calls on one ACL stream; synchronize once"

    def prefix_values(self, sample):
        """Map named image/token arrays to the loaded prefix contract.

        Args:
            sample: Named normalized images, availability masks, tokens and token_mask.
        Returns:
            Correctly ordered/dtyped prefix arrays. Ineligible specialization is refused.
        """
        specifications = self.prefix.io["inputs"]
        count = len(specifications)
        if count == 6:
            mask = sample["token_mask"]
            if mask[:, TEXT_SPECIALIZATION_SLOTS:].any() or not mask[:, :TEXT_SPECIALIZATION_SLOTS].any():
                raise ValueError("Observation is not eligible for the text64 specialized OM")
            slots = specifications[4]["shape"][1]
            values = [
                sample["image0"],
                sample["image1"],
                np.asarray([True]),
                np.asarray([True]),
                sample["tokens"][:, :slots],
                mask[:, :slots],
            ]
        elif count == 8:
            values = [
                sample[name]
                for name in ("image0", "image1", "image2", "mask0", "mask1", "mask2", "tokens", "token_mask")
            ]
        else:
            raise ValueError(f"Unsupported prefix input contract: {count} inputs")
        return [
            np.ascontiguousarray(value, dtype=DTYPES[spec["dtype_id"]])
            for value, spec in zip(values, specifications)
        ]

    def predict(self, payload):
        """Run one prefix and the unchanged ten denoising steps.

        Args:
            payload: Named inputs, observation state in degrees/mm, and optional RTC prefix/weights.
        Returns:
            Complete normalized [1,50,32], absolute [50,7] targets and latency components.
        """
        self.prepare_guidance(payload)
        input_started = time.monotonic()
        values = self.prefix_values(payload["inputs"])
        started = time.monotonic()
        input_cast_ms = (started - input_started) * 1000.0
        prefix_upload = self.prefix.feed(dict(enumerate(values)))
        prefix_ms = self.prefix.run()
        if self.device_euler:
            return self._predict_device_euler(payload, started, prefix_ms, prefix_upload, input_cast_ms)
        value, times, capture = self.noise.copy(), [], None
        for step in range(DEFAULT_DENOISING_STEPS):
            value = self.project_state(value, step)
            timestep = (
                np.asarray([step], np.int64)
                if self.suffix.io["inputs"][3]["dtype_id"] == 9
                else np.asarray([1 - step / DEFAULT_DENOISING_STEPS], np.float32)
            )
            self.suffix.feed({2: value, 3: timestep})
            times.append(self.suffix.run())
            velocity = self.suffix.read(0)
            if velocity.shape != self.noise.shape or not np.isfinite(velocity).all():
                raise RuntimeError("Nonfinite or incomplete action velocity")
            if payload.get("capture_step") and step == DEFAULT_DENOISING_STEPS // 2:
                capture = dict(
                    noise=value.copy(), timestep=np.asarray([0.5], np.float32), velocity=velocity.copy()
                )
            velocity = self.guide_velocity(velocity, step)
            value = value + np.float32(-1.0 / DEFAULT_DENOISING_STEPS) * velocity.astype(np.float32)
        value = self.project_state(value, DEFAULT_DENOISING_STEPS)
        inference_ms = (time.monotonic() - started) * 1000
        if not np.isfinite(value).all():
            raise RuntimeError("Nonfinite complete 50x32 action prediction")
        state = np.asarray(payload["state"], np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("Expected a finite observation state [7]")
        targets = (value[0, :, :7] + 1) * self.span + self.low
        targets[:, :6] += state[:6]
        if capture is not None:
            capture.update(past_kv_tensor=self.prefix.read(0), prefix_pad_masks=self.prefix.read(1))
            np.savez(payload["capture_step"], **capture)
        return dict(
            normalized=value,
            targets=targets,
            prefix_ms=prefix_ms,
            input_cast_ms=input_cast_ms,
            prefix_upload=prefix_upload,
            suffix_steps_ms=times,
            inference_ms=inference_ms,
            guidance_steps_ms=getattr(self, "guidance_times", []),
            vjp_execute_steps_ms=getattr(self, "vjp_execute_times", []),
            prefix_projection_ms=getattr(self, "projection_ms", 0.0),
        )

    def _predict_device_euler(self, payload, started, prefix_ms, prefix_upload, input_cast_ms):
        """在同一 NPU 流依次提交十步去噪和 FP16 更新，完成后读取最终状态。

        Args:
            payload: 观测及可选的中间结果保存路径。
            started: 本轮模型执行的开始时间。
            prefix_ms, prefix_upload, input_cast_ms: 前缀模型已有的计时信息。
        Returns:
            归一化动作、物理目标和耗时；中间状态按 FP16 舍入。
        """
        capture_step = payload.get("capture_step")
        self.suffix.feed({2: self.noise})
        enqueue_started = time.monotonic()
        for executor in self.device_euler_executors:
            executor.run_async(self.device_euler_stream)
            self.device_euler_pending = True
        enqueue_ms = (time.monotonic() - enqueue_started) * 1000.0
        synchronize_started = time.monotonic()
        check(self.prefix.acl.rt.synchronize_stream(self.device_euler_stream), "serial Euler synchronize")
        self.device_euler_pending = False
        synchronize_ms = (time.monotonic() - synchronize_started) * 1000.0
        final_executor = self.device_euler_executors[-1]
        value = final_executor.read(1).astype(np.float32)
        inference_ms = (time.monotonic() - started) * 1000.0
        if not np.isfinite(value).all():
            raise RuntimeError("Nonfinite complete 50x32 action prediction")
        state = np.asarray(payload["state"], np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("Expected a finite observation state [7]")
        targets = (value[0, :, :7] + 1) * self.span + self.low
        targets[:, :6] += state[:6]
        if capture_step:
            step = DEFAULT_DENOISING_STEPS // 2
            previous = self.noise if step == 0 else self.device_euler_executors[step - 1].read(1)
            capture = dict(
                noise=previous,
                timestep=np.asarray([0.5], np.float32),
                velocity=self.device_euler_executors[step].read(0),
                past_kv_tensor=self.prefix.read(0),
                prefix_pad_masks=self.prefix.read(1),
            )
            np.savez(capture_step, **capture)
        return dict(
            normalized=value,
            targets=targets,
            prefix_ms=prefix_ms,
            input_cast_ms=input_cast_ms,
            prefix_upload=prefix_upload,
            suffix_steps_ms=[],
            inference_ms=inference_ms,
            suffix_enqueue_ms=enqueue_ms,
            suffix_synchronize_ms=synchronize_ms,
            guidance_steps_ms=getattr(self, "guidance_times", []),
            vjp_execute_steps_ms=getattr(self, "vjp_execute_times", []),
            prefix_projection_ms=getattr(self, "projection_ms", 0.0),
        )

    def prepare_guidance(self, payload):
        """Reject RTC tensors for the ordinary forward-only model.

        Args:
            payload: One request dictionary.
        Returns:
            None. RTC subclasses override this explicitly.
        """
        if "rtc_prefix" in payload or "rtc_weights" in payload:
            raise ValueError("This OM policy has no RTC input-gradient model")

    def guide_velocity(self, velocity, step):
        """Return ordinary velocity; RTC subclasses override this hook.

        Args:
            velocity: Current [1,50,32] velocity.
            step: Denoising index.
        Returns:
            The same velocity array.
        """
        return velocity

    def project_state(self, value, step):
        """Return the ordinary action state without prefix projection.

        Args:
            value: Current normalized action/noise state.
            step: Denoising index.
        Returns:
            The same state array.
        """
        return value

    def close(self):
        """Release model/context before releasing this worker's device lock.

        Args:
            None.
        Returns:
            None. Safe after partial construction or repeated close calls.
        """
        if self.device_euler_stream is not None:
            if self.device_euler_pending:
                check(
                    self.prefix.acl.rt.synchronize_stream(self.device_euler_stream),
                    "serial Euler synchronize before release",
                )
                self.device_euler_pending = False
            stream, self.device_euler_stream = self.device_euler_stream, None
            check(self.prefix.acl.rt.destroy_stream(stream), "serial Euler destroy stream")
        for executor in reversed(self.device_euler_executors[1:]):
            executor.close()
        self.device_euler_executors = []
        try:
            if self.context is not None:
                context, self.context = self.context, None
                context.__exit__(None, None, None)
        finally:
            if self.lock is not None:
                self.lock.close()
                self.lock = None
