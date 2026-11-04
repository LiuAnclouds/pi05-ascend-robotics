"""Keep the ten FP32 RTC Euler states on device between shared VJP calls."""

import json
import time

import numpy as np

from inference.core.backends.acl_executor import check
from inference.core.rtc.shared_forward import SharedForwardPolicy


class DeviceEulerPolicy(SharedForwardPolicy):
    def __init__(self, *args, **kwargs):
        """Load shared FP16 RTC and validate the compiled FP32 Euler contracts.

        Args: SharedForwardPolicy constructor options.
        Returns: None; mismatched coefficients or state layouts are rejected.
        """
        super().__init__(*args, **kwargs)
        try:
            specification_path = self.gradient.directory / "euler.json"
            specification = json.loads(specification_path.read_text())
            if (
                specification.get("kind") != "device_fp32_euler"
                or specification["max_guidance"] != self.max_guidance
                or specification["cotangent_scale"] != float(self.cotangent_scale)
                or specification["denoising_steps"] != 10
            ):
                raise ValueError("Device Euler coefficients differ from the requested RTC")
            for output, destination in (("rtc_next_noise", "noise"), ("rtc_finite_out", "rtc_finite_in")):
                if self.gradient.values[output][1] != self.gradient.values[destination][1]:
                    raise ValueError("Device Euler state contracts differ")
            self.gradient.feed(dict(rtc_finite_in=np.asarray([True])))
            self.metadata["rtc"].update(
                euler_location="NPU_FP32",
                per_step_host_state_reads=0,
                finite_guard="All guided velocities/corrections, every step; accumulated on device",
                state_transfer="One initial H2D, device-to-device between steps, one final state D2H",
            )
        except BaseException:
            self.close()
            raise

    def _copy_state(self, output, destination):
        """Copy one contract-matched device output into the next step's input.

        Args: Manifest source and destination tensor names.
        Returns: None; performs one device-to-device copy without host staging.
        """
        pointer, spec = self.gradient.values[destination]
        source, source_spec = self.gradient.values[output]
        if spec != source_spec:
            raise ValueError("Device Euler copy contracts differ")
        check(
            self.gradient.acl.rt.memcpy(pointer, spec["bytes"], source, spec["bytes"], 3),
            "RTC device-to-device state update",
        )

    def predict(self, payload):
        """Run ten shared action/VJP steps with FP32 Euler state held on the NPU.

        Args: Named observation inputs, source state and optional RTC reference.
        Returns: Complete normalized actions, absolute targets and wall timings.
        """
        if "rtc_prefix" not in payload:
            return super().predict(payload)
        self.prepare_guidance(payload)
        input_started = time.monotonic()
        values = self.prefix_values(payload["inputs"])
        started = time.monotonic()
        input_cast_ms = (started - input_started) * 1000.0
        prefix_upload = self.prefix.feed(dict(enumerate(values)))
        prefix_ms = self.prefix.run()
        self.gradient.feed(dict(noise=self.noise, rtc_finite_in=np.asarray([True])))
        for step in range(10):
            step_started = time.monotonic()
            self.gradient.feed(
                dict(
                    step_index=np.asarray([step], np.int64),
                    timestep=np.asarray([np.float32(1 - step / 10)], np.float32),
                )
            )
            self.vjp_execute_times.append(self.gradient.run())
            if step < 9:
                self._copy_state("rtc_next_noise", "noise")
                self._copy_state("rtc_finite_out", "rtc_finite_in")
            self.guidance_times.append((time.monotonic() - step_started) * 1000)
        finite = bool(self.gradient.read("rtc_finite_out")[0])
        value = self.gradient.read("rtc_next_noise").astype(np.float32)
        inference_ms = (time.monotonic() - started) * 1000
        if not finite or not np.isfinite(value).all():
            raise RuntimeError("Nonfinite ten-step device RTC velocity/correction/action")
        targets = (value[0, :, :7] + 1) * self.span + self.low
        targets[:, :6] += np.asarray(payload["state"], np.float32)[:6]
        return dict(
            normalized=value,
            targets=targets,
            prefix_ms=prefix_ms,
            input_cast_ms=input_cast_ms,
            prefix_upload=prefix_upload,
            suffix_steps_ms=[],
            inference_ms=inference_ms,
            guidance_steps_ms=self.guidance_times,
            vjp_execute_steps_ms=self.vjp_execute_times,
            shared_forward_vjp_steps_ms=self.vjp_execute_times,
            action_timing_scope="Ten shared FP16 forward+VJP calls including NPU FP32 Euler and per-step finite guards",
            prefix_projection_ms=0.0,
        )
