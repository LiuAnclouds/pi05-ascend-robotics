"""One NPU FP16 action forward shared with its RTC input pullback; no STE."""

import time
from pathlib import Path

import numpy as np

from inference.core.backends.om_policy import OMPolicy
from inference.core.rtc.guidance_math import (
    AUTHOR_RTC_COMMIT,
    REFERENCE_MAX_GUIDANCE,
    guidance_scale,
)
from inference.core.rtc.partitioned_vjp import PartitionedVJP
from inference.include.definitions import COTANGENT_SCALE, DEFAULT_SEED


class SharedForwardPolicy(OMPolicy):
    def __init__(
        self,
        root=None,
        pair_directory=None,
        action_q01=None,
        action_q99=None,
        guidance_directory=None,
        max_guidance=REFERENCE_MAX_GUIDANCE,
        vjp_execution_mode="single-stream",
        *,
        part1_om=None,
        part2_om=None,
        seed=DEFAULT_SEED,
        noise_file=None,
    ):
        """Load a frozen action forward/VJP graph beside the selected model pair.

        Args: Asset root, pair or explicit OM paths, action quantiles, VJP
            directory, stream mode, and optional noise seed/reference archive.
        Returns: None; this object owns the resident pair and VJP resources.
        """
        guidance_scale(1.0, max_guidance)
        if guidance_directory is None:
            raise ValueError("Provide the matching compiled RTC guidance directory")
        self.gradient = None
        self.max_guidance = max_guidance
        self.cotangent_scale = np.float32(COTANGENT_SCALE)
        self.guidance_active = False
        self.guidance_times = []
        self.vjp_execute_times = []
        super().__init__(
            root,
            pair_directory,
            action_q01,
            action_q99,
            part1_om=part1_om,
            part2_om=part2_om,
            seed=seed,
            noise_file=noise_file,
        )
        try:
            bindings = {
                name: (self.prefix.addresses["outputs"][index], self.prefix.io["outputs"][index])
                for index, name in enumerate(("past_kv_tensor", "prefix_pad_masks"))
            }
            self.gradient = PartitionedVJP(
                self.prefix.acl, Path(guidance_directory), bindings, execution_mode=vjp_execution_mode
            )
            self.gradient.telemetry.update(
                derivative="Ordinary frozen-FP16 endpoint input VJP, no STE",
                precision_mode="origin",
                shared_forward=True,
            )
            self.metadata.update(
                quantization="vision/language W8 dynamic A8; action original W16A16",
                rtc=dict(
                    author_commit=AUTHOR_RTC_COMMIT,
                    max_guidance=max_guidance,
                    prior_std=1.0,
                    parameter_updates=0,
                    activation_ste=False,
                    gripper_reference="exact predicted endpoint",
                    forward_recomputed_for_vjp=False,
                    native_velocity_input=False,
                    internal_cotangent_scale=float(self.cotangent_scale),
                    cotangent_unscaled_after_vjp=True,
                    partitions=self.gradient.telemetry,
                ),
            )
        except BaseException:
            self.close()
            raise

    def prepare_guidance(self, payload):
        """Validate and stage this request's complete RTC reference tensors.

        Args: Request mapping with both rtc_prefix/rtc_weights or neither.
        Returns: None; resets per-request guidance timings.
        """
        self.guidance_times, self.vjp_execute_times = [], []
        state = np.asarray(payload["state"], np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("Expected a finite observation state [7]")
        self.guidance_active = "rtc_prefix" in payload
        if self.guidance_active != ("rtc_weights" in payload):
            raise ValueError("RTC prefix and weights must be paired")
        if not self.guidance_active:
            return
        prefix, weights = (np.asarray(payload[name], np.float32) for name in ("rtc_prefix", "rtc_weights"))
        if any(v.shape != (1, 50, 32) or not np.isfinite(v).all() for v in (prefix, weights)):
            raise ValueError("Invalid complete RTC prefix/weight tensors")
        if np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("RTC temporal weights must lie in [0,1]")
        lower = upper = prefix
        self.gradient.feed(
            dict(prefix_lower=lower, prefix_upper=upper, weights=weights * self.cotangent_scale)
        )

    def predict(self, payload):
        """Run the ten action steps and one shared forward/VJP per step.

        Args: Named prefix inputs, model-unit state and optional RTC tensors.
        Returns: Normalized action chunk, absolute targets and timing breakdown.
        """
        self.prepare_guidance(payload)
        if not self.guidance_active:
            return super().predict(payload)
        input_started = time.monotonic()
        values = self.prefix_values(payload["inputs"])
        started = time.monotonic()
        input_cast_ms = (started - input_started) * 1000.0
        prefix_upload = self.prefix.feed(dict(enumerate(values)))
        prefix_ms = self.prefix.run()
        value = self.noise.copy()
        for step in range(10):
            step_started = time.monotonic()
            timestep = np.float32(1 - step / 10)
            self.gradient.feed(
                dict(
                    noise=value,
                    step_index=np.asarray([step], np.int64),
                    timestep=np.asarray([timestep], np.float32),
                )
            )
            self.vjp_execute_times.append(self.gradient.run())
            velocity = self.gradient.read("base_velocity").astype(np.float32)
            correction = self.gradient.read("input_correction").astype(np.float32) / self.cotangent_scale
            guided = velocity - np.float32(guidance_scale(float(timestep), self.max_guidance)) * correction
            if (
                velocity.shape != (1, 50, 32)
                or not np.isfinite(guided).all()
                or not np.isfinite(correction).all()
            ):
                raise RuntimeError("Invalid shared-forward RTC action/gradient at step " + str(step))
            value = value + np.float32(-0.1) * guided
            self.guidance_times.append((time.monotonic() - step_started) * 1000)
        inference_ms = (time.monotonic() - started) * 1000
        if not np.isfinite(value).all():
            raise RuntimeError("Nonfinite complete RTC action chunk")
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
            action_timing_scope="Ten combined FP16 forward+VJP calls; no separate native action forward",
            prefix_projection_ms=0.0,
        )

    def close(self):
        """Release the VJP pipeline before its borrowed prefix buffers.

        Args: None. Returns: None; repeated calls are safe.
        """
        if self.gradient is not None:
            self.gradient.close()
            self.gradient = None
        super().close()
