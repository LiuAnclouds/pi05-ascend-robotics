"""Frozen OM plus forward-only joint-prefix inpainting, NOT full RTC VJP.

Uses the same ten native denoising forwards and weights, without gradients or
training. At flow time t, constrain the prefix towards t*noise+(1-t)*old_target.
This is a deliberately labelled approximation: soft overwriting is not the
input-Jacobian correction used by full RTC. No physical safety is implied.
"""

import time

import numpy as np

from inference.core.backends.om_policy import OMPolicy
from inference.core.rtc.guidance_math import prefix_weights


class PrefixProjectionPolicy(OMPolicy):
    def __init__(self, *args, **kwargs):
        """Initialize the base OM pair and mark this approximate projection mode.

        Args: OMPolicy constructor arguments.
        Returns: None; no extra forward or weight update is introduced.
        """
        super().__init__(*args, **kwargs)
        self.metadata.update(
            prefix_conditioning="forward_only_flow_prefix_projection_not_full_RTC",
            extra_model_forwards=0,
            parameter_updates=0,
            gripper_conditioned=False,
            extra_committed_joint_points=3,
        )

    def prepare_guidance(self, payload):
        """Build six-joint overlap weights from a complete old action reference.

        Args: Request with rtc_prefix, overlap and delay, or no RTC input.
        Returns: None; gripper and padded dimensions receive no projection.
        """
        self.projection_ms = 0.0
        self.old_prefix, self.weights = None, None
        if "rtc_prefix" not in payload:
            return
        prefix = np.asarray(payload["rtc_prefix"], np.float32)
        if prefix.shape != self.noise.shape or not np.isfinite(prefix).all():
            raise ValueError("Expected a finite normalized [1,50,32] old prefix")
        overlap, delay = int(payload["prefix_overlap_steps"]), int(payload["prefix_delay_steps"])
        if overlap < delay + 3:
            raise ValueError("Insufficient old prefix for three committed handover samples")
        temporal = prefix_weights(delay + 3, overlap)
        self.weights = np.zeros_like(self.noise)
        self.weights[0, :, :6] = temporal[:, None]
        self.old_prefix = prefix.copy()

    def project_state(self, value, step):
        """Project the current noise state toward the frozen old joint prefix.

        Args: Current normalized [1,50,32] state and denoising index.
        Returns: Projected state, or the original array when projection is disabled.
        """
        if self.old_prefix is None:
            return value
        started = time.perf_counter()
        t = np.float32(1 - step / 10)
        old_flow_state = t * self.noise + (1 - t) * self.old_prefix
        projected = value + self.weights * (old_flow_state - value)
        self.projection_ms += (time.perf_counter() - started) * 1000
        return projected
