"""RTC prefix preparation, independent of ACL, cameras and robot commands.

The prefix is aligned at observation time. Positions already scheduled during
inference receive strong guidance; later shared positions fade to zero weight.
All model calls still predict 50 positions. See Physical Intelligence's
real-time-chunking-kinetix src/model.py, get_prefix_weights/realtime_action.
"""

from __future__ import annotations

import math

import numpy as np

AUTHOR_RTC_COMMIT = "9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b"
REFERENCE_MAX_GUIDANCE = 5.0
REFERENCE_PRIOR_STD = 1.0
REFERENCE_MIN_EXECUTION_HORIZON = 25
REFERENCE_DELAY_BUFFER = 10


def guidance_scale(
    noise_time: float, max_guidance: float = REFERENCE_MAX_GUIDANCE, prior_std: float = REFERENCE_PRIOR_STD
) -> float:
    """Paper Eq. 2/4 with OpenPI noise time t=1-tau, integrated 1 -> 0.

    The scalar expression is shared by the native-OM adapters; it is not a
    replacement for the endpoint VJP. beta=5 matches the paper's real-world
    baseline, not a universal optimum. prior_std=1 is the original RTC prior;
    an explicitly supplied smaller value is the conditional-prior extension
    described by Soare (2025), not a change to model weights or precision.
    No denoiser call is made at t=0.
    """
    if (
        not math.isfinite(noise_time)
        or not 0 <= noise_time <= 1
        or not math.isfinite(max_guidance)
        or max_guidance <= 0
        or not math.isfinite(prior_std)
        or prior_std <= 0
    ):
        raise ValueError("Expected time in [0,1], a positive finite guidance bound and prior std")
    if noise_time in (0.0, 1.0):
        return float(max_guidance)  # continuous clipped endpoint limit
    variance = prior_std**2
    return min(
        float(max_guidance),
        (noise_time**2 + variance * (1 - noise_time) ** 2) / (variance * noise_time * (1 - noise_time)),
    )


def prefix_weights(delay: int, overlap: int, horizon: int = 50) -> np.ndarray:
    """Return exponential RTC time weights [horizon] for committed/soft prefix.

    Args: delay is the number of already committed control slots; overlap is
        the number of real old targets available from the NEW observation.
    Returns: ones before delay, exponential taper until overlap, then zeros.
        No prefix is invented beyond the old trajectory.
    """
    if not 0 <= overlap <= horizon or delay < 0:
        raise ValueError("RTC requires nonnegative delay and 0 <= overlap <= horizon")
    # The author's get_prefix_weights explicitly makes end take precedence
    # over start. A late previous reply can leave fewer than d old points;
    # constrain only the available prefix, never invent repeated endpoints.
    delay = min(delay, overlap)
    index = np.arange(horizon, dtype=np.float32)
    linear = np.clip((delay - 1 - index) / (overlap - delay + 1) + 1, 0, 1)
    weights = linear * np.expm1(linear) / np.expm1(np.float32(1))
    weights[overlap:] = 0
    return weights.astype(np.float32)


def prepare_prefix(targets, observation_state, action_q01, action_q99, delay):
    """Express overlapping physical targets in the new observation's coordinates.

    Args:
        targets: Old absolute Piper targets [overlap, 7], aligned to the first
            command slot at/after the new observation; degrees and millimetres.
        observation_state: New image/state observation's seven raw values.
        action_q01, action_q99: The checkpoint's seven action quantiles.
        delay: Control slots committed to old targets while inference runs.
    Returns:
        (prefix, weights), each FP32 [1, 50, 32]. Unused 25 dimensions receive
        zero weight; clipping is intentionally avoided for physical continuity.
    """
    targets = np.asarray(targets, dtype=np.float32)
    state = np.asarray(observation_state, dtype=np.float32)
    low, high = np.asarray(action_q01, dtype=np.float32), np.asarray(action_q99, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 7 or state.shape != (7,):
        raise ValueError("Expected RTC targets [overlap, 7] and observation state [7]")
    if low.shape != (7,) or high.shape != (7,) or np.any(high <= low):
        raise ValueError("RTC requires seven non-degenerate action quantile intervals")
    if not all(np.isfinite(a).all() for a in (targets, state, low, high)):
        raise ValueError("RTC prefix inputs must be finite")
    temporal = prefix_weights(delay, len(targets))
    delta = targets.copy()
    delta[:, :6] -= state[:6]
    prefix = np.zeros((1, 50, 32), dtype=np.float32)
    prefix[0, : len(targets), :7] = 2 * (delta - low) / (high - low) - 1
    weights = np.zeros_like(prefix)
    weights[0, :, :7] = temporal[:, None]
    return prefix, weights
