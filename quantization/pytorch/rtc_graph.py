"""Inference-time RTC guidance for the official OpenPI denoiser.

Equations follow Physical-Intelligence/real-time-chunking-kinetix,
src/model.py, FlowPolicy.realtime_action (9296f31d62d5). OpenPI integrates
noise time from 1 to 0, reversing the velocity/guidance signs in that source.
No training, weight gradients, or LeRobot runtime are required.
"""

from __future__ import annotations

import torch

from quantization.include.definitions import DEFAULT_RTC_MAX_GUIDANCE as REFERENCE_MAX_GUIDANCE


def endpoint_input_vjp(denoiser, x, timestep, prefix, weights, *, native_velocity=None, prefix_upper=None):
    """One differentiable forward reused by its complete endpoint pullback.

    With native_velocity=None this is the author's full VJP, with OpenPI's
    reversed time convention. An optional external native velocity changes
    only the error's endpoint: its Jacobian is still the supplied denoiser's
    Jacobian and MUST be labelled a surrogate for a separate quantized OM.
    When prefix_upper is supplied, prefix is the lower endpoint bound. The
    cotangent is the negative gradient of squared distance to that interval.
    Equal bounds reproduce the ordinary exact-target RTC loss.
    """
    t = timestep.float().reshape(-1, 1, 1)

    def endpoint(action):
        """Estimate clean action from one noise state; return endpoint and velocity."""
        velocity = denoiser(action)
        return action.float() - t * velocity.float(), velocity

    clean, pullback, velocity = torch.func.vjp(endpoint, x, has_aux=True)
    if native_velocity is not None:
        clean = x.float() - t * native_velocity.float()
    reference = prefix.float()
    if prefix_upper is not None:
        reference = torch.minimum(torch.maximum(clean, reference), prefix_upper.float())
    error = ((reference - clean) * weights.float()).detach()
    return pullback(error)[0].float(), velocity


def guided_velocity(
    denoiser, x, timestep, prefix, weights, max_guidance=REFERENCE_MAX_GUIDANCE, *, prefix_upper=None
):
    """Return velocity and full VJP correction at one denoising timestep.

    Args:
        denoiser: Differentiable callback x -> velocity, with cache/observation fixed.
        x: Current normalized noise/action tensor [1, 50, 32].
        timestep: Noise level [1], integrated from 1 towards 0.
        prefix: Previous physical targets expressed in the NEW observation's
            normalized delta coordinates, padded to x.shape.
        weights: Same shape as x; zero outside the overlapping time/real dimensions.
        max_guidance: Upper bound on the paper's time-dependent guidance scale.
        prefix_upper: Optional upper bounds; prefix supplies the matching lower
            bounds for execution-equivalent references.

    Returns:
        (guided velocity, ordinary velocity, VJP correction), shaped like x.
        Gradients include the denoiser Jacobian. Only treating velocity as a
        constant would reduce this to a different, approximate algorithm.
    """
    t = timestep.float().reshape(-1, 1, 1)

    correction, velocity = endpoint_input_vjp(
        denoiser, x, timestep, prefix, weights, prefix_upper=prefix_upper
    )
    scale = torch.clamp(
        (t.square() + (1.0 - t).square()) / torch.clamp(t * (1.0 - t), min=1e-6),
        max=max_guidance,
    )
    return velocity.float() - scale * correction, velocity, correction


class RTCGraph(torch.nn.Module):
    """Wrap DenoiseGraph for ahead-of-time capture of the RTC input gradient."""

    def __init__(self, denoiser):
        """Store a frozen official Part2 graph; accept the normal graph instance."""
        super().__init__()
        self.denoiser = denoiser
        self.denoiser.requires_grad_(False)

    def forward(self, past_kv, prefix_pad_masks, noise, timestep, prefix_lower, prefix_upper, weights):
        """Export RTC velocities/correction for normalized reference intervals.

        Args: Frozen observation cache/mask, current noise/time, lower/upper
            executable endpoint bounds and temporal weights.
        Returns: Guided velocity, base velocity and the full input VJP.
        """
        return guided_velocity(
            lambda x: self.denoiser(past_kv, prefix_pad_masks, x, timestep),
            noise,
            timestep,
            prefix_lower,
            weights,
            prefix_upper=prefix_upper,
        )
