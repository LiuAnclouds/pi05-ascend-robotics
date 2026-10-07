"""Shared official-model construction and Ascend graph adaptations."""

from __future__ import annotations

import types
from pathlib import Path

import torch

from include.export_defs import DEFAULT_ACTION_DIM, DEFAULT_ACTION_HORIZON, add_official_source


def patch_ascend_rotary(model: torch.nn.Module) -> None:
    """Patch rotary embeddings to use the validated Ascend310P1 arithmetic.

    Args:
        model: Loaded PyTorch Pi0.5 model whose rotary modules are patched.

    Returns:
        None. Matching rotary modules are modified in place.
    """
    def rotary_forward(self, x, position_ids):
        """Compute FP32 sine/cosine rotary factors and cast them to input dtype.

        Args:
            x: Rotary input tensor; used to obtain the target device and dtype.
            position_ids: Integer token positions with shape ``[batch, length]``.

        Returns:
            A ``(cos, sin)`` tuple matching the model's rotary embedding API.
        """
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        positions = position_ids[:, None, :].float()
        freqs = (inv_freq @ positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        scale = getattr(self, "attention_scaling", 1.0)
        sin = torch.sin(emb) * scale
        cos = torch.sin(emb + torch.pi / 2) * scale
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    for module in model.modules():
        if hasattr(module, "inv_freq") and module.__class__.__name__.endswith("RotaryEmbedding"):
            module.forward = types.MethodType(rotary_forward, module)


def build_pi05_model(weights: Path, dtype: str):
    """Load the official Pi0.5 checkpoint and apply export-only adaptations.

    Args:
        weights: Directory containing ``model.safetensors``.
        dtype: Export arithmetic dtype, either ``float32`` or ``float16``.

    Returns:
        An evaluation-mode ``PI0Pytorch`` model ready for graph export.
    """
    add_official_source()
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from safetensors.torch import load_model
    model = PI0Pytorch(Pi0Config(pi05=True, action_dim=DEFAULT_ACTION_DIM, action_horizon=DEFAULT_ACTION_HORIZON,
                                 dtype="float32", pytorch_compile_mode=None))
    load_model(model, str(weights / "model.safetensors"))
    if dtype == "float16":
        model.half()
    patch_ascend_rotary(model)
    model.eval()
    return model
