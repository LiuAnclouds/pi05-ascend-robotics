"""Export graph wrappers kept separate from command-line entry points."""

from __future__ import annotations

import torch


class DenoiseGraph(torch.nn.Module):
    """Expose one official OpenPI action-expert denoising step."""
    def __init__(self, model):
        """Create a one-step action-expert graph.

        Args:
            model: Loaded official ``PI0Pytorch`` model.

        Returns:
            None. The wrapped model is stored on the graph instance.
        """
        super().__init__(); self.model = model
    def __repr__(self):
        """Return a stable human-readable graph name.

        Returns:
            The graph name used in diagnostics and exporter messages.
        """
        return "DenoiseGraph(official_openpi_pi05)"
    def forward(self, past_kv, prefix_pad_masks, noise, timestep):
        """Run one official action-expert denoising step.

        Args:
            past_kv: Flattened Part1 key/value cache tensor.
            prefix_pad_masks: Boolean valid-token mask from Part1.
            noise: Current normalized action state.
            timestep: Current diffusion timestep.

        Returns:
            The predicted action velocity tensor for the current step.
        """
        from transformers.cache_utils import DynamicCache
        legacy = tuple((past_kv[2 * i].unsqueeze(0), past_kv[2 * i + 1].unsqueeze(0)) for i in range(18))
        cache = DynamicCache.from_legacy_cache(legacy)
        return self.model.denoise_step(
            state=torch.zeros((noise.shape[0], 32), dtype=noise.dtype, device=noise.device),
            prefix_pad_masks=prefix_pad_masks, past_key_values=cache, x_t=noise, timestep=timestep)


class ManualPrefixGraph(torch.nn.Module):
    """Explicit Gemma-layer Part1 graph used by the high-performance candidate."""
    def __init__(self, model):
        """Create the explicit-layer Part1 graph.

        Args:
            model: Loaded official Pi0.5 model.

        Returns:
            None. The wrapped model is stored on the graph instance.
        """
        super().__init__(); self.model = model
    def __repr__(self):
        """Return a stable human-readable graph name.

        Returns:
            The graph name used in diagnostics and exporter messages.
        """
        return "ManualPrefixGraph(official_openpi_pi05)"
    def forward(self, image0, image1, image2, mask0, mask1, mask2, tokens, token_mask):
        """Run explicit Gemma layers and return the Part1 KV cache.

        Args:
            image0, image1, image2: Three normalized image tensors.
            mask0, mask1, mask2: Boolean availability masks for the images.
            tokens: Token IDs for the task/state prompt.
            token_mask: Boolean validity mask for ``tokens``.

        Returns:
            A tuple ``(past_kv_tensor, prefix_pad_masks)`` for Part2.
        """
        from transformers.models.gemma import modeling_gemma
        model = self.model
        embs, pad, _ = model.embed_prefix([image0, image1, image2], [mask0, mask1, mask2], tokens, token_mask)
        position_ids = torch.cumsum(pad.to(torch.int64), dim=1) - 1
        pad_float = pad.to(torch.float32)
        valid = pad_float[:, None, :] * pad_float[:, :, None]
        attention_mask = ((1.0 - valid) * (-2.3819763e38))[:, None, :, :]
        language_model = model.paligemma_with_expert.paligemma.language_model
        hidden, cache = embs, []
        for layer in language_model.layers:
            residual = hidden
            hidden, gate = layer.input_layernorm(hidden, None)
            hidden_shape = (*hidden.shape[:-1], -1, layer.self_attn.head_dim)
            q = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
            k = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
            v = layer.self_attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
            dummy = torch.zeros(q.shape[0], q.shape[2], q.shape[-1], device=q.device, dtype=q.dtype)
            cos, sin = language_model.rotary_emb(dummy, position_ids)
            q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin)
            cache.append((k, v))
            att_out, _ = modeling_gemma.eager_attention_forward(layer.self_attn, q, k, v, attention_mask, scaling=layer.self_attn.scaling)
            att_out = layer.self_attn.o_proj(att_out.reshape(hidden.shape[0], hidden.shape[1], -1).contiguous())
            hidden = modeling_gemma._gated_residual(residual, att_out, gate)
            residual = hidden
            hidden, gate = layer.post_attention_layernorm(hidden, None)
            hidden = layer.mlp(hidden)
            hidden = modeling_gemma._gated_residual(residual, hidden, gate)
        return torch.cat([x for pair in cache for x in pair], dim=0), pad
