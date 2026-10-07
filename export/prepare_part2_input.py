#!/usr/bin/env python3
"""Prepare deterministic official Pi0.5 Part2 inputs from a real sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from include.project_paths import DEFAULT_PART1_INPUT, DEFAULT_PART2_INPUT, DEFAULT_WEIGHTS

def parse_args() -> argparse.Namespace:
    """Parse Part2 input preparation arguments.

    Returns:
        Parsed weights, Part1 input, and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Prepare deterministic Part2 tensors from a Part1 input file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS,
                        help='Checkpoint directory containing model.safetensors.')
    parser.add_argument('--part1-input', type=Path, default=DEFAULT_PART1_INPUT,
                        help='Prepared Part1 tensor file (.pt).')
    parser.add_argument('--output', type=Path, default=DEFAULT_PART2_INPUT,
                        help='Destination Part2 tensor file (.pt).')
    return parser.parse_args()


def make_att_2d_masks(pad_masks, att_masks):
    """Build the official two-dimensional causal/padding attention mask.

    Args:
        pad_masks: Boolean valid-token mask.
        att_masks: One-dimensional causal attention mask.

    Returns:
        Boolean two-dimensional attention mask.
    """
    import torch

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


def main() -> None:
    """Prepare deterministic Part2 tensors from a saved Part1 input.

    Returns:
        None. The prepared tensor file and a JSON summary are written.
    """
    args = parse_args()
    import torch
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'openpi' / 'src'))
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from safetensors.torch import load_model

    config = Pi0Config(pi05=True, action_dim=32, action_horizon=50, dtype='float32', pytorch_compile_mode=None)
    model = PI0Pytorch(config)
    load_model(model, str(args.weights / 'model.safetensors'))
    model.eval()
    sample = torch.load(args.part1_input, map_location='cpu', weights_only=False)
    images = [sample['image0'].float(), sample['image1'].float(), sample['image2'].float()]
    image_masks = [sample['mask0'].bool(), sample['mask1'].bool(), sample['mask2'].bool()]
    tokens = sample['tokens'].long()
    token_mask = sample['token_mask'].bool()
    with torch.inference_mode():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, image_masks, tokens, token_mask)
        prefix_att_2d = model._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = 'eager'
        _, cache = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=torch.cumsum(prefix_pad_masks, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        legacy = cache.to_legacy_cache()
        past_kv = torch.cat([tensor.float() for pair in legacy for tensor in pair], dim=0)
    if not torch.isfinite(past_kv).all():
        raise RuntimeError('official Part1 cache is non-finite')
    generator = torch.Generator(device='cpu').manual_seed(0)
    noise = torch.randn((1, 50, 32), generator=generator, dtype=torch.float32)
    timestep = torch.tensor([0.5], dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'past_kv': past_kv,
        'prefix_pad_masks': prefix_pad_masks.bool(),
        'noise': noise,
        'timestep': timestep,
        'meta': {'framework': 'official_openpi', 'model': 'pi05', 'seed': 0, 'timestep': 0.5},
    }, args.output)
    print(json.dumps({'output': str(args.output), 'past_kv': list(past_kv.shape), 'noise': list(noise.shape)}, indent=2))


if __name__ == '__main__':
    main()
