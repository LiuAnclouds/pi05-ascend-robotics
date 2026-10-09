#!/usr/bin/env python3
"""Prepare one real Piper sample for the official OpenPI Pi0.5 graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from include.project_paths import DEFAULT_PART1_INPUT, DEFAULT_STATS, DEFAULT_TOKENIZER
from runtime.preprocessing import encode_task_state


def resize_with_pad(image, size=224):
    """Resize an HWC image with centered padding and OpenPI normalization.

    Args:
        image: NumPy uint8 image in HWC layout.
        size: Target square edge length.

    Returns:
        A normalized PyTorch tensor shaped ``[1, 3, size, size]``.
    """
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    h, w = x.shape[-2:]
    ratio = max(w / size, h / size)
    rh, rw = max(1, int(h / ratio)), max(1, int(w / ratio))
    x = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False)
    ph, pw = size - rh, size - rw
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=0.0) * 2.0 - 1.0


def main() -> None:
    """Convert one recorded Piper sample into a Part1 input tensor file.

    Args:
        None. The sample, tokenizer, and output paths come from the CLI.

    Returns:
        None. A ``.pt`` file and a short JSON summary are produced.
    """
    import numpy as np
    import sentencepiece
    import torch

    parser = argparse.ArgumentParser(
        description="Convert one recorded Piper sample into official Part1 tensors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample", type=Path, required=True,
                        help="Recorded sample .npz containing images, state, and prompt.")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER,
                        help="SentencePiece tokenizer.model used by the checkpoint.")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS,
                        help="Normalization statistics JSON used for the Piper state.")
    parser.add_argument("--output", type=Path, default=DEFAULT_PART1_INPUT,
                        help="Destination Part1 tensor file (.pt).")
    args = parser.parse_args()

    sample = np.load(args.sample, allow_pickle=True)
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    state = np.asarray(sample["state"], dtype=np.float32)
    s = stats["norm_stats"]["state"]
    q01 = np.asarray(s["q01"], dtype=np.float32)
    q99 = np.asarray(s["q99"], dtype=np.float32)
    state_norm = np.clip(2.0 * (state - q01) / np.maximum(q99 - q01, 1e-8) - 1.0, -1.0, 1.0)

    prompt = str(sample["prompt"].item()).strip().replace("_", " ").replace("\n", " ")
    sp = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer))
    tokens, token_mask = encode_task_state(sp, prompt, state, q01, q99)
    tokens, token_mask = torch.from_numpy(tokens), torch.from_numpy(token_mask)

    image0 = resize_with_pad(np.asarray(sample["image"], dtype=np.uint8))
    image1 = resize_with_pad(np.asarray(sample["wrist_image"], dtype=np.uint8))
    image2 = torch.zeros_like(image1)
    mask0 = torch.ones((1,), dtype=torch.bool)
    mask1 = torch.ones((1,), dtype=torch.bool)
    mask2 = torch.zeros((1,), dtype=torch.bool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image0": image0,
            "image1": image1,
            "image2": image2,
            "mask0": mask0,
            "mask1": mask1,
            "mask2": mask2,
            "tokens": tokens,
            "token_mask": token_mask,
            "meta": {"prompt": prompt, "state_raw": state.tolist(), "state_normalized": state_norm.tolist()},
        },
        args.output,
    )
    print(json.dumps({"output": str(args.output), "prompt": prompt, "token_count": int(token_mask.sum())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
