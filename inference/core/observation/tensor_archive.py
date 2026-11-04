"""Saved tensor input loading for repeatable OM validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


def load_saved_inputs(path: Path) -> list[np.ndarray]:
    """Load the eight Part1 tensors in the exact OM input order.

    Args:
        path: Saved ``.pt`` file produced by the input preparation command.

    Returns:
        A list of contiguous NumPy arrays matching the Part1 OM input dtypes.
    """
    if torch is None:
        raise RuntimeError("Install the PyTorch export dependencies to load .pt inputs")
    data = torch.load(path, map_location="cpu", weights_only=False)
    return [
        np.ascontiguousarray(data[key].numpy().astype(dtype))
        for key, dtype in (
            ("image0", np.float16),
            ("image1", np.float16),
            ("image2", np.float16),
            ("mask0", np.bool_),
            ("mask1", np.bool_),
            ("mask2", np.bool_),
            ("tokens", np.int64),
            ("token_mask", np.bool_),
        )
    ]
