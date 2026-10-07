#!/usr/bin/env python3
"""Runtime adapter for the verified official Pi0.5 Part1/Part2 OM pair."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ATTENTION_MASK_VALUE = -2.3819763e38


def resize_with_pad(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize, pad, normalize, and transpose one camera image.

    Args:
        image: HWC uint8 BGR or RGB image.
        size: Target square image edge length.

    Returns:
        Contiguous FP16 tensor shaped ``[1, 3, size, size]``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB/BGR image, got {image.shape}")
    height, width = image.shape[:2]
    scale = max(width / size, height / size)
    resized_width = max(1, int(width / scale))
    resized_height = max(1, int(height / scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return np.ascontiguousarray((canvas.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None].astype(np.float16))


def delta_actions_to_absolute(action_delta: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Add current Piper joint angles to six deltas; keep the gripper absolute."""
    delta = np.asarray(action_delta, dtype=np.float32)
    current = np.asarray(state, dtype=np.float32)
    if delta.ndim < 2 or delta.shape[-1] != 7:
        raise ValueError(f"expected actions ending in seven values, got {delta.shape}")
    if current.shape != (7,):
        raise ValueError(f"expected seven current Piper state values, got {current.shape}")
    if not np.isfinite(delta).all() or not np.isfinite(current).all():
        raise ValueError("action deltas and current Piper state must be finite")
    absolute = delta.copy()
    absolute[..., :6] += current[:6]
    return absolute


class OfficialOMPolicy:
    """Own ACL resources and execute the verified official Pi0.5 OM pair."""

    def __init__(self, part1_om: Path, part2_om: Path, tokenizer: Path, stats: Path, steps: int = 10):
        """Initialize ACL resources, OM handles, tokenizer, and normalization.

        Args:
            part1_om: Compiled Part1 prefix OM path.
            part2_om: Compiled Part2 denoising OM path.
            tokenizer: SentencePiece tokenizer model path.
            stats: JSON normalization statistics path.
            steps: Number of Part2 denoising steps per prediction.

        Returns:
            None. ACL model resources are allocated for this policy instance.
        """
        if steps <= 0:
            raise ValueError("steps must be positive")
        import sentencepiece
        from acllite.acllite_model import AclLiteModel
        from acllite.acllite_resource import AclLiteResource

        self.steps = steps
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
        payload = json.loads(stats.read_text(encoding="utf-8"))
        self.state_q01 = np.asarray(payload["norm_stats"]["state"]["q01"], dtype=np.float32)
        self.state_q99 = np.asarray(payload["norm_stats"]["state"]["q99"], dtype=np.float32)
        self.action_q01 = np.asarray(payload["norm_stats"]["actions"]["q01"], dtype=np.float32)
        self.action_q99 = np.asarray(payload["norm_stats"]["actions"]["q99"], dtype=np.float32)
        if self.state_q01.shape != (7,) or self.action_q01.shape != (7,):
            raise ValueError("official Piper normalization statistics must have seven dimensions")
        self.resource = AclLiteResource()
        self.resource.init()
        self.part1 = AclLiteModel(str(part1_om))
        self.part2 = AclLiteModel(str(part2_om))

    def close(self) -> None:
        """Release model handles and ACL resources.

        Args:
            None.

        Returns:
            None. References are cleared so ACL cleanup can run.
        """
        self.part1 = None
        self.part2 = None
        self.resource = None

    def _tokens(self, task: str, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build the exact Task/State/Action token prompt.

        Args:
            task: Natural-language task instruction.
            state: Seven-dimensional raw Piper state.

        Returns:
            ``(tokens, token_mask)`` arrays shaped ``[1, 200]``.
        """
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (7,):
            raise ValueError(f"expected seven Piper state values, got {state.shape}")
        normalized = np.clip(
            2.0 * (state - self.state_q01) / np.maximum(self.state_q99 - self.state_q01, 1e-8) - 1.0,
            -1.0,
            1.0,
        )
        discrete = np.digitize(normalized, np.linspace(-1.0, 1.0, 257)[:-1]).astype(np.int64) - 1
        clean_task = str(task).strip().replace("_", " ").replace("\n", " ")
        text = f"Task: {clean_task}, State: {' '.join(map(str, discrete))};\nAction: "
        ids = list(self.tokenizer.encode(text, add_bos=True))[:200]
        tokens = np.zeros((1, 200), dtype=np.int64)
        token_mask = np.zeros((1, 200), dtype=np.bool_)
        tokens[0, : len(ids)] = np.asarray(ids, dtype=np.int64)
        token_mask[0, : len(ids)] = True
        return tokens, token_mask

    def make_inputs(self, third_person: np.ndarray, wrist: np.ndarray, state: np.ndarray, task: str) -> list[np.ndarray]:
        """Prepare camera frames and Piper state as Part1 OM inputs.

        Args:
            third_person: HWC uint8 third-person camera image.
            wrist: HWC uint8 wrist-camera image.
            state: Seven-dimensional raw Piper state.
            task: Natural-language task instruction.

        Returns:
            Eight arrays in the exact order expected by the Part1 OM.
        """
        tokens, token_mask = self._tokens(task, state)
        image0 = resize_with_pad(third_person)
        image1 = resize_with_pad(wrist)
        image2 = np.zeros_like(image1)
        return [
            image0,
            image1,
            image2,
            np.ones((1,), dtype=np.bool_),
            np.ones((1,), dtype=np.bool_),
            np.zeros((1,), dtype=np.bool_),
            tokens,
            token_mask,
        ]

    def predict_inputs(
        self, inputs: list[np.ndarray], seed: int = 0, state: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Run Part1 once and Part2 for the configured denoising steps.

        Args:
            inputs: Eight Part1 OM input arrays.
            seed: Random seed used to initialize the normalized action noise.
            state: Current seven-dimensional Piper state for converting joint deltas to targets.

        Returns:
            Normalized actions, denormalized deltas, optional absolute targets, timings, and cache data.
        """
        if len(inputs) != 8:
            raise ValueError(f"expected eight Part1 inputs, got {len(inputs)}")
        part1_started = time.perf_counter()
        prefix_result = self.part1.execute(inputs)
        part1_ms = (time.perf_counter() - part1_started) * 1000.0
        past_kv = np.ascontiguousarray(np.asarray(prefix_result[0], dtype=np.float16))
        prefix_pad_masks = np.ascontiguousarray(np.asarray(prefix_result[1], dtype=np.bool_))
        rng = np.random.default_rng(seed)
        action = np.ascontiguousarray(rng.standard_normal((1, 50, 32), dtype=np.float32).astype(np.float16))
        dt = np.float16(-1.0 / self.steps)
        part2_ms: list[float] = []
        for index in range(self.steps):
            timestep = np.asarray([1.0 - index / self.steps], dtype=np.float16)
            started = time.perf_counter()
            velocity = np.asarray(
                self.part2.execute(
                    [past_kv, prefix_pad_masks, action, np.ascontiguousarray(timestep)]
                )[0],
                dtype=np.float16,
            )
            part2_ms.append((time.perf_counter() - started) * 1000.0)
            action = np.ascontiguousarray(action + dt * velocity)
        action_normalized = action.astype(np.float32)
        # The first six de-normalized values are joint deltas; the gripper stays absolute.
        action_delta = (
            (action_normalized[..., :7] + 1.0)
            * 0.5
            * (self.action_q99 - self.action_q01)
            + self.action_q01
        )
        action_target = delta_actions_to_absolute(action_delta, state) if state is not None else None
        action_raw = action_target if action_target is not None else action_delta
        return {
            "action_normalized": action_normalized,
            "action_delta": action_delta.astype(np.float32),
            "action_target": None if action_target is None else action_target.astype(np.float32),
            "action_raw": action_raw.astype(np.float32),
            "part1_ms": part1_ms,
            "part2_ms": part2_ms,
            "past_kv": past_kv,
            "prefix_pad_masks": prefix_pad_masks,
        }
