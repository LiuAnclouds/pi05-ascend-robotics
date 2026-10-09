"""Pure state/prompt preprocessing shared by sample preparation and inference."""

import numpy as np


def encode_task_state(tokenizer, task: str, state: np.ndarray,
                      q01: np.ndarray, q99: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the trained Task/State/Action token format without model or device access.

    Args:
        tokenizer: SentencePiece processor matching the checkpoint.
        task: Natural-language task instruction.
        state: Seven raw Piper values: six joint angles (degrees), gripper (mm).
        q01: Training state 1st-percentile values, shape [7].
        q99: Training state 99th-percentile values, shape [7].

    Returns:
        INT64 tokens and BOOL validity mask, both shaped [1, 200].
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape != (7,):
        raise ValueError(f"expected seven Piper state values, got {state.shape}")
    normalized = np.clip(
        2.0 * (state - q01) / np.maximum(q99 - q01, 1e-8) - 1.0, -1.0, 1.0,
    )
    discrete = np.digitize(normalized, np.linspace(-1.0, 1.0, 257)[:-1]).astype(np.int64) - 1
    clean_task = str(task).strip().replace("_", " ").replace("\n", " ")
    text = f"Task: {clean_task}, State: {' '.join(map(str, discrete))};\nAction: "
    ids = list(tokenizer.encode(text, add_bos=True))[:200]
    tokens = np.zeros((1, 200), dtype=np.int64)
    token_mask = np.zeros((1, 200), dtype=np.bool_)
    tokens[0, :len(ids)] = np.asarray(ids, dtype=np.int64)
    token_mask[0, :len(ids)] = True
    return tokens, token_mask
