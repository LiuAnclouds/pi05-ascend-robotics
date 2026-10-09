#!/usr/bin/env python3
"""Download a published Hugging Face model bundle into the project layout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    "models/paligemma-3b-pt-224/tokenizer.model",
    "config/norm_stats.json",
    "outputs/om/part1.om",
    "outputs/om/part2.om",
)
WEIGHTS_FILE = "models/weights/instrction_9.14_float32/model.safetensors"


def download_models(repo: str, revision: str, weights: bool = False) -> None:
    """Install one published bundle with matching tokenizer, statistics and OMs.

    Args:
        repo: Hugging Face model repository ID (owner/name).
        revision: Branch, tag or commit to resolve once for the whole bundle.
        weights: Also download the PyTorch checkpoint for ONNX export.

    Returns:
        None. Files are placed under models/, config/ and outputs/om/.
        Existing matching paths are replaced after all downloads succeed.
    """
    from huggingface_hub import HfApi, hf_hub_download

    commit = HfApi().model_info(repo, revision=revision).sha
    files = (*RUNTIME_FILES, WEIGHTS_FILE) if weights else RUNTIME_FILES
    print(f"======== Model download | {repo} @ {commit} ========", flush=True)
    # Download the whole bundle before replacing any installed asset.
    with tempfile.TemporaryDirectory(prefix=".model-download-", dir=PROJECT_ROOT) as folder:
        for name in files:
            hf_hub_download(repo_id=repo, filename=name, revision=commit, local_dir=folder)
        for name in files:
            source = Path(folder) / name
            if not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError(f"Downloaded file is missing or empty: {name}")
        for name in files:
            target = PROJECT_ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            # models/config/outputs are separate Docker bind mounts. Atomic
            # replacement must stage within the target's own filesystem.
            with tempfile.NamedTemporaryFile(prefix=".download-", dir=target.parent, delete=False) as staged:
                pending = Path(staged.name)
                try:
                    with (Path(folder) / name).open("rb") as source_file:
                        shutil.copyfileobj(source_file, staged)
                    staged.flush()
                    os.fsync(staged.fileno())
                except BaseException:
                    pending.unlink(missing_ok=True)
                    raise
            try:
                os.replace(pending, target)
            finally:
                pending.unlink(missing_ok=True)
            print(f"Saved: {name}", flush=True)


def main() -> None:
    """Parse repository/revision options and download deployment files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repo", required=True, help="Published Hugging Face model repository: owner/name.")
    parser.add_argument("--revision", default="main", help="Model repository tag, branch or commit.")
    parser.add_argument("--weights", action="store_true", help="Also download the checkpoint for ONNX export.")
    args = parser.parse_args()
    download_models(args.repo, args.revision, args.weights)


if __name__ == "__main__":
    main()
