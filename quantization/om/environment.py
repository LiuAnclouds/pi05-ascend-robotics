"""CANN discovery and environment initialization; no device work at import time."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def cann_root() -> Path:
    """Find the active CANN toolkit without project-specific environment overrides.

    Args: None; uses CANN's standard variables and the active atc executable.
    Returns: Toolkit directory containing set_env.sh; ambiguous installs raise.
    """
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(name)
        if not value:
            continue
        candidate = Path(value).expanduser().resolve()
        for directory in (candidate, candidate.parent):
            if (directory / "set_env.sh").is_file():
                return directory
    compiler = shutil.which("atc")
    if compiler:
        for candidate in Path(compiler).resolve().parents:
            if (candidate / "set_env.sh").is_file():
                return candidate
    candidates = set()
    for parent in (Path("/usr/local/Ascend"), Path.home() / "Ascend"):
        for pattern in ("ascend-toolkit/latest", "cann/latest", "cann-*", "latest"):
            for candidate in parent.glob(pattern):
                if (candidate / "set_env.sh").is_file():
                    candidates.add(candidate.resolve())
    if len(candidates) == 1:
        return candidates.pop()
    detail = ", ".join(str(path) for path in sorted(candidates)) or "none found"
    raise FileNotFoundError(f"Activate the intended CANN set_env.sh before compilation ({detail})")


def cann_environment() -> dict[str, str]:
    """Load CANN but keep the invoking Python environment, not a named Conda env."""
    prepared = subprocess.check_output(
        ["bash", "-c", 'source "$1" >&2 && env -0', "pi05-cann-env", str(cann_root() / "set_env.sh")],
        env=os.environ.copy(),
    )
    result = os.environ.copy()
    result.update(os.fsdecode(item).split("=", 1) for item in prepared.split(b"\0") if item)
    result["PATH"] = str(Path(sys.executable).absolute().parent) + os.pathsep + result.get("PATH", "")
    return result
