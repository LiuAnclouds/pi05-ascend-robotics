"""Operator-facing terminal formatting; independent of model and robot control."""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
import unicodedata
from collections.abc import Sequence

_COLORS = {"title": "1;36", "ok": "1;32", "wait": "1;33", "error": "1;31", "normal": "0"}


def _paint(text: str, style: str) -> str:
    """Return colored text on a terminal; keep redirected logs free of ANSI codes."""
    if not sys.stdout.isatty() or "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return text
    return f"\033[{_COLORS[style]}m{text}\033[0m"


def _line(label: str, value: str, style: str = "normal") -> str:
    """Return one aligned row from a label, value, and semantic color name."""
    width = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in label)
    prefix = f"  {label}{' ' * max(1, 12 - width)}"
    return prefix + _paint(value, style)


def number(value: object) -> str:
    """Format optional millisecond values with one decimal place for display."""
    return "--" if value is None else f"{float(value):.1f}"


def section(title: str, subtitle: str | None = None) -> None:
    """Print a cyan title between separators; no data or control side effects."""
    width = min(76, max(40, shutil.get_terminal_size((80, 24)).columns - 2))
    rule = "=" * width
    rows = ["", _paint(rule, "title"), _paint(f"  {title}", "title")]
    if subtitle:
        rows.append(f"  {subtitle}")
    rows.append(_paint(rule, "title"))
    print("\n".join(rows), flush=True)


def status(label: str, value: str, style: str = "normal") -> None:
    """Print one status row, wrapping long prompts and paths in narrow terminals."""
    width = max(26, min(76, shutil.get_terminal_size((80, 24)).columns - 2) - 14)
    parts = textwrap.wrap(str(value), width=width, break_long_words=True, break_on_hyphens=False) or [""]
    print("\n".join(_line(label if i == 0 else "", part, style) for i, part in enumerate(parts)), flush=True)


def format_log_block(title: str, rows: Sequence[str], style: str = "title") -> str:
    """Build one separated ROS log message without printing or changing runtime state.

    Args: Human-readable title, body rows, and an existing semantic color style.
    Returns: One multiline string; color is disabled by NO_COLOR or TERM=dumb.
    """
    rule = "=" * 76
    color = "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
    begin, end = (f"\033[{_COLORS[style]}m", "\033[0m") if color else ("", "")
    return "\n" + "\n".join(
        [f"{begin}{rule}", f"  {title}{end}", *rows, f"{begin}{rule}{end}", ""]
    )


def format_joint_table(rows: Sequence[tuple[str, Sequence[float]]]) -> list[str]:
    """Align seven physical values for a ROS log, without transforming the values.

    Args: Labeled seven-value rows; joints are degrees and the gripper is millimetres.
    Returns: Header and numeric rows. Inputs and control commands are unchanged.
    """
    columns = (*[f"J{index + 1}" for index in range(6)], "G(mm)")
    return [_line("关节(度)", " ".join(f"{name:>8}" for name in columns))] + [
        _line(label, " ".join(f"{float(value):8.2f}" for value in values))
        for label, values in rows
    ]


def print_iteration(record: dict, steps: int) -> None:
    """Print a completed inference round and its measured timings.

    Args:
        record: Existing runtime result record; never modified by this function.
        steps: Number of Part2 denoising calls for this prediction.

    Returns:
        None. Full action arrays remain in the existing JSON result file.
    """
    timing = record["timing_ms"]
    finite = bool(record["action_finite"])
    outcome = record.get("outcome", "completed")
    style = "ok" if finite and outcome == "completed" else "error"
    # One line per prediction. Full timing, camera and joint data stay in reports.
    text = (
        f"  [{record['iteration'] + 1:04d}] infer={number(timing['total'])} ms | "
        f"sent={record['motion_points_sent']}/{record['motion_horizon']} | "
        f"gap={number(timing.get('between_chunks'))} ms | "
        f"{'finite' if finite else 'NONFINITE'} | {outcome}"
    )
    print(_paint(text, style), flush=True)
