"""Incremental inference journal and readable final report, without control logic."""

from __future__ import annotations

import contextlib
from datetime import datetime
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo


def timestamp() -> str:
    """Return the report timestamp in Asia/Shanghai with millisecond precision."""
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="milliseconds")


def _json_value(value):
    """Convert Path/nonfinite numbers to valid JSON values without rounding floats."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def pretty_json(value, depth: int = 0) -> str:
    """Format objects with indentation and each numeric vector on one line."""
    pad = "  " * depth
    if isinstance(value, dict) and value:
        return "{\n" + ",\n".join(
            "  " + pad + json.dumps(key) + ": " + pretty_json(item, depth + 1)
            for key, item in value.items()) + "\n" + pad + "}"
    if isinstance(value, list) and any(isinstance(item, (list, dict)) for item in value):
        return "[\n" + ",\n".join("  " + pad + pretty_json(item, depth + 1)
                                   for item in value) + "\n" + pad + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class RunReport:
    """Persist predictions before sending, then finalize a structured JSON report.

    The JSONL journal is flushed/fsynced at each event. It remains useful after
    abrupt termination; the final JSON is assembled in a streaming pass, so a
    continuous run does not retain all trajectories in memory. The temporary
    journal is removed only after the complete report has been saved.
    """

    def __init__(self, path: Path, settings: dict):
        """Create report/journal/diagnostic files for the supplied run settings."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.path.with_suffix(".jsonl")
        self.log_path = self.path.with_suffix(".log")
        for output in (self.path, self.journal_path, self.log_path):
            if output.exists():
                raise FileExistsError(f"Run output already exists: {output}. Choose a new --output.")
        self.session = {"started_at": timestamp(), "settings": _json_value(settings),
                        "journal": str(self.journal_path), "diagnostic_log": str(self.log_path),
                        "dimensions": ["J1_deg", "J2_deg", "J3_deg", "J4_deg", "J5_deg", "J6_deg", "gripper_mm"],
                        "action_convention": "First six delta values are relative to the observation state; gripper is absolute.",
                        "feedback_note": "Command counts are host calls, not controller acknowledgements or grasp success.",
                        "nonfinite_encoding": "Nonfinite values become null; finite flags remain false."}
        self._journal = self.journal_path.open("w", encoding="utf-8")
        self._log = self.log_path.open("w", encoding="utf-8", buffering=1)
        self.event("session_started", self.session)
        self.path.write_text(pretty_json({"schema_version": 2, "status": "running",
                                         "session": self.session,
                                         "note": "Live predictions and control events are in the JSONL journal."}) + "\n",
                             encoding="utf-8")

    def event(self, name: str, data: dict) -> None:
        """Append a timestamped event and durably flush it to the JSONL journal."""
        item = _json_value({"event": name, "timestamp": timestamp(), "data": data})
        self._journal.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        self._journal.flush()
        os.fsync(self._journal.fileno())

    @contextlib.contextmanager
    def diagnostics(self):
        """Capture Python SDK stdout/stderr during model loading/cleanup to .log."""
        with contextlib.redirect_stdout(self._log), contextlib.redirect_stderr(self._log):
            yield self._log

    def finish(self, outcome: str, summary: dict) -> None:
        """Stream completed/failed rounds and control events to an atomic final JSON."""
        self.event("session_finished", {"outcome": outcome, **summary})
        self._journal.close()
        temp = self.path.with_suffix(".json.tmp")
        with temp.open("w", encoding="utf-8") as target:
            session = {key: value for key, value in self.session.items() if key != "journal"}
            header = {"schema_version": 2, "status": outcome, "session": session,
                      "finished_at": timestamp(), "summary": _json_value(summary)}
            target.write(pretty_json(header)[:-2] + ',\n  "rounds": [\n')
            first = True
            pending = None
            with self.journal_path.open(encoding="utf-8") as source:
                for line in source:
                    item = json.loads(line)
                    if item["event"] == "prediction":
                        pending = item["data"]
                    elif item["event"] == "round_finished":
                        record = item["data"]
                        target.write(("" if first else ",\n") + "    " + pretty_json(record, 2))
                        first, pending = False, None
            if pending is not None:
                pending["outcome"] = "interrupted_before_execution_record"
                target.write(("" if first else ",\n") + "    " + pretty_json(pending, 2))
            target.write('\n  ],\n  "events": [\n')
            first = True
            with self.journal_path.open(encoding="utf-8") as source:
                for line in source:
                    item = json.loads(line)
                    if item["event"] not in ("prediction", "round_finished", "session_started", "session_finished"):
                        target.write(("" if first else ",\n") + "    " + pretty_json(item, 2))
                        first = False
            target.write("\n  ]\n}\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp, self.path)
        self._log.close()
        self.journal_path.unlink()
