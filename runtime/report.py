"""Incremental inference journal and readable final report, without control logic."""

from __future__ import annotations

import contextlib
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import threading
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
    """Queue prediction snapshots, then finalize a structured JSON report.

    A single bounded writer serializes and fsyncs every event outside the
    control path. Normal shutdown drains all events; sudden power loss can
    lose queued/in-flight events. A full queue or writer error is surfaced,
    never silently dropped. Finalization streams the journal to limit memory.
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
                        "journal_note": "Background writer, up to 8 queued events; shutdown drains all, abrupt power loss may lose pending events.",
                        "nonfinite_encoding": "Nonfinite values become null; finite flags remain false."}
        self._journal = self.journal_path.open("w", encoding="utf-8")
        self._log = self.log_path.open("w", encoding="utf-8", buffering=1)
        self._queue = queue.Queue(maxsize=8)
        self._error = None
        self._closed = False
        self._writer = threading.Thread(target=self._write_events, name="inference-report", daemon=True)
        self._writer.start()
        self.event("session_started", self.session)
        self.path.write_text(pretty_json({"schema_version": 2, "status": "running",
                                         "session": self.session,
                                         "note": "Live predictions and control events are in the JSONL journal."}) + "\n",
                             encoding="utf-8")

    def event(self, name: str, data: dict) -> None:
        """Snapshot an event into a bounded queue; return before serialization/fsync.

        Args: name identifies the event; data is JSON-compatible mutable data.
        Later caller mutations cannot change the queued snapshot. Queue/write
        failures raise RuntimeError so control can stop and preserve evidence.
        """
        self.check_writer()
        if self._closed:
            raise RuntimeError("Report writer is closed")
        item = _json_value({"event": name, "timestamp": timestamp(), "data": data})
        try:
            self._queue.put_nowait(item)
        except queue.Full as error:
            raise RuntimeError("Report writer queue is full; check output storage") from error

    def check_writer(self) -> None:
        """Raise any background I/O failure without waiting for disk operations."""
        if self._error is not None:
            raise RuntimeError(f"Report writer failed: {self._error}") from self._error

    def _write_events(self) -> None:
        """Consume FIFO snapshots, fsync each, and expose errors to the main loop."""
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._error is None:
                    self._journal.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                    self._journal.flush()
                    os.fsync(self._journal.fileno())
            except Exception as error:
                self._error = error
            finally:
                self._queue.task_done()

    @contextlib.contextmanager
    def diagnostics(self):
        """Capture Python SDK stdout/stderr during model loading/cleanup to .log."""
        with contextlib.redirect_stdout(self._log), contextlib.redirect_stderr(self._log):
            yield self._log

    def finish(self, outcome: str, summary: dict) -> None:
        """Stream completed/failed rounds and control events to an atomic final JSON."""
        try:
            self._queue.join()
            self.event("session_finished", {"outcome": outcome, **summary})
        finally:
            self._closed = True
            self._queue.put(None)
            self._writer.join()
            self._journal.close()
            self._log.close()
        self.check_writer()
        temp = self.path.with_suffix(".json.tmp")
        with temp.open("w", encoding="utf-8") as target:
            session = {key: value for key, value in self.session.items() if key != "journal"}
            header = {"schema_version": 2, "status": outcome, "session": session,
                      "finished_at": timestamp(), "summary": _json_value(summary)}
            target.write(pretty_json(header)[:-2] + ',\n  "rounds": [\n')
            first = True
            pending = {}
            with self.journal_path.open(encoding="utf-8") as source:
                for line in source:
                    item = json.loads(line)
                    if item["event"] == "prediction":
                        pending[item["data"]["iteration"]] = item["data"]
                    elif item["event"] == "round_finished":
                        record = item["data"]
                        target.write(("" if first else ",\n") + "    " + pretty_json(record, 2))
                        first = False
                        pending.pop(record["iteration"], None)
            for record in pending.values():
                record["outcome"] = "interrupted_before_execution_record"
                target.write(("" if first else ",\n") + "    " + pretty_json(record, 2))
                first = False
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
