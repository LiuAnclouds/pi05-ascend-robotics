"""Dual-camera acquisition helpers.

Each camera is drained continuously by one reader thread. The application keeps
only its newest frame. Timestamps measure read completion, not sensor exposure.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

_MAX_FRAME_AGE = 0.5
_FRAME_TIMEOUT = 2.0


@dataclass(frozen=True)
class FrameSnapshot:
    """One latest frame and timing information for a camera stream."""

    frame: np.ndarray
    frame_id: int
    received_monotonic: float


class _LatestFrameReader:
    """Continuously read one V4L2 device while retaining only its latest frame."""

    def __init__(self, path: str):
        """Open a camera and start its single producer thread.

        Args:
            path: V4L2 device path.

        Returns:
            None. The reader owns one camera handle and one daemon thread.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self.path = path
        self.cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"cannot open camera: {path}")
        # This is a best-effort driver hint. Continuous draining below is the
        # actual stale-frame protection because some V4L2 backends ignore it.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._frame = None
        self._frame_id = 0
        self._received_monotonic = 0.0
        self._last_error: str | None = None
        self._thread = threading.Thread(target=self._run, name=f"camera-{path}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Drain the driver queue and atomically replace the latest frame."""
        try:
            while not self._stop.is_set():
                ok, frame = self.cap.read()
                received = time.monotonic()
                if not ok or frame is None:
                    raise RuntimeError("camera frame read failed")
                with self._condition:
                    self._frame = frame
                    self._frame_id += 1
                    self._received_monotonic = received
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._last_error = str(error)
                self._condition.notify_all()
        finally:
            # Only the producer touches the handle: release must not race read().
            self.cap.release()

    def snapshot(self, timeout: float = _FRAME_TIMEOUT, min_frame_id: int = 1) -> FrameSnapshot:
        """Copy the newest frame and its timestamp without queueing frames.

        Args:
            timeout: Maximum seconds to wait for a recent frame.
            min_frame_id: Minimum number of captured frames; used for warm-up.

        Returns:
            A copied latest frame with a monotonically increasing frame id.

        Raises:
            RuntimeError: If the reader failed, stopped, or has no recent frame.
        """
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._last_error or self._stop.is_set():
                    raise RuntimeError(f"camera {self.path}: {self._last_error or 'reader stopped'}")
                now = time.monotonic()
                if (self._frame is not None and self._frame_id >= min_frame_id
                        and now - self._received_monotonic <= _MAX_FRAME_AGE):
                    return FrameSnapshot(self._frame.copy(), self._frame_id, self._received_monotonic)
                remaining = deadline - now
                if remaining <= 0:
                    raise RuntimeError(f"camera {self.path}: no recent frame received")
                self._condition.wait(remaining)

    def close(self) -> None:
        """Stop the reader and release its camera handle."""
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=_FRAME_TIMEOUT)


class CameraPair:
    """Keep two V4L2 cameras draining and read one latest-frame pair."""

    def __init__(self, first: str, second: str, warmup: int = 5):
        """Open both devices and wait for their first usable frames.

        Args:
            first: Device path for the third-person camera.
            second: Device path for the wrist camera.
            warmup: Number of actual frames discarded before the first snapshot.

        Returns:
            None. Only one latest frame per camera is retained.
        """
        self.paths = (first, second)
        self.readers = []
        self._last_metadata: dict[str, object] | None = None
        try:
            for path in self.paths:
                self.readers.append(_LatestFrameReader(path))
            for reader in self.readers:
                reader.snapshot(timeout=max(_FRAME_TIMEOUT, (warmup + 1) / 15.0),
                                min_frame_id=max(0, warmup) + 1)
        except BaseException:
            self.close()
            raise

    def read(self):
        """Return the newest third-person and wrist frames.

        Returns:
            ``(third_person_bgr, wrist_bgr, snapshot_time_ms)``. Timing metadata
            for the same pair is available through :meth:`metadata`.
        """
        started = time.monotonic()
        # If one stream needed to wait, refresh the other snapshot as well.
        while True:
            pair = [reader.snapshot(timeout=max(0.0, started + _FRAME_TIMEOUT - time.monotonic()))
                    for reader in self.readers]
            now = time.monotonic()
            if all(now - frame.received_monotonic <= _MAX_FRAME_AGE for frame in pair):
                break
            if now - started >= _FRAME_TIMEOUT:
                raise RuntimeError("camera pair has no recent frames")
        first, second = pair
        self._last_metadata = {
            name: {"path": path, "frame_id": frame.frame_id,
                   "age_ms": (now - frame.received_monotonic) * 1000.0,
                   "read_complete_monotonic": frame.received_monotonic}
            for name, path, frame in zip(("a", "b"), self.paths, pair)
        }
        self._last_metadata.update({
            "timestamp_source": "host_read_complete_not_sensor_exposure",
            "pair_skew_ms": abs(first.received_monotonic - second.received_monotonic) * 1000.0,
            "snapshot_ms": (now - started) * 1000.0,
        })
        return first.frame, second.frame, self._last_metadata["snapshot_ms"]

    def metadata(self) -> dict[str, object] | None:
        """Return timing metadata for the most recent frame pair."""
        return None if self._last_metadata is None else {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in self._last_metadata.items()
        }

    def close(self) -> None:
        """Stop both producer threads and release both camera handles."""
        for reader in getattr(self, "readers", []):
            reader.close()
