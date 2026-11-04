"""Ordered, cancellable Piper target sender; no model code runs on this thread."""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from inference.core.motion.piper_interface import send_motion_chunk, send_motion_timeline


class AsyncChunkSender:
    """Send a trajectory without running ACL/model code on the sender thread."""

    def __init__(
        self, piper, speed: int, fps: float, check_pending: Callable | None = None, *, command_lock=None
    ):
        """Store SDK controls, health callback, and optional shared command lock.

        Args: piper is connected; speed/fps specify sending; callbacks never block.
        Returns: None. No CAN command is sent until start is called.
        """
        self.piper, self.speed, self.fps = piper, speed, fps
        self.check_pending = check_pending
        self.command_lock = command_lock
        self.thread = None
        self.stop_event = threading.Event()
        self.progress = {}
        self.error = None
        self.expected_points = 0

    def start(self, action: np.ndarray) -> dict:
        """Copy the trajectory, start a sender and return its live progress record."""
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("A trajectory is already being sent")
        trajectory = np.ascontiguousarray(action, dtype=np.float32).copy()
        self.expected_points = len(trajectory)
        self.stop_event.clear()
        self.progress, self.error = {}, None

        def run():
            """Keep command order and retain worker exceptions for the owner."""
            self.progress["sender_started_at"] = time.perf_counter()
            try:
                send_motion_chunk(
                    self.piper,
                    trajectory,
                    self.speed,
                    self.fps,
                    prepared=True,
                    progress=self.progress,
                    check_pending=self._check,
                    command_lock=self.command_lock,
                    cancel_wait=self.stop_event.wait,
                )
            except BaseException as error:
                self.error = error
            finally:
                self.progress["sender_finished_at"] = time.perf_counter()

        self.thread = threading.Thread(target=run, name="piper-trajectory", daemon=True)
        self.thread.start()
        return self.progress

    def start_timeline(self, timeline, on_replan, *, timeline_lock=None) -> dict:
        """Start sending a live RTC queue whose future can be replaced on reply.

        Args: Primed timestamped timeline, callback for its owned replan snapshot,
            and optional lock shared with the ROS reply callback.
        Returns: Mutable progress telemetry; no second sender is started.
        Side effects: Sends targets until cancellation; empty queues wait for prediction.
        """
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("A trajectory is already being sent")
        self.stop_event.clear()
        self.progress, self.error = {}, None

        def run():
            """Keep the control clock and Piper SDK on this dedicated sender thread."""
            self.progress["sender_started_at"] = time.perf_counter()
            try:
                send_motion_timeline(
                    self.piper,
                    timeline,
                    self.speed,
                    self.fps,
                    progress=self.progress,
                    check_pending=self._check,
                    on_replan=on_replan,
                    command_lock=self.command_lock,
                    timeline_lock=timeline_lock,
                    cancel_wait=self.stop_event.wait,
                )
            except BaseException as error:
                self.error = error
            finally:
                self.progress["sender_finished_at"] = time.perf_counter()

        self.thread = threading.Thread(target=run, name="piper-rtc-timeline", daemon=True)
        self.thread.start()
        return self.progress

    def _check(self):
        """Cancel before the next command or propagate a reporting error."""
        if self.stop_event.is_set():
            raise RuntimeError("Trajectory sender stopped")
        if self.check_pending is not None:
            self.check_pending()

    def raise_if_failed(self):
        """Surface the original exception without consuming it."""
        if self.error is not None:
            raise self.error

    def wait_until_points(self, points: int):
        """Wait for sent point count or failure; no robot commands are added."""
        while self.thread is not None and self.thread.is_alive():
            self.raise_if_failed()
            if self.progress.get("points_sent", 0) >= points:
                return
            time.sleep(0.002)
        self.raise_if_failed()

    def wait(self):
        """Join the sender and return measured execution telemetry."""
        while self.thread is not None and self.thread.is_alive():
            self.thread.join(0.02)
        self.raise_if_failed()
        return self.progress

    def stop(self):
        """Cancel and join before CAN teardown; detect a stuck SDK call explicitly."""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                raise RuntimeError("CAN sender did not stop; SDK call is still pending")
