"""Timestamp-aligned action scheduling, with no camera, ACL or robot imports.

One owner consumes integer control ticks and merges worker replies. Action zero
of a prediction is anchored at the request's observation tick, not reply time.
This is the asynchronous substrate for RTC, not an RTC guidance algorithm.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from inference.core.rtc.guidance_math import (
    REFERENCE_DELAY_BUFFER,
    REFERENCE_MIN_EXECUTION_HORIZON,
    prepare_prefix,
)


@dataclass(frozen=True)
class ChunkRequest:
    """Owned snapshot of the old future and the new observation's anchor."""

    request_id: int
    generation: int
    observation_tick: int
    captured_at: float
    state: np.ndarray
    old_targets: np.ndarray
    delay_steps: int


class ActionTimeline:
    """Single-owner, single-flight queue of absolute seven-axis targets.

    ``consume`` never waits. A missing target returns ``None`` and records an
    idle slot: the sender waits for a reply without extrapolating targets.
    Capture the observation before consuming its named control tick. Workers
    must not modify this object; their replies are polled by its owner.
    """

    def __init__(
        self,
        fps=30.0,
        horizon=50,
        replan_after=30,
        *,
        initial_latency=0.2,
        margin_steps=2,
        latency_quantile=99,
        handover_policy="commit_budget",
    ):
        """Create a single-owner queue and its observation-to-reply delay policy.

        Args: Control rate/horizon, replan point, initial delay and timing margins.
        Returns: None; no sender, model, device or control loop is started.
        """
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if not 0 < replan_after < horizon or margin_steps < 0:
            raise ValueError("Invalid horizon, replan interval or margin")
        if not math.isfinite(initial_latency) or initial_latency < 0:
            raise ValueError("Invalid initial latency")
        if not 0 < latency_quantile <= 100:
            raise ValueError("Invalid latency quantile")
        if handover_policy not in ("commit_budget", "author_rtc_on_ready"):
            raise ValueError("Unknown handover policy")
        self.fps, self.horizon = float(fps), int(horizon)
        self.replan_after, self.margin_steps = int(replan_after), int(margin_steps)
        self.initial_latency, self.latency_quantile = initial_latency, latency_quantile
        self.handover_policy = handover_policy
        self.latencies = deque(maxlen=64)
        self.generation = self.tick = self.underruns = 0
        self._serial = 0
        self._queue = {}
        self.pending = None

    @property
    def remaining(self):
        """Count contiguous future commands, excluding already consumed ticks."""
        end = self.tick
        while end in self._queue:
            end += 1
        return end - self.tick

    @property
    def estimated_delay(self):
        """Conservative observation-to-result budget, including IPC/preprocessing."""
        seconds = (
            float(np.percentile(self.latencies, self.latency_quantile))
            if self.latencies
            else self.initial_latency
        )
        return max(1, math.ceil(seconds * self.fps - 1e-9) + self.margin_steps)

    def _targets(self, values):
        """Take an owned snapshot of a complete absolute action chunk.

        Args: Horizon-by-seven finite targets in degrees/mm.
        Returns: Copied FP32 array; invalid shape or nonfinite values are rejected.
        """
        array = np.array(values, dtype=np.float32, copy=True)
        if array.shape != (self.horizon, 7) or not np.isfinite(array).all():
            raise ValueError(f"Expected finite ({self.horizon}, 7) absolute targets")
        return array

    def prime(self, targets):
        """Install the initial prediction before starting the control clock."""
        if self._queue or self.pending is not None:
            raise RuntimeError("Timeline is already primed or has an in-flight request")
        array = self._targets(targets)
        self._queue = {self.tick + i: (row.copy(), 0) for i, row in enumerate(array)}

    def should_request(self):
        """Start near step 30, or earlier when the measured latency needs it."""
        if self.pending is not None or not self.remaining:
            return False
        threshold = max(self.horizon - self.replan_after, self.estimated_delay + 2)
        return self.remaining <= min(self.horizon, threshold)

    def begin_request(self, state, captured_at):
        """Snapshot old targets on the same tick as the newly captured observation."""
        return self._begin_request(state, captured_at, require_complete_budget=True)

    def _begin_request(self, state, captured_at, *, require_complete_budget):
        """Capture old queued targets and the matching observation at one tick.

        Args: State, monotonic capture time and whether the full delay must fit.
        Returns: Immutable request record; only one request may be pending.
        """
        if self.pending is not None:
            raise RuntimeError("Only one inference request may be in flight")
        state = np.array(state, dtype=np.float32, copy=True)
        if state.shape != (7,) or not np.isfinite(state).all() or not math.isfinite(captured_at):
            raise ValueError("Expected a finite state and monotonic capture timestamp")
        overlap = self.remaining
        delay = self.estimated_delay
        if require_complete_budget and (delay >= self.horizon or delay > overlap):
            raise RuntimeError("Too little queued prefix for the latency budget; refill earlier")
        if overlap == 0 and require_complete_budget:
            raise RuntimeError("No old trajectory is available for this prefix-only scheduler")
        old = (
            np.stack([self._queue[self.tick + i][0] for i in range(overlap)])
            if overlap
            else np.empty((0, 7), dtype=np.float32)
        )
        state.setflags(write=False)
        old.setflags(write=False)
        self._serial += 1
        self.pending = ChunkRequest(
            self._serial, self.generation, self.tick, float(captured_at), state, old, delay
        )
        return self.pending

    def rtc_prefix(self, request, action_q01, action_q99):
        """Re-anchor physical targets; padded dimensions receive no guidance."""
        if self.horizon != 50:
            raise ValueError("The current OpenPI RTC prefix contract has horizon 50")
        return prepare_prefix(request.old_targets, request.state, action_q01, action_q99, request.delay_steps)

    def merge(self, request, targets, received_at):
        """Install only future, time-aligned commands; never average gripper events.

        The historical async-only baseline retains the old latency budget.
        Author RTC instead swaps immediately on readiness (paper Algorithm 1,
        line21): a conservative d sets generation guidance, not an extra wait.
        Already consumed commands are never changed under either policy.
        An empty-reference request is a normal serial refill: install its complete
        prediction at reply time, because no old trajectory was executing.
        """
        if request.generation != self.generation:
            return {"accepted": False, "reason": "reset_generation", "request_id": request.request_id}
        if self.pending is None or request.request_id != self.pending.request_id:
            return {"accepted": False, "reason": "not_pending", "request_id": request.request_id}
        latency = float(received_at) - request.captured_at
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("Reply timestamp predates the observation")
        array = self._targets(targets)
        self.latencies.append(latency)
        self.pending = None
        serial_refill = len(request.old_targets) == 0
        switch_tick = (
            self.tick
            if self.handover_policy == "author_rtc_on_ready"
            else max(self.tick, request.observation_tick + request.delay_steps)
        )
        prediction_tick = switch_tick if serial_refill else request.observation_tick
        skipped = switch_tick - prediction_tick
        record = dict(
            request_id=request.request_id,
            observation_tick=request.observation_tick,
            received_tick=self.tick,
            switch_tick=switch_tick,
            skipped_points=skipped,
            installed_points=max(0, self.horizon - skipped),
            actual_delay_steps=self.tick - request.observation_tick,
            serial_refill=serial_refill,
            latency_ms=latency * 1000,
            budget_steps=request.delay_steps,
            overlap_at_request=len(request.old_targets),
            overlap_at_switch=max(0, len(request.old_targets) - skipped),
            method="serial_refill_on_ready" if serial_refill else "timestamp_aligned_queue_swap",
            handover_policy=self.handover_policy,
        )
        if skipped >= self.horizon:
            return {**record, "accepted": False, "reason": "expired"}
        self._queue = {tick: value for tick, value in self._queue.items() if self.tick <= tick < switch_tick}
        for i in range(skipped, self.horizon):
            self._queue[prediction_tick + i] = (array[i].copy(), request.request_id)
        return {**record, "accepted": True, "reason": "merged"}

    def cancel(self, request_id, generation):
        """Revoke one in-flight reply without changing the remaining old targets.

        Args: Request identity from the paired RTC ROS message.
        Returns: Whether it matched and cleared the active request.
        Side effects: A later reply for this request can no longer replace the queue.
        """
        request = self.pending
        if request is None or request.request_id != request_id or request.generation != generation:
            return False
        self.pending = None
        return True

    def consume(self):
        """Pop one control tick without sleeping, including on queue underrun."""
        value = self._queue.pop(self.tick, None)
        self.tick += 1
        if value is None:
            self.underruns += 1
            return None
        return value[0].copy(), value[1]

    def reset(self):
        """Revoke old replies; the external worker still needs to drain its job."""
        self.generation += 1
        self._queue.clear()
        self.pending = None


class AuthorRTCTimeline(ActionTimeline):
    """Paper Algorithm 1 scheduling, isolated from the old async baseline.

    Replan after s_min ticks since the current chunk's observation, or as soon
    as a longer in-flight call finishes. Estimate d using max of the last ten
    observed CONTROL-TICK delays, initially seeded by d_init. Swap on arrival.
    No additional margin, percentile estimate or early-refill policy is used.
    An empty queue waits for its pending reply. If that reply has no future
    left, request a fresh observation and resume with a normal serial refill.
    """

    def __init__(
        self, fps=30.0, horizon=50, *, initial_latency=0.2, replan_after=REFERENCE_MIN_EXECUTION_HORIZON
    ):
        """Initialize author RTC scheduling and its bounded delay-history buffer.

        Args: Control rate/horizon, startup delay and explicit replan interval.
        Returns: None; no inference or robot actions are started.
        """
        if not 0 < replan_after < horizon:
            raise ValueError("RTC replan interval must be inside the action horizon")
        super().__init__(
            fps=fps,
            horizon=horizon,
            replan_after=replan_after,
            initial_latency=initial_latency,
            margin_steps=0,
            latency_quantile=100,
            handover_policy="author_rtc_on_ready",
        )
        self._initial_delay = math.ceil(initial_latency * self.fps - 1e-9)
        self.delay_ticks = deque([self._initial_delay], maxlen=REFERENCE_DELAY_BUFFER)
        self.chunk_observation_tick = self.tick

    @property
    def estimated_delay(self):
        """Read the maximum recent observed tick delay.

        Args: None. Returns: Nonnegative integer delay in control ticks.
        """
        return max(self.delay_ticks)

    def should_request(self):
        """Request at the replan point, or refill an empty queue after a late reply.

        Args: None. Returns: True when no request is pending and work is needed.
        """
        return self.pending is None and (
            self.remaining == 0
            or self.tick - self.chunk_observation_tick >= self.replan_after
        )

    def begin_request(self, state, captured_at):
        """Capture the real overlap, or an empty reference for a serial refill.

        Args: Source-unit state and monotonic observation timestamp.
        Returns: Owned ChunkRequest; no old targets are invented for an empty queue.
        """
        # get_prefix_weights clamps start to end when d exceeds available
        # overlap. The paper does not pad that missing prefix with fake data.
        return self._begin_request(state, captured_at, require_complete_budget=False)

    def merge(self, request, targets, received_at):
        """Swap a valid ready reply and update the author control-tick delay history.

        Args: Pending request, complete absolute targets and monotonic reply time.
        Returns: Merge record; stale/revoked replies do not replace future commands.
        """
        result = super().merge(request, targets, received_at)
        if result["accepted"] or result["reason"] == "expired":
            self.delay_ticks.append(result["actual_delay_steps"])
            result["delay_estimator"] = "max_last_10_control_tick_delays"
        if result["accepted"]:
            self.chunk_observation_tick = (
                result["switch_tick"] if result["serial_refill"] else request.observation_tick
            )
        return result

    def reset(self):
        """Revoke pending replies and restore the initial delay estimate.

        Args: None. Returns: None; callers must separately drain any worker request.
        """
        super().reset()
        self.chunk_observation_tick = self.tick
        self.delay_ticks = deque([self._initial_delay], maxlen=REFERENCE_DELAY_BUFFER)


def motion_diagnostics(targets, fps):
    """Command-space changes only, never measured robot tracking error."""
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 7 or not np.isfinite(targets).all():
        raise ValueError("Expected finite [ticks, 7] commands")

    def distribution(values):
        """Summarize absolute command differences.

        Args: Difference samples. Returns: MAE/percentiles, or None for an empty set.
        """
        values = np.abs(values)
        if not values.size:
            return None
        return dict(
            mae=float(values.mean()),
            percentiles={str(q): float(np.percentile(values, q)) for q in (50, 75, 90, 95, 99, 100)},
        )

    axes = {}
    for i in range(7):
        axis = targets[:, i]
        axes[f"J{i + 1}" if i < 6 else "gripper"] = dict(
            unit="deg" if i < 6 else "mm",
            step=distribution(np.diff(axis)),
            velocity=distribution(np.diff(axis) * fps),
            acceleration=distribution(np.diff(axis, n=2) * fps**2),
            jerk=distribution(np.diff(axis, n=3) * fps**3),
        )
    return dict(axes=axes, scope="Command differences/finite differences; no physical robot measurements")
