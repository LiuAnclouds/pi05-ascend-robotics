"""Bounded online input-path latency statistics, independent of device/robot libraries."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from inference.include.definitions import LATENCY_WINDOW_SIZE


class InputLatencyWindow:
    """Bounded input-path measurements; summaries stay off the per-frame path."""

    def __init__(self, capacity=LATENCY_WINDOW_SIZE):
        """Create a fixed-size timing window.

        Args: Positive maximum number of recent requests.
        Returns: None. No files, device libraries or background tasks are opened.
        """
        if capacity <= 0:
            raise ValueError("Latency window capacity must be positive")
        self.samples = deque(maxlen=capacity)
        self.total = 0

    def record(self, milliseconds):
        """Retain one independent snapshot of named timing measurements.

        Args: Mapping of component names to finite, nonnegative milliseconds.
        Returns: None; the oldest sample is dropped when the window is full.
        """
        values = {name: float(value) for name, value in milliseconds.items()}
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("Input latency measurements must be finite and nonnegative")
        self.samples.append(values)
        self.total += 1

    def summary(self):
        """Summarize the measured requests without conflating nested intervals.

        Args: None.
        Returns: Sample counts, latest values and per-component mean/P50/P95/P99.
            Components such as H2D are inside worker time, and are not additive.
        """
        distributions = {}
        for name in self.samples[-1] if self.samples else ():
            values = np.asarray([sample[name] for sample in self.samples if name in sample], np.float64)
            p50, p95, p99 = np.percentile(values, [50, 95, 99])
            distributions[name] = dict(
                mean=float(values.mean()), P50=float(p50), P95=float(p95), P99=float(p99)
            )
        return dict(
            requests_total=self.total,
            window_samples=len(self.samples),
            capacity=self.samples.maxlen,
            last_ms=dict(self.samples[-1]) if self.samples else {},
            distributions_ms=distributions,
        )
