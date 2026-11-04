"""Single-flight spawned inference worker with nonblocking control-loop polling.

The factory and device libraries are imported only in the child. No robot or
camera module is imported here. All payloads are owned snapshots, so the queue's
background serializer cannot race with the next observation.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import multiprocessing as mp
import queue
import time
import traceback


def _worker_main(factory_path, factory_kwargs, requests, replies, log_path):
    """Own the model in a spawned child and process one queued snapshot at a time.

    Args: Importable factory, constructor options, request/reply queues and log path.
    Returns: None; sends ready/result/failure messages and closes the model on exit.
    """
    model = None
    with contextlib.ExitStack() as stack:
        if log_path:
            log = stack.enter_context(open(log_path, "a", encoding="utf-8", buffering=1))
            stack.enter_context(contextlib.redirect_stdout(log))
            stack.enter_context(contextlib.redirect_stderr(log))
        try:
            module, name = factory_path.rsplit(":", 1)
            model = getattr(importlib.import_module(module), name)(**factory_kwargs)
            replies.put(dict(kind="ready", metadata=getattr(model, "metadata", {})))
            while True:
                message = requests.get()
                if message is None:
                    return
                request_id, payload = message
                started = time.monotonic()
                result = model.predict(payload)
                replies.put(
                    dict(
                        kind="result",
                        request_id=request_id,
                        result=result,
                        worker_started_at=started,
                        worker_finished_at=time.monotonic(),
                    )
                )
        except BaseException:
            replies.put(dict(kind="failed", traceback=traceback.format_exc()))
        finally:
            if model is not None and hasattr(model, "close"):
                model.close()


class InferenceWorker:
    """Only startup and explicit ``wait`` block; ``submit``/``poll`` do not wait."""

    def __init__(self, factory_path, factory_kwargs=None, *, log_path=None, progress_callback=None):
        """Start a child with isolated device ownership and wait for its ready reply.

        Args: module:factory name, constructor options and optional log path.
        Returns: None; constructor failure closes this worker and its IPC channels.
        """
        context = mp.get_context("spawn")
        self.requests = context.Queue(maxsize=1)
        self.replies = context.Queue(maxsize=1)
        self.pending = None
        self.closed = False
        self.process = context.Process(
            target=_worker_main,
            args=(factory_path, factory_kwargs or {}, self.requests, self.replies, log_path),
            name="pi05-inference-only",
        )
        try:
            self.process.start()
            started_at = time.monotonic()
            spinner = "|/-\\"
            bar_width = 24
            frame = 0
            last_progress_at = started_at - 1.0
            while True:
                try:
                    message = self.replies.get(timeout=0.08)
                except queue.Empty:
                    now = time.monotonic()
                    if progress_callback is None or now - last_progress_at >= 1.0:
                        travel = bar_width - 4
                        phase = frame % (2 * travel)
                        position = phase if phase <= travel else 2 * travel - phase
                        bar = "." * position + "===" + ">" + "." * (bar_width - position - 4)
                        elapsed = now - started_at
                        progress = (
                            f"模型加载中 [{bar}] {spinner[frame % len(spinner)]} | {elapsed:.1f}s"
                        )
                        if progress_callback is None:
                            print(f"\r{progress}\033[K", end="", flush=True)
                        else:
                            progress_callback(progress)
                            last_progress_at = now
                    frame += 1
                    continue
                if message["kind"] != "ready":
                    if progress_callback is None:
                        print(flush=True)
                    raise RuntimeError(message.get("traceback", repr(message)))
                self.metadata = message["metadata"]
                elapsed = time.monotonic() - started_at
                result = f"模型加载完成 | 用时 {elapsed:.1f}s"
                if progress_callback is None:
                    print(f"\r{result}\033[K", flush=True)
                else:
                    progress_callback(result)
                break
        except BaseException:
            if progress_callback is None:
                print(flush=True)
            self.close()
            raise

    def submit(self, request_id, payload):
        """Reject a second request rather than queue stale observations."""
        if self.pending is not None:
            raise RuntimeError("Inference request already in flight")
        owned = copy.deepcopy(payload)
        self.requests.put_nowait((request_id, owned))
        self.pending = request_id

    def poll(self):
        """Return a reply or None immediately; an absent reply remains pending."""
        try:
            message = self.replies.get_nowait()
        except queue.Empty:
            return None
        if message["kind"] == "failed":
            raise RuntimeError(message["traceback"])
        if message["kind"] != "result" or message["request_id"] != self.pending:
            raise RuntimeError(f"Inference protocol mismatch: {message.get('kind')}")
        self.pending = None
        message["received_at"] = time.monotonic()
        return message

    def wait(self, timeout=60):
        """Explicit blocking helper for loading/warm-up, never the control loop."""
        deadline = time.monotonic() + timeout
        while True:
            result = self.poll()
            if result is not None:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError("Inference result timed out")
            time.sleep(0.002)

    def close(self):
        """Stop only this worker and release IPC resources; safe after partial init."""
        if self.closed:
            return
        self.closed = True
        if self.process.pid is not None:
            if self.process.is_alive():
                try:
                    self.requests.put_nowait(None)
                except queue.Full:
                    pass
                self.process.join(2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(2)
        for channel in (self.requests, self.replies):
            channel.cancel_join_thread()
            channel.close()

    def __enter__(self):
        """Enter scoped worker ownership; input none, output this worker."""
        return self

    def __exit__(self, *_):
        """Close scoped worker ownership; input exception details, output None."""
        self.close()
