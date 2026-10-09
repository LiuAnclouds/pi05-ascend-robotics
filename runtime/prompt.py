"""Exchange task requests over a local socket shared by host and Docker."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import socket
import threading

DEFAULT_SOCKET = Path(__file__).resolve().parents[1] / "outputs/control/prompt.sock"
MAX_MESSAGE = 16384


def read_message(connection) -> dict:
    """Read one bounded UTF-8 JSON line from a socket; return its object."""
    data = bytearray()
    while b"\n" not in data:
        chunk = connection.recv(4096)
        if not chunk:
            raise ValueError("Connection closed before a complete message")
        data.extend(chunk)
        if len(data) > MAX_MESSAGE:
            raise ValueError("Prompt message is too long")
    result = json.loads(data.split(b"\n", 1)[0])
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def send_prompt(task: str, path: Path = DEFAULT_SOCKET) -> dict:
    """Submit a task to running inference; return a queued acknowledgement."""
    return _exchange({"task": task}, path)


def send_continue(path: Path = DEFAULT_SOCKET) -> dict:
    """Release the current waiting gate; never queue permission for a future task."""
    return _exchange({"command": "continue"}, path)


def _exchange(message: dict, path: Path) -> dict:
    """Send one local request and return its acknowledgement or raise rejection."""
    request = json.dumps(message, ensure_ascii=False).encode() + b"\n"
    if len(request) > MAX_MESSAGE:
        raise ValueError("Prompt message is too long")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(2.0)
        connection.connect(str(path))
        connection.sendall(request)
        result = read_message(connection)
    if result.get("status") not in ("queued", "accepted"):
        raise ValueError(result.get("error", "Prompt rejected"))
    return result


class PromptController:
    """Keep the newest task for the next complete trajectory-block boundary."""

    def __init__(self, path: Path = DEFAULT_SOCKET):
        """Listen locally; a lifetime file lock prevents duplicate receivers."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._guard = self.path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self._guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._guard.close()
            raise RuntimeError("Another inference process owns the prompt receiver") from None
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.path.unlink(missing_ok=True)
            self._server.bind(str(self.path))
            self._server.listen(4)
            self._server.settimeout(0.1)
        except BaseException:
            self._server.close()
            self._guard.close()
            raise
        self._pending = None
        self._running = False
        self._waiting = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, name="prompt-receiver", daemon=True)
        self._thread.start()

    def request(self, line: str) -> str:
        """Validate and queue text; return it. The newest queued request wins."""
        if not isinstance(line, str):
            raise ValueError("Task must be text")
        text = line.strip()
        if text == "/prompt" or text.startswith("/prompt "):
            text = text[len("/prompt"):].strip()
        if not text:
            raise ValueError("Task must not be empty")
        with self._lock:
            self._pending = text
            self._running = False
            self._waiting = False
        return text

    def take(self) -> str | None:
        """Return and clear the latest request without blocking robot control."""
        with self._lock:
            pending, self._pending = self._pending, None
        return pending

    def poll_gate(self) -> tuple[str | None, bool]:
        """At a block boundary, consume task updates and expose the W waiting gate."""
        with self._lock:
            pending, self._pending = self._pending, None
            self._waiting = not self._running
            return pending, self._running

    def continue_task(self) -> None:
        """Accept W only at a waiting boundary; a new task always resets permission."""
        with self._lock:
            if not self._waiting:
                raise ValueError("Not waiting yet. Wait for 'Waiting for W' in terminal 1.")
            self._running = True
            self._waiting = False

    def _read(self) -> None:
        """Receive requests and acknowledge queuing, not application or completion."""
        while not self._stop.is_set():
            try:
                connection, _ = self._server.accept()
            except socket.timeout:
                continue
            with connection:
                connection.settimeout(0.3)
                try:
                    message = read_message(connection)
                    if message.get("command") == "continue":
                        self.continue_task()
                        reply = {"status": "accepted", "message": "W accepted. Continuing the current task."}
                    elif "command" in message:
                        raise ValueError("Unknown command")
                    else:
                        task = self.request(message.get("task"))
                        reply = {"status": "queued", "task": task,
                                 "message": "Wait for 'Waiting for W' in terminal 1, then enter W to start."}
                except (OSError, ValueError) as error:
                    reply = {"status": "rejected", "error": str(error)}
                try:
                    connection.sendall(json.dumps(reply).encode() + b"\n")
                except OSError:
                    pass

    def close(self) -> None:
        """Stop the receiver and remove its socket; no hardware commands are sent."""
        self._stop.set()
        self._thread.join()
        self._server.close()
        self.path.unlink(missing_ok=True)
        self._guard.close()
