#!/usr/bin/env python3
"""Enter tasks in a second board terminal; requires only standard Python."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.prompt import DEFAULT_SOCKET, send_prompt, send_continue


def main() -> int:
    """Read interactive tasks or --task once; print acceptance/errors, send no motion."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--task", help="Queue a task and pause at the next block; omit for interactive input.")
    actions.add_argument("--continue", dest="resume", action="store_true",
                         help="Send W once to release the current waiting gate.")
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET,
                        help="Override the inference socket for a separate project instance.")
    args = parser.parse_args()

    def submit(task=None):
        """Submit text and report queued status; return success as a boolean."""
        try:
            reply = send_continue(args.socket) if task is None else send_prompt(task, args.socket)
        except (OSError, ValueError) as error:
            print(f"Not accepted: {error}", flush=True)
            return False
        if 'task' in reply:
            print(f"Queued: {reply['task']}", flush=True)
        print(reply['message'], flush=True)
        return True

    if args.task is not None:
        return 0 if submit(args.task) else 1
    if args.resume:
        return 0 if submit() else 1
    print("Enter a new task to queue it. Enter W to continue when terminal 1 is waiting.")
    print("/quit or Ctrl+C closes this input program only.")
    while True:
        try:
            text = input("Task > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nPrompt input closed. Inference is unchanged.")
            return 0
        if text == "/quit":
            return 0
        if text.lower() == "w":
            submit()
            continue
        if text:
            submit(text)


if __name__ == "__main__":
    raise SystemExit(main())
