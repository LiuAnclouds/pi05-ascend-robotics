#!/usr/bin/env python3
"""Run official Pi0.5 OM inference from saved tensors or live cameras."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

# Keep the documented ``python runtime/realtime_inference.py`` invocation
# working without requiring callers to set PYTHONPATH first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.camera import CameraPair
from runtime.console import number, print_iteration, section, status
from include.runtime_defs import (
    CAMERA_A_DEFAULT, CAMERA_B_DEFAULT, DEFAULT_PART1_OM, DEFAULT_PART2_OM,
    DEFAULT_STATS, DEFAULT_TOKENIZER,
)
from runtime.input_data import load_saved_inputs
from runtime.piper import (
    connect_piper, parse_state, quick_stop, read_piper_state,
    send_motion_chunk, prepare_motion, require_motion_ready, MotionNotReady,
)
from runtime.report import RunReport, timestamp
from include.project_paths import DEFAULT_RUN_DIR


def parse_args() -> argparse.Namespace:
    """Parse the real-time inference command-line arguments.

    Args:
        None. Arguments are read from ``sys.argv``.

    Returns:
        Parsed runtime options as an ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description="Run official Pi0.5 OM inference from saved tensors or live cameras.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common = parser.add_argument_group(
        "common input",
        "Values normally chosen for each task; model paths use the verified project defaults.",
    )
    common.add_argument("--task", required=True,
                        help="Required, non-empty task instruction; no default prompt.")
    common.add_argument("--input", type=Path,
                        help="Saved Part1 input (.pt); when set, cameras and Piper state are not read. Omit for live cameras.")
    common.add_argument("--steps", type=int, default=10,
                        help="Number of Part2 denoising steps per prediction.")
    common.add_argument("--iterations", type=int, default=0,
                        help="Prediction count; 0 means run continuously until Ctrl+C.")
    common.add_argument("--output", type=Path,
                        help="JSON report path; defaults to outputs/runs/Infer_report_<timestamp>/result.json.")

    hardware = parser.add_argument_group(
        "hardware input",
        "Camera and Piper controls. Motion is opt-in and remains disabled by default.",
    )
    hardware.add_argument("--camera-a", default=CAMERA_A_DEFAULT,
                          help="Third-person camera device.")
    hardware.add_argument("--camera-b", default=CAMERA_B_DEFAULT,
                          help="Wrist camera device.")
    hardware.add_argument("--can", default="can0",
                          help="SocketCAN interface used to read Piper state.")
    hardware.add_argument("--send-motion", action="store_true",
                          help="Send predicted trajectory to Piper; omitted means shadow mode.")
    hardware.add_argument("--motion-speed", type=int, default=15,
                          help="Piper command speed when --send-motion is enabled.")
    hardware.add_argument("--action-fps", type=float, default=30.0,
                          help="Trajectory command rate when --send-motion is enabled.")

    advanced = parser.add_argument_group(
        "advanced overrides",
        "Developer-only overrides for candidate OM, diagnostics, warm-up, and reproducibility.",
    )
    advanced.add_argument("--part1-om", type=Path, default=DEFAULT_PART1_OM,
                          help="Advanced: override the verified Part1 OM candidate.")
    advanced.add_argument("--part2-om", type=Path, default=DEFAULT_PART2_OM,
                          help="Advanced: override the verified Part2 OM candidate.")
    advanced.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER,
                          help="Advanced: override the tokenizer model.")
    advanced.add_argument("--stats", type=Path, default=DEFAULT_STATS,
                          help="Advanced: override normalization statistics JSON.")
    advanced.add_argument("--state",
                          help="Seven current state values for saved-input shadow inference; motion always reads Piper.")
    advanced.add_argument("--warmup", type=int, default=5,
                          help="Advanced: camera frames discarded before the first prediction.")
    advanced.add_argument("--seed", type=int, default=0,
                          help="Advanced: deterministic initial action-noise seed.")
    args = parser.parse_args()
    if not args.task.strip():
        parser.error("--task must contain a non-empty task instruction")
    if not np.isfinite(args.action_fps) or args.action_fps <= 0:
        parser.error("--action-fps must be finite and positive")
    return args


def main() -> int:
    """Run inference, persist each prediction/execution, and return a process exit code.

    Inputs come from CLI/devices. Reports include full actions even if sending
    fails or Ctrl+C interrupts a block. Warm-up never sends motion commands.
    """
    args = parse_args()
    started_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    if args.output is None:
        args.output = DEFAULT_RUN_DIR / f"Infer_report_{started_at:%Y%m%d_%H%M%S_%f}" / "result.json"
    report = RunReport(args.output, vars(args))
    section("Pi0.5 OpenPI | Real-time inference")
    status("Task", args.task)
    status("Cameras", f"A: {args.camera_a} (front) | B: {args.camera_b} (wrist)" if not args.input else "Saved tensor input")
    status("Mode", f"{'MOTION' if args.send_motion else 'INFERENCE ONLY'} | {args.steps} denoising steps | 50 points | {args.action_fps:g} Hz | speed {args.motion_speed}")
    status("Report", str(args.output), "title")
    policy = cameras = piper = None
    motion = args.send_motion
    iteration = completed = 0
    previous_last_command = None
    motion_requested = False
    outcome, failure = "completed", None
    stage = "device_setup"

    def motion_event(name, data):
        """Persist control transitions and print only meaningful operator messages."""
        report.event(name, data)
        messages = {"startup_recovery": "Recovering startup stop/mode; motor torque may briefly drop.",
                    "motion_ready": "CAN_CTRL | NORMAL | MOVE_J | enabled 6/6 (stable feedback)"}
        if name in messages:
            status("Arm", messages[name], "ok" if name == "motion_ready" else "wait")

    try:
        section("1/4 | Device setup", "Connect cameras and prepare the arm before loading models.")
        report.event("stage_started", {"stage": stage})
        state = parse_state(args.state) if args.state and not args.send_motion else None
        if args.input is None:
            cameras = CameraPair(args.camera_a, args.camera_b, args.warmup)
            status("Cameras", "Front and wrist connected", "ok")
        else:
            status("Devices", "Saved inputs; live cameras not required")
        if args.send_motion or (args.input is None and state is None):
            piper = connect_piper(args.can, announce=False)
            state = read_piper_state(piper)
            status("CAN", f"{args.can} connected")
        if motion:
            motion_requested = True
            prepare_motion(piper, args.motion_speed, on_event=motion_event)
        report.event("devices_ready", {"motion_enabled": motion})
        status("Devices", "Ready", "ok")

        stage = "model_loading"
        section("2/4 | Model loading", "Loading Part1 / Part2 OM...")
        report.event("stage_started", {"stage": stage})
        saved_inputs = load_saved_inputs(args.input) if args.input else None
        with report.diagnostics():
            from runtime.official_om_policy import OfficialOMPolicy
            policy = OfficialOMPolicy(args.part1_om, args.part2_om, args.tokenizer, args.stats, args.steps)
        report.event("models_ready", {})
        stage = "warmup"
        section("3/4 | Warm-up")
        report.event("stage_started", {"stage": stage})
        status("Warm-up", "One prediction; output discarded, no motion.")
        warmup_started = time.perf_counter()
        if saved_inputs is not None:
            warmup_inputs, warmup_state = saved_inputs, state
        else:
            first, second, _ = cameras.read()
            warmup_state = read_piper_state(piper) if piper is not None else state
            warmup_inputs = policy.make_inputs(cv2.cvtColor(first, cv2.COLOR_BGR2RGB),
                                              cv2.cvtColor(second, cv2.COLOR_BGR2RGB), warmup_state, args.task)
        warmup_result = policy.predict_inputs(warmup_inputs, seed=args.seed, state=warmup_state)
        if not np.isfinite(warmup_result["action_delta"]).all():
            raise RuntimeError("Model warm-up produced nonfinite actions")
        report.event("warmup_finished", {"duration_ms": (time.perf_counter() - warmup_started) * 1000.0})
        del warmup_inputs, warmup_result
        # Loading can take tens of seconds. Check once that the startup state
        # still holds, but never reset/re-enable here or inside the action loop.
        stage = "before_inference"
        if motion:
            require_motion_ready(piper)
        stage = "inference"
        report.event("stage_started", {"stage": stage})
        section("4/4 | Inference running", "Press Ctrl+C to stop. Full results are saved in the report journal.")

        while args.iterations == 0 or iteration < args.iterations:
            cycle_started = time.perf_counter()
            capture_ms, camera_metadata = 0.0, None
            observation_at = timestamp()
            observation_started = time.perf_counter()
            if saved_inputs is not None:
                inputs = saved_inputs
                if piper is not None:
                    state = read_piper_state(piper)
            else:
                first, second, capture_ms = cameras.read()
                camera_metadata = cameras.metadata()
                if piper is not None:
                    state = read_piper_state(piper)
                inputs = policy.make_inputs(cv2.cvtColor(first, cv2.COLOR_BGR2RGB),
                                            cv2.cvtColor(second, cv2.COLOR_BGR2RGB), state, args.task)
            inference_started_at = timestamp()
            started = time.perf_counter()
            result = policy.predict_inputs(inputs, seed=args.seed + iteration, state=state)
            total_ms = (time.perf_counter() - started) * 1000.0
            action_target = result["action_target"]
            action_raw = action_target if action_target is not None else result["action_delta"]
            record = {
                "iteration": iteration, "timestamp": timestamp(), "observation_at": observation_at,
                "inference_started_at": inference_started_at, "inference_finished_at": timestamp(),
                "seed": args.seed + iteration,
                "mode": "saved_input" if saved_inputs is not None else ("live_motion" if motion else "live_shadow"),
                "task": args.task, "action_shape": list(action_raw.shape),
                "action_finite": bool(np.isfinite(action_raw).all()),
                "action_space": "absolute_piper_targets" if action_target is not None else "joint_delta_gripper_absolute",
                "piper_state_raw": state.tolist() if state is not None else None,
                "action_normalized": result["action_normalized"].tolist(),
                "action_delta": result["action_delta"].tolist(), "action_raw": action_raw.tolist(),
                "first_action_delta": result["action_delta"][0, 0].tolist(),
                "first_action_target": action_target[0, 0].tolist() if action_target is not None else None,
                "motion_sent": False, "motion_horizon": int(action_raw.shape[1]), "motion_points_sent": 0,
                "action_fps": args.action_fps, "motion_speed": args.motion_speed, "denoising_steps": args.steps,
                "camera_observation": camera_metadata, "execution": {},
                "timing_ms": {"capture": capture_ms, "part1": result["part1_ms"],
                              "part2_mean": float(np.mean(result["part2_ms"])),
                              "part2_steps": result["part2_ms"], "total": total_ms,
                              "preprocess": (started - observation_started) * 1000.0 - capture_ms,
                              "motion": 0.0, "between_chunks": None, "between_chunks_extra": None},
                "outcome": "predicted",
            }
            write_started = time.perf_counter()
            report.event("prediction", record)
            record["timing_ms"]["prediction_journal"] = (time.perf_counter() - write_started) * 1000.0
            motion_started = None
            try:
                if not record["action_finite"]:
                    raise RuntimeError("Model prediction contains nonfinite actions")
                if motion:
                    if action_target is None:
                        raise RuntimeError("Cannot send delta actions without the current Piper state")
                    motion_started = time.perf_counter()
                    send_motion_chunk(piper, action_target[0], args.motion_speed, args.action_fps,
                                      prepared=True, progress=record["execution"])
                    record["outcome"] = "completed"
                else:
                    record["outcome"] = "completed"
            except MotionNotReady as error:
                record.update(outcome="control_interrupted", error=str(error), error_feedback=error.feedback)
                raise
            except BaseException as error:
                record.update(outcome="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                              error=f"{type(error).__name__}: {error}")
                raise
            finally:
                execution = record["execution"]
                record["motion_points_sent"] = execution.get("points_sent", 0)
                record["motion_sent"] = execution.get("joint_calls_completed", 0) > 0
                if motion_started is not None:
                    record["timing_ms"]["motion"] = (time.perf_counter() - motion_started) * 1000.0
                first_command = execution.get("first_command_time")
                if first_command is not None:
                    record["timing_ms"]["observation_to_first_command"] = (first_command - observation_started) * 1000.0
                    record["timing_ms"]["max_command_interval"] = execution["max_command_interval_ms"]
                    if previous_last_command is not None:
                        gap = (first_command - previous_last_command) * 1000.0
                        record["timing_ms"].update(between_chunks=gap, between_chunks_extra=max(0.0, gap - 1000.0 / args.action_fps))
                    previous_last_command = execution["last_command_time"]
                record["timing_ms"]["cycle"] = (time.perf_counter() - cycle_started) * 1000.0
                report.event("round_finished", record)
                print_iteration(record, args.steps)
                iteration += 1
                completed += int(record["outcome"] == "completed")
    except KeyboardInterrupt:
        outcome = "interrupted"
        status("Stop", "Ctrl+C received", "wait")
    except Exception as error:
        outcome, failure = "failed", f"{type(error).__name__}: {error}"
        report.event("failure", {"stage": stage, "error": failure, "feedback": getattr(error, "feedback", None)})
        with report.diagnostics():
            traceback.print_exc()
        status("Error", f"{stage}: {error}", "error")
    finally:
        # Each cleanup runs even when another cleanup operation raises.
        cleanup = []
        if motion_requested and piper is not None:
            cleanup.append(("quick_stop", lambda: quick_stop(piper)))
        if cameras is not None:
            cleanup.append(("cameras_close", cameras.close))
        if piper is not None:
            cleanup.append(("can_disconnect", piper.DisconnectPort))
        if policy is not None:
            cleanup.append(("model_close", policy.close))
        for name, operation in cleanup:
            try:
                with report.diagnostics():
                    operation()
                report.event(name, {"result": "request_sent" if name == "quick_stop" else "finished"})
            except Exception as error:
                outcome, failure = "failed", failure or f"{name}: {error}"
                report.event("cleanup_error", {"operation": name, "error": str(error)})
        report.finish(outcome, {"predictions": iteration, "completed_rounds": completed,
                                "last_stage": stage, "error": failure})
        section("Inference stopped")
        status("Result", f"{outcome} | predictions={iteration} | completed={completed}")
        status("Report", str(args.output), "title")
    return 1 if outcome == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
