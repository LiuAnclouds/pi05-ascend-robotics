"""Piper connection, state conversion, and motion helpers."""

from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np

from include.runtime_defs import PIPER_STATE_DIM


def parse_state(value: str) -> np.ndarray:
    """Parse seven comma- or space-separated raw Piper values.

    Args:
        value: Text containing six joint values and one gripper value.

    Returns:
        A float32 array with shape ``[7]`` in model state units.
    """
    values = [float(item.strip()) for item in value.replace(" ", ",").split(",") if item.strip()]
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (PIPER_STATE_DIM,):
        raise ValueError(f"--state must contain exactly {PIPER_STATE_DIM} values")
    return state


def connect_piper(can: str, announce: bool = True):
    """Connect to Piper on ``can`` without enabling motion.

    Args:
        can: SocketCAN interface name, normally ``can0``.
        announce: Print the legacy JSON connection message when true. The
            formatted real-time entry disables it and prints its own status.

    Returns:
        An initialized Piper interface object with motion still disabled.
    """
    from piper_sdk import C_PiperInterface_V2

    piper = C_PiperInterface_V2(can)
    piper.ConnectPort()
    time.sleep(0.3)
    if announce:
        print(json.dumps({"piper": "connected", "can": can, "motion": "disabled"}), flush=True)
    return piper


def read_piper_state(piper) -> np.ndarray:
    """Read Piper joints and gripper in the model's raw units.

    Args:
        piper: Connected Piper interface object.

    Returns:
        A float32 array with six joint values and one gripper value.
    """
    joints = piper.GetArmJointMsgs().joint_state
    gripper = piper.GetArmGripperMsgs()
    raw = np.asarray(
        [joints.joint_1, joints.joint_2, joints.joint_3, joints.joint_4, joints.joint_5, joints.joint_6,
         gripper.gripper_state.grippers_angle], dtype=np.float32
    )
    return raw / 1000.0


def read_motion_status(piper) -> dict:
    """Return copied controller/driver feedback; input is an SDK connection.

    No CAN commands are sent. Timestamps are SDK receive times; enable bits
    describe driver state, not proof that a target has been reached.
    """
    message = piper.GetArmStatus()
    arm = message.arm_status
    result = {"control_mode": int(arm.ctrl_mode), "control_mode_name": str(arm.ctrl_mode),
              "arm_status": int(arm.arm_status), "arm_status_name": str(arm.arm_status),
              "move_mode": int(arm.mode_feed), "teach_status": int(arm.teach_status),
              "error_code": int(arm.err_code), "status_timestamp": float(message.time_stamp)}
    message = piper.GetArmLowSpdInfoMsgs()
    codes = [int(getattr(message, f"motor_{i}").foc_status_code) for i in range(1, 7)]
    result.update(driver_timestamp=float(message.time_stamp),
                  enabled=[bool(code & 0x40) for code in codes],
                  driver_faults=[code & 0xBF for code in codes])
    return result


class MotionNotReady(RuntimeError):
    """Carry actual controller feedback when motion must not continue."""

    def __init__(self, reason: str, feedback: dict):
        """Store the reason and snapshot; no control commands are issued."""
        self.feedback = feedback
        super().__init__(f"{reason}: {feedback['control_mode_name']} / "
                         f"{feedback['arm_status_name']}; enabled={sum(feedback['enabled'])}/6; "
                         f"error={feedback['error_code']}; driver_faults={feedback['driver_faults']}")


def _validate_feedback(state: dict, startup: bool = False) -> None:
    """Validate freshness/faults; startup alone may accept stop/teaching states."""
    ages = [time.time() - state[key] for key in ("status_timestamp", "driver_timestamp")]
    if not all(0 <= age <= 1.0 for age in ages):
        raise MotionNotReady("Controller feedback is missing or older than 1 second", state)
    allowed = (0, 1, 11, 12, 13) if startup else (0,)
    if state["arm_status"] not in allowed or state["error_code"] or any(state["driver_faults"]):
        raise MotionNotReady("Controller stopped or faulted; automatic fault reset is disabled", state)
    if (state["control_mode"] not in range(8) or state["move_mode"] not in range(6)
            or state["teach_status"] not in range(8)):
        raise MotionNotReady("Unknown controller mode", state)


def _ready(state: dict) -> bool:
    """Return whether feedback confirms CAN/MOVE_J, normal and six enabled joints."""
    return (state["control_mode"] == 1 and state["arm_status"] == 0
            and state["move_mode"] == 1 and state["teach_status"] == 0
            and all(state["enabled"]))


def require_motion_ready(piper) -> dict:
    """Read and validate motion feedback; return it or raise without recovery."""
    state = read_motion_status(piper)
    _validate_feedback(state)
    if not _ready(state):
        raise MotionNotReady("Controller mode or enable state changed", state)
    return state


def prepare_motion(piper, speed: int, *, on_event: Callable | None = None) -> dict:
    """Switch to CAN/MOVE_J and require 0.5 seconds of fresh, stable feedback.

    Args:
        piper: Connected Piper interface object.
        speed: Piper motion speed from 1 to 100.
        on_event: Optional callback accepting an event name and JSON-compatible data.

    Returns:
        Before/after feedback and recovery details. Raises within six seconds
        on faults or unsuccessful transition. No joint/gripper target is sent.

    The SDK's recovery opcode may temporarily remove motor torque. Normal
    standby preparation only enables/selects mode; it never sends this opcode.
    Call only during device setup, before allocating model resources.
    """
    initial = read_motion_status(piper)
    _validate_feedback(initial, startup=True)
    emit = on_event or (lambda name, data: None)
    emit("motion_prepare", {"feedback": initial})
    teaching = (initial["control_mode"] in (2, 6) or initial["arm_status"] in (11, 12, 13)
                or initial["teach_status"] != 0)
    if teaching or initial["control_mode"] == 7:
        teach_command = (2 if initial["teach_status"] == 1 or initial["arm_status"] == 11
                         else 6) if teaching else 0
        piper.MotionCtrl_1(0, 6, teach_command)
        emit("teaching_trajectory_terminated", {})
        time.sleep(0.05)
    reset = initial["arm_status"] == 1 or teaching or initial["move_mode"] == 4
    if reset:
        emit("startup_recovery", {"note": "SDK reset may briefly remove motor torque"})
        piper.MotionCtrl_1(2, 0, 0)
        # Do not immediately treat old enable flags as confirmation of reset.
        time.sleep(0.5)
    deadline = time.perf_counter() + 6.0
    requested_at = time.time()
    stable_since = None
    last_stamp = 0.0
    mode_sent = False
    recovered = not reset
    piper.EnablePiper()
    time.sleep(0.05)
    state = initial
    while time.perf_counter() < deadline:
        state = read_motion_status(piper)
        _validate_feedback(state, startup=not recovered)
        if state["arm_status"] == 0:
            recovered = True
        new_feedback = min(state["status_timestamp"], state["driver_timestamp"])
        if _ready(state) and mode_sent and new_feedback > requested_at:
            if new_feedback > last_stamp:
                if stable_since is None:
                    stable_since = time.perf_counter()
                if time.perf_counter() - stable_since >= 0.5:
                    result = {"before": initial, "after": state, "reset_requested": reset}
                    emit("motion_ready", result)
                    return result
                last_stamp = new_feedback
        else:
            stable_since = None
            if not all(state["enabled"]):
                piper.EnablePiper()
            elif new_feedback > requested_at:
                piper.ModeCtrl(1, 1, max(1, min(100, speed)), 0)
                if not mode_sent:
                    requested_at = time.time()
                    mode_sent = True
        time.sleep(0.05)
    raise MotionNotReady("Timed out waiting for stable CAN/MOVE_J and enabled joints", state)


def send_motion_chunk(
    piper, action: np.ndarray, speed: int, fps: float, prepared: bool = False,
    progress: dict | None = None,
) -> dict:
    """Send a denormalized seven-dimensional action trajectory to Piper.

    Args:
        piper: Connected Piper interface object.
        action: Array shaped ``[trajectory_length, 7]`` in raw Piper units.
        speed: Piper motion speed from 1 to 100.
        fps: Command interval in frames per second.
        prepared: Whether :func:`prepare_motion` was already called for this
            session. Keeping the default ``False`` preserves standalone callers.
        progress: Optional mutable record, updated even if sending is interrupted.

    Returns:
        Command count, first/last command start times from ``perf_counter``,
        and the largest command interval in milliseconds. These are host-side
        send times, not controller acknowledgements or arrival at a target.
    """
    trajectory = np.asarray(action, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != PIPER_STATE_DIM or not np.isfinite(trajectory).all():
        raise ValueError(f"invalid Piper action shape/value: {trajectory.shape}")
    if len(trajectory) == 0:
        raise ValueError("motion trajectory must contain at least one point")
    if fps <= 0:
        raise ValueError("action fps must be positive")
    if not prepared:
        prepare_motion(piper, speed)
    progress = progress if progress is not None else {}
    progress.update(points_sent=0, joint_calls_completed=0, gripper_calls_completed=0,
                    first_command_time=None, last_command_time=None,
                    max_command_interval_ms=0.0, feedback_samples=[])
    interval = 1.0 / fps
    next_tick = time.perf_counter()
    first_command = last_command = None
    max_interval = 0.0
    for index, point in enumerate(trajectory):
        feedback = read_motion_status(piper)
        progress["last_feedback"] = feedback
        _validate_feedback(feedback)
        if not _ready(feedback):
            raise MotionNotReady("Controller mode or enable state changed during sending", feedback)
        sample = {"point": index, "host_time": time.time(),
                  "state_raw": read_piper_state(piper).tolist(), "controller": feedback}
        progress["feedback_samples"].append(sample)
        joints = np.rint(point[:6] * 1000.0).astype(np.int32)
        command_started = time.perf_counter()
        if first_command is None:
            first_command = command_started
        if last_command is not None:
            max_interval = max(max_interval, command_started - last_command)
        last_command = command_started
        piper.JointCtrl(*[int(value) for value in joints])
        progress.update(joint_calls_completed=index + 1, first_command_time=first_command,
                        last_command_time=last_command, max_command_interval_ms=max_interval * 1000.0)
        piper.GripperCtrl(int(round(float(point[6]) * 1000.0)), 1000, 0x01, 0x00)
        progress.update(points_sent=index + 1, gripper_calls_completed=index + 1, first_command_time=first_command,
                        last_command_time=last_command, max_command_interval_ms=max_interval * 1000.0)
        next_tick += interval
        time.sleep(max(0.0, next_tick - time.perf_counter()))
    progress["after_feedback"] = read_motion_status(piper)
    progress["state_after_raw"] = read_piper_state(piper).tolist()
    return progress


def quick_stop(piper) -> None:
    """Issue Piper quick-stop for Ctrl+C handling.

    Args:
        piper: Connected Piper interface object.

    Returns:
        None. A quick-stop command is sent to the physical arm.
    """
    piper.MotionCtrl_1(0x01, 0x00, 0x00)
