"""Piper connection, state conversion, and motion helpers."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from threading import RLock
from typing import Callable

import numpy as np

from inference.include.definitions import PIPER_FEEDBACK_TIMEOUT_SECONDS, PIPER_STATE_DIM

try:
    from piper_sdk import C_PiperInterface_V2
except ModuleNotFoundError:
    C_PiperInterface_V2 = None


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


def connect_piper(
    can: str,
    announce: bool = True,
    *,
    joint_limits: bool = False,
    gripper_limits: bool = False,
):
    """Connect to Piper on ``can`` without enabling motion.

    Args:
        can: SocketCAN interface name, normally ``can0``.
        announce: Print the legacy JSON connection message when true. The
            formatted real-time entry disables it and prints its own status.
        joint_limits: Enable Piper SDK joint target/feedback limits.
        gripper_limits: Enable Piper SDK gripper target/feedback limits.

    Returns:
        An initialized Piper interface object with motion still disabled.
    """
    if C_PiperInterface_V2 is None:
        raise RuntimeError("Install the Piper SDK before using Piper hardware control")

    piper = C_PiperInterface_V2(
        can,
        start_sdk_joint_limit=joint_limits,
        start_sdk_gripper_limit=gripper_limits,
    )
    try:
        piper.ConnectPort()
    except BaseException:
        piper.DisconnectPort()
        raise
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
        [
            joints.joint_1,
            joints.joint_2,
            joints.joint_3,
            joints.joint_4,
            joints.joint_5,
            joints.joint_6,
            gripper.gripper_state.grippers_angle,
        ],
        dtype=np.float32,
    )
    return raw / 1000.0


def fit_serial_trajectory(
    action: np.ndarray,
    current_state: np.ndarray,
    jerk_penalty_weight: float,
    terminal_preserve_points: int = 10,
    jitter_range_deg: tuple[float, ...] = (0.1,) * 6,
    jitter_net_ratio: float = 0.25,
) -> tuple[np.ndarray, dict]:
    """平滑机械臂轨迹，并保留末段持续运动和压平局部微抖动。

    Args:
        action: 六轴角度和夹爪毫米值组成的 [N,7] 模型动作。
        current_state: Piper 当前的 [7] 轴实测值。
        jerk_penalty_weight: 三阶差分惩罚权重；0 表示保留原预测。
        terminal_preserve_points: 末段逐渐保留模型原始关节目标的点数。
        jitter_range_deg: J1 至 J6 各自的五点抖动角度范围。
        jitter_net_ratio: 抖动窗口首尾位移与总变化量的最大比例。
    Returns:
        拟合后的 [N,7] 动作及调整量、拟合输入输出记录。
    Side effects: None.
    """
    source = np.asarray(action, dtype=np.float64)
    start = np.asarray(current_state, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != PIPER_STATE_DIM or len(source) == 0:
        raise ValueError("Expected a nonempty serial action with seven axes")
    if start.shape != (PIPER_STATE_DIM,):
        raise ValueError("Expected seven measured Piper axes")
    if not np.isfinite(source).all() or not np.isfinite(start).all():
        raise ValueError("Serial action and measured state must be finite")
    if (
        type(jerk_penalty_weight) not in (int, float)
        or not np.isfinite(jerk_penalty_weight)
        or jerk_penalty_weight < 0
    ):
        raise ValueError("Trajectory jerk penalty weight must be finite and nonnegative")
    jitter_range = np.asarray(jitter_range_deg, dtype=np.float64)
    if jitter_range.shape != (6,) or not np.isfinite(jitter_range).all() or np.any(jitter_range <= 0):
        raise ValueError("Expected six positive finite per-joint jitter ranges")
    if type(terminal_preserve_points) is not int or terminal_preserve_points < 5:
        raise ValueError("Terminal preservation must cover at least five action points")
    if type(jitter_net_ratio) not in (int, float) or not np.isfinite(jitter_net_ratio) or not 0 < jitter_net_ratio < 1:
        raise ValueError("Jitter net ratio must be finite and between zero and one")

    model_joints = source[:, :6]
    fitted_joints = model_joints.copy()
    point_count = len(source)
    if jerk_penalty_weight > 0 and point_count >= 3:
        waypoints = np.vstack((start[:6], model_joints))
        jerk = np.diff(np.eye(len(waypoints), dtype=np.float64), n=3, axis=0)
        interior_jerk = jerk[:, 1:-1]
        fixed_endpoints = jerk[:, [0, -1]] @ np.vstack((start[:6], model_joints[-1]))
        system = np.eye(point_count - 1) + jerk_penalty_weight * (interior_jerk.T @ interior_jerk)
        right_hand_side = model_joints[:-1] - jerk_penalty_weight * (interior_jerk.T @ fixed_endpoints)
        interior = np.linalg.solve(system, right_hand_side)
        fitted_joints = np.vstack((start[:6], interior, model_joints[-1]))[1:]

    # 末段只在原始轨迹持续同向时，逐渐减少拟合改动。
    preserve_count = min(terminal_preserve_points, point_count) if point_count >= 5 else 0
    preserve_start = point_count - preserve_count
    for point_index in range(preserve_start, point_count):
        left = min(max(point_index - 2, 0), point_count - 5)
        window = model_joints[left : left + 5]
        delta = np.diff(window, axis=0)
        path_length = np.sum(np.abs(delta), axis=0)
        coherence = np.divide(
            np.abs(window[-1] - window[0]),
            path_length,
            out=np.zeros(6, dtype=np.float64),
            where=path_length > 0,
        )
        progress = (point_index - preserve_start + 1) / preserve_count
        ramp = progress * progress * (3 - 2 * progress)
        model_weight = ramp * coherence
        fitted_joints[point_index] += model_weight * (
            model_joints[point_index] - fitted_joints[point_index]
        )

    # 五点窗口只用来确认反复变化；通过后将中间三点压成一个角度。
    jitter_flattened_points = 0
    for joint_index in range(6):
        threshold = jitter_range[joint_index]
        epsilon = threshold / 10
        centers = []
        for center in range(2, point_count - 2):
            window = model_joints[center - 2 : center + 3, joint_index]
            deltas = np.diff(window)
            significant = deltas[np.abs(deltas) > epsilon]
            path_length = np.sum(np.abs(deltas))
            if (
                np.ptp(window) <= threshold
                and len(significant) >= 3
                and np.count_nonzero(np.diff(np.sign(significant))) >= 2
                and abs(window[-1] - window[0]) <= jitter_net_ratio * path_length
            ):
                centers.append(center)

        segment_start = segment_end = None
        for center in (*centers, None):
            if center is not None:
                start_index, end_index = center - 1, center + 1
                joins_segment = segment_end is not None and start_index <= segment_end + 1
                combined_start = segment_start if joins_segment else start_index
                combined_end = max(segment_end, end_index) if joins_segment else end_index
                segment = model_joints[combined_start : combined_end + 1, joint_index]
                if joins_segment and np.ptp(segment) <= threshold:
                    segment_end = combined_end
                    continue
            if segment_start is not None:
                segment_values = fitted_joints[segment_start : segment_end + 1, joint_index]
                plateau = np.median(segment_values)
                jitter_flattened_points += int(np.count_nonzero(segment_values != plateau))
                fitted_joints[segment_start : segment_end + 1, joint_index] = plateau
            if center is None:
                segment_start = segment_end = None
            else:
                segment_start, segment_end = start_index, end_index

    fitted = source.copy()
    fitted[:, :6] = fitted_joints
    adjustment = fitted_joints - model_joints
    joint_steps = np.diff(np.vstack((start[:6], fitted_joints)), axis=0)
    return np.ascontiguousarray(fitted, dtype=np.float32), {
        "source_points": point_count,
        "command_points": point_count,
        "jerk_penalty_weight": float(jerk_penalty_weight),
        "terminal_preserve_points": preserve_count,
        "jitter_flattened_points": jitter_flattened_points,
        "max_joint_adjustment_deg": np.max(np.abs(adjustment), axis=0).tolist(),
        "rms_joint_adjustment_deg": np.sqrt(np.mean(adjustment**2, axis=0)).tolist(),
        "max_joint_step_deg": np.max(np.abs(joint_steps), axis=0).tolist(),
    }


def limit_serial_joint_steps(
    action: np.ndarray,
    current_state: np.ndarray,
    max_joint_step_deg: float,
) -> tuple[np.ndarray, dict]:
    """Limit each fitted arm-joint target change while preserving gripper commands.

    Args:
        action: Fitted absolute targets [N,7] in degrees and millimetres.
        current_state: Measured Piper state [7] in the same units.
        max_joint_step_deg: Maximum consecutive target change for each arm joint.
    Returns:
        Limited targets and the largest target difference before and after limiting.
    Side effects: None. The limit changes command targets, not physical velocity.
    """
    source = np.asarray(action, dtype=np.float64)
    start = np.asarray(current_state, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != PIPER_STATE_DIM or len(source) == 0:
        raise ValueError("Expected a nonempty serial action with seven axes")
    if start.shape != (PIPER_STATE_DIM,) or not np.isfinite(source).all() or not np.isfinite(start).all():
        raise ValueError("Expected finite seven-axis action and measured state")
    if not np.isfinite(max_joint_step_deg) or max_joint_step_deg <= 0:
        raise ValueError("Maximum joint target step must be finite and positive")

    limited = source.copy()
    previous = start[:6].copy()
    for point in limited:
        point[:6] = previous + np.clip(point[:6] - previous, -max_joint_step_deg, max_joint_step_deg)
        previous = point[:6].copy()
    original_steps = np.diff(np.vstack((start[:6], source[:, :6])), axis=0)
    limited_steps = np.diff(np.vstack((start[:6], limited[:, :6])), axis=0)
    return np.ascontiguousarray(limited, dtype=np.float32), {
        "max_joint_step_before_limit_deg": np.max(np.abs(original_steps), axis=0).tolist(),
        "max_joint_step_after_limit_deg": np.max(np.abs(limited_steps), axis=0).tolist(),
        "max_joint_step_deg": np.max(np.abs(limited_steps), axis=0).tolist(),
    }



def read_piper_feedback(piper) -> dict:
    """Copy joint/gripper values and their SDK receive timestamps; send nothing.

    Args: piper is the connected SDK interface.
    Returns: raw state (degrees/mm), host sample time, individual feedback
        receive times and ages in milliseconds. These are not exposure times.
    """
    message = piper.GetArmJointMsgs()
    joints = message.joint_state
    joint_stamp = float(message.time_stamp)
    values = [getattr(joints, f"joint_{index}") for index in range(1, 7)]
    gripper = piper.GetArmGripperMsgs()
    values.append(gripper.gripper_state.grippers_angle)
    gripper_stamp = float(gripper.time_stamp)
    now = time.time()
    return {
        "host_time": now,
        "state_raw": (np.asarray(values, dtype=np.float32) / 1000.0).tolist(),
        "joint_timestamp": joint_stamp,
        "gripper_timestamp": gripper_stamp,
        "joint_age_ms": (now - joint_stamp) * 1000.0,
        "gripper_age_ms": (now - gripper_stamp) * 1000.0,
    }


def read_motion_status(piper) -> dict:
    """Return copied controller/driver feedback; input is an SDK connection.

    No CAN commands are sent. Timestamps are SDK receive times; enable bits
    describe driver state, not proof that a target has been reached.
    """
    message = piper.GetArmStatus()
    arm = message.arm_status
    result = {
        "control_mode": int(arm.ctrl_mode),
        "control_mode_name": str(arm.ctrl_mode),
        "arm_status": int(arm.arm_status),
        "arm_status_name": str(arm.arm_status),
        "move_mode": int(arm.mode_feed),
        "motion_status": int(arm.motion_status),
        "teach_status": int(arm.teach_status),
        "error_code": int(arm.err_code),
        "status_timestamp": float(message.time_stamp),
    }
    message = piper.GetArmLowSpdInfoMsgs()
    codes = [int(getattr(message, f"motor_{i}").foc_status_code) for i in range(1, 7)]
    result.update(
        driver_timestamp=float(message.time_stamp),
        enabled=[bool(code & 0x40) for code in codes],
        driver_faults=[code & 0xBF for code in codes],
    )
    return result


def require_piper_feedback(piper) -> dict:
    """Read finite, recent joint/gripper feedback without sending commands.

    Args: piper is a connected SDK interface.
    Returns: A feedback snapshot in degrees/mm, or raises when it is unusable.
    """
    feedback = read_piper_feedback(piper)
    state = np.asarray(feedback["state_raw"], dtype=np.float32)
    ages = (feedback["joint_age_ms"], feedback["gripper_age_ms"])
    if (
        state.shape != (PIPER_STATE_DIM,)
        or not np.isfinite(state).all()
        or not all(0 <= age <= PIPER_FEEDBACK_TIMEOUT_SECONDS * 1000.0 for age in ages)
    ):
        raise MotionNotReady("Joint/gripper feedback is missing, invalid or stale", read_motion_status(piper))
    return feedback


class MotionNotReady(RuntimeError):
    """Carry actual controller feedback when motion must not continue."""

    def __init__(self, reason: str, feedback: dict):
        """Store the reason and snapshot; no control commands are issued."""
        self.feedback = feedback
        super().__init__(
            f"{reason}: {feedback['control_mode_name']} / "
            f"{feedback['arm_status_name']}; enabled={sum(feedback['enabled'])}/6; "
            f"error={feedback['error_code']}; driver_faults={feedback['driver_faults']}"
        )


class RecoveryConfirmationRequired(MotionNotReady):
    """Separate a recoverable startup stop from faults that must remain blocked."""


def validate_motion_feedback(state: dict, startup: bool = False, recovering: bool = False) -> None:
    """Reject stale/faulted controller snapshots; send no commands.

    Args: state is SDK feedback; startup permits inspection of stopped/teach modes;
        recovering permits only the documented transient reset stop/comm status.
    Returns: None, or raises MotionNotReady without clearing controller faults.
    """
    ages = [time.time() - state[key] for key in ("status_timestamp", "driver_timestamp")]
    if not all(0 <= age <= PIPER_FEEDBACK_TIMEOUT_SECONDS for age in ages):
        raise MotionNotReady("Controller feedback is missing or older than 1 second", state)
    allowed = (0, 1, 11, 12, 13) if startup else (0,)
    reset_transition = recovering and state["arm_status"] in (1, 5) and state["error_code"] in (0, 63)
    if (not reset_transition and (state["arm_status"] not in allowed or state["error_code"])) or any(
        state["driver_faults"]
    ):
        raise MotionNotReady("Controller stopped or faulted; automatic fault reset is disabled", state)
    if (
        state["control_mode"] not in range(8)
        or state["move_mode"] not in range(6)
        or state["teach_status"] not in range(8)
        or state["motion_status"] not in (0, 1)
    ):
        raise MotionNotReady("Unknown controller mode", state)


def motion_recovery_required(state: dict) -> bool:
    """Classify feedback that must not be reset without on-site confirmation.

    Args: state is a validated controller snapshot.
    Returns: True for emergency-stop, teach, MIT or trajectory-replay states.
        The SDK cannot distinguish a physical emergency stop from a software one.
    """
    return (
        state["arm_status"] in (1, 11, 12, 13)
        or state["control_mode"] in (2, 6, 7)
        or state["teach_status"] != 0
        or state["move_mode"] == 4
    )


def _ready(state: dict) -> bool:
    """Return whether feedback confirms CAN/MOVE_J, normal and six enabled joints."""
    return (
        state["control_mode"] == 1
        and state["arm_status"] == 0
        and state["move_mode"] == 1
        and state["teach_status"] == 0
        and all(state["enabled"])
    )


def require_motion_ready(piper) -> dict:
    """Read and validate motion feedback; return it or raise without recovery."""
    state = read_motion_status(piper)
    validate_motion_feedback(state)
    if not _ready(state):
        raise MotionNotReady("Controller mode or enable state changed", state)
    return state


def prepare_motion(
    piper,
    speed: int,
    *,
    confirm_recovery: Callable[[dict], bool] | None = None,
    on_event: Callable | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    command_lock: RLock | None = None,
) -> dict:
    """Prepare the controller through acknowledged recovery, enable and mode stages.

    Args:
        piper: Connected Piper interface object.
        speed: Piper motion speed from 1 to 100.
        confirm_recovery: Operator confirmation required before a reset that
            may remove motor torque. Missing or negative confirmation fails closed.
        on_event: Optional callback accepting an event name and JSON-compatible data.
        cancel_requested: Optional nonblocking cancellation check between feedback polls.
        command_lock: Session lock shared with the target sender and stop path.

    Returns:
        Before/after feedback and recovery details. Each transition has a
        six-second limit. No joint/gripper target is sent.

    The SDK reset removes torque. Wait for normal, disabled feedback before
    enabling again; a fixed sleep does not establish that reset has finished.
    Failed transitions are reported without an automatic torque-removing
    disable/enable retry. Call only during device setup.
    """
    if type(speed) is not int or not 1 <= speed <= 100:
        raise ValueError("Piper motion speed must be an integer from 1 to 100")
    initial = read_motion_status(piper)
    validate_motion_feedback(initial, startup=True)
    emit = on_event or (lambda name, data: None)
    emit("motion_prepare", {"feedback": initial})

    def send(command, *, startup=False):
        """Serialize a setup command; inputs are a callable and validation mode.

        Returns None. Cancellation and fresh feedback are checked under the
        same lock used by stop, so a cancelled setup cannot re-enable afterward.
        """
        with command_lock if command_lock is not None else nullcontext():
            state = read_motion_status(piper)
            if cancel_requested is not None and cancel_requested():
                raise MotionNotReady("Startup cancelled before command", state)
            validate_motion_feedback(state, startup=startup)
            emit("motion_command_started", {})
            command()

    def wait_for(phase, predicate, command=None, recovery=False, stable_for=0.2):
        """Wait for fresh stable feedback, optionally retrying one transition command.

        Reset alone may temporarily report emergency-stop or joint communication
        status. No enable/mode/target command is issued during this recovery wait.
        Persistent recovery errors time out and remain visible in the report.
        """
        started = time.perf_counter()
        requested_at = time.time()
        last_send, last_stamp, stable_since = -float("inf"), 0.0, None
        emit("motion_stage_started", {"stage": phase})
        while time.perf_counter() - started < 6.0:
            if cancel_requested is not None and cancel_requested():
                raise MotionNotReady(f"Startup {phase} cancelled", read_motion_status(piper))
            state = read_motion_status(piper)
            transient_reset = (
                recovery
                and state["arm_status"] in (1, 5)
                and state["error_code"] in (0, 63)
                and not any(state["driver_faults"])
            )
            validate_motion_feedback(state, recovering=recovery)
            stamp = min(state["status_timestamp"], state["driver_timestamp"])
            now = time.perf_counter()
            if not transient_reset and predicate(state) and stamp > requested_at:
                if stamp > last_stamp:
                    stable_since = now if stable_since is None else stable_since
                    if now - stable_since >= stable_for:
                        emit(
                            "motion_stage_ready",
                            {"stage": phase, "feedback": state, "duration_ms": (now - started) * 1000.0},
                        )
                        return state
                    last_stamp = stamp
            else:
                stable_since = None
                if command is not None and now - last_send >= 0.1:
                    send(command)
                    last_send = now
            time.sleep(0.05)
        raise MotionNotReady(f"Timed out during startup {phase}", state)

    teaching = (
        initial["control_mode"] in (2, 6)
        or initial["arm_status"] in (11, 12, 13)
        or initial["teach_status"] != 0
    )
    reset = motion_recovery_required(initial)
    if reset and (confirm_recovery is None or not confirm_recovery(initial)):
        emit("startup_recovery_declined", {"feedback": initial})
        raise RecoveryConfirmationRequired("Startup recovery requires on-site operator confirmation", initial)
    if initial["control_mode"] not in (0, 1) and not reset:
        raise MotionNotReady("Another control mode owns the arm; automatic takeover is disabled", initial)
    if teaching or initial["control_mode"] == 7:
        teach_command = (
            (2 if initial["teach_status"] == 1 or initial["arm_status"] == 11 else 6) if teaching else 0
        )
        send(lambda: piper.MotionCtrl_1(0, 6, teach_command), startup=True)
        emit("teaching_trajectory_terminated", {})
        time.sleep(0.05)
    if reset:
        emit("startup_recovery", {"note": "SDK reset may briefly remove motor torque"})
        send(lambda: piper.MotionCtrl_1(2, 0, 0), startup=True)
        wait_for("recovery_disabled", lambda state: not any(state["enabled"]), recovery=True, stable_for=0.5)
    wait_for("enabled", lambda state: all(state["enabled"]), command=piper.EnablePiper)

    def mode():
        """Request validated CAN/MOVE_J speed; no targets are sent. Args/Returns: None."""
        piper.ModeCtrl(1, 1, speed, 0)

    # Always send speed/mode once even if the previous session was already ready.
    send(mode)
    state = wait_for("can_mode", _ready, command=mode, stable_for=0.5)
    result = {"before": initial, "after": state, "reset_requested": reset}
    emit("motion_ready", result)
    return result


def stop_motion(piper, speed: int, timeout: float = 2.0, *, command_lock: RLock | None = None) -> dict:
    """Request normal standby and confirm fresh feedback before returning.

    Args:
        piper: Connected Piper interface; its target sender must already be stopped.
        speed: Configured Piper motion speed from 1 to 100.
        timeout: Maximum seconds to wait for normal standby feedback.
        command_lock: Session lock shared with startup and trajectory commands.
    Returns:
        Fresh controller state with standby mode and all six drivers enabled.
    Side effects: Terminates the plan and requests standby, without reset or disable.
        Failure raises MotionNotReady; the caller reports the unconfirmed stop.
    """
    if type(speed) is not int or not 1 <= speed <= 100:
        raise ValueError("Piper motion speed must be an integer from 1 to 100")
    if timeout <= 0:
        raise ValueError("Stop timeout must be positive")
    requested_at = time.time()
    with command_lock if command_lock is not None else nullcontext():
        piper.MotionCtrl_1(0, 6, 0)
        piper.ModeCtrl(0, 1, speed, 0)
    deadline = time.perf_counter() + timeout
    last_stamp = requested_at
    stable_since = None
    state = read_motion_status(piper)
    while time.perf_counter() < deadline:
        state = read_motion_status(piper)
        validate_motion_feedback(state)
        stamp = min(state["status_timestamp"], state["driver_timestamp"])
        stopped = state["control_mode"] == 0 and state["teach_status"] == 0 and all(state["enabled"])
        # SDK motion_status is a target-arrival flag, not a velocity/standstill
        # measurement. Only acknowledge the requested standby mode here.
        now = time.perf_counter()
        if stopped and stamp > last_stamp:
            stable_since = now if stable_since is None else stable_since
            last_stamp = stamp
            if now - stable_since >= 0.1:
                return state
        elif not stopped:
            stable_since = None
            with command_lock if command_lock is not None else nullcontext():
                piper.ModeCtrl(0, 1, speed, 0)
        time.sleep(0.02)
    raise MotionNotReady("Normal stop was not confirmed; caller must escalate safely", state)


def send_motion_chunk(
    piper,
    action: np.ndarray,
    speed: int,
    fps: float,
    prepared: bool = False,
    progress: dict | None = None,
    check_pending: Callable | None = None,
    command_lock: RLock | None = None,
    cancel_wait: Callable[[float], bool] | None = None,
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
        check_pending: Optional nonblocking background-error check; must not
            perform disk I/O. Exceptions stop sending before the next point.
        command_lock: Shared session lock around each joint/gripper command pair.
        cancel_wait: Optional interruptible wait, normally threading.Event.wait.

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
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("action fps must be positive")
    if type(speed) is not int or not 1 <= speed <= 100:
        raise ValueError("Piper motion speed must be an integer from 1 to 100")
    if not prepared:
        prepare_motion(piper, speed)
    progress = progress if progress is not None else {}
    progress.update(
        points_sent=0,
        commands_attempted=False,
        joint_calls_completed=0,
        gripper_calls_completed=0,
        first_command_time=None,
        last_command_time=None,
        max_command_interval_ms=0.0,
        feedback_samples=[],
    )
    interval = 1.0 / fps
    next_tick = time.perf_counter()
    first_command = last_command = None
    max_interval = 0.0
    for index, point in enumerate(trajectory):
        if check_pending is not None:
            check_pending()
        sample = {
            "point": index,
            **read_piper_feedback(piper),
            "target_raw": point.tolist(),
        }
        progress["feedback_samples"].append(sample)
        joints = np.rint(point[:6] * 1000.0).astype(np.int32)
        command_started = time.perf_counter()
        if first_command is None:
            first_command = command_started
        sample["send_elapsed_seconds"] = command_started - first_command
        if last_command is not None:
            max_interval = max(max_interval, command_started - last_command)
        last_command = command_started
        with command_lock if command_lock is not None else nullcontext():
            if check_pending is not None:
                check_pending()
            # Record the attempt before the SDK call: an exception can follow a
            # partially transmitted multi-frame JointCtrl command.
            progress["commands_attempted"] = True
            piper.JointCtrl(*[int(value) for value in joints])
            progress["joint_calls_completed"] = index + 1
            if check_pending is not None:
                check_pending()
            piper.GripperCtrl(int(round(float(point[6]) * 1000.0)), 1000, 0x01, 0x00)
        progress.update(
            points_sent=index + 1,
            gripper_calls_completed=index + 1,
            first_command_time=first_command,
            last_command_time=last_command,
            max_command_interval_ms=max_interval * 1000.0,
        )
        next_tick += interval
        delay = max(0.0, next_tick - time.perf_counter())
        if cancel_wait is None:
            time.sleep(delay)
        elif cancel_wait(delay):
            if check_pending is not None:
                check_pending()
            raise RuntimeError("Trajectory sender cancelled during wait")
    progress["state_after_raw"] = read_piper_state(piper).tolist()
    return progress


def send_motion_timeline(
    piper,
    timeline,
    speed: int,
    fps: float,
    *,
    progress: dict | None = None,
    check_pending: Callable | None = None,
    on_replan: Callable | None = None,
    command_lock: RLock | None = None,
    timeline_lock=None,
    cancel_wait: Callable[[float], bool] | None = None,
) -> dict:
    """Send the actual RTC queue and request guidance at its owned control tick.

    Args: Connected/ready Piper, primed timeline, send rate and nonblocking callbacks.
    Returns: Live send telemetry; an empty queue waits until prediction or cancellation.
    Side effects: Sends joint/gripper targets. The replan callback receives the
        owned request and matching measured feedback; it must not block or send CAN.
        While waiting, no new targets or standby commands are sent; the control
        clock continues so a late reply still discards its elapsed prefix.
    """
    if not np.isfinite(fps) or fps <= 0 or type(speed) is not int or not 1 <= speed <= 100:
        raise ValueError("Invalid Piper timeline speed or frequency")
    if on_replan is None:
        raise ValueError("RTC timeline sending requires an on_replan callback")
    progress = progress if progress is not None else {}
    progress.update(
        points_sent=0,
        commands_attempted=False,
        joint_calls_completed=0,
        gripper_calls_completed=0,
        first_command_time=None,
        last_command_time=None,
        max_command_interval_ms=0.0,
        feedback_samples=[],
        waiting_for_prediction=False,
    )
    interval = 1.0 / fps
    next_deadline = time.perf_counter()
    first_command = last_command = None
    max_interval = 0.0
    while True:
        if check_pending is not None:
            check_pending()
        measured = read_piper_feedback(piper)
        with timeline_lock if timeline_lock is not None else nullcontext():
            request = None
            if timeline.should_request():
                request = timeline.begin_request(measured["state_raw"], time.monotonic())
            command = timeline.consume()
        if request is not None:
            on_replan(request, measured)
        command_started = time.perf_counter()
        progress["waiting_for_prediction"] = command is None
        if command is not None:
            point, source_request = command
            index = progress["points_sent"]
            sample = {
                "tick": timeline.tick - 1,
                "source_request": source_request,
                **measured,
                "target_raw": point.tolist(),
                "controller": feedback,
            }
            samples = progress["feedback_samples"]
            samples.append(sample)
            if len(samples) > 64:
                del samples[:32]
            joints = np.rint(point[:6] * 1000.0).astype(np.int32)
            command_started = time.perf_counter()
            if first_command is None:
                first_command = command_started
            if last_command is not None:
                max_interval = max(max_interval, command_started - last_command)
            last_command = command_started
            with command_lock if command_lock is not None else nullcontext():
                if check_pending is not None:
                    check_pending()
                progress["commands_attempted"] = True
                piper.JointCtrl(*[int(value) for value in joints])
                progress["joint_calls_completed"] = index + 1
                if check_pending is not None:
                    check_pending()
                piper.GripperCtrl(int(round(float(point[6]) * 1000.0)), 1000, 0x01, 0x00)
            progress.update(
                points_sent=index + 1,
                gripper_calls_completed=index + 1,
                first_command_time=first_command,
                last_command_time=last_command,
                max_command_interval_ms=max_interval * 1000.0,
                last_timeline_tick=timeline.tick - 1,
            )
        # A late SDK call must not cause a burst of closely spaced catch-up
        # targets. Empty slots advance the same clock without sending a target.
        next_deadline = max(next_deadline + interval, command_started + interval)
        now = time.perf_counter()
        delay = max(0.0, next_deadline - now)
        if cancel_wait is None:
            time.sleep(delay)
        elif cancel_wait(delay):
            if check_pending is not None:
                check_pending()
            raise RuntimeError("RTC trajectory sender cancelled during wait")
