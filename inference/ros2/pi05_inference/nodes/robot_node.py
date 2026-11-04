"""Own Piper feedback and either serial chunks or an RTC action timeline."""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory
from pi05_interfaces.msg import RTCAction, RTCRequest

from inference.core.motion.chunk_dispatcher import AsyncChunkSender
from inference.core.motion.piper_interface import (
    RecoveryConfirmationRequired,
    connect_piper,
    fit_serial_trajectory,
    limit_serial_joint_steps,
    motion_recovery_required,
    prepare_motion,
    read_motion_status,
    read_piper_feedback,
    require_piper_feedback,
    stop_motion,
    validate_motion_feedback,
)
from inference.core.observation.observation_builder import targets_to_ros, trajectory_from_ros
from inference.core.streaming.action_timeline import AuthorRTCTimeline, ChunkRequest
from inference.core.telemetry.terminal_view import format_joint_table, format_log_block
from inference.include.definitions import (
    ACTION_HORIZON,
    POLL_INTERVAL_SECONDS,
    ROBOT_STATUS_INTERVAL_SECONDS,
)
from pi05_inference.config import RuntimeConfig


class RobotNode(Node):
    """Own the CAN connection; inference and prompt nodes never send robot commands."""

    def __init__(self) -> None:
        """Open the opt-in Piper bridge and check the controller before preparation.

        Args: None. The shared config_file selects robot and trajectory settings.
        Returns: None. robot.enabled=false refuses CAN access.
        Side effects: A healthy controller may be enabled; stopped/teach states
            require the on-site prepare service. No trajectory is sent at startup.
        """
        super().__init__("pi05_robot")
        self.config = RuntimeConfig.from_node(self)
        self.robot, self.inference, self.topics = self.config.robot, self.config.inference, self.config.topics
        self.rtc_mode = self.config.execution.mode == "rtc"
        if not self.robot.enabled:
            raise RuntimeError("robot_node requires robot.enabled: true in config/config.yaml")
        if self.get_parameter("use_sim_time").value:
            raise RuntimeError("Physical Piper execution requires the real clock, not use_sim_time")
        self.piper = self.sender = self.prepare_thread = None
        self.timeline = None
        self.serial_fit_info = None
        self.timeline_lock = threading.RLock()
        self.replan_snapshot = None
        self.rtc_stamp = None
        self.execution_generation = time.time_ns()
        self.model_ready = False
        self.prompt = None
        self.prepare_cancel = threading.Event()
        self.command_lock = threading.RLock()
        self.state_lock = threading.RLock()
        self.closing = False
        self.motion_commands_issued = False
        self.motion_state = "STARTING"
        self.ready_after_ns = self.last_action_stamp_ns = 0
        self.last_report = self.last_feedback_error = None
        self.chunk_number = 0

        status_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_publisher = self.create_publisher(String, self.topics.robot_status, status_qos)
        state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.state_publisher = self.create_publisher(JointState, self.topics.state, state_qos)
        self.create_subscription(JointTrajectory, self.topics.action, self._action, 1)
        if self.rtc_mode:
            self.rtc_request_publisher = self.create_publisher(RTCRequest, self.topics.rtc_request, 1)
            self.create_subscription(RTCAction, self.topics.rtc_action, self._rtc_action, 1)
        self.create_subscription(String, self.topics.model_status, self._model_status, status_qos)
        self.create_subscription(String, self.topics.prompt, self._prompt, status_qos)
        self.create_service(SetBool, self.robot.prepare_service, self._prepare_service)
        self.create_service(Trigger, self.robot.stop_service, self._stop_service)
        self.create_timer(1.0 / self.robot.state_publish_hz, self._publish_state)
        self.create_timer(ROBOT_STATUS_INTERVAL_SECONDS, self._monitor_sender)
        if self.rtc_mode:
            self.create_timer(POLL_INTERVAL_SECONDS, self._publish_pending_rtc_request)

        try:
            self.piper = connect_piper(
                self.robot.can_interface, announce=False, joint_limits=True, gripper_limits=True
            )
            initial = read_motion_status(self.piper)
            validate_motion_feedback(initial, startup=True)
            if motion_recovery_required(initial):
                self._set_motion_state(
                    "WAITING_CONFIRMATION",
                    "Inspect the arm/workspace before requesting stopped/teach recovery.",
                )
            else:
                self._set_motion_state("WAITING_MODEL", "Waiting for the selected inference model to load.")
        except Exception as error:
            self._set_motion_state("FAULT", f"Piper startup check failed: {error}")

    def _publish_status(self) -> None:
        """Publish a retained state heartbeat; input/output none, no CAN commands.

        Skip ROS publishing after SIGINT shuts down its context, so SDK cleanup
        can still stop active motion without a publisher exception.
        """
        if rclpy.ok(context=self.context):
            message = String()
            message.data = self.motion_state
            self.status_publisher.publish(message)

    def _model_status(self, message: String) -> None:
        """Require a loaded, live model before preparing or continuing motion.

        Args: Model availability heartbeat. Returns: None; faults close execution.
        """
        self.model_ready = message.data == "READY"
        if self.closing:
            return
        if self.model_ready and self.motion_state == "WAITING_MODEL":
            self._start_prepare(confirmed=False)
        elif not self.model_ready and (
            self.motion_state in ("READY", "BUSY", "PREPARING")
            or self.motion_state == "WAITING_MODEL"
            and message.data in ("FAULT", "STOPPED")
        ):
            self.prepare_cancel.set()
            self._set_motion_state("FAULT", f"Model became unavailable: {message.data}")
            if self.sender is not None or self.motion_commands_issued:
                self._stop_active_chunk(self.sender)

    def _prompt(self, message: String) -> None:
        """Apply a new task to the next serial prediction without interrupting this chunk.

        Args: Retained task text. Returns: None. Changing task never clears faults.
        """
        prompt = message.data.strip()
        if prompt == self.prompt:
            return
        self.prompt = prompt
        self.ready_after_ns = self.get_clock().now().nanoseconds
        if self.sender is not None and not self.rtc_mode:
            self.get_logger().info("任务已切换；当前串行动作块发送完成后使用新任务预测。")

    def _set_motion_state(self, state: str, detail: str) -> None:
        """Update the gate and publish state; log only changes, send no CAN commands.

        Args: state is the execution state; detail is the operator-facing cause.
        Returns: None. A new READY epoch invalidates older queued action chunks.
        """
        with self.state_lock:
            previous = self.motion_state
            changed = state != self.motion_state or detail != self.last_report
            if state == "READY" and self.motion_state != "READY":
                self.ready_after_ns = self.get_clock().now().nanoseconds
            if state in ("FAULT", "STOPPING", "STANDBY", "SHUTDOWN"):
                self.prepare_cancel.set()
            self.motion_state, self.last_report = state, detail
            self._publish_status()
        if changed and state != "BUSY" and not (state == "READY" and previous == "BUSY"):
            logger = self.get_logger()
            # Humble fixes severity per call site, even across state changes.
            if state in ("FAULT", "WAITING_CONFIRMATION"):
                logger.warning(f"Robot state {state}: {detail}")
            else:
                logger.info(f"Robot state {state}: {detail}")

    def _start_prepare(self, *, confirmed: bool) -> bool:
        """Start one cancellable startup handshake without sending joint targets.

        Args: confirmed records the operator's on-site stopped-state confirmation.
        Returns: Whether accepted. A live sender/preparation or shutdown rejects it.
        Side effects: May enable a healthy controller; reset requires confirmation.
        """
        with self.state_lock:
            if self.closing or self.piper is None or self.sender is not None:
                return False
            if self.motion_state not in (
                "STARTING",
                "WAITING_MODEL",
                "STANDBY",
                "WAITING_CONFIRMATION",
                "FAULT",
            ):
                return False
            if self.prepare_thread is not None and self.prepare_thread.is_alive():
                return False
            self.prepare_cancel.clear()
            self._set_motion_state("PREPARING", "Checking fresh feedback, motor enable and CAN joint mode.")
            self.prepare_thread = threading.Thread(
                target=self._prepare_worker, args=(confirmed,), name="piper-startup-check", daemon=True
            )
            self.prepare_thread.start()
        return True

    def _prepare_worker(self, confirmed: bool) -> None:
        """Prepare healthy Piper feedback and publish READY only if not cancelled.

        Args: confirmed permits a stopped-state reset after physical inspection.
        Returns: None. Failures latch a non-executing state; no automatic retry.
        """
        try:
            require_piper_feedback(self.piper)
            prepare_motion(
                self.piper,
                self.robot.motion_speed,
                confirm_recovery=lambda _feedback: confirmed,
                cancel_requested=self.prepare_cancel.is_set,
                command_lock=self.command_lock,
                on_event=lambda name, data: self.get_logger().info(f"Piper {name}: {data}"),
            )
            require_piper_feedback(self.piper)
            with self.state_lock:
                if not self.closing and not self.prepare_cancel.is_set() and self.motion_state == "PREPARING":
                    self._set_motion_state(
                        "READY", "Fresh feedback confirms CAN joint mode and six enabled joints."
                    )
        except RecoveryConfirmationRequired as error:
            if not self.closing and not self.prepare_cancel.is_set():
                self._set_motion_state("WAITING_CONFIRMATION", str(error))
        except Exception as error:
            with self.state_lock:
                if not self.closing and self.motion_state == "PREPARING":
                    state = "STANDBY" if self.prepare_cancel.is_set() else "FAULT"
                    self._set_motion_state(state, f"Piper preparation did not complete: {error}")

    def _prepare_service(self, request: SetBool.Request, response: SetBool.Response):
        """Accept an on-site prepare request, never an automatic fault-reset loop.

        Args: true confirms physical inspection; false leaves the controller unchanged.
        Returns: Request acceptance, not proof that preparation or motion completed.
        """
        response.success = bool(request.data) and self._start_prepare(confirmed=True)
        response.message = (
            f"Prepare accepted; wait for {self.topics.robot_status}=READY."
            if response.success
            else f"No change; inspect the arm and request true when idle (current state: {self.motion_state})."
        )
        return response

    def _stop_service(self, _request: Trigger.Request, response: Trigger.Response):
        """Latch execution off and stop an active chunk or cancel startup.

        Args: Trigger has no payload.
        Returns: Whether cancellation/ordinary stop was confirmed. Idle stop sends
            no emergency-stop and does not disable motors; prepare resumes later.
        """
        self.prepare_cancel.set()
        if self.sender is not None or self.motion_commands_issued:
            response.success, response.message = self._stop_active_chunk(self.sender)
        elif self.motion_state == "READY":
            response.success, response.message = self._stop_active_chunk(None)
        elif self.motion_state in ("PREPARING", "STARTING", "WAITING_MODEL"):
            self._set_motion_state(
                "STANDBY", "Execution paused; no action target or emergency-stop was sent."
            )
            if self.prepare_thread is not None:
                self.prepare_thread.join(timeout=2.0)
            if self.prepare_thread is not None and self.prepare_thread.is_alive():
                self._set_motion_state(
                    "FAULT", "Startup thread did not finish cancellation; no action target was sent."
                )
                response.success, response.message = (
                    False,
                    "Preparation is still cancelling; execution remains blocked.",
                )
            else:
                response.success, response.message = True, "Execution paused; prepare is required to resume."
        else:
            response.success, response.message = (
                True,
                f"No active targets; state remains {self.motion_state}.",
            )
        return response

    def _stop_active_chunk(self, sender: AsyncChunkSender | None) -> tuple[bool, str]:
        """Cancel targets and request ordinary standby.

        Args: sender is the active/finished sender, or None for a lingering target.
        Returns: Confirmation and details. No reset, disable or quick-stop is issued.
        """
        self._set_motion_state("STOPPING", "Cancelling targets before requesting normal standby.")
        if sender is not None:
            try:
                sender.stop()
            except Exception as error:
                self._set_motion_state("FAULT", f"Target cancellation failed; no quick-stop sent: {error}")
                return False, "Target cancellation failed; no quick-stop was sent."
            with self.state_lock:
                if self.sender is sender:
                    self.sender = None
        with self.timeline_lock:
            self.replan_snapshot = self.rtc_stamp = None
            if self.timeline is not None:
                self.timeline.reset()
                self.timeline = None
        try:
            state = stop_motion(self.piper, self.robot.motion_speed, command_lock=self.command_lock)
            self.motion_commands_issued = False
            self._set_motion_state("STANDBY", "Normal standby confirmed; six motor drivers remain enabled.")
            return True, f"Standby confirmed; enabled={sum(state['enabled'])}/6."
        except Exception as error:
            self._set_motion_state("FAULT", f"Normal standby unconfirmed; no quick-stop sent: {error}")
            return False, "Normal standby unconfirmed; inspect controller feedback. No quick-stop was sent."

    def _sender_allowed(self) -> None:
        """Reject the next target if execution was cancelled/faulted; send nothing.

        Args/Returns: None. Raises before the sender's next joint/gripper SDK call.
        """
        if self.closing or self.motion_state != "BUSY":
            raise RuntimeError(f"Piper execution gate closed: {self.motion_state}")

    def _action(self, message: JointTrajectory) -> None:
        """接收并发送当前一轮完整动作。

        Args: message 包含 50 点绝对目标，六关节为弧度，夹爪为米。
        Returns: None。串行模式先拟合并限制单步角度，再顺序发送全部目标。
        Side effects: READY 时启动发送线程；只有 RTC 模式在发送期间重规划。
        """
        if self.closing or self.motion_state != "READY" or self.piper is None:
            return
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if stamp_ns < self.ready_after_ns or stamp_ns <= self.last_action_stamp_ns:
            self.get_logger().warning(
                "Discarding action from an earlier readiness epoch or duplicate observation."
            )
            return
        try:
            trajectory = trajectory_from_ros(
                message,
                self.inference.joint_names,
                self.inference.gripper_name,
                ACTION_HORIZON,
            )
            feedback = read_piper_feedback(self.piper)
            serial_fit_info = None
            if not self.rtc_mode:
                trajectory, serial_fit_info = fit_serial_trajectory(
                    trajectory,
                    feedback["state_raw"],
                    self.config.serial.trajectory_fit.jerk_penalty_weight,
                    self.config.serial.trajectory_fit.terminal_preserve_points,
                    self.config.serial.trajectory_fit.jitter_range_deg,
                    self.config.serial.trajectory_fit.jitter_net_ratio,
                )
                trajectory, step_info = limit_serial_joint_steps(
                    trajectory,
                    feedback["state_raw"],
                    self.config.serial.trajectory_fit.max_joint_step_deg,
                )
                serial_fit_info.update(step_info)
        except Exception as error:
            self._set_motion_state("FAULT", f"Action rejected before execution: {error}")
            if self.motion_commands_issued:
                self._stop_active_chunk(self.sender)
            return
        with self.state_lock:
            if self.closing or self.motion_state != "READY" or self.sender is not None:
                return
            sender = AsyncChunkSender(
                self.piper,
                self.robot.motion_speed,
                self.config.rtc.command_hz if self.rtc_mode else self.config.serial.command_hz,
                check_pending=self._sender_allowed,
                command_lock=self.command_lock,
            )
            timeline = None
            if self.rtc_mode:
                timeline = AuthorRTCTimeline(
                    fps=self.config.rtc.command_hz,
                    horizon=ACTION_HORIZON,
                    initial_latency=self.config.rtc.replanning.initial_latency_seconds,
                    replan_after=self.config.rtc.replanning.request_after_points,
                )
                self.execution_generation += 1
                timeline.generation = self.execution_generation
                timeline.prime(trajectory)
                self.timeline = timeline
            self.sender, self.last_action_stamp_ns = sender, stamp_ns
            self.serial_fit_info = serial_fit_info
            self.motion_commands_issued = True
            try:
                if self.timeline is not None:
                    detail = "RTC 执行中：在配置的控制点发起下一轮引导预测。"
                else:
                    detail = "串行执行中：发送完整 50 点，发送完成后再预测下一块。"
                self._set_motion_state("BUSY", detail)
                if self.timeline is not None:
                    sender.start_timeline(
                        timeline, self._queue_rtc_request, timeline_lock=self.timeline_lock
                    )
                else:
                    sender.start(trajectory)
                self.chunk_number += 1
            except Exception as error:
                self.sender = None
                self.serial_fit_info = None
                self.motion_commands_issued = False
                self.timeline = None
                self._set_motion_state("FAULT", f"Could not start the target sender: {error}")

    def _queue_rtc_request(self, request: ChunkRequest, feedback: dict) -> None:
        """Pass a sender snapshot to the ROS thread without blocking the CAN clock.

        Args: Tick-aligned request and matching SDK feedback ages. Returns: None.
        """
        with self.timeline_lock:
            if self.timeline is not None and self.timeline.pending is request:
                self.replan_snapshot = (
                    request,
                    max(feedback["joint_age_ms"], feedback["gripper_age_ms"]) * 1e-3,
                )

    def _publish_pending_rtc_request(self) -> None:
        """Publish a queued snapshot with original capture/feedback timestamps.

        Args/Returns: None; no SDK/model work occurs on this timer.
        """
        with self.timeline_lock:
            snapshot, self.replan_snapshot = self.replan_snapshot, None
            if snapshot is None or self.timeline is None or self.timeline.pending is not snapshot[0]:
                return
        if self.closing or self.motion_state != "BUSY":
            return
        request, feedback_age = snapshot
        elapsed = time.monotonic() - request.captured_at
        stamp_ns = self.get_clock().now().nanoseconds - round(elapsed * 1_000_000_000)
        message = RTCRequest()
        message.request_id = request.request_id
        message.generation = request.generation
        message.observation_tick = request.observation_tick
        message.observation_stamp.sec, message.observation_stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
        state_ns = max(0, stamp_ns - round(feedback_age * 1_000_000_000))
        message.state_stamp.sec, message.state_stamp.nanosec = divmod(state_ns, 1_000_000_000)
        message.captured_monotonic = request.captured_at
        message.delay_steps = request.delay_steps
        message.overlap_count = len(request.old_targets)
        message.state = request.state.tolist()
        message.old_targets = request.old_targets.reshape(-1).tolist()
        self.rtc_stamp = stamp_ns
        self.rtc_request_publisher.publish(message)
        self.get_logger().info(
            f"RTC 预测开始 | 控制点 {request.observation_tick} | "
            f"未发送参考点 {len(request.old_targets)} | 预测期间继续发送。"
        )

    def _rtc_action(self, message: RTCAction) -> None:
        """Swap only the unsent timeline suffix after a matching RTC reply.

        Args: RTCAction paired with the sender's active request identity.
        Returns: None. Cancelled, stale and malformed replies never modify targets.
        Side effects: A valid reply updates queued targets; failure stops execution.
        """
        if self.closing or self.motion_state != "BUSY":
            return
        request_mode = "RTC"
        failure = None
        with self.timeline_lock:
            timeline = self.timeline
            request = None if timeline is None else timeline.pending
            matches = (
                request is not None
                and request.request_id == message.request_id
                and request.generation == message.generation
                and request.observation_tick == message.observation_tick
            )
            if not matches:
                self.get_logger().warning(f"Discarding stale {request_mode} prediction reply.")
                return
            if message.cancelled:
                timeline.cancel(message.request_id, message.generation)
                failure = f"{request_mode} prediction cancelled: {message.reason}"
            else:
                try:
                    stamp_ns = (
                        message.observation_stamp.sec * 1_000_000_000 + message.observation_stamp.nanosec
                    )
                    if stamp_ns != self.rtc_stamp:
                        raise ValueError(f"{request_mode} reply does not match its request timestamp")
                    targets = np.asarray(message.targets, dtype=np.float32)
                    if targets.size != ACTION_HORIZON * 7:
                        raise ValueError(f"{request_mode} reply must contain a complete 50x7 action")
                    targets = targets.reshape(ACTION_HORIZON, 7)
                    result = timeline.merge(request, targets, time.monotonic())
                    if not result.get("accepted") and result.get("reason") != "expired":
                        raise ValueError(f"{request_mode} reply rejected: {result.get('reason')}")
                    self.rtc_stamp = None
                except Exception as error:
                    failure = f"{request_mode} action replacement rejected: {error}"
        # Never join a sender while holding its timeline lock.
        if failure is not None:
            self._set_motion_state("FAULT", failure)
            self._stop_active_chunk(self.sender)
            return
        if result["reason"] == "expired":
            self.get_logger().info(
                "RTC 预测已过时；动作队列继续等待新预测。"
            )
            return
        self.get_logger().info(
            format_log_block(
                "动作队列已更新 | RTC",
                [
                    f"  当前控制点   {result['received_tick']}  | 请求观测点 {result['observation_tick']}",
                    f"  丢弃过时点   {result['skipped_points']}  | 安装新点 {result['installed_points']}",
                    f"  覆盖旧点     {result['overlap_at_switch']}  | 请求耗时 {result['latency_ms']:.1f} ms",
                    "  接续方式：新预测替换未来队列，无返回后加权融合。",
                ],
                style="ok",
            )
        )

    def _publish_state(self) -> None:
        """Publish the latest measured joints in ROS units.

        Args/Returns: None. Header time accounts for SDK receive age, not sample time.
        Side effects: Reads feedback only; it does not change execution state.
        """
        if self.piper is None or self.closing:
            return
        try:
            feedback = read_piper_feedback(self.piper)
            state = np.asarray(feedback["state_raw"], dtype=np.float32)
            message = JointState()
            receive_age_ns = round(max(feedback["joint_age_ms"], feedback["gripper_age_ms"]) * 1_000_000)
            stamp_ns = max(0, self.get_clock().now().nanoseconds - receive_age_ns)
            message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
            message.name = [*self.inference.joint_names, self.inference.gripper_name]
            message.position = targets_to_ros(state.reshape(1, 7))[0].tolist()
            self.state_publisher.publish(message)
            self.last_feedback_error = None
        except Exception as error:
            if str(error) != self.last_feedback_error:
                self.get_logger().warning(f"Could not read Piper feedback: {error}")
            self.last_feedback_error = str(error)

    def _monitor_sender(self) -> None:
        """Check controller/sender state and publish a continuous readiness heartbeat.

        Args/Returns: None. Full send means commands transmitted, not physical arrival.
        Side effects: Partial sends, including a failed first JointCtrl, are stopped.
        """
        if self.closing:
            return
        sender = self.sender
        if sender is not None:
            waiting = sender.progress.get("waiting_for_prediction", False)
            if waiting != sender.progress.get("waiting_reported", False):
                sender.progress["waiting_reported"] = waiting
                if waiting:
                    mode = "RTC" if self.rtc_mode else "串行"
                    self.get_logger().info(f"{mode}目标队列已发完，等待新预测并在返回后继续。")
        if sender is not None and sender.thread is not None and not sender.thread.is_alive():
            if (
                not self.rtc_mode
                and self.timeline is None
                and sender.error is None
                and sender.progress.get("points_sent") == sender.expected_points
            ):
                with self.state_lock:
                    if self.sender is sender:
                        self.sender = None
                progress = sender.progress
                elapsed_ms = (
                    progress["sender_finished_at"] - progress["sender_started_at"]
                ) * 1000.0
                fit_info = self.serial_fit_info or {}
                max_adjustment = max(fit_info.get("max_joint_adjustment_deg", [0.0]), default=0.0)
                max_joint_step = max(fit_info.get("max_joint_step_deg", [0.0]), default=0.0)
                self.get_logger().info(
                    format_log_block(
                        f"发送完成 #{self.chunk_number:04d} | {progress.get('points_sent', 0)}/{sender.expected_points} 点",
                        [
                            f"  发送耗时 {elapsed_ms:.1f} ms | 频率 {self.config.serial.command_hz:g} Hz | 最大点间隔 {progress.get('max_command_interval_ms', 0.0):.1f} ms",
                            f"  拟合调整 最大 {max_adjustment:.4f} deg | 点间最大变化 {max_joint_step:.4f} deg",
                            f"  微抖动压平 {fit_info.get('jitter_flattened_points', 0)} 个关节点",
                            "",
                            *format_joint_table([("结束时反馈", progress["state_after_raw"])]),
                            "  状态：目标已发送；物理到达时间未提供。",
                        ],
                        style="ok",
                    )
                )
                # Sending all targets does not confirm physical arrival. Keep
                # motion_commands_issued set until an ordinary stop is confirmed.
                self._set_motion_state("READY", "Serial action chunk sent; waiting for the next prediction.")
                self.serial_fit_info = None
                self._publish_status()
                return
            mode = "RTC" if self.rtc_mode else "Serial"
            reason = sender.error or RuntimeError(f"{mode} sender exited while execution was active")
            if sender.error is not None:
                self.get_logger().error(f"{mode} sender failed: {reason}")
            self._stop_active_chunk(sender)
        self._publish_status()

    def destroy_node(self):
        """Cancel setup/targets before closing CAN; idle shutdown sends no emergency-stop.

        Args: None. Returns: Base rclpy destruction result.
        Side effects: Cancels targets and requests normal standby; no emergency-stop
            or disable command is sent by this method.
        """
        self.closing = True
        self.prepare_cancel.set()
        if self.sender is not None or self.motion_commands_issued:
            self._stop_active_chunk(self.sender)
        elif self.motion_state == "READY":
            self._stop_active_chunk(None)
        if self.prepare_thread is not None and self.prepare_thread.is_alive():
            self.prepare_thread.join(timeout=2.0)
        threads = (self.prepare_thread, self.sender.thread if self.sender is not None else None)
        if any(thread is not None and thread.is_alive() for thread in threads):
            self.get_logger().fatal(
                "SDK thread is blocked; CAN retained until process exit. Inspect the arm."
            )
        elif self.piper is not None:
            try:
                self.piper.DisconnectPort()
                self.piper = None
            except Exception as error:
                self.get_logger().error(f"Could not close Piper CAN: {error}")
        self._set_motion_state("SHUTDOWN", "No further targets or startup commands are permitted.")
        return super().destroy_node()


def main(args=None) -> None:
    """Run the opt-in robot node, stopping active targets even after ROS SIGINT.

    Args: args are optional ROS CLI arguments.
    Returns: None. Prediction-only configuration never invokes this entrypoint.
    """
    rclpy.init(args=args)
    node = None
    try:
        node = RobotNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
