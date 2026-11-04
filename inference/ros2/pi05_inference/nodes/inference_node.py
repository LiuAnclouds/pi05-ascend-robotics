"""Serve bootstrap predictions and sender-clocked RTC without blocking ROS."""

import json
import math
import time

import numpy as np
import rclpy
from pi05_interfaces.msg import RTCAction, RTCRequest
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from include.project_paths import PROJECT_ROOT
from inference.core.observation.observation_builder import (
    InputPreprocessor,
    image_from_message,
    state_from_joints,
    targets_to_ros,
)
from inference.core.rtc.guidance_math import prepare_prefix
from inference.core.streaming.latency_accounting import InputLatencyWindow
from inference.core.streaming.shared_memory_transport import SharedMemoryInferenceWorker
from inference.core.telemetry.terminal_view import format_joint_table, format_log_block
from inference.include.definitions import (
    ACTION_DIM,
    ACTION_HORIZON,
    LATENCY_REPORT_INTERVAL_SECONDS,
    MAX_FRAME_AGE_SECONDS,
    MAX_FRAME_SKEW_SECONDS,
    POLL_INTERVAL_SECONDS,
    ROBOT_STATUS_INTERVAL_SECONDS,
    STATUS_REPORT_INTERVAL_SECONDS,
)
from pi05_inference.config import RuntimeConfig


class InferenceNode(Node):
    """Own observations and one model worker; the robot owns the action clock."""

    def __init__(self) -> None:
        """Load YAML, observations and the model before announcing readiness.

        Args: None; config_file selects the shared YAML.
        Returns: None. This node never opens CAN or sends physical commands.
        """
        super().__init__("pi05_inference")
        self.config = RuntimeConfig.from_node(self)
        self.execute_on_robot = self.config.robot.enabled
        self.rtc_mode = self.config.execution.mode == "rtc"
        model, inference, topics = self.config.model, self.config.inference, self.config.topics
        self.worker = self.preprocessor = None
        self.front = self.wrist = self.state = self.state_stamp = None
        self.front_at = self.wrist_at = self.state_at = 0.0
        self.prompt, self.generation, self.request_id = "", 0, 0
        self.pending = self.rtc_request = self.last_rtc_identity = self.last_attempt = None
        self.robot_state, self.robot_status_at = "UNKNOWN", 0.0
        self.robot_ready_after, self.robot_generation = 0.0, 0
        self.awaiting_action_start = False
        self.last_status, self.last_status_at = None, -float("inf")
        self.latency_window = InputLatencyWindow()
        self.last_latency_report = time.monotonic()
        self.last_model_status_at = -float("inf")
        status_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(JointTrajectory, topics.action, 1)
        self.rtc_action_publisher = (
            self.create_publisher(RTCAction, topics.rtc_action, 1)
            if self.execute_on_robot and self.rtc_mode else None
        )
        self.latency_publisher = self.create_publisher(String, topics.latency, 1)
        self.model_status_publisher = self.create_publisher(String, topics.model_status, status_qos)
        self._publish_model_status("STARTING")
        self.create_subscription(String, topics.prompt, self._prompt, status_qos)
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, topics.front_image, self._front, sensor_qos)
        self.create_subscription(Image, topics.wrist_image, self._wrist, sensor_qos)
        self.create_subscription(JointState, topics.state, self._state, sensor_qos)
        if self.execute_on_robot:
            self.create_subscription(String, topics.robot_status, self._robot_status, status_qos)
        if self.execute_on_robot and self.rtc_mode:
            self.create_subscription(RTCRequest, topics.rtc_request, self._rtc_request, 1)
        try:
            self.preprocessor = InputPreprocessor(model.tokenizer, model.stats, parallel_resize=True)
            factory = "inference.core.backends.om_policy:OMPolicy"
            options = dict(
                root=str(PROJECT_ROOT),
                part1_om=str(model.part1_om),
                part2_om=str(model.part2_om),
                action_q01=self.preprocessor.action_q01.tolist(),
                action_q99=self.preprocessor.action_q99.tolist(),
                seed=inference.seed,
            )
            if not self.rtc_mode:
                options["device_euler"] = True
            if self.execute_on_robot and self.rtc_mode:
                factory = "inference.core.rtc.device_euler:DeviceEulerPolicy"
                options.update(
                    guidance_directory=str(self.config.rtc.guidance.model_directory),
                    max_guidance=self.config.rtc.guidance.max_guidance,
                )
            self.worker = SharedMemoryInferenceWorker(
                factory, options, progress_callback=lambda message: self.get_logger().info(message)
            )
            self.preprocessor.configure_model_inputs(
                self.worker.metadata["inputs"][0], buffer_allocator=self.worker.allocate_shared_array
            )
            self._publish_model_status("READY")
            self.create_timer(POLL_INTERVAL_SECONDS, self._tick)
        except BaseException:
            self.destroy_node()
            raise
        if not self.execute_on_robot:
            mode = "prediction-only; CAN closed"
        else:
            mode = "串行：整块发送完成后再预测" if not self.rtc_mode else "sender-clocked RTC"
        self.get_logger().info(f"Pi0.5 model ready: {mode}.")

    def _publish_model_status(self, state: str) -> None:
        """Publish model availability; input status, output none, no robot commands."""
        if rclpy.ok(context=self.context):
            message = String()
            message.data = state
            self.model_status_publisher.publish(message)
        self.last_model_status_at = time.monotonic()

    def _robot_status(self, message: String) -> None:
        """Track epochs without revoking a request on normal READY -> BUSY.

        Args: Piper status heartbeat. Returns: None; sends no commands.
        """
        state, previous = message.data.strip(), self.robot_state
        if state != previous:
            if state == "READY":
                self.robot_generation += 1
                self.robot_ready_after = self.get_clock().now().nanoseconds * 1e-9
            elif state != "BUSY" or previous not in ("READY", "BUSY"):
                self.robot_generation += 1
            if state == "BUSY":
                self.awaiting_action_start = False
            if state not in ("READY", "BUSY"):
                self.rtc_request = None
                self.awaiting_action_start = False
        self.robot_state, self.robot_status_at = state, time.monotonic()

    def _robot_available(self, *, rtc: bool = False) -> bool:
        """Check whether the robot node has announced the state needed for this request."""
        return not self.execute_on_robot or (
            self.robot_state == ("BUSY" if rtc else "READY")
        )

    def _prompt(self, message: String) -> None:
        """Accept task text and revoke old task results; input prompt, output none."""
        prompt = message.data.strip()
        if prompt != self.prompt:
            self.prompt, self.generation = prompt, self.generation + 1
            if self.rtc_request is not None:
                self._cancel_rtc(self.rtc_request, "Task changed before RTC submission")
                self.rtc_request = None

    def _front(self, message: Image) -> None:
        """Retain the newest front image; input message, output none, no pixel copy."""
        self.front, self.front_at = message, time.monotonic()

    def _wrist(self, message: Image) -> None:
        """Retain the newest wrist image; input message, output none, no pixel copy."""
        self.wrist, self.wrist_at = message, time.monotonic()

    def _state(self, message: JointState) -> None:
        """Convert ROS state into degrees/mm; invalid input clears readiness."""
        try:
            self.state = state_from_joints(
                message.name,
                message.position,
                self.config.inference.joint_names,
                self.config.inference.gripper_name,
            )
            self.state_at, self.state_stamp = time.monotonic(), message.header.stamp
        except ValueError as error:
            self.state = None
            self._report_status(f"Invalid JointState: {error}", warning=True)

    def _rtc_request(self, message: RTCRequest) -> None:
        """Accept one real sender snapshot; reject duplicate/overlapping requests.

        Args: Typed state and unsent queue in degrees/mm, plus its control tick.
        Returns: None; this node never starts a second action clock.
        """
        identity = (message.generation, message.request_id, message.observation_tick)
        if identity == self.last_rtc_identity:
            return
        self.last_rtc_identity = identity
        request_mode = "RTC"
        try:
            if not self._robot_available(rtc=True):
                raise ValueError("Piper is not sending an active trajectory")
            if self.pending is not None or self.rtc_request is not None:
                raise ValueError("Another model request is in flight")
            if not math.isfinite(message.captured_monotonic):
                raise ValueError(f"{request_mode} request has an invalid capture timestamp")
            state, old = np.asarray(message.state, np.float32), np.asarray(message.old_targets, np.float32)
            if (
                state.shape != (7,)
                or not np.isfinite(state).all()
                or not 0 <= message.overlap_count <= ACTION_HORIZON
                or old.size != message.overlap_count * 7
                or not np.isfinite(old).all()
            ):
                raise ValueError(f"Invalid {request_mode} state/action-queue request")
        except ValueError as error:
            self._cancel_rtc(message, str(error))
            return
        self.rtc_request = message

    def _cancel_rtc(self, request: RTCRequest, reason: str) -> None:
        """Reject a paired request so the executor can stop; inputs request/cause."""
        if not rclpy.ok(context=self.context):
            return
        response = RTCAction()
        response.request_id, response.generation = request.request_id, request.generation
        response.observation_tick = request.observation_tick
        response.observation_stamp = request.observation_stamp
        response.cancelled, response.reason = True, reason
        self.rtc_action_publisher.publish(response)
        request_mode = "RTC"
        self._report_status(f"{request_mode} prediction request cancelled: {reason}", warning=True)

    def _report_status(self, message: str, *, warning: bool = False) -> None:
        """Rate-limit unchanged status; input text/severity, output none."""
        now = time.monotonic()
        if message == self.last_status and now - self.last_status_at < STATUS_REPORT_INTERVAL_SECONDS:
            return
        logger = self.get_logger()
        # Humble caches severity per call site; keep the two sites distinct.
        if warning:
            logger.warning(message)
        else:
            logger.info(message)
        self.last_status, self.last_status_at = message, now

    def _tick(self) -> None:
        """Poll once, then submit an eligible bootstrap or sender-clocked RTC job.

        Args/Returns: None. Model/IPC failure exits without automatic retry.
        """
        if self.worker is None:
            return
        try:
            reply = self.worker.poll()
            if reply is not None:
                pending, self.pending = self.pending, None
                if pending is None:
                    raise RuntimeError("Model reply has no matching observation")
                current = pending["generation"] == self.generation and (
                    not self.execute_on_robot or pending["robot_generation"] == self.robot_generation
                )
                if current and self._publish(reply["result"], pending):
                    self._record_latency(reply, pending, time.monotonic())
                elif pending["rtc_request"] is not None:
                    self._cancel_rtc(pending["rtc_request"], "Task or execution changed")
            if self.pending is not None:
                return
            now = time.monotonic()
            if now - self.last_model_status_at >= ROBOT_STATUS_INTERVAL_SECONDS:
                self._publish_model_status("READY")
            request = self.rtc_request
            if not self._robot_available(rtc=request is not None):
                if self.robot_state != "BUSY":
                    self._report_status(f"等待机械臂就绪：{self.robot_state}")
                return
            if self.execute_on_robot and request is None and self.awaiting_action_start:
                return
            prepared = self._prepare_observation(request)
            if prepared is None:
                return
            snapshot, pending = prepared
            self.worker.submit(self.request_id, snapshot)
            self.request_id += 1
            pending.update(
                prediction_id=self.request_id,
                submit_ms=self.worker.last_submit_ms,
                request_bytes=self.worker.last_payload_bytes,
                snapshot_copy_bytes=self.worker.last_copied_bytes,
            )
            self.pending, self.rtc_request = pending, None
        except ValueError as error:
            if self.rtc_request is not None:
                self._cancel_rtc(self.rtc_request, str(error))
                self.rtc_request = None
            self._report_status(f"Observation rejected: {error}", warning=True)
        except Exception as error:
            self._publish_model_status("FAULT")
            self.get_logger().fatal(f"Inference stopped; no automatic retry: {error}")
            raise RuntimeError("Model or IPC failure") from error

    def _prepare_observation(self, request: RTCRequest | None):
        """Prepare synchronized inputs anchored to bootstrap or the real control tick.

        Args: Optional sender snapshot of actual postprocessed targets.
        Returns: Payload/metadata pair, or None while waiting for usable inputs.
        Side effects: Writes only currently idle shared-memory input buffers.
        """
        if (
            not self.prompt
            or self.front is None
            or self.wrist is None
            or (request is None and self.state is None)
        ):
            self._report_status("Waiting for prompt, both images and measured state.")
            return None
        now = time.monotonic()
        receipts = (self.front_at, self.wrist_at) if request is not None else (
            self.front_at,
            self.wrist_at,
            self.state_at,
        )
        if any(now - received > MAX_FRAME_AGE_SECONDS for received in receipts):
            self._report_status("Waiting for fresh image/state messages.", warning=True)
            return None
        image_stamps = [m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for m in (self.front, self.wrist)]
        state_stamp = request.state_stamp if request is not None else self.state_stamp
        state_seconds = state_stamp.sec + state_stamp.nanosec * 1e-9
        stamps = (*image_stamps, state_seconds)
        ros_now = self.get_clock().now().nanoseconds * 1e-9
        if (
            max(image_stamps) - min(image_stamps) > MAX_FRAME_SKEW_SECONDS
            or abs(state_seconds - min(image_stamps)) > MAX_FRAME_AGE_SECONDS
            or any(
                stamp <= 0 or ros_now - stamp > MAX_FRAME_AGE_SECONDS or stamp > ros_now + 0.05
                for stamp in stamps
            )
        ):
            self._report_status("Waiting for synchronized, fresh image/state timestamps.", warning=True)
            return None
        if self.execute_on_robot and min(stamps) < self.robot_ready_after:
            return None
        if request is not None:
            anchor = request.observation_stamp.sec + request.observation_stamp.nanosec * 1e-9
            if anchor <= 0:
                raise ValueError("RTC request has an invalid observation timestamp")
            if max(abs(stamp - anchor) for stamp in image_stamps) > MAX_FRAME_SKEW_SECONDS:
                self._report_status("Waiting for cameras aligned to the RTC snapshot.", warning=True)
                return None
        attempt = (
            *image_stamps,
            self.generation,
            None
            if request is None
            else (
                request.generation,
                request.request_id,
                request.observation_tick,
            ),
        )
        if attempt == self.last_attempt:
            return None
        observation_started = time.monotonic()
        state = np.array(self.state if request is None else request.state, dtype=np.float32, copy=True)
        front, wrist = image_from_message(self.front), image_from_message(self.wrist)
        decoded_at = time.monotonic()
        snapshot = dict(
            state=state,
            inputs=self.preprocessor.make_inputs(
                front,
                wrist,
                state,
                self.prompt,
                front_bgr=self.front.encoding == "bgr8",
                wrist_bgr=self.wrist.encoding == "bgr8",
            ),
        )
        if self.rtc_mode and request is not None and request.overlap_count > 0:
            prefix, weights = prepare_prefix(
                np.asarray(request.old_targets, np.float32).reshape(request.overlap_count, 7),
                state,
                self.preprocessor.action_q01,
                self.preprocessor.action_q99,
                request.delay_steps,
            )
            snapshot.update(rtc_prefix=prefix, rtc_weights=weights)
        submitted_at = time.monotonic()
        self.last_attempt = attempt
        return snapshot, dict(
            observed_state=state.tolist(),
            generation=self.generation,
            robot_generation=self.robot_generation,
            rtc_request=request,
            stamp=self.front.header.stamp,
            request_started=observation_started if request is None else request.captured_monotonic,
            observation_started=observation_started,
            submitted_at=submitted_at,
            decode_ms=(decoded_at - observation_started) * 1000.0,
            preprocess_ms=(submitted_at - decoded_at) * 1000.0,
            oldest_frame_stamp=min(image_stamps),
            frame_residence_ms=(observation_started - min(self.front_at, self.wrist_at)) * 1000.0,
        )

    def _publish(self, result: dict, pending: dict) -> bool:
        """Publish bootstrap or RTC commands, never both for the same result.

        Args: Complete model output and its observation/request metadata.
        Returns: False after execution changes; otherwise publishes and returns True.
        """
        request = pending["rtc_request"]
        if not self._robot_available(rtc=request is not None):
            return False
        normalized = np.asarray(result["normalized"])
        targets = np.array(result["targets"], dtype=np.float32, copy=True)
        if (
            normalized.shape != (1, ACTION_HORIZON, ACTION_DIM)
            or not np.isfinite(normalized).all()
            or targets.shape != (ACTION_HORIZON, 7)
            or not np.isfinite(targets).all()
        ):
            raise RuntimeError("Expected complete finite 50x32 output and 50x7 physical targets")
        if request is not None:
            response = RTCAction()
            response.request_id, response.generation = request.request_id, request.generation
            response.observation_tick = request.observation_tick
            response.observation_stamp = request.observation_stamp
            response.targets = targets.reshape(-1).tolist()
            self.rtc_action_publisher.publish(response)
        else:
            message = JointTrajectory()
            message.header.stamp = self.get_clock().now().to_msg()
            message.joint_names = [*self.config.inference.joint_names, self.config.inference.gripper_name]
            for index, row in enumerate(targets_to_ros(targets)):
                point = JointTrajectoryPoint()
                point.positions = row.tolist()
                message.points.append(point)
            self.publisher.publish(message)
            if self.execute_on_robot:
                self.awaiting_action_start = True
        return True

    def _record_latency(self, reply: dict, pending: dict, published_at: float) -> None:
        """Record nested timings without adding overlapped intervals twice.

        Args: Matching reply, input metadata and monotonic publication time.
        Returns: None; publishes bounded P50/P95/P99 latency statistics periodically.
        """
        result, upload = reply["result"], reply["result"].get("prefix_upload", {})
        values = {
            name: pending[name]
            for name in (
                "decode_ms",
                "preprocess_ms",
                "submit_ms",
                "frame_residence_ms",
                "snapshot_copy_bytes",
            )
        }
        values.update(
            pre_inference_ms=(reply["worker_started_at"] - pending["observation_started"]) * 1000.0,
            input_ipc_ms=(reply["worker_started_at"] - pending["submitted_at"]) * 1000.0,
            worker_ms=(reply["worker_finished_at"] - reply["worker_started_at"]) * 1000.0,
            reply_poll_ms=(reply["received_at"] - reply["worker_finished_at"]) * 1000.0,
            publish_ms=(published_at - reply["received_at"]) * 1000.0,
            observation_to_publish_ms=(published_at - pending["observation_started"]) * 1000.0,
            request_to_publish_ms=(published_at - pending["request_started"]) * 1000.0,
            frame_to_publish_ms=max(
                0.0, (self.get_clock().now().nanoseconds * 1e-9 - pending["oldest_frame_stamp"]) * 1000.0
            ),
        )
        values.update({name: result[name] for name in ("inference_ms", "input_cast_ms") if name in result})
        if upload:
            values.update(prefix_upload_ms=upload["total_ms"], prefix_h2d_ms=upload["h2d_ms"])
        self.latency_window.record(values)
        inference_ms = values.get("inference_ms", values["worker_ms"])
        camera_to_model_ms = max(0.0, values["frame_to_publish_ms"] - (
            published_at - reply["worker_started_at"]
        ) * 1000.0)
        targets = np.asarray(result["targets"])
        mode = ("串行" if not self.rtc_mode else "RTC 异步") if self.execute_on_robot else "仅预测"
        self.get_logger().info(
            format_log_block(
                f"预测 #{pending['prediction_id']:04d} | {mode} | {ACTION_HORIZON} 点",
                [
                    f"  相机→模型    {camera_to_model_ms:7.1f} ms  （从相机读帧完成计时）",
                    f"  预处理       {values['preprocess_ms']:7.1f} ms  | 模型推理 {inference_ms:.1f} ms",
                    f"  观测→发布    {values['observation_to_publish_ms']:7.1f} ms  | 相机→发布 {values['frame_to_publish_ms']:.1f} ms",
                    "",
                    *format_joint_table([
                        ("观测反馈", pending["observed_state"]),
                        ("预测首点", targets[0]),
                        ("预测末点", targets[-1]),
                    ]),
                ],
            )
        )
        if published_at - self.last_latency_report >= LATENCY_REPORT_INTERVAL_SECONDS:
            summary = self.latency_window.summary()
            summary.update(
                source="inference_node",
                request_bytes=pending["request_bytes"],
                prefix_upload_bytes=upload.get("bytes"),
                transport="Shared-memory images; state/RTC snapshots; process-queue descriptors",
                timing_scope="Request/input headers through publish; excludes physical execution",
                nested_intervals="Submit inside IPC; H2D/inference inside worker; do not add twice",
            )
            message = String()
            message.data = json.dumps(summary, separators=(",", ":"), allow_nan=False)
            self.latency_publisher.publish(message)
            self.last_latency_report = published_at

    def destroy_node(self):
        """Revoke model readiness then release the worker; no robot SDK calls."""
        if getattr(self, "model_status_publisher", None) is not None:
            self._publish_model_status("STOPPED")
        if self.preprocessor is not None:
            self.preprocessor.close()
            self.preprocessor = None
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        return super().destroy_node()


def main(args=None) -> None:
    """Run ROS inference; input optional CLI args, output none, cleanup on exit."""
    rclpy.init(args=args)
    node = None
    try:
        node = InferenceNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
