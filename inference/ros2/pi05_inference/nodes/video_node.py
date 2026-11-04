"""Publish two camera topics; independent of policy and robot execution."""

import json
import time
from array import array

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from inference.core.observation.camera_stream import CameraPair
from inference.core.streaming.latency_accounting import InputLatencyWindow
from inference.include.definitions import LATENCY_REPORT_INTERVAL_SECONDS
from pi05_inference.config import RuntimeConfig


class VideoNode(Node):
    """Continuously drain each camera and publish only its latest frame."""

    def __init__(self) -> None:
        """Initialize cameras from the shared typed YAML configuration.

        Args:
            None; the read-only ROS config_file parameter selects the YAML file.
        Returns:
            None. This node publishes frames and never loads an OM.
        """
        super().__init__("pi05_video")
        self.config = RuntimeConfig.from_node(self)
        camera, topics = self.config.camera, self.config.topics
        fps, width, height = camera.fps, camera.width, camera.height
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.front_publisher = self.create_publisher(Image, topics.front_image, sensor_qos)
        self.wrist_publisher = self.create_publisher(Image, topics.wrist_image, sensor_qos)
        self.latency_publisher = self.create_publisher(String, topics.latency, 1)
        self.cameras = None
        self.last_frame_ids = {}
        self.latency_window = InputLatencyWindow()
        self.last_latency_report = time.monotonic()
        try:
            self.cameras = CameraPair(
                camera.front_device,
                camera.wrist_device,
                width=width,
                height=height,
                fps=fps,
            )
            self.create_timer(1.0 / fps, self._publish)
        except BaseException:
            self.destroy_node()
            raise

    def _publish(self) -> None:
        """Publish native bgr8 frames with separate host-read timestamps.

        Args:
            None; data comes from CameraPair's continuously drained snapshots.
        Returns:
            None. Unchanged frame IDs are skipped; errors are logged.
        """
        try:
            started = time.monotonic()
            front, wrist, snapshot_ms = self.cameras.read(copy=False)
            metadata = self.cameras.metadata()
            ros_now = self.get_clock().now().nanoseconds
            host_now = time.monotonic()
            messages = []
            for frame, publisher, frame_id, key in (
                (front, self.front_publisher, "front_camera", "a"),
                (wrist, self.wrist_publisher, "wrist_camera", "b"),
            ):
                if metadata[key]["frame_id"] == self.last_frame_ids.get(key):
                    continue
                message = Image()
                stamp = ros_now - int((host_now - metadata[key]["read_complete_monotonic"]) * 1_000_000_000)
                message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
                message.header.frame_id = frame_id
                message.height, message.width = frame.shape[:2]
                message.encoding, message.is_bigendian = "bgr8", 0
                message.step = frame.shape[1] * 3
                # One owned ROS buffer, without an intermediate bytes object or
                # full-rate RGB conversion. Convert only frames used for inference.
                pixels = array("B")
                pixels.frombytes(memoryview(np.ascontiguousarray(frame)).cast("B"))
                message.data = pixels
                messages.append((publisher, message))
            for publisher, message in messages:
                publisher.publish(message)
            for key in ("a", "b"):
                self.last_frame_ids[key] = metadata[key]["frame_id"]
            finished = time.monotonic()
            if messages:
                self.latency_window.record(
                    {
                        "camera_pair_snapshot_ms": snapshot_ms,
                        "publish_path_ms": (finished - started) * 1000.0,
                        "front_read_to_publish_ms": (finished - metadata["a"]["read_complete_monotonic"])
                        * 1000.0,
                        "wrist_read_to_publish_ms": (finished - metadata["b"]["read_complete_monotonic"])
                        * 1000.0,
                        "camera_pair_skew_ms": metadata["pair_skew_ms"],
                    }
                )
            if finished - self.last_latency_report >= LATENCY_REPORT_INTERVAL_SECONDS:
                summary = self.latency_window.summary()
                summary.update(
                    source="video_node",
                    width=front.shape[1],
                    height=front.shape[0],
                    encoding="bgr8",
                    timing_scope="host camera read completion through ROS publish return",
                )
                latency = String()
                latency.data = json.dumps(summary, separators=(",", ":"), allow_nan=False)
                self.latency_publisher.publish(latency)
                self.last_latency_report = finished
        except Exception as error:
            self.get_logger().error(f"Camera pair unavailable: {error}")

    def destroy_node(self):
        """Release the camera threads, then destroy the ROS node.

        Args:
            None.
        Returns:
            The base rclpy destruction result.
        """
        if getattr(self, "cameras", None) is not None:
            self.cameras.close()
            self.cameras = None
        return super().destroy_node()


def main(args=None) -> None:
    """Spin camera publishers and close handles after shutdown.

    Args:
        args: Optional ROS2 arguments.
    Returns:
        None; no policy or robot messages are sent.
    """
    rclpy.init(args=args)
    node = None
    try:
        node = VideoNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
