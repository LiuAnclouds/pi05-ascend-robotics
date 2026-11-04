"""ROS2 task prompt publisher; independent of camera and model code."""

import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from pi05_inference.config import RuntimeConfig


class PromptNode(Node):
    def __init__(self) -> None:
        super().__init__("pi05_prompt")
        self.config = RuntimeConfig.from_node(self)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(String, self.config.topics.prompt, qos)
        self.get_logger().info(
            "Enter a task instruction and press Enter to publish; enter another line to switch."
        )

    def publish_prompt(self, prompt: str) -> None:
        prompt = prompt.strip()
        if prompt:
            self.publisher.publish(String(data=prompt))
            self.get_logger().info(f"Prompt changed: {prompt}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    spin_thread = None
    try:
        node = PromptNode()
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()
        while rclpy.ok():
            node.publish_prompt(input("Task instruction> "))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join()
        if node is not None:
            node.destroy_node()
