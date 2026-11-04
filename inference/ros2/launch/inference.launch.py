"""Launch video/inference and the optional Piper bridge from one YAML file."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from pi05_inference.config import DEFAULT_CONFIG, RuntimeConfig


def _launch_runtime(context) -> list:
    """Create connected nodes, enabling CAN only from robot.enabled in the YAML.

    Args: context is the launch context resolving the shared config argument.
    Returns: Node actions and exit handlers. Exiting any launched runtime node
        shuts down the others, allowing the robot node to cancel/stop its sender.
    Side effects: Nodes start only after launch executes the returned actions.
    """
    config = LaunchConfiguration("config").perform(context)
    settings = RuntimeConfig.from_yaml(config)
    parameters = [{"config_file": ParameterValue(config, value_type=str)}]
    video = Node(
        package="pi05_inference",
        executable="video_node",
        name="pi05_video",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_video_node")),
        parameters=parameters,
    )
    model = Node(
        package="pi05_inference",
        executable="inference_node",
        name="pi05_inference",
        output="screen",
        emulate_tty=True,
        parameters=parameters,
    )
    nodes = [video, model]
    if settings.robot.enabled:
        nodes.append(
            Node(
                package="pi05_inference",
                executable="robot_node",
                name="pi05_robot",
                output="screen",
                parameters=parameters,
                sigterm_timeout="10",
            )
        )
    handlers = [
        RegisterEventHandler(
            OnProcessExit(
                target_action=node, on_exit=[EmitEvent(event=Shutdown(reason="Runtime node exited"))]
            )
        )
        for node in nodes
    ]
    return [*handlers, *nodes]


def generate_launch_description() -> LaunchDescription:
    """Describe the common launch entry; task prompt remains a separate node.

    Args: None. config and use_video_node are the public launch arguments.
    Returns: A launch description; robot.enabled defaults to false in the YAML.
    """
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config", default_value=str(DEFAULT_CONFIG), description="Runtime YAML for Pi0.5 inference."
            ),
            DeclareLaunchArgument(
                "use_video_node",
                default_value="true",
                description="Open the configured cameras; false consumes external image topics.",
            ),
            OpaqueFunction(function=_launch_runtime),
        ]
    )
