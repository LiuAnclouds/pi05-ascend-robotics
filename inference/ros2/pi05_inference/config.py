"""Typed YAML settings shared by the ROS2 runtime nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml
from rcl_interfaces.msg import ParameterDescriptor

from include.project_paths import PROJECT_ROOT
from inference.include.definitions import ACTION_HORIZON

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class CameraConfig:
    """Camera devices and requested capture format."""

    front_device: str
    wrist_device: str
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class ModelConfig:
    """Model, tokenizer, and normalization asset paths."""

    part1_om: Path
    part2_om: Path
    tokenizer: Path
    stats: Path


@dataclass(frozen=True)
class InferenceConfig:
    """Initial noise seed and action names shared by execution modes."""

    seed: int
    joint_names: tuple[str, ...]
    gripper_name: str


@dataclass(frozen=True)
class SerialTrajectoryFitConfig:
    """Serial trajectory fitting and per-command joint-angle limit."""

    jerk_penalty_weight: float
    max_joint_step_deg: float
    terminal_preserve_points: int = 10
    jitter_range_deg: tuple[float, ...] = (0.1,) * 6
    jitter_net_ratio: float = 0.25


@dataclass(frozen=True)
class SerialConfig:
    """串行模式的目标发送频率、轨迹拟合和单步角度限制。"""

    command_hz: float
    trajectory_fit: SerialTrajectoryFitConfig


@dataclass(frozen=True)
class RTCReplanningConfig:
    """When RTC starts the next prediction."""

    request_after_points: int
    initial_latency_seconds: float


@dataclass(frozen=True)
class RTCGuidanceConfig:
    """RTC compiled model assets and guidance ceiling."""

    model_directory: Path
    max_guidance: float


@dataclass(frozen=True)
class RTCConfig:
    """RTC command clock, replanning policy, and model guidance."""

    command_hz: float
    replanning: RTCReplanningConfig
    guidance: RTCGuidanceConfig


@dataclass(frozen=True)
class TopicConfig:
    """Topics shared between prompt, video, inference and the optional Piper bridge."""

    prompt: str
    front_image: str
    wrist_image: str
    state: str
    action: str
    latency: str
    robot_status: str
    rtc_request: str
    rtc_action: str
    model_status: str


@dataclass(frozen=True)
class RobotConfig:
    """Opt-in Piper connection and motion settings."""

    enabled: bool
    can_interface: str
    motion_speed: int
    state_publish_hz: float
    prepare_service: str
    stop_service: str


@dataclass(frozen=True)
class ExecutionConfig:
    """Select serial or RTC execution and their separate parameter blocks."""

    mode: str = "rtc"


@dataclass(frozen=True)
class RuntimeConfig:
    """Read one YAML file; validate each typed section only when it is used."""

    path: Path
    _values: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG) -> RuntimeConfig:
        """Read configuration without requiring unrelated node settings.

        Args:
            path: Configuration filename; relative assets use the project root.
        Returns:
            Settings whose camera/model/inference/topics properties validate on first access.
        Raises:
            ValueError: If the document is not a YAML mapping.
        """
        config_path = Path(path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file)
        if not isinstance(raw, dict):
            raise ValueError(f"Runtime config must be a YAML mapping: {config_path}")
        return cls(config_path, raw)

    def _section(self, name: str, settings_type: type) -> Any:
        """Build one settings object and attach the filename to schema errors.

        Args:
            name: YAML section name.
            settings_type: Dataclass describing that section's supported fields.
        Returns:
            Constructed settings object; missing/unknown fields raise ValueError.
        """
        try:
            return settings_type(**self._values[name])
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid {name} settings in {self.path}: {error}") from error

    @cached_property
    def camera(self) -> CameraConfig:
        """Return validated camera settings; only the video node needs this section.

        Args: None. Returns: Camera settings; no device is opened.
        """
        camera = self._section("camera", CameraConfig)
        for name, value in (("camera.width", camera.width), ("camera.height", camera.height)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(camera.fps) not in (int, float) or not math.isfinite(camera.fps) or camera.fps <= 0:
            raise ValueError("camera.fps must be finite and positive")
        if not all(
            isinstance(value, str) and value.strip() for value in (camera.front_device, camera.wrist_device)
        ):
            raise ValueError("Camera device names must be nonempty strings")
        return CameraConfig(
            camera.front_device.strip(), camera.wrist_device.strip(), camera.width, camera.height, camera.fps
        )

    @cached_property
    def model(self) -> ModelConfig:
        """Resolve model assets without requiring camera or joint settings.

        Args: None. Returns: Absolute asset paths; files are not opened.
        """
        model = self._section("model", ModelConfig)
        return ModelConfig(
            _asset_path(model.part1_om),
            _asset_path(model.part2_om),
            _asset_path(model.tokenizer),
            _asset_path(model.stats),
        )

    @cached_property
    def inference(self) -> InferenceConfig:
        """Return the shared noise seed and joint names for prediction.

        Args: None. Returns: Inference settings; no model is loaded.
        """
        inference = self._section("inference", InferenceConfig)
        if type(inference.seed) is not int:
            raise ValueError("inference.seed must be an integer")
        names = inference.joint_names
        if not isinstance(names, list) or len(names) != 6:
            raise ValueError("inference.joint_names must contain six names")
        if not all(isinstance(value, str) and value.strip() for value in (inference.gripper_name, *names)):
            raise ValueError("Joint and gripper names must be nonempty strings")
        joint_names = tuple(name.strip() for name in names)
        gripper_name = inference.gripper_name.strip()
        if len(set((*joint_names, gripper_name))) != 7:
            raise ValueError("Joint and gripper names must be distinct")

        return InferenceConfig(inference.seed, joint_names, gripper_name)

    @cached_property
    def serial(self) -> SerialConfig:
        """Return serial send-rate and trajectory-fit settings.

        Args: None. Returns: Validated serial controls; no device is opened.
        """
        execution = self._values.get("execution", {})
        values = execution.get("serial") if isinstance(execution, dict) else None
        if not isinstance(values, dict):
            raise ValueError(f"Missing execution.serial settings in {self.path}")
        try:
            command_hz = values["command_hz"]
            fit_values = dict(values["trajectory_fit"])
            fit_values["jitter_range_deg"] = tuple(fit_values.get("jitter_range_deg", (0.1,) * 6))
            trajectory_fit = SerialTrajectoryFitConfig(**fit_values)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid execution.serial settings in {self.path}: {error}") from error
        if type(command_hz) not in (int, float) or not math.isfinite(command_hz) or command_hz <= 0:
            raise ValueError("execution.serial.command_hz must be finite and positive")
        if (
            type(trajectory_fit.jerk_penalty_weight) not in (int, float)
            or not math.isfinite(trajectory_fit.jerk_penalty_weight)
            or trajectory_fit.jerk_penalty_weight < 0
        ):
            raise ValueError("execution.serial.trajectory_fit.jerk_penalty_weight must be finite and nonnegative")
        if (
            type(trajectory_fit.max_joint_step_deg) not in (int, float)
            or not math.isfinite(trajectory_fit.max_joint_step_deg)
            or trajectory_fit.max_joint_step_deg <= 0
        ):
            raise ValueError("execution.serial.trajectory_fit.max_joint_step_deg must be finite and positive")
        if (
            type(trajectory_fit.terminal_preserve_points) is not int
            or not 5 <= trajectory_fit.terminal_preserve_points <= ACTION_HORIZON
        ):
            raise ValueError("execution.serial.trajectory_fit.terminal_preserve_points must be between 5 and the action horizon")
        if (
            len(trajectory_fit.jitter_range_deg) != 6
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in trajectory_fit.jitter_range_deg)
        ):
            raise ValueError("execution.serial.trajectory_fit.jitter_range_deg must contain six positive finite values in J1-J6 order")
        if (
            type(trajectory_fit.jitter_net_ratio) not in (int, float)
            or not math.isfinite(trajectory_fit.jitter_net_ratio)
            or not 0 < trajectory_fit.jitter_net_ratio < 1
        ):
            raise ValueError("execution.serial.trajectory_fit.jitter_net_ratio must be between zero and one")
        return SerialConfig(float(command_hz), trajectory_fit)

    @cached_property
    def rtc(self) -> RTCConfig:
        """Return grouped RTC timing, replan, and guidance settings.

        Args: None. Returns: Validated RTC settings and resolved model path.
        """
        execution = self._values.get("execution", {})
        values = execution.get("rtc") if isinstance(execution, dict) else None
        if not isinstance(values, dict):
            raise ValueError(f"Missing execution.rtc settings in {self.path}")
        try:
            command_hz = values["command_hz"]
            replanning = RTCReplanningConfig(**values["replanning"])
            guidance_values = dict(values["guidance"])
            guidance_values["model_directory"] = _asset_path(guidance_values["model_directory"])
            guidance = RTCGuidanceConfig(**guidance_values)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid execution.rtc settings in {self.path}: {error}") from error
        if type(command_hz) not in (int, float) or not math.isfinite(command_hz) or command_hz <= 0:
            raise ValueError("execution.rtc.command_hz must be finite and positive")
        if type(replanning.request_after_points) is not int or not 0 < replanning.request_after_points < ACTION_HORIZON:
            raise ValueError(
                f"execution.rtc.replanning.request_after_points must be between 1 and {ACTION_HORIZON - 1}"
            )
        if (
            type(replanning.initial_latency_seconds) not in (int, float)
            or not math.isfinite(replanning.initial_latency_seconds)
            or replanning.initial_latency_seconds < 0
        ):
            raise ValueError("execution.rtc.replanning.initial_latency_seconds must be finite and nonnegative")
        if (
            type(guidance.max_guidance) not in (int, float)
            or not math.isfinite(guidance.max_guidance)
            or guidance.max_guidance <= 0
        ):
            raise ValueError("execution.rtc.guidance.max_guidance must be finite and positive")
        return RTCConfig(float(command_hz), replanning, guidance)

    @cached_property
    def topics(self) -> TopicConfig:
        """Return the common topic contract used to connect the independent nodes.

        Args: None. Returns: Nonempty topic names; no subscriptions are created.
        """
        topics = self._section("topics", TopicConfig)
        if not all(isinstance(value, str) and value.strip() for value in vars(topics).values()):
            raise ValueError("Topic names must be nonempty strings")
        return TopicConfig(**{name: value.strip() for name, value in vars(topics).items()})

    @cached_property
    def robot(self) -> RobotConfig:
        """Return Piper settings; opening CAN remains the robot node's job.

        Args: None. Returns: Device and motion settings; no CAN access occurs.
        """
        robot = self._section("robot", RobotConfig)
        if type(robot.enabled) is not bool:
            raise ValueError("robot.enabled must be boolean")
        if not isinstance(robot.can_interface, str) or not robot.can_interface.strip():
            raise ValueError("robot.can_interface must be nonempty")
        if type(robot.motion_speed) is not int or not 1 <= robot.motion_speed <= 100:
            raise ValueError("robot.motion_speed must be an integer from 1 to 100")
        if (
            type(robot.state_publish_hz) not in (int, float)
            or not math.isfinite(robot.state_publish_hz)
            or robot.state_publish_hz <= 0
        ):
            raise ValueError("robot.state_publish_hz must be finite and positive")
        services = (robot.prepare_service, robot.stop_service)
        if not all(isinstance(value, str) and value.strip() for value in services):
            raise ValueError("Robot service names must be nonempty strings")
        return RobotConfig(
            robot.enabled,
            robot.can_interface.strip(),
            robot.motion_speed,
            float(robot.state_publish_hz),
            robot.prepare_service.strip(),
            robot.stop_service.strip(),
        )

    @cached_property
    def execution(self) -> ExecutionConfig:
        """Return the selected execution schedule without opening any device.

        Args: None. Returns: Validated serial or RTC mode.
        """
        execution = self._values.get("execution", {})
        if not isinstance(execution, dict):
            raise ValueError(f"Invalid execution settings in {self.path}: expected a mapping")
        mode = execution.get("mode", "rtc")
        if mode == "rtc_async":
            mode = "rtc"
        if mode not in ("serial", "rtc"):
            raise ValueError("execution.mode must be 'serial' or 'rtc'")
        return ExecutionConfig(mode)

    @classmethod
    def from_node(cls, node: Any) -> RuntimeConfig:
        """Declare the one read-only ROS file parameter and load its settings.

        Args:
            node: ROS2 node that will retain the returned configuration.
        Returns:
            Runtime settings validated as the node accesses them; no device is opened.
        """
        node.declare_parameter("config_file", str(DEFAULT_CONFIG), ParameterDescriptor(read_only=True))
        return cls.from_yaml(node.get_parameter("config_file").value)


def _asset_path(value: str | Path) -> Path:
    """Resolve an asset filename relative to the project root.

    Args:
        value: Nonempty filename from the model configuration section.
    Returns:
        Absolute path; the file is not opened or changed.
    """
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("Model asset paths must be nonempty strings")
    path = Path(str(value).strip()).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
