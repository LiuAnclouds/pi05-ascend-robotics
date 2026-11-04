"""ROS-independent observation preparation and explicit unit conversions."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import sentencepiece
except ModuleNotFoundError:
    sentencepiece = None

from inference.include.definitions import IMAGE_RESIZE_WORKERS, IMAGE_SIZE, TOKEN_SLOTS


def image_from_message(message) -> np.ndarray:
    """Decode a padded rgb8/bgr8 ROS Image without a cv_bridge dependency.

    Args:
        message: Image-like object exposing width, height, step, encoding and data.
    Returns:
        Read-only HWC uint8 pixels in the message's native channel order. The
        view retains its ROS message buffer owner; no full-frame channel copy is made.
        Unsupported encodings/layouts raise ValueError.
    """
    width, height, step = int(message.width), int(message.height), int(message.step)
    if message.encoding not in ("rgb8", "bgr8") or min(width, height) <= 0 or step < width * 3:
        raise ValueError("Expected nonempty rgb8/bgr8 image with a valid row stride")
    data = np.frombuffer(message.data, dtype=np.uint8)
    if data.size != step * height:
        raise ValueError("Image payload length differs from its row stride/height")
    image = data.reshape(height, step)[:, : width * 3].reshape(height, width, 3)
    image.flags.writeable = False
    return image


def state_from_joints(names, positions, joint_names, gripper_name: str) -> np.ndarray:
    """Convert standard ROS JointState units into the checkpoint's physical state.

    Args:
        names: Feedback joint names, in arbitrary order.
        positions: Corresponding radians for joints and metres for the gripper.
        joint_names: Six model-ordered arm joint names.
        gripper_name: Feedback name for absolute gripper opening.
    Returns:
        FP32 state [7] in degrees and millimetres, without clipping or angle wrapping.
    """
    expected = (*joint_names, gripper_name)
    if len(joint_names) != 6 or len(set(expected)) != 7 or len(set(names)) != len(names):
        raise ValueError("Expected six unique model joints and a distinct gripper name")
    if len(names) != len(positions):
        raise ValueError("JointState names and positions must have equal lengths")
    indices = {name: index for index, name in enumerate(names)}
    if any(name not in indices for name in expected):
        raise ValueError("JointState is missing a required arm/gripper joint")
    state = np.asarray([positions[indices[name]] for name in expected], dtype=np.float64)
    if not np.isfinite(state).all():
        raise ValueError("JointState contains nonfinite positions")
    state[:6] = np.rad2deg(state[:6])
    state[6] *= 1000.0
    return state.astype(np.float32)


def targets_to_ros(targets: np.ndarray) -> np.ndarray:
    """Convert absolute model targets to standard ROS trajectory positions.

    Args:
        targets: Finite [...,7] targets with six joint degrees and gripper millimetres.
    Returns:
        A new FP64 array in radians/metres. Input predictions are unchanged.
    """
    result = np.asarray(targets, dtype=np.float64).copy()
    if result.ndim < 2 or result.shape[-1] != 7 or not np.isfinite(result).all():
        raise ValueError("Expected finite action targets ending in seven values")
    result[..., :6] = np.deg2rad(result[..., :6])
    result[..., 6] /= 1000.0
    return result


def trajectory_from_ros(
    message,
    joint_names: tuple[str, ...],
    gripper_name: str,
    point_count: int,
) -> np.ndarray:
    """Validate and convert one complete ROS trajectory to Piper degree/mm targets.

    Args:
        message: JointTrajectory with ordered names and positions.
        joint_names: The six Piper joint names in model order.
        gripper_name: Configured gripper joint name.
        point_count: Required complete horizon length.
    Returns:
        Contiguous FP32 [point_count,7] absolute positions in degrees/mm.
    Side effects: None; malformed or partial trajectories are rejected.
    """
    expected = (*joint_names, gripper_name)
    if len(joint_names) != 6 or len(set(expected)) != 7 or tuple(message.joint_names) != expected:
        raise ValueError("Trajectory joint names/order differ from the configured Piper mapping")
    if type(point_count) is not int or point_count <= 0 or len(message.points) != point_count:
        raise ValueError(f"Expected exactly {point_count} complete trajectory points")
    ros_positions = np.empty((point_count, 7), dtype=np.float64)
    for index, point in enumerate(message.points):
        if len(point.positions) != 7:
            raise ValueError(f"Trajectory point {index} must contain seven positions")
        ros_positions[index] = point.positions
    if not np.isfinite(ros_positions).all():
        raise ValueError("Trajectory contains nonfinite positions")
    ros_positions[:, :6] = np.rad2deg(ros_positions[:, :6])
    ros_positions[:, 6] *= 1000.0
    return np.ascontiguousarray(ros_positions, dtype=np.float32)


@lru_cache(maxsize=1)
def _fp16_normalized_pixels() -> np.ndarray:
    """Cache the exact FP16 results of the original two FP32 pixel operations.

    Args:
        None. The input domain is all 256 possible uint8 pixel values.
    Returns:
        Read-only FP16 [256] lookup, including black padding mapped to -1.
    """
    values = np.arange(256, dtype=np.float32)
    np.divide(values, np.float32(127.5), out=values)
    np.subtract(values, np.float32(1), out=values)
    values = values.astype(np.float16)
    values.flags.writeable = False
    return values


def resize_rgb(
    image: np.ndarray,
    size: int = IMAGE_SIZE,
    *,
    out: np.ndarray | None = None,
    bgr: bool = False,
) -> np.ndarray:
    """Apply the recorded quantized-model path's original PIL bilinear padding.

    Args:
        image: HWC uint8 RGB frame.
        size: Compiled graph image edge; not a live resolution tuning parameter.
        out: Optional writable, contiguous FP32/FP16 NCHW destination.
        bgr: The input channels are BGR; write them into the RGB-ordered output directly.
    Returns:
        Normalized NCHW [1,3,size,size], with black padding mapped to -1.
        Resize and both normalization operations match the original FP32 path;
        FP16 destinations use a lookup generated by those same FP32 operations
        followed by FP16 conversion, avoiding repeated half-precision casting on ARM.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) <= 0:
        raise ValueError("Expected a nonempty HWC uint8 RGB frame")
    height, width = image.shape[:2]
    ratio = max(width / size, height / size)
    resized_width, resized_height = max(1, int(width / ratio)), max(1, int(height / ratio))
    resized = Image.fromarray(image).resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (size, size), 0)
    canvas.paste(resized, ((size - resized_width) // 2, (size - resized_height) // 2))
    shape = (1, 3, size, size)
    if out is None:
        out = np.empty(shape, np.float32)
    if out.shape != shape or out.dtype not in (np.float32, np.float16) or not out.flags.c_contiguous:
        raise ValueError("Image destination must be contiguous FP32/FP16 NCHW")
    canvas_array = np.asarray(canvas)
    if out.dtype == np.float16:
        indices = canvas_array.transpose(2, 0, 1)
        if bgr:
            indices = indices[::-1]
        # uint8 indices are already in [0,255]. clip avoids np.take's default
        # extra output buffer; it does not clip any source pixel value here.
        np.take(_fp16_normalized_pixels(), indices, out=out[0], mode="clip")
        return out
    if bgr:
        for source_channel, output_channel in ((2, 0), (1, 1), (0, 2)):
            np.copyto(out[0, output_channel], canvas_array[..., source_channel], casting="safe")
    else:
        np.copyto(out[0], canvas_array.transpose(2, 0, 1), casting="safe")
    np.divide(out, np.float32(127.5), out=out)
    np.subtract(out, np.float32(1), out=out)
    return out


class InputPreprocessor:
    """Prepare observations without loading PyTorch, ONNX or any device runtime."""

    def __init__(self, tokenizer: Path, stats: Path, *, parallel_resize: bool = False):
        """Load matching tokenizer and seven-axis state/action statistics.

        Args:
            tokenizer: Source SentencePiece tokenizer model.
            stats: Checkpoint normalization JSON, not recalibrated statistics.
            parallel_resize: Resize the independent camera images on two host threads.
        Returns:
            None. No hardware is opened and no model weights are loaded. When enabled,
            close() must be called to release the bounded resize worker pool.
        """
        if sentencepiece is None:
            raise RuntimeError("Install inference/requirements.txt to load the tokenizer")
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
        payload = json.loads(stats.read_text(encoding="utf-8"))["norm_stats"]
        self.state_q01, self.state_q99 = (
            np.asarray(payload["state"][key], np.float32) for key in ("q01", "q99")
        )
        self.action_q01, self.action_q99 = (
            np.asarray(payload["actions"][key], np.float32) for key in ("q01", "q99")
        )
        for low, high in ((self.state_q01, self.state_q99), (self.action_q01, self.action_q99)):
            if (
                low.shape != (7,)
                or high.shape != (7,)
                or not np.isfinite([low, high]).all()
                or np.any(high <= low)
            ):
                raise ValueError("Expected seven finite nondegenerate normalization quantiles")
        self.state_bins = np.linspace(-1, 1, 257)[:-1]
        self.configure_model_inputs()
        self.resize_executor = (
            ThreadPoolExecutor(max_workers=IMAGE_RESIZE_WORKERS, thread_name_prefix="pi05-image")
            if parallel_resize
            else None
        )

    def configure_model_inputs(self, specifications=None, *, buffer_allocator=None) -> None:
        """Allocate reusable host inputs matching the loaded prefix OM.

        Args:
            specifications: ACL-style input shape/dtype descriptors from worker
                metadata, or None for the original three-image FP32 contract.
            buffer_allocator: Optional function allocating a writable array with
                signature ``(shape, dtype)``; used for direct shared-memory inputs.
        Returns:
            None. Replaces host buffers; call only with no inference in flight.
            A two-camera OM has no third-image host allocation or IPC payload.
        """
        image_shape = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
        if specifications is None:
            specifications = (
                [dict(shape=image_shape, dtype_id=0) for _ in range(3)]
                + [dict(shape=[1], dtype_id=12) for _ in range(3)]
                + [dict(shape=[1, TOKEN_SLOTS], dtype_id=3), dict(shape=[1, TOKEN_SLOTS], dtype_id=12)]
            )
        if len(specifications) not in (6, 8):
            raise ValueError("Expected a two-camera or three-camera prefix OM")
        cameras = (len(specifications) - 2) // 2
        names = [f"image{index}" for index in range(cameras)]
        names += [f"mask{index}" for index in range(cameras)] + ["tokens", "token_mask"]
        token_shape = specifications[-2]["shape"]
        if (
            len(token_shape) != 2
            or token_shape[0] != 1
            or not 0 < token_shape[1] <= TOKEN_SLOTS
            or specifications[-1]["shape"] != token_shape
        ):
            raise ValueError("Unsupported prefix token shape")
        dtypes = {0: np.float32, 1: np.float16, 3: np.int32, 9: np.int64, 12: np.bool_}
        for index, spec in enumerate(specifications):
            if index < cameras:
                valid = spec["shape"] == image_shape and spec["dtype_id"] in (0, 1)
            elif index < 2 * cameras:
                valid = spec["shape"] == [1] and spec["dtype_id"] == 12
            else:
                valid = spec["dtype_id"] in ((3, 9) if index == 2 * cameras else (12,))
            if not valid:
                raise ValueError(f"Unsupported prefix input {names[index]}: {spec}")
        self.inputs = {}
        for name, spec in zip(names, specifications):
            dtype = dtypes[spec["dtype_id"]]
            value = (
                buffer_allocator(spec["shape"], dtype)
                if buffer_allocator is not None
                else np.empty(spec["shape"], dtype)
            )
            if value.shape != tuple(spec["shape"]) or value.dtype != dtype or not value.flags.c_contiguous:
                raise ValueError(f"Input buffer allocator returned an incompatible array for {name}")
            value.fill(0)
            self.inputs[name] = value
        self.inputs["mask0"][:] = self.inputs["mask1"][:] = True
        if any(self.inputs[name].dtype == np.float16 for name in ("image0", "image1")):
            _fp16_normalized_pixels()
        self.last_text = None

    def close(self) -> None:
        """Stop the optional host resize pool and wait for in-flight pixel work.

        Args:
            None.
        Returns:
            None. Safe to call more than once.
        """
        if self.resize_executor is not None:
            executor, self.resize_executor = self.resize_executor, None
            executor.shutdown(wait=True, cancel_futures=True)
        self.inputs = {}

    def make_inputs(
        self,
        front: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
        prompt: str,
        *,
        front_bgr: bool = False,
        wrist_bgr: bool = False,
    ) -> dict:
        """Prepare source-normalized images and the exact Task/State/Action prompt.

        Args:
            front, wrist: Raw HWC uint8 RGB camera frames.
            state: Six joint degrees and gripper millimetres.
            prompt: Nonempty task instruction.
            front_bgr, wrist_bgr: Input channel order flags. BGR frames are
                converted during the final resize copy, without an extra full-frame allocation.
        Returns:
            Reusable named prefix buffers. The caller must snapshot/submit them
            before the next call; concurrent preprocessing is not supported.
        """
        state = np.asarray(state, np.float32)
        if state.shape != (7,) or not np.isfinite(state).all() or not prompt.strip():
            raise ValueError("Expected a finite seven-axis state and a nonempty task")
        normalized = (state - self.state_q01) / (self.state_q99 - self.state_q01 + np.float32(1e-6)) * 2 - 1
        discrete = np.digitize(normalized, self.state_bins) - 1
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        text = f"Task: {cleaned}, State: {' '.join(map(str, discrete))};\nAction: "
        if text != self.last_text:
            ids = self.tokenizer.encode(text, add_bos=True)
            if not ids or len(ids) > self.inputs["tokens"].shape[1]:
                raise ValueError("Task/state prompt exceeds the compiled input token contract")
            self.inputs["tokens"].fill(0)
            self.inputs["token_mask"].fill(False)
            self.inputs["tokens"][0, : len(ids)] = ids
            self.inputs["token_mask"][0, : len(ids)] = True
            self.last_text = text
        resize_jobs = (
            (front, self.inputs["image0"], front_bgr),
            (wrist, self.inputs["image1"], wrist_bgr),
        )
        if self.resize_executor is None:
            for image, output, is_bgr in resize_jobs:
                resize_rgb(image, out=output, bgr=is_bgr)
        else:
            futures = [
                self.resize_executor.submit(resize_rgb, image, out=output, bgr=is_bgr)
                for image, output, is_bgr in resize_jobs
            ]
            # Finish both independent writes before surfacing either failure, so
            # a rejected frame cannot leave a background task modifying reusable buffers.
            wait(futures)
            for future in futures:
                future.result()
        return self.inputs
