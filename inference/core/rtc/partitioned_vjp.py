"""Device-resident execution of exact topological RTC-VJP partitions.

The final 50x32 correction is copied to host for the existing FP32 Euler loop.
Native velocity, already read by that loop, is converted FP16->FP32 and fed to
the endpoint. All internal activations/gradients are directly borrowed on device.
This still uses the frozen STE Jacobian and recomputes its expert forward; it
does not claim exact native-integer gradients or a fused forward/backward OM.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx.reference import ReferenceEvaluator
except ModuleNotFoundError:
    onnx = ReferenceEvaluator = None
from inference.core.backends.acl_executor import BufferedACL, check, model_io
from inference.core.backends.om_policy import OMPolicy
from inference.core.rtc.guidance_math import AUTHOR_RTC_COMMIT, REFERENCE_MAX_GUIDANCE, guidance_scale
from inference.include.definitions import COTANGENT_SCALE, DEFAULT_SEED

ONNX_TO_ACL = {1: 0, 10: 1, 3: 2, 6: 3, 7: 9, 9: 12}
ACL_DTYPES = {0: np.float32, 1: np.float16, 2: np.int8, 3: np.int32, 9: np.int64, 12: np.bool_}


def contract(value):
    """Translate a manifest entry into its ACL bytes/shape/dtype contract.

    Args: Tensor metadata containing ONNX type and dimensions.
    Returns: ACL contract dictionary; unsupported types raise KeyError.
    """
    return dict(bytes=value["bytes"], shape=value["shape"], dtype_id=ONNX_TO_ACL[value["onnx_dtype"]])


def constant_boundary_values(path, values):
    """Evaluate ONLY proven input-independent outputs pruned by the compiler.

    GE can omit a constant NetOutput (e.g. FP16 initializer -> FP32 Cast).
    Restore that exact value once, without offloading any live activation or
    replacing a gradient computation. Refuse any dependency on graph inputs.
    """
    if onnx is None or ReferenceEvaluator is None:
        raise RuntimeError("Install the optional ONNX dependency to read RTC ONNX partitions")
    model = onnx.load(str(path), load_external_data=False)
    initializers = {value.name: value for value in model.graph.initializer}
    definitions = {name: node for node in model.graph.node for name in node.output}
    required_nodes, required_initializers = set(), set()

    def visit(name):
        """Collect constant-only producers needed to restore a boundary value.

        Args: Tensor name. Returns: None; any live external dependency is rejected.
        """
        if name in initializers:
            required_initializers.add(name)
        elif name in definitions:
            node = definitions[name]
            if node.name not in required_nodes:
                for dependency in node.input:
                    if dependency:
                        visit(dependency)
                required_nodes.add(node.name)
        else:
            raise ValueError("Compiler omitted a NONCONSTANT boundary: " + name)

    for value in values:
        visit(value["name"])
    graph = onnx.helper.make_model(
        onnx.helper.make_graph(
            [node for node in model.graph.node if node.name in required_nodes],
            "constant_partition_boundaries",
            [],
            [
                onnx.helper.make_tensor_value_info(value["name"], value["onnx_dtype"], value["shape"])
                for value in values
            ],
            initializer=[initializers[name] for name in sorted(required_initializers)],
        ),
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
    )
    onnx.external_data_helper.load_external_data_for_model(graph, str(path.parent))
    arrays = ReferenceEvaluator(graph).run(None, {})
    for value, array in zip(values, arrays):
        expected = contract(value)
        if (
            list(array.shape) != expected["shape"]
            or array.nbytes != expected["bytes"]
            or array.dtype != ACL_DTYPES[expected["dtype_id"]]
        ):
            raise ValueError("Constant boundary type changed: " + value["name"])
    return dict(zip((value["name"] for value in values), arrays))


class PartitionedVJP:
    def __init__(self, acl, directory, bindings=None, execution_mode="synchronous"):
        """Load connected gradient partitions and declare borrowed tensor buffers.

        Args: ACL API, partition directory, optional external bindings and mode.
        Returns: None; owns only buffers allocated inside this pipeline.
        """
        if execution_mode not in ("synchronous", "single-stream"):
            raise ValueError("Unknown VJP execution mode: " + execution_mode)
        self.acl = acl
        self.execution_mode = execution_mode
        self.stream = None
        self.stream_pending = False
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.models, self.descriptors, self.executors, self.memory, self.input_memory = [], [], [], [], []
        self.values = dict(bindings or {})
        self.externals = {value["name"]: contract(value) for value in self.manifest["inputs"]}
        self.borrowed_external = set(self.values)
        self.owners = {}
        self.constant_outputs = {}
        self.telemetry = dict(
            parts=[],
            boundary_copy_bytes_per_step=0,
            derivative="unchanged complete frozen-STE input VJP",
            precision_mode="cube_fp16in_fp32out",
            execution_mode=execution_mode,
        )
        try:
            for name, spec in self.externals.items():
                if name in self.values:
                    if self.values[name][1] != spec:
                        raise ValueError(
                            f"Native binding differs for {name}: {self.values[name][1]} vs {spec}"
                        )
                else:
                    pointer, code = acl.rt.malloc(spec["bytes"], 2)
                    check(code, "VJP external input allocation")
                    self.input_memory.append(pointer)
                    self.values[name] = (pointer, spec)
            for view in self.manifest.get("input_views", []):
                name, source = view["name"], view["source"]
                pointer, parent = self.values[source]
                spec, offset = contract(view), view["offset_bytes"]
                if (
                    name in self.values
                    or offset < 0
                    or offset + spec["bytes"] > parent["bytes"]
                    or spec["dtype_id"] != parent["dtype_id"]
                ):
                    raise ValueError("Invalid device-only cache input view: " + name)
                self.values[name] = (pointer + offset, spec)
            self.telemetry["device_only_input_views"] = len(self.manifest.get("input_views", []))
            for row in self.manifest["parts"]:
                path = self.directory / f"part_{row['index']:02d}.om"
                work, weight, code = acl.mdl.query_size(str(path))
                check(code, "partition query_size")
                pointers = []
                for size in (work, weight):
                    pointer = 0
                    if size:
                        pointer, code = acl.rt.malloc(size, 2)
                        check(code, "partition work/weight allocation")
                        self.memory.append(pointer)
                    pointers.append(pointer)
                model, code = acl.mdl.load_from_file_with_mem(
                    str(path), pointers[0], work, pointers[1], weight
                )
                check(code, "partition load")
                self.models.append(model)
                io = model_io(acl, model)
                self.descriptors.append(io)
                expected_inputs = [contract(value) for value in row["inputs"]]
                expected_outputs = [contract(value) for value in row["outputs"]]
                actual_output_count = len(io["outputs"])
                if (
                    io["inputs"] != expected_inputs
                    or actual_output_count > len(expected_outputs)
                    or io["outputs"] != expected_outputs[:actual_output_count]
                ):
                    differences = [
                        dict(kind=kind, index=index, actual=actual, expected=expected)
                        for kind, expected_items in (
                            ("inputs", expected_inputs),
                            ("outputs", expected_outputs),
                        )
                        for index, (actual, expected) in enumerate(zip(io[kind], expected_items))
                        if actual != expected
                    ]
                    raise ValueError(
                        f"Partition {row['index']} changed boundary contracts: "
                        f"input counts {len(io['inputs'])}/{len(expected_inputs)}, "
                        f"output counts {actual_output_count}/{len(expected_outputs)}, "
                        f"first differences {differences[:3]}"
                    )
                missing = row["outputs"][actual_output_count:]
                constants = constant_boundary_values(path.with_suffix(".onnx"), missing) if missing else {}
                borrowed = {index: self.values[value["name"]] for index, value in enumerate(row["inputs"])}
                executor = BufferedACL(acl, model, io, borrowed=borrowed, malloc_policy=2)
                self.executors.append(executor)
                for index, value in enumerate(row["outputs"][:actual_output_count]):
                    self.values[value["name"]] = (executor.addresses["outputs"][index], io["outputs"][index])
                    self.owners[value["name"]] = (executor, index)
                for value in missing:
                    name, spec = value["name"], contract(value)
                    array = np.ascontiguousarray(constants[name])
                    pointer, code = acl.rt.malloc(spec["bytes"], 2)
                    check(code, "constant boundary allocation")
                    self.input_memory.append(pointer)
                    raw = array.tobytes()
                    check(
                        acl.rt.memcpy(pointer, spec["bytes"], acl.util.bytes_to_ptr(raw), len(raw), 1),
                        "constant boundary one-time H2D",
                    )
                    self.values[name] = (pointer, spec)
                    self.constant_outputs[name] = array
                self.telemetry["parts"].append(
                    dict(
                        index=row["index"],
                        work_bytes=work,
                        weight_bytes=weight,
                        input_count=len(row["inputs"]),
                        output_count=len(row["outputs"]),
                        constant_outputs_materialized_once=[
                            dict(name=name, sha256=hashlib.sha256(array.tobytes()).hexdigest())
                            for name, array in constants.items()
                        ],
                    )
                )
        except BaseException:
            self.close()
            raise

    def feed(self, arrays):
        """Copy named external input arrays into the corresponding device buffers.

        Args: Mapping of partition input names to contract-matched numeric arrays.
        Returns: None; borrowed native buffers cannot be overwritten.
        """
        for name, array in arrays.items():
            if name in self.borrowed_external:
                raise ValueError("Cannot overwrite a native borrowed buffer: " + name)
            pointer, spec = self.values[name]
            value = np.ascontiguousarray(array, dtype=ACL_DTYPES[spec["dtype_id"]])
            if list(value.shape) != spec["shape"] or value.nbytes != spec["bytes"]:
                raise ValueError(f"VJP input {name} contract differs: {value.shape} vs {spec}")
            raw = value.tobytes()
            check(
                self.acl.rt.memcpy(pointer, spec["bytes"], self.acl.util.bytes_to_ptr(raw), len(raw), 1),
                "partition external H2D",
            )

    def run(self):
        """Run dependent partitions in order, never in parallel with each other.

        A single ACL stream preserves all producer/consumer dependencies while
        avoiding a host synchronization after each block. The final synchronize
        is mandatory before reading, overwriting inputs, or freeing buffers.
        Enqueue timings are NOT reported as device compute timings.
        """
        # Alternating sync/async on these loaded OMs was rejected by CANN
        # (500002). Compare with separately loaded instances, keeping each
        # model on one execution path throughout its lifetime.
        mode = self.execution_mode
        if mode not in ("synchronous", "single-stream"):
            raise ValueError("Unknown VJP execution mode: " + str(mode))
        self.last_times, self.last_submission_ms = [], []
        self.last_synchronize_ms = None
        if mode == "synchronous":
            self.last_times = [executor.run() for executor in self.executors]
            return sum(self.last_times)
        if self.stream is None:
            self.stream, code = self.acl.rt.create_stream()
            check(code, "VJP create shared stream")
        started = time.monotonic()
        try:
            for executor in self.executors:
                submitted = time.monotonic()
                check(
                    self.acl.mdl.execute_async(executor.model_id, *executor.datasets, self.stream),
                    "VJP asynchronous partition submit",
                )
                self.stream_pending = True
                self.last_submission_ms.append((time.monotonic() - submitted) * 1000)
        finally:
            if self.stream_pending:
                synchronized = time.monotonic()
                check(self.acl.rt.synchronize_stream(self.stream), "VJP final stream synchronize")
                self.stream_pending = False
                self.last_synchronize_ms = (time.monotonic() - synchronized) * 1000
        return (time.monotonic() - started) * 1000

    def read(self, name):
        """Read one named gradient/model output after execution.

        Args: Name in the partition manifest.
        Returns: Owned NumPy array, or the exact cached constant boundary.
        """
        if name in self.constant_outputs:
            return self.constant_outputs[name].copy()
        executor, index = self.owners[name]
        return executor.read(index)

    def close(self):
        """Finish outstanding stream work and release only pipeline-owned memory.

        Args: None. Returns: None; borrowed OM buffers remain with their owner.
        """
        if self.stream is not None:
            if self.stream_pending:
                check(self.acl.rt.synchronize_stream(self.stream), "VJP synchronize before release")
                self.stream_pending = False
            self.acl.rt.destroy_stream(self.stream)
            self.stream = None
        for executor in reversed(self.executors):
            executor.close()
        self.executors = []
        for io in reversed(self.descriptors):
            self.acl.mdl.destroy_desc(io["desc"])
        self.descriptors = []
        for model in reversed(self.models):
            self.acl.mdl.unload(model)
        self.models = []
        for pointer in reversed(self.memory + self.input_memory):
            self.acl.rt.free(pointer)
        self.memory, self.input_memory = [], []


class PartitionedVJPPolicy(OMPolicy):
    def __init__(
        self,
        root,
        pair_directory,
        action_q01,
        action_q99,
        guidance_directory,
        max_guidance=REFERENCE_MAX_GUIDANCE,
        vjp_execution_mode="synchronous",
        *,
        seed=DEFAULT_SEED,
        noise_file=None,
    ):
        """Attach full frozen-STE VJP partitions to a resident OM pair.

        Args: Asset root, paired forward OMs, action quantiles, VJP directory,
            execution mode and optional initial-noise source.
        Returns: None; no source weights are updated.
        """
        guidance_scale(1.0, max_guidance)
        self.max_guidance = max_guidance
        # GEMM's FP16 inputs underflow for tiny unscaled cotangents. A VJP is
        # linear in its cotangent: scale internally, then divide the COMPLETE
        # correction by the same power of two. This is not a new RTC beta,
        # prefix mask, training update or change to the native quantized model.
        self.cotangent_scale = np.float32(COTANGENT_SCALE)
        self.gradient = None
        self.guidance_times, self.partition_times = [], []
        self.vjp_execute_times = []
        self.guidance_active = False
        super().__init__(root, pair_directory, action_q01, action_q99, seed=seed, noise_file=noise_file)
        try:
            bindings = {}
            for name, executor, kind, index in (
                ("past_kv_tensor", self.prefix, "outputs", 0),
                ("prefix_pad_masks", self.prefix, "outputs", 1),
                ("noise", self.suffix, "inputs", 2),
            ):
                bindings[name] = (executor.addresses[kind][index], executor.io[kind][index])
            self.gradient = PartitionedVJP(
                self.prefix.acl, guidance_directory, bindings, execution_mode=vjp_execution_mode
            )
            self.metadata["rtc"] = dict(
                author_commit=AUTHOR_RTC_COMMIT,
                max_guidance=max_guidance,
                prior_std=1.0,
                parameter_updates=0,
                forward_recomputed_for_vjp=True,
                native_endpoint_h2d_bytes_per_step=50 * 32 * 4,
                internal_cotangent_scale=float(self.cotangent_scale),
                cotangent_unscaled_after_vjp=True,
                partitions=self.gradient.telemetry,
            )
        except BaseException:
            self.close()
            raise

    def prepare_guidance(self, payload):
        """Validate and feed paired full RTC reference/weight tensors.

        Args: One inference request containing both RTC arrays or neither.
        Returns: None; resets per-request timings.
        """
        self.guidance_times, self.partition_times = [], []
        self.vjp_execute_times = []
        self.guidance_active = "rtc_prefix" in payload
        if self.guidance_active != ("rtc_weights" in payload):
            raise ValueError("RTC prefix and weights must be paired")
        if not self.guidance_active:
            return
        prefix, weights = (np.asarray(payload[name], np.float32) for name in ("rtc_prefix", "rtc_weights"))
        if any(value.shape != (1, 50, 32) or not np.isfinite(value).all() for value in (prefix, weights)):
            raise ValueError("Invalid complete RTC tensors")
        if np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("Invalid RTC temporal weights")
        self.gradient.feed(dict(prefix=prefix, weights=weights * self.cotangent_scale))

    def guide_velocity(self, velocity, step):
        """Apply the complete partitioned frozen-STE correction for one step.

        Args: Native velocity [1,50,32] and denoising step index.
        Returns: Finite guided FP32 velocity, or the native value without RTC.
        """
        if not self.guidance_active:
            return velocity
        started = time.monotonic()
        timestep = np.float32(1 - step / 10)
        self.gradient.feed(
            dict(timestep=np.asarray([timestep], np.float32), native_velocity=velocity.astype(np.float32))
        )
        self.vjp_execute_times.append(self.gradient.run())
        correction = self.gradient.read("input_correction").astype(np.float32) / self.cotangent_scale
        result = (
            velocity.astype(np.float32)
            - np.float32(guidance_scale(float(timestep), self.max_guidance)) * correction
        )
        if not np.isfinite(correction).all() or not np.isfinite(result).all():
            raise RuntimeError("Nonfinite complete partitioned RTC result")
        self.partition_times.append(self.gradient.last_times)
        self.guidance_times.append((time.monotonic() - started) * 1000)
        return result

    def close(self):
        """Release gradient partitions before the forward OMs they may borrow.

        Args: None. Returns: None; repeated calls are safe.
        """
        if self.gradient is not None:
            self.gradient.close()
            self.gradient = None
        super().close()
