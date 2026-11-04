#!/usr/bin/env python3
"""Preallocated ACL execution and shared device buffers, without robot I/O."""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

try:
    import acl
except ModuleNotFoundError:
    acl = None

DTYPES = {0: np.float32, 1: np.float16, 2: np.int8, 3: np.int32, 9: np.int64, 11: np.float64, 12: np.bool_}


def check(code, operation):
    """Raise with the ACL operation when the runtime returns a failure code."""
    if code != 0:
        raise RuntimeError(f"{operation}: ACL return {code}")


def model_io(acl, model_id):
    """Read one model descriptor and return input/output dtype, shape, bytes."""
    desc = acl.mdl.create_desc()
    check(acl.mdl.get_desc(desc, model_id), "get_desc")
    result = {"desc": desc, "inputs": [], "outputs": []}
    for kind, target, count, size_fn, dim_fn, dtype_fn in [
        (
            "input",
            result["inputs"],
            acl.mdl.get_num_inputs(desc),
            acl.mdl.get_input_size_by_index,
            acl.mdl.get_input_dims,
            acl.mdl.get_input_data_type,
        ),
        (
            "output",
            result["outputs"],
            acl.mdl.get_num_outputs(desc),
            acl.mdl.get_output_size_by_index,
            acl.mdl.get_output_dims,
            acl.mdl.get_output_data_type,
        ),
    ]:
        for i in range(count):
            dims, ret = dim_fn(desc, i)
            check(ret, f"{kind} dims {i}")
            target.append(
                {"bytes": size_fn(desc, i), "shape": list(dims["dims"]), "dtype_id": dtype_fn(desc, i)}
            )
    return result


class BufferedACL:
    """Fixed-shape synchronous executor with reusable buffers and borrowed inputs.

    Args: ACL API, loaded model id, descriptor, and optional input-index to
    (device address, tensor specification) mapping. Borrowed addresses are not
    freed here; their owner must outlive this executor. No robot dependencies.
    """

    def __init__(self, acl, model_id, io, borrowed=None, malloc_policy=0):
        """Allocate persistent I/O for one loaded, fixed-shape OM.

        Args: ACL API, model handle/descriptor, borrowed input bindings and policy.
        Returns: None; only owned allocations are freed by close().
        """
        self.acl, self.model_id, self.io = acl, model_id, io
        self.datasets, self.buffers, self.owned, self.host = [], [], [], []
        self.addresses = {"inputs": [], "outputs": []}
        self.borrowed = borrowed or {}
        try:
            for kind in ("inputs", "outputs"):
                dataset = acl.mdl.create_dataset()
                self.datasets.append(dataset)
                for index, spec in enumerate(io[kind]):
                    if kind == "inputs" and index in self.borrowed:
                        address, contract = self.borrowed[index]
                        if contract != spec:
                            raise ValueError(f"Incompatible borrowed input {index}: {contract} vs {spec}")
                    else:
                        address, ret = acl.rt.malloc(spec["bytes"], malloc_policy)
                        check(ret, "buffered malloc")
                        self.owned.append(address)
                    self.addresses[kind].append(address)
                    buffer = acl.create_data_buffer(address, spec["bytes"])
                    self.buffers.append(buffer)
                    _, ret = acl.mdl.add_dataset_buffer(dataset, buffer)
                    check(ret, "buffered add_dataset_buffer")
            for spec in io["outputs"]:
                pointer, ret = acl.rt.malloc_host(spec["bytes"])
                check(ret, "buffered malloc_host")
                self.host.append(pointer)
        except Exception:
            self.close()
            raise

    def feed(self, values):
        """Synchronously upload input arrays from their existing host pointers.

        Args: Input-index/array pairs; shape and dtype must match the OM.
        Returns: Host preparation/upload milliseconds and bytes transferred.
            The local ndarray owns the source until memcpy returns. No bytes
            staging allocation is made; borrowed device inputs remain untouched.
        """
        started = time.perf_counter()
        h2d_ms, byte_count = 0.0, 0
        for index, array in values.items():
            if index in self.borrowed:
                raise ValueError(f"Cannot overwrite borrowed input {index}")
            spec = self.io["inputs"][index]
            value = np.ascontiguousarray(array, dtype=DTYPES[spec["dtype_id"]])
            if list(value.shape) != spec["shape"] or value.nbytes != spec["bytes"]:
                raise ValueError(f"Input {index}: {value.shape} vs {spec}")
            copied_at = time.perf_counter()
            check(
                self.acl.rt.memcpy(
                    self.addresses["inputs"][index],
                    spec["bytes"],
                    int(value.ctypes.data),
                    value.nbytes,
                    1,
                ),
                "buffered H2D",
            )
            h2d_ms += (time.perf_counter() - copied_at) * 1000.0
            byte_count += value.nbytes
        self.last_feed = dict(
            total_ms=(time.perf_counter() - started) * 1000.0,
            h2d_ms=h2d_ms,
            bytes=byte_count,
        )
        return self.last_feed

    def run(self):
        """Execute synchronously; return execute-only milliseconds."""
        start = time.perf_counter()
        check(self.acl.mdl.execute(self.model_id, *self.datasets), "buffered execute")
        return (time.perf_counter() - start) * 1000.0

    def run_async(self, stream):
        """向指定 ACL 流提交一次执行请求。

        Args:
            stream: 前后依赖的模型共用的 ACL 流。
        Returns:
            无；调用者在读取输出或释放缓存前等待执行完成。
        """
        check(
            self.acl.mdl.execute_async(self.model_id, *self.datasets, stream),
            "buffered asynchronous execute",
        )

    def read(self, index):
        """Copy one output to a persistent host buffer and return an owned array."""
        spec = self.io["outputs"][index]
        pointer = self.host[index]
        check(
            self.acl.rt.memcpy(pointer, spec["bytes"], self.addresses["outputs"][index], spec["bytes"], 2),
            "buffered D2H",
        )
        raw = self.acl.util.ptr_to_bytes(pointer, spec["bytes"])
        return np.frombuffer(raw, dtype=DTYPES[spec["dtype_id"]]).reshape(spec["shape"]).copy()

    def close(self):
        """Release only owned resources; borrowers must be closed first."""
        for buffer in self.buffers:
            self.acl.destroy_data_buffer(buffer)
        for dataset in self.datasets:
            self.acl.mdl.destroy_dataset(dataset)
        for pointer in self.host:
            self.acl.rt.free_host(pointer)
        for pointer in self.owned:
            self.acl.rt.free(pointer)
        self.buffers, self.datasets, self.host, self.owned = [], [], [], []


@contextmanager
def resident_model_pair(part1, part2, allocation="default", telemetry=None):
    """Load two OMs and yield ACL, handles, descriptors and resident executors.

    Inputs: two OM paths; default loader or explicit normal-page work/weights/IO.
    Outputs: one shared prefix-buffer execution context; optional load telemetry.
    NORMAL_ONLY=2 is from installed CANN 8.5 acl_rt.h, not a system hugepage setting.
    No hardware-control imports. Owned resources are released on normal/error exit.
    """
    if allocation not in ("default", "normal"):
        raise ValueError(f"Unknown allocation strategy: {allocation}")
    if acl is None:
        raise RuntimeError("Load the CANN Python environment before creating an ACL model pair")
    handles, descriptors, buffered = [], [], []
    model_memory = []
    malloc_policy = 2 if allocation == "normal" else 0
    telemetry = telemetry if telemetry is not None else {}
    telemetry.update(strategy=allocation, owned_io_malloc_policy=malloc_policy, models=[])
    context = None
    initialized = False
    try:
        check(acl.init(), "acl.init")
        initialized = True
        check(acl.rt.set_device(0), "set_device")
        context, ret = acl.rt.create_context(0)
        check(ret, "context")
        for path in (part1, part2):
            started = time.perf_counter()
            work_size, weight_size, ret = acl.mdl.query_size(str(path))
            check(ret, f"query_size {path}")
            model_log = {"path": str(path), "work_bytes": work_size, "weight_bytes": weight_size}
            telemetry["models"].append(model_log)
            if allocation == "normal":
                pointers = []
                for size in (work_size, weight_size):
                    if size:
                        pointer, ret = acl.rt.malloc(size, malloc_policy)
                        check(ret, f"model malloc {size}")
                        model_memory.append(pointer)
                    else:
                        pointer = 0
                    pointers.append(pointer)
                handle, ret = acl.mdl.load_from_file_with_mem(
                    str(path), pointers[0], work_size, pointers[1], weight_size
                )
            else:
                handle, ret = acl.mdl.load_from_file(str(path))
            check(ret, f"load {path}")
            model_log["load_and_allocation_ms"] = (time.perf_counter() - started) * 1000
            handles.append(handle)
            descriptors.append(model_io(acl, handle))
        buffered.append(BufferedACL(acl, handles[0], descriptors[0], malloc_policy=malloc_policy))
        borrowed = {i: (buffered[0].addresses["outputs"][i], descriptors[0]["outputs"][i]) for i in (0, 1)}
        buffered.append(BufferedACL(acl, handles[1], descriptors[1], borrowed, malloc_policy=malloc_policy))
        yield acl, handles, descriptors, buffered
    finally:
        for executor in reversed(buffered):
            executor.close()
        for io in descriptors:
            acl.mdl.destroy_desc(io["desc"])
        for handle in reversed(handles):
            acl.mdl.unload(handle)
        # Explicit work and weight storage must outlive every loaded model.
        for pointer in reversed(model_memory):
            acl.rt.free(pointer)
        if context is not None:
            acl.rt.destroy_context(context)
        if initialized:
            acl.rt.reset_device(0)
            acl.finalize()
