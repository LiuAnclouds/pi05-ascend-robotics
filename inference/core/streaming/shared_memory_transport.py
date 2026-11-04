"""Single-flight shared-memory snapshots for resident model inference."""

import importlib
import time
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

from inference.core.streaming.inference_process import InferenceWorker
from inference.include.definitions import INPUT_SHARED_MEMORY_BYTES


@dataclass(frozen=True)
class SharedArray:
    offset: int
    shape: tuple
    dtype: str


class SharedPayloadBackend:
    """Map a request arena inside the existing exclusive inference process."""

    def __init__(self, factory_path, factory_kwargs, memory_name, memory_bytes):
        """Attach one existing request arena and construct its model adapter.

        Args: Backend factory/options plus the shared-memory name and capacity.
        Returns: None; this child closes but never unlinks the parent's arena.
        """
        self.memory = shared_memory.SharedMemory(name=memory_name)
        self.backend = None
        try:
            if self.memory.size < memory_bytes:
                raise ValueError("Shared request arena is smaller than its contract")
            module, name = factory_path.rsplit(":", 1)
            self.backend = getattr(importlib.import_module(module), name)(**factory_kwargs)
            self.metadata = dict(
                getattr(self.backend, "metadata", {}),
                request_transport=dict(
                    kind="shared-memory",
                    capacity_bytes=memory_bytes,
                    ownership="Registered buffers are filled in the arena; arena reused only after complete reply",
                ),
            )
        except BaseException:
            self.memory.close()
            raise

    def _restore(self, value):
        """Map numeric descriptors recursively into read-only shared array views.

        Args: Nested request values. Returns: Matching structure with array views.
        """
        if isinstance(value, SharedArray):
            dtype = np.dtype(value.dtype)
            if dtype.hasobject or dtype.fields is not None:
                raise TypeError("Shared requests require plain numeric arrays")
            size = int(np.prod(value.shape, dtype=np.int64)) * dtype.itemsize
            if value.offset < 0 or value.offset + size > self.memory.size:
                raise ValueError("Shared request array exceeds its arena")
            array = np.ndarray(value.shape, dtype=dtype, buffer=self.memory.buf, offset=value.offset)
            array.flags.writeable = False
            return array
        if isinstance(value, dict):
            return {key: self._restore(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._restore(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._restore(item) for item in value)
        return value

    def predict(self, payload):
        """Pass a restored shared-memory snapshot to the resident model.

        Args: Nested request descriptors. Returns: The backend prediction mapping.
        """
        return self.backend.predict(self._restore(payload))

    def close(self):
        """Close backend resources and detach this process's arena mapping.

        Args: None. Returns: None; arena ownership remains with the parent.
        """
        try:
            if self.backend is not None and hasattr(self.backend, "close"):
                self.backend.close()
        finally:
            self.memory.close()


class SharedMemoryInferenceWorker(InferenceWorker):
    """Keep the existing worker protocol; send only array descriptors in its queue."""

    def __init__(
        self, factory_path, factory_kwargs=None, *, memory_bytes=INPUT_SHARED_MEMORY_BYTES, **kwargs
    ):
        """Create one reusable request arena and start the isolated model child.

        Args: Backend factory/options, arena capacity and InferenceWorker options.
        Returns: None; failed startup releases both child and shared-memory handle.
        """
        if memory_bytes <= 0:
            raise ValueError("Shared request arena must have positive capacity")
        self.request_memory = shared_memory.SharedMemory(create=True, size=memory_bytes)
        self.last_payload_bytes = 0
        self.last_submit_ms = 0.0
        self.last_copied_bytes = 0
        self._next_shared_offset = 0
        self._shared_allocations = {}
        self._pending_shared_arrays = []
        try:
            super().__init__(
                "inference.core.streaming.shared_memory_transport:SharedPayloadBackend",
                dict(
                    factory_path=factory_path,
                    factory_kwargs=factory_kwargs or {},
                    memory_name=self.request_memory.name,
                    memory_bytes=memory_bytes,
                ),
                **kwargs,
            )
        except BaseException:
            self.close()
            raise

    def submit(self, request_id, payload):
        """Submit registered inputs directly; snapshot other arrays into free arena space.

        Args: Unique request ID and nested numeric payload.
        Returns: None; registered arrays remain read-only until a complete reply.
            A second in-flight request or oversized payload is rejected.
        """
        if self.pending is not None:
            raise RuntimeError("Inference request already in flight")
        started = time.monotonic()
        position = 0
        copied_bytes = 0
        direct_arrays = []

        def pack(value):
            """Copy one nested value into aligned arena slots.

            Args: Numeric array/container/scalar. Returns: Descriptor/container/scalar.
            """
            nonlocal position, copied_bytes
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject or value.dtype.fields is not None:
                    raise TypeError("Shared requests require plain numeric arrays")
                registered = self._shared_allocations.get(int(value.ctypes.data))
                if registered is not None:
                    shared_offset, shape, dtype, byte_count = registered
                    if (
                        tuple(value.shape) != shape
                        or value.dtype.str != dtype
                        or value.nbytes != byte_count
                        or not value.flags.c_contiguous
                    ):
                        raise ValueError("Registered shared input buffer changed its shape, dtype or layout")
                    position = max(position, shared_offset + byte_count)
                    if position > self.request_memory.size:
                        raise BufferError("Request exceeds the fixed shared-memory arena")
                    value.flags.writeable = False
                    direct_arrays.append(value)
                    return SharedArray(shared_offset, shape, dtype)
                offset = (max(position, self._next_shared_offset) + 63) // 64 * 64
                position = offset + value.nbytes
                if position > self.request_memory.size:
                    raise BufferError("Request exceeds the fixed shared-memory arena")
                target = np.ndarray(
                    value.shape, dtype=value.dtype, buffer=self.request_memory.buf, offset=offset
                )
                np.copyto(target, value, casting="no")
                copied_bytes += value.nbytes
                return SharedArray(offset, tuple(value.shape), value.dtype.str)
            if isinstance(value, dict):
                return {key: pack(item) for key, item in value.items()}
            if isinstance(value, list):
                return [pack(item) for item in value]
            if isinstance(value, tuple):
                return tuple(pack(item) for item in value)
            return value

        try:
            descriptors = pack(payload)
            super().submit(request_id, descriptors)
        except BaseException:
            for array in direct_arrays:
                array.flags.writeable = True
            raise
        self._pending_shared_arrays = direct_arrays
        self.last_payload_bytes = position
        self.last_copied_bytes = copied_bytes
        self.last_submit_ms = (time.monotonic() - started) * 1000.0

    def allocate_shared_array(self, shape, dtype):
        """Reserve a stable, aligned input array directly in the request arena.

        Args:
            shape: Fixed positive shape for one reusable numeric input.
            dtype: Plain NumPy dtype compatible with the model input contract.
        Returns:
            Writable NumPy array whose storage belongs to this worker's arena.
            The array is made read-only during an in-flight request and writable
            again when its matching reply is received.
        """
        if self.pending is not None:
            raise RuntimeError("Cannot allocate request buffers while inference is in flight")
        shape = tuple(int(dimension) for dimension in shape)
        dtype = np.dtype(dtype)
        if not shape or any(dimension <= 0 for dimension in shape) or dtype.hasobject or dtype.fields:
            raise ValueError("Shared input arrays require a positive shape and plain numeric dtype")
        byte_count = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        offset = (self._next_shared_offset + 63) // 64 * 64
        if byte_count <= 0 or offset + byte_count > self.request_memory.size:
            raise BufferError("Shared input allocation exceeds the fixed request arena")
        array = np.ndarray(shape, dtype=dtype, buffer=self.request_memory.buf, offset=offset)
        self._next_shared_offset = offset + byte_count
        self._shared_allocations[int(array.ctypes.data)] = (offset, shape, dtype.str, byte_count)
        return array

    def _release_shared_arrays(self):
        """Return input ownership after a reply; takes no arguments and returns None."""
        for array in self._pending_shared_arrays:
            array.flags.writeable = True
        self._pending_shared_arrays = []

    def poll(self):
        """Poll without arguments; return a reply/None and unlock inputs only on reply."""
        result = super().poll()
        if result is not None:
            self._release_shared_arrays()
        return result

    def close(self):
        """Stop the child before detaching and unlinking the owned request arena.

        Args: None. Returns: None; repeated calls are safe.
        """
        try:
            if hasattr(self, "process"):
                super().close()
        finally:
            self._pending_shared_arrays = []
            self._shared_allocations.clear()
            memory = getattr(self, "request_memory", None)
            if memory is not None:
                self.request_memory = None
                try:
                    memory.close()
                finally:
                    memory.unlink()
