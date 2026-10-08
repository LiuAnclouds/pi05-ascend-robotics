"""Share the prefix OM's device outputs with the denoising OM, without copies."""

from __future__ import annotations


class DevicePrefixCache:
    """Borrow ACLLite output buffers; their owner and lifetime remain Part1.

    Both OMs must run sequentially in the same ACL context/thread. Part1 must
    not run again until all Part2 steps finish. No model arithmetic is changed.
    ACLLite internals are accessed only here, not in the control loop.
    """

    def __init__(self, part1, part2):
        """Check shape/dtype/size compatibility and borrow KV/mask device pointers.

        Args: part1 and part2 are initialized ACLLite models in one context.
        Returns: None; the two borrowed buffer descriptors are stored locally.
        """
        import acl

        self.acl, self.part1 = acl, part1
        self.buffers = []
        for index in range(2):
            output_shape, ret = acl.mdl.get_output_dims(part1._model_desc, index)
            self._check(ret, 'get_output_dims')
            input_shape, ret = acl.mdl.get_input_dims(part2._model_desc, index)
            self._check(ret, 'get_input_dims')
            output_type = acl.mdl.get_output_data_type(part1._model_desc, index)
            input_type = acl.mdl.get_input_data_type(part2._model_desc, index)
            output_size = acl.mdl.get_output_size_by_index(part1._model_desc, index)
            input_size = acl.mdl.get_input_size_by_index(part2._model_desc, index)
            if (output_shape['dims'] != input_shape['dims'] or
                    output_type != input_type or output_size != input_size):
                raise ValueError(f'Part1 output {index} does not match Part2 input {index}')
            buffer = acl.mdl.get_dataset_buffer(part1._output_dataset, index)
            self.buffers.append({'data': acl.get_data_buffer_addr(buffer), 'size': output_size})

    @staticmethod
    def _check(code: int, operation: str) -> None:
        """Raise on an ACL error code; return None on successful operation."""
        if code != 0:
            raise RuntimeError(f'ACL {operation} failed: {code}')

    def execute(self, inputs: list) -> list[dict]:
        """Run Part1 on host inputs and return device KV/mask buffer descriptors.

        The descriptors are accepted by ACLLite Part2.execute directly. They
        are borrowed, not freed here, and stay valid until Part1 is closed.
        """
        model = self.part1
        try:
            self._check(model._gen_input_dataset(inputs), 'Part1 input dataset')
            self._check(self.acl.mdl.execute(model._model_id, model._input_dataset,
                                             model._output_dataset), 'Part1 execute')
        finally:
            model._release_dataset(model._input_dataset)
            model._input_dataset = None
        return self.buffers

    def to_host(self) -> list:
        """Copy the latest KV/mask to NumPy only when a diagnostic caller needs them."""
        return self.part1._output_dataset_to_numpy()
