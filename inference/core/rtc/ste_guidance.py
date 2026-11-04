"""Optional frozen-STE RTC policy for recorded-input replay only.

The ordinary W8/dynamic-A8 OMs are unchanged. The additional gradient OM
borrows their KV/noise/velocity addresses, never round-tripping the KV through
the control process. This is explicitly approximate inference-time RTC.
"""

import math
import time

import numpy as np

from inference.core.backends.acl_executor import BufferedACL, check, model_io
from inference.core.backends.om_policy import OMPolicy
from inference.core.rtc.guidance_math import (
    AUTHOR_RTC_COMMIT,
    REFERENCE_MAX_GUIDANCE,
    REFERENCE_PRIOR_STD,
    guidance_scale,
)
from inference.include.definitions import DEFAULT_SEED


class STEGuidancePolicy(OMPolicy):
    def __init__(
        self,
        root,
        pair_directory,
        action_q01,
        action_q99,
        guidance_om,
        max_guidance=REFERENCE_MAX_GUIDANCE,
        *,
        seed=DEFAULT_SEED,
        noise_file=None,
    ):
        """Load a frozen-STE gradient OM beside the native forward model pair.

        Args: Asset root, paired forward OMs, action quantiles, full VJP OM,
            guidance bound and optional exact initial-noise source.
        Returns: None; owns ACL allocations until close().
        """
        if not math.isfinite(max_guidance) or max_guidance <= 0:
            raise ValueError("Invalid guidance bound")
        self.max_guidance = float(max_guidance)
        self.gradient = self.gradient_model = self.gradient_io = None
        self.gradient_memory = []
        self.guidance_active = False
        self.guidance_times = []
        super().__init__(root, pair_directory, action_q01, action_q99, seed=seed, noise_file=noise_file)
        self.acl = self.prefix.acl
        try:
            work, weight, ret = self.acl.mdl.query_size(str(guidance_om))
            check(ret, "query gradient model")
            pointers = []
            for size in (work, weight):
                pointer = 0
                if size:
                    pointer, ret = self.acl.rt.malloc(size, 2)
                    check(ret, "normal-page gradient allocation")
                    self.gradient_memory.append(pointer)
                pointers.append(pointer)
            self.gradient_model, ret = self.acl.mdl.load_from_file_with_mem(
                str(guidance_om), pointers[0], work, pointers[1], weight
            )
            check(ret, "load gradient model")
            self.gradient_io = model_io(self.acl, self.gradient_model)
            names = [
                self.acl.mdl.get_input_name_by_index(self.gradient_io["desc"], i)
                for i in range(len(self.gradient_io["inputs"]))
            ]
            names = [name.decode() if isinstance(name, bytes) else name for name in names]
            if names != [
                "past_kv_tensor",
                "prefix_pad_masks",
                "noise",
                "timestep",
                "native_velocity",
                "prefix",
                "weights",
            ]:
                raise ValueError(f"Gradient model input order differs: {names}")
            borrowed = {}
            for index, executor, kind, source in (
                (0, self.prefix, "outputs", 0),
                (1, self.prefix, "outputs", 1),
                (2, self.suffix, "inputs", 2),
                (4, self.suffix, "outputs", 0),
            ):
                contract = executor.io[kind][source]
                if contract != self.gradient_io["inputs"][index]:
                    raise ValueError(
                        f"Gradient input {index} is not compatible with the native device buffer"
                    )
                borrowed[index] = (executor.addresses[kind][source], contract)
            self.gradient = BufferedACL(
                self.acl, self.gradient_model, self.gradient_io, borrowed=borrowed, malloc_policy=2
            )
            self.metadata["rtc"] = dict(
                om=str(guidance_om),
                derivative="frozen W8/A8 identity-STE surrogate",
                max_guidance=self.max_guidance,
                work_bytes=work,
                weight_bytes=weight,
                prior_std=REFERENCE_PRIOR_STD,
                author_commit=AUTHOR_RTC_COMMIT,
                forward_recomputed_for_vjp=True,
                shared_device_inputs=[0, 1, 2, 4],
                parameter_updates=0,
            )
        except BaseException:
            self.close()
            raise

    def prepare_guidance(self, payload):
        """Validate RTC input tensors and copy them to the VJP model buffers.

        Args: Request with paired normalized prefix and temporal weights.
        Returns: None; no weights or source artifacts are changed.
        """
        self.guidance_times = []
        self.guidance_active = "rtc_prefix" in payload
        if self.guidance_active != ("rtc_weights" in payload):
            raise ValueError("RTC prefix and weights must be provided together")
        if not self.guidance_active:
            return  # Bootstrap has no old trajectory to condition on.
        arrays = [np.asarray(payload[name], np.float32) for name in ("rtc_prefix", "rtc_weights")]
        if any(value.shape != (1, 50, 32) or not np.isfinite(value).all() for value in arrays):
            raise ValueError("Invalid complete RTC prefix/weight tensors")
        if np.any(arrays[1] < 0) or np.any(arrays[1] > 1):
            raise ValueError("RTC temporal weights must be in [0, 1]")
        self.gradient.feed({5: arrays[0], 6: arrays[1]})

    def guide_velocity(self, velocity, step):
        """Apply one finite VJP correction to the ordinary native velocity.

        Args: Native action velocity [1,50,32] and denoising step index.
        Returns: FP32 guided velocity, or unchanged velocity when RTC is absent.
        """
        if not self.guidance_active:
            return velocity
        started = time.monotonic()
        timestep = np.float32(1 - step / 10)
        self.gradient.feed({3: np.asarray([timestep], np.float32)})
        self.gradient.run()
        correction = self.gradient.read(0).astype(np.float32)
        scale = np.float32(guidance_scale(float(timestep), self.max_guidance))
        guided = velocity.astype(np.float32) - scale * correction
        if not np.isfinite(correction).all() or not np.isfinite(guided).all():
            raise RuntimeError("Nonfinite RTC correction/complete 50x32 guided velocity")
        self.guidance_times.append((time.monotonic() - started) * 1000)
        return guided

    def close(self):
        """Unload the gradient OM and allocations before closing the base pair.

        Args: None. Returns: None; repeated calls are safe.
        """
        if self.gradient is not None:
            self.gradient.close()
            self.gradient = None
        if self.gradient_io is not None:
            self.acl.mdl.destroy_desc(self.gradient_io["desc"])
            self.gradient_io = None
        if self.gradient_model is not None:
            self.acl.mdl.unload(self.gradient_model)
            self.gradient_model = None
        for pointer in reversed(self.gradient_memory):
            self.acl.rt.free(pointer)
        self.gradient_memory = []
        super().close()
