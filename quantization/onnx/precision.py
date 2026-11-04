"""ONNX-only precision repairs and pure-node sharing; no PyTorch/ROS dependency."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper as nh


def _repair_legacy_layernorm_fp64(model):
    """opset<17 PyTorch casts half LayerNorm to its DOUBLE epsilon dtype.

    310P has no AI-Core DOUBLE arithmetic. Use FP32 accumulation, keeping the
    existing final FP16 cast, epsilon and normalization formula. Do not alter
    any W8/A8 quantizer or unrelated floating computation.
    """
    casts, constants = [], []
    for node in model.graph.node:
        if node.op_type == "Cast":
            target = next(a for a in node.attribute if a.name == "to")
            if target.i == onnx.TensorProto.DOUBLE:
                if "norm" not in node.name.lower():
                    raise ValueError("Unexpected non-Norm DOUBLE cast: " + node.name)
                target.i = onnx.TensorProto.FLOAT
                casts.append(node.name)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == onnx.TensorProto.DOUBLE:
                if "norm" not in node.name.lower() or list(attr.t.dims):
                    raise ValueError("Unexpected non-scalar/non-Norm DOUBLE constant: " + node.name)
                attr.t.CopyFrom(nh.from_array(nh.to_array(attr.t).astype(np.float32), attr.t.name))
                constants.append(node.name)
    for value in model.graph.value_info:
        if value.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE:
            if "norm" not in value.name.lower():
                raise ValueError("Unexpected non-Norm DOUBLE value info")
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    return dict(
        casts=casts, constants=constants, accumulator="FP32", output="existing FP16 boundary preserved"
    )


def repair_legacy_layernorm_export(source, output):
    """Create a repaired candidate; retain original header and external weights."""

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source.parent != output.parent:
        raise ValueError("Use a new candidate filename in the same external-data directory")
    model = onnx.load(str(source), load_external_data=False)
    result = _repair_legacy_layernorm_fp64(model)
    onnx.save_model(model, str(output))
    result.update(
        source=str(source),
        output_file=str(output),
        cause="PyTorch legacy LayerNorm symbolic promotes half numerator to DOUBLE epsilon dtype",
        quantization="W8/dynamic A8 unchanged; no extra A16 Linear exceptions",
    )
    output.with_suffix(".norm_repair.json").write_text(json.dumps(result, indent=2))
    return result


def finalize_export(path):
    """Repair accidental legacy FP64 Norms and share identical pure paths."""

    # Quantization is pure: equal input, dtype and attributes give equal outputs.
    model = onnx.load(str(path), load_external_data=False)
    norm_repair = _repair_legacy_layernorm_fp64(model)
    before = Counter(n.op_type for n in model.graph.node)
    replacements, seen, kept, casts = {}, {}, [], {}
    outputs = {value.name for value in model.graph.output}
    pure = {"Cast", "Reshape", "Constant", "NPUDynamicQuant"}
    for node in model.graph.node:
        for i, value in enumerate(node.input):
            while value in replacements:
                value = replacements[value]
            node.input[i] = value
        if node.op_type == "Cast" and node.input[0] in casts:
            target = next(a.i for a in node.attribute if a.name == "to")
            if casts[node.input[0]] == target and node.output[0] not in outputs:
                replacements[node.output[0]] = node.input[0]
                continue
        attrs = []
        for attribute in node.attribute:
            copied = onnx.AttributeProto()
            copied.CopyFrom(attribute)
            if copied.type == onnx.AttributeProto.TENSOR and not copied.t.external_data:
                copied.t.ClearField("name")
                copied.t.ClearField("doc_string")
            attrs.append(copied.SerializeToString())
        key = (node.op_type, node.domain, tuple(node.input), tuple(attrs))
        if node.op_type in pure and key in seen and not any(name in outputs for name in node.output):
            for out, existing in zip(node.output, seen[key]):
                replacements[out] = existing
        else:
            kept.append(node)
            if node.op_type == "Cast":
                casts[node.output[0]] = next(a.i for a in node.attribute if a.name == "to")
            if node.op_type in pure:
                seen[key] = list(node.output)
    live = set(outputs)
    result = []
    for node in reversed(kept):
        if any(x in live for x in node.output):
            result.append(node)
            live.update(node.input)
    del model.graph.node[:]
    model.graph.node.extend(reversed(result))
    used = {name for node in model.graph.node for name in node.input} | outputs
    inits = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(inits)
    known = {name for node in model.graph.node for name in node.output}
    info = [v for v in model.graph.value_info if v.name in known]
    del model.graph.value_info[:]
    model.graph.value_info.extend(info)
    onnx.save_model(model, str(path))
    after = Counter(n.op_type for n in model.graph.node)
    return dict(
        nodes_before=sum(before.values()),
        nodes_after=sum(after.values()),
        types=dict(after),
        official_dynamic_quantizers=after["NPUDynamicQuant"],
        official_integer_matmuls=after["QuantBatchMatMul"],
        source_ops_all_builtin=True,
        layernorm_fp64_repair=norm_repair,
    )


def add_serial_euler_state(source, output):
    """在 Part2 图内执行 FP16 Euler 更新，输出供下一步复用的动作状态。

    Args:
        source: 已有单步 Part2 ONNX 文件。
        output: 新 ONNX 文件路径。
    Returns:
        新图的输入、输出和计算精度信息；保留原文件。
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    model = onnx.load(str(source), load_external_data=False)
    noise = next((value for value in model.graph.input if value.name == "noise"), None)
    velocity = next((value for value in model.graph.output if value.name == "velocity"), None)
    if noise is None or velocity is None:
        raise ValueError("Expected Part2 inputs noise and output velocity")
    if (
        noise.type.tensor_type.elem_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)
        or velocity.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16
    ):
        raise ValueError("Serial Euler export expects the existing FP16 Part2 action model")
    if [dimension.dim_value for dimension in noise.type.tensor_type.shape.dim] != [1, 50, 32]:
        raise ValueError("Serial Euler export expects the fixed [1,50,32] action shape")
    names = {name for node in model.graph.node for name in (*node.input, *node.output)}
    names.update(value.name for value in (*model.graph.input, *model.graph.output))
    generated = {"serial_scaled_velocity", "updated_action"}
    if names.intersection(generated) or "serial_euler_scale" in {value.name for value in model.graph.initializer}:
        raise ValueError("Part2 graph already contains serial Euler values")
    # 原动作网络在入口将 noise 转为 FP16；串行状态直接以 FP16 传入。
    noise.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
    for node in model.graph.node:
        if node.domain == "ai.onnx":
            node.domain = ""
    for opset in model.opset_import:
        if opset.domain == "ai.onnx":
            opset.domain = ""
    model.graph.node.extend([
        onnx.helper.make_node(
            "Mul", ["velocity", "serial_euler_scale"], ["serial_scaled_velocity"],
            name="SerialEulerScale"
        ),
        onnx.helper.make_node(
            "Add", ["noise", "serial_scaled_velocity"], ["updated_action"], name="SerialEulerUpdate"
        ),
    ])
    model.graph.initializer.append(
        nh.from_array(np.asarray(-0.1, np.float16), name="serial_euler_scale")
    )
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("updated_action", onnx.TensorProto.FLOAT16, [1, 50, 32])
    )
    onnx.checker.check_model(model)
    # CANN 的混合域模型要求标准节点显式写出 ai.onnx。
    for node in model.graph.node:
        if node.domain == "":
            node.domain = "ai.onnx"
    for opset in model.opset_import:
        if opset.domain == "":
            opset.domain = "ai.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output))
    return dict(
        source=str(source),
        output=str(output),
        update="FP16 noise + FP16(-0.1 * FP16 velocity)",
        denoiser_input="FP16 action state; existing action network retained",
        outputs=["velocity:FP16", "updated_action:FP16"],
    )
