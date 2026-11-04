#include "graph/operator_reg.h"
namespace ge {
REG_OP(Pi05OfficialGeluMulDynamicQuant)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(other, TensorType({DT_FLOAT16}))
    .OUTPUT(a, TensorType({DT_INT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(Pi05OfficialGeluMulDynamicQuant)

IMPLEMT_COMMON_INFERFUNC(Pi05OfficialGeluQuantShape) {
    const auto shape = op.GetInputDescByName("x").GetShape();
    if (shape.GetDimNum() != 2 || shape.GetDim(1) != 16384 || shape.GetDim(0) % 8 ||
        shape.GetDims() != op.GetInputDescByName("other").GetShape().GetDims()) return GRAPH_FAILED;
    auto a = op.GetOutputDescByName("a");
    a.SetShape(shape); a.SetDataType(DT_INT8); a.SetFormat(FORMAT_ND);
    if (op.UpdateOutputDesc("a", a) != GRAPH_SUCCESS) return GRAPH_FAILED;
    auto scale = op.GetOutputDescByName("scale");
    scale.SetShape(Shape({shape.GetDim(0)})); scale.SetDataType(DT_FLOAT); scale.SetFormat(FORMAT_ND);
    return op.UpdateOutputDesc("scale", scale);
}
COMMON_INFER_FUNC_REG(Pi05OfficialGeluMulDynamicQuant, Pi05OfficialGeluQuantShape);

REG_OP(Pi05OfficialChunkGeluMulDynamicQuant)
    .INPUT(g0, TensorType({DT_FLOAT16}))
    .INPUT(g1, TensorType({DT_FLOAT16}))
    .INPUT(g2, TensorType({DT_FLOAT16}))
    .INPUT(g3, TensorType({DT_FLOAT16}))
    .INPUT(u0, TensorType({DT_FLOAT16}))
    .INPUT(u1, TensorType({DT_FLOAT16}))
    .INPUT(u2, TensorType({DT_FLOAT16}))
    .INPUT(u3, TensorType({DT_FLOAT16}))
    .OUTPUT(a, TensorType({DT_INT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(Pi05OfficialChunkGeluMulDynamicQuant)

IMPLEMT_COMMON_INFERFUNC(Pi05OfficialChunkGeluQuantShape) {
    const auto shape = op.GetInputDescByName("g0").GetShape();
    if (shape.GetDimNum() != 2 || shape.GetDim(1) != 4096 || shape.GetDim(0) % 8) return GRAPH_FAILED;
    for (const auto name : {"g1", "g2", "g3", "u0", "u1", "u2", "u3"}) {
        if (shape.GetDims() != op.GetInputDescByName(name).GetShape().GetDims()) return GRAPH_FAILED;
    }
    auto a = op.GetOutputDescByName("a");
    a.SetShape(Shape({shape.GetDim(0), 16384})); a.SetDataType(DT_INT8); a.SetFormat(FORMAT_ND);
    if (op.UpdateOutputDesc("a", a) != GRAPH_SUCCESS) return GRAPH_FAILED;
    auto scale = op.GetOutputDescByName("scale");
    scale.SetShape(Shape({shape.GetDim(0)})); scale.SetDataType(DT_FLOAT); scale.SetFormat(FORMAT_ND);
    return op.UpdateOutputDesc("scale", scale);
}
COMMON_INFER_FUNC_REG(Pi05OfficialChunkGeluMulDynamicQuant, Pi05OfficialChunkGeluQuantShape);

REG_OP(Pi05OfficialVisionGeluDynamicQuant)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(a, TensorType({DT_INT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(Pi05OfficialVisionGeluDynamicQuant)

IMPLEMT_COMMON_INFERFUNC(Pi05OfficialVisionGeluQuantShape) {
    const auto shape = op.GetInputDescByName("x").GetShape();
    if (shape.GetDimNum() != 2 || shape.GetDim(1) != 4304 || shape.GetDim(0) % 8) return GRAPH_FAILED;
    auto a = op.GetOutputDescByName("a");
    a.SetShape(Shape({shape.GetDim(0), 4320})); a.SetDataType(DT_INT8); a.SetFormat(FORMAT_ND);
    if (op.UpdateOutputDesc("a", a) != GRAPH_SUCCESS) return GRAPH_FAILED;
    auto scale = op.GetOutputDescByName("scale");
    scale.SetShape(Shape({shape.GetDim(0)})); scale.SetDataType(DT_FLOAT); scale.SetFormat(FORMAT_ND);
    return op.UpdateOutputDesc("scale", scale);
}
COMMON_INFER_FUNC_REG(Pi05OfficialVisionGeluDynamicQuant, Pi05OfficialVisionGeluQuantShape);

REG_OP(Pi05PipelinedChunkGeluMulDynamicQuant)
    .INPUT(g0, TensorType({DT_FLOAT16}))
    .INPUT(g1, TensorType({DT_FLOAT16}))
    .INPUT(g2, TensorType({DT_FLOAT16}))
    .INPUT(g3, TensorType({DT_FLOAT16}))
    .INPUT(u0, TensorType({DT_FLOAT16}))
    .INPUT(u1, TensorType({DT_FLOAT16}))
    .INPUT(u2, TensorType({DT_FLOAT16}))
    .INPUT(u3, TensorType({DT_FLOAT16}))
    .OUTPUT(a, TensorType({DT_INT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(Pi05PipelinedChunkGeluMulDynamicQuant)
COMMON_INFER_FUNC_REG(Pi05PipelinedChunkGeluMulDynamicQuant, Pi05OfficialChunkGeluQuantShape);

REG_OP(Pi05PipelinedVisionGeluDynamicQuant)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(a, TensorType({DT_INT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(Pi05PipelinedVisionGeluDynamicQuant)
COMMON_INFER_FUNC_REG(Pi05PipelinedVisionGeluDynamicQuant, Pi05OfficialVisionGeluQuantShape);
}
