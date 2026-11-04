#include "pi05_mul_prepare_a8.h"
namespace ge {
IMPLEMT_COMMON_INFERFUNC(Pi05MulPrepareA8InferShape) {
    const auto shape = op.GetInputDescByName("x").GetShape();
    if (shape.GetDimNum() != 2 || shape.GetDim(1) != 16384 ||
            shape.GetDims() != op.GetInputDescByName("other").GetShape().GetDims()) return GRAPH_FAILED;
    const auto rows = (shape.GetDim(0) + 15) / 16 * 16;
    const auto columns = shape.GetDim(1);
    auto a = op.GetOutputDescByName("a");
    a.SetShape(Shape({columns / 32, rows / 16, 16, 32}));
    a.SetDataType(DT_INT8);
    a.SetFormat(FORMAT_ND);
    if (op.UpdateOutputDesc("a", a) != GRAPH_SUCCESS) return GRAPH_FAILED;
    for (const char* name : {"row_sum", "activation_scale"}) {
        auto d = op.GetOutputDescByName(name);
        d.SetShape(Shape({rows, 1}));
        d.SetDataType(DT_FLOAT);
        d.SetFormat(FORMAT_ND);
        if (op.UpdateOutputDesc(name, d) != GRAPH_SUCCESS) return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(Pi05MulPrepareA8, Pi05MulPrepareA8InferShape);
COMMON_INFER_FUNC_REG(Pi05GeluMulPrepareA8, Pi05MulPrepareA8InferShape);
}
