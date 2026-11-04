#include "pi05_dynamic_a8.h"
namespace ge {
IMPLEMT_COMMON_INFERFUNC(Pi05DynamicA8InferShape) {
    auto scale = op.GetOutputDescByName("scale");
    scale.SetShape(op.GetInputDescByName("bound").GetShape());
    scale.SetDataType(DT_FLOAT);
    scale.SetFormat(FORMAT_ND);
    if (op.UpdateOutputDesc("scale", scale) != GRAPH_SUCCESS) return GRAPH_FAILED;
    auto clipped = op.GetOutputDescByName("clipped");
    clipped.SetShape(op.GetInputDescByName("x").GetShape());
    clipped.SetDataType(DT_FLOAT);
    clipped.SetFormat(FORMAT_ND);
    return op.UpdateOutputDesc("clipped", clipped);
}
COMMON_INFER_FUNC_REG(Pi05DynamicA8, Pi05DynamicA8InferShape);
}
