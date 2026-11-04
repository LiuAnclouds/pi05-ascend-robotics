#include "pi05_perchannel_post.h"

namespace ge {
IMPLEMT_COMMON_INFERFUNC(Pi05PerchannelPostInferShape) {
    const auto rows = op.GetInputDescByName("row_sum").GetShape().GetDim(0);
    const auto columns = op.GetInputDescByName("weight_zero").GetShape().GetDim(0);
    auto result = op.GetOutputDescByName("y");
    result.SetShape(Shape({rows, columns}));
    result.SetDataType(DT_FLOAT16);
    result.SetFormat(FORMAT_ND);
    return op.UpdateOutputDesc("y", result);
}
COMMON_INFER_FUNC_REG(Pi05PerchannelPost, Pi05PerchannelPostInferShape);
}  // namespace ge
