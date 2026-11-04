#include "pi05_perchannel_linear.h"
namespace ge {
IMPLEMT_COMMON_INFERFUNC(Pi05PerchannelLinearInferShape) {
    auto y = op.GetOutputDescByName("y");
    int64_t rows = 0, columns = 0;
    op.GetAttr("out_rows", rows);
    op.GetAttr("out_columns", columns);
    if (rows == 0) rows = op.GetInputDescByName("row_sum").GetShape().GetDim(0);
    if (columns == 0) columns = op.GetInputDescByName("weight_zero").GetShape().GetDim(0);
    y.SetShape(Shape({rows, columns}));
    y.SetDataType(DT_FLOAT16);
    y.SetFormat(FORMAT_ND);
    return op.UpdateOutputDesc("y", y);
}
COMMON_INFER_FUNC_REG(Pi05PerchannelLinear, Pi05PerchannelLinearInferShape);
}
