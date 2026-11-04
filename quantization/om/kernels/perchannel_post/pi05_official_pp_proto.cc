// Reconstruct the graph-level signature from the installed CANN 8.5 310P
// ops-info-legacy.json. No computation/tiling/kernel implementation is added.
#include "graph/operator_reg.h"

namespace ge {
REG_OP(PpMatmulW8A8)
    .INPUT(A, TensorType({DT_INT8}))
    .INPUT(B, TensorType({DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(descale, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(perTokenScale, TensorType({DT_FLOAT}))
    .OUTPUT(C, TensorType({DT_FLOAT16}))
    .ATTR(transposeA, Bool, false)
    .ATTR(transposeB, Bool, false)
    .OP_END_FACTORY_REG(PpMatmulW8A8)

IMPLEMT_COMMON_INFERFUNC(OfficialPpMatmulInferShape) {
    const auto a = op.GetInputDescByName("A").GetShape();
    const auto b = op.GetInputDescByName("B").GetShape();
    if (a.GetDimNum() != b.GetDimNum() || a.GetDimNum() < 2 || a.GetDimNum() > 3) return GRAPH_FAILED;
    const int64_t rank = a.GetDimNum();
    bool ta = false, tb = false;
    op.GetAttr("transposeA", ta);
    op.GetAttr("transposeB", tb);
    const int64_t rows = a.GetDim(rank - (ta ? 1 : 2));
    const int64_t columns = b.GetDim(rank - (tb ? 2 : 1));
    if (a.GetDim(rank - (ta ? 2 : 1)) != b.GetDim(rank - (tb ? 1 : 2))) return GRAPH_FAILED;
    auto c = op.GetOutputDescByName("C");
    c.SetShape(rank == 3 ? Shape({a.GetDim(0), rows, columns}) : Shape({rows, columns}));
    c.SetDataType(DT_FLOAT16);
    return op.UpdateOutputDesc("C", c);
}
COMMON_INFER_FUNC_REG(PpMatmulW8A8, OfficialPpMatmulInferShape);
}
