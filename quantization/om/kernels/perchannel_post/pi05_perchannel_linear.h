#ifndef GE_OP_PI05_PERCHANNEL_LINEAR_H
#define GE_OP_PI05_PERCHANNEL_LINEAR_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(Pi05PerchannelLinear)
    .INPUT(a, TensorType({DT_INT8}))
    .INPUT(w, TensorType({DT_INT8}))
    .INPUT(row_sum, TensorType({DT_FLOAT}))
    .INPUT(weight_zero, TensorType({DT_FLOAT}))
    .INPUT(weight_scale, TensorType({DT_FLOAT}))
    .INPUT(activation_scale, TensorType({DT_FLOAT}))
    .INPUT(bias, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(out_rows, Int, 0)
    .ATTR(out_columns, Int, 0)
    .OP_END_FACTORY_REG(Pi05PerchannelLinear)
}
#endif
