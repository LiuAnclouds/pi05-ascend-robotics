#ifndef GE_OP_PI05_PREPARE_A8_H
#define GE_OP_PI05_PREPARE_A8_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(Pi05PrepareA8)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(a, TensorType({DT_INT8}))
    .OUTPUT(row_sum, TensorType({DT_FLOAT}))
    .OUTPUT(activation_scale, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(Pi05PrepareA8)
}
#endif
