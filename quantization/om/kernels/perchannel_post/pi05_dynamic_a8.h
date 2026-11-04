#ifndef GE_OP_PI05_DYNAMIC_A8_H
#define GE_OP_PI05_DYNAMIC_A8_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(Pi05DynamicA8)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(bound, TensorType({DT_FLOAT}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(clipped, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(Pi05DynamicA8)
}
#endif
