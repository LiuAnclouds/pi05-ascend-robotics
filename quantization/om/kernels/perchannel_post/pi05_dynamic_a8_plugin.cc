#include "register/register.h"
namespace domi {
Status ParseParamPi05DynamicA8(const ge::Operator&, ge::Operator&) { return SUCCESS; }
REGISTER_CUSTOM_OP("Pi05DynamicA8")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::17::Pi05DynamicA8")
    .ParseParamsByOperatorFn(ParseParamPi05DynamicA8);
}
