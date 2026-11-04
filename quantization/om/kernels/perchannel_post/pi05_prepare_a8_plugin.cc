#include "register/register.h"
namespace domi {
Status ParseParamPi05PrepareA8(const ge::Operator&, ge::Operator&) { return SUCCESS; }
REGISTER_CUSTOM_OP("Pi05PrepareA8")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::17::Pi05PrepareA8")
    .ParseParamsByOperatorFn(ParseParamPi05PrepareA8);
}
