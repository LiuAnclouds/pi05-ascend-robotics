#include "register/register.h"
namespace domi {
Status ParseParamPi05MulPrepareA8(const ge::Operator&, ge::Operator&) { return SUCCESS; }
REGISTER_CUSTOM_OP("Pi05MulPrepareA8")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::17::Pi05MulPrepareA8")
    .ParseParamsByOperatorFn(ParseParamPi05MulPrepareA8);
REGISTER_CUSTOM_OP("Pi05GeluMulPrepareA8")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::17::Pi05GeluMulPrepareA8")
    .ParseParamsByOperatorFn(ParseParamPi05MulPrepareA8);
}
