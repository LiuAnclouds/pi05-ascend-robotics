#include "register/register.h"

namespace domi {
Status ParseParamPi05PerchannelPost(const ge::Operator&, ge::Operator&) {
    return SUCCESS;
}
REGISTER_CUSTOM_OP("Pi05PerchannelPost")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::17::Pi05PerchannelPost")
    .ParseParamsByOperatorFn(ParseParamPi05PerchannelPost);
}  // namespace domi
