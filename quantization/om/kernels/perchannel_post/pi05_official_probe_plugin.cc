// ONNX import adapters only. All calculations use unmodified built-in CANN operators.
#include "register/register.h"

namespace domi {
Status ParseOfficialPpMatmul(const ge::Operator& source, ge::Operator& target) {
    const auto result = AutoMappingByOpFn(source, target);
    if (result != SUCCESS) return result;
    target.SetAttr("transposeA", false);
    target.SetAttr("transposeB", true);
    return SUCCESS;
}
REGISTER_CUSTOM_OP("PpMatmulW8A8")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::16::Pi05OfficialPpMatmul")
    .ParseParamsByOperatorFn(ParseOfficialPpMatmul);
}
