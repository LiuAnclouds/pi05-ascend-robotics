#include "register/register.h"
namespace domi {
Status ParsePi05OfficialGeluQuant(const ge::Operator&, ge::Operator&) { return SUCCESS; }
REGISTER_CUSTOM_OP("Pi05OfficialGeluMulDynamicQuant")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::16::Pi05OfficialGeluMulDynamicQuant")
    .ParseParamsByOperatorFn(ParsePi05OfficialGeluQuant);
REGISTER_CUSTOM_OP("Pi05OfficialChunkGeluMulDynamicQuant")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::16::Pi05OfficialChunkGeluMulDynamicQuant")
    .ParseParamsByOperatorFn(ParsePi05OfficialGeluQuant);
REGISTER_CUSTOM_OP("Pi05OfficialVisionGeluDynamicQuant")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::16::Pi05OfficialVisionGeluDynamicQuant")
    .ParseParamsByOperatorFn(ParsePi05OfficialGeluQuant);
REGISTER_CUSTOM_OP("Pi05PipelinedChunkGeluMulDynamicQuant")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::16::Pi05PipelinedChunkGeluMulDynamicQuant")
    .ParseParamsByOperatorFn(ParsePi05OfficialGeluQuant);
REGISTER_CUSTOM_OP("Pi05PipelinedVisionGeluDynamicQuant")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::16::Pi05PipelinedVisionGeluDynamicQuant")
    .ParseParamsByOperatorFn(ParsePi05OfficialGeluQuant);
}
