#include "register/register.h"
#include <cstdio>
#include <cstdlib>
#include <regex>
namespace domi {
Status ParseParamPi05PerchannelLinear(const ge::Operator& source, ge::Operator& target) {
    const auto result = AutoMappingByOpFn(source, target);
    if (result != SUCCESS) return result;
    int64_t rows = 0, columns = 0;
    auto row_status = source.GetAttr("out_rows", rows);
    auto column_status = source.GetAttr("out_columns", columns);
    // CANN 8.5's ONNX ByOperator interface keeps ONNX AttributeProto values
    // in one JSON string named "attribute", not as native GE attributes.
    // Only the two flat integer attributes of this operator are accepted.
    std::string encoded;
    if (source.GetAttr("attribute", encoded) == ge::GRAPH_SUCCESS) {
        const std::regex object_pattern("\\{[^{}]*\\}");
        const std::regex name_pattern("\"name\"\\s*:\\s*\"(out_rows|out_columns)\"");
        const std::regex integer_pattern("\"i\"\\s*:\\s*([0-9]+)");
        const std::regex type_pattern("\"type\"\\s*:\\s*2\\b");
        for (auto it = std::sregex_iterator(encoded.begin(), encoded.end(), object_pattern);
             it != std::sregex_iterator(); ++it) {
            const std::string item = it->str();
            std::smatch name, value;
            if (!std::regex_search(item, name, name_pattern)) continue;
            if (!std::regex_search(item, value, integer_pattern)
                    || !std::regex_search(item, type_pattern)) return FAILED;
            try {
                const int64_t parsed = std::stoll(value[1].str());
                if (parsed <= 0) return FAILED;
                if (name[1] == "out_rows") {
                    rows = parsed;
                    row_status = ge::GRAPH_SUCCESS;
                } else {
                    columns = parsed;
                    column_status = ge::GRAPH_SUCCESS;
                }
            } catch (...) { return FAILED; }
        }
    }
    if (row_status == ge::GRAPH_SUCCESS) target.SetAttr("out_rows", rows);
    if (column_status == ge::GRAPH_SUCCESS) target.SetAttr("out_columns", columns);
    if (std::getenv("PI05_TRACE_PARSER")) {
        std::fprintf(stderr, "Pi05 valid-output source attrs: rows=%lld status=%u columns=%lld status=%u\n",
                     static_cast<long long>(rows), static_cast<unsigned>(row_status),
                     static_cast<long long>(columns), static_cast<unsigned>(column_status));
    }
    return SUCCESS;
}
REGISTER_CUSTOM_OP("Pi05PerchannelLinear")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::17::Pi05PerchannelLinear")
    .ParseParamsByOperatorFn(ParseParamPi05PerchannelLinear);

// Keep one ONNX domain/opset throughout the model. CANN rejects mixed domain
// versions when a graph also contains default-domain custom operators.
Status ParseParamPi05GeluTanh(const ge::Operator& source, ge::Operator& target) {
    const auto result = AutoMappingByOpFn(source, target);
    if (result != SUCCESS) return result;
    target.SetAttr("approximate", std::string("tanh"));
    return SUCCESS;
}
REGISTER_CUSTOM_OP("GeluV2")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::17::Pi05GeluTanh")
    .ParseParamsByOperatorFn(ParseParamPi05GeluTanh);
}
