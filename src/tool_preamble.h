// Tools preamble + ChatML control-token stripping, shared by the CUDA
// server (api_common.h chatml_prompt) and the Metal server's chat-template
// path. One definition so the two backends render byte-identical tool
// schemas — the Metal server previously never rendered tools at all
// (parity audit follow-up, 2026-07-17).
#pragma once

#include "../third_party/json.hpp"

#include <string>

namespace q27 {

// Strip ChatML role delimiters from untrusted content/roles so they can't forge
// prompt structure (Security #7): the tokenizer matches <|im_start|>/<|im_end|>
// as control tokens anywhere, so a document or tool result containing them would
// otherwise become real role boundaries. Operator content that legitimately
// includes the literal markers loses them -- the safe tradeoff vs injection.
inline std::string strip_ctrl(std::string s) {
    for (const std::string& m : {std::string("<|im_start|>"), std::string("<|im_end|>")})
        for (size_t p; (p = s.find(m)) != std::string::npos;) s.erase(p, m.size());
    return s;
}

// Tools preamble, verbatim structure from the chat template. `tools` entries
// must already be in {"type":"function","function":{...}} shape.
inline std::string tools_preamble(const nlohmann::json& tools) {
    std::string s = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    // tool declarations carry caller-controlled (and often third-party-
    // authored) description strings -- same forgery surface as message
    // content (review 2026-07-09 P1 #5)
    for (auto& t : tools) s += "\n" + strip_ctrl(t.dump());
    s += "\n</tools>\n\nFor each function call, return a JSON object with the function name "
         "and arguments inside <tool_call></tool_call> tags:\n<tool_call>\n{\"name\": "
         "<function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n<IMPORTANT>\n"
         "- Required parameters MUST be specified.\n- You may provide optional reasoning "
         "before the function call, but never after it.\n- If no function call is needed, "
         "answer normally and do not mention the tool interface.\n</IMPORTANT>";
    return s;
}

} // namespace q27
