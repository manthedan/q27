// Tools preamble + ChatML control-token stripping, shared by the CUDA
// server (api_common.h chatml_prompt) and the Metal server's chat-template
// path. One definition so the two backends render byte-identical tool
// schemas — the Metal server previously never rendered tools at all
// (parity audit follow-up, 2026-07-17).
#pragma once

#include "../third_party/json.hpp"
#include "strip_ctrl.h"

#include <string>

namespace q27 {

// strip_ctrl now lives in strip_ctrl.h (single definition shared with
// tokenizer.cpp's apply_chat_template; see that header's note).

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
