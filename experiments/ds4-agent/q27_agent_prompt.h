// Qwen3.8 native transcript adapter. Generation, counting, save and restore
// must all render the same bytes; the closed prefix comes from stable_off.
#pragma once
#include "../../src/api_common.h"
#include <utility>

namespace q27::native_agent {

struct RenderedPrompt {
    std::string text;
    size_t stable_offset = 0;
    bool thinking_open = false;
};

inline RenderedPrompt render_qwen38_prompt(
        const std::vector<std::pair<std::string, std::string>>& chat,
        bool think, bool xml_dialect) {
    std::vector<q27::Msg> messages;
    for (const auto& [role, bytes] : chat) {
        q27::Msg message{role, bytes};
        // The native transcript stores the inline stream, whereas api_common
        // takes reasoning separately. Preserve it before history normalization.
        if (role == "assistant" && bytes.rfind("<think>", 0) == 0) {
            const size_t close = bytes.find("</think>", 7);
            message.reasoning = bytes.substr(7, close == std::string::npos ? close : close - 7);
            message.content = close == std::string::npos ? "" : bytes.substr(close + 8);
        } else if (role == "tool") {
            message.role = "user";
            message.content = "<tool_response>\n" + bytes + "\n</tool_response>";
        }
        messages.push_back(std::move(message));
    }
    q27::TemplateOpts opts;
    // This helper is only for Qwen3.8-family artifacts. Effort defaults to
    // xhigh even when the operator overrides the tool dialect to JSON, just
    // as in HTTP. Do not let another engine's process-global boot bit decide.
    const char* effort = std::getenv("Q27_REASONING_EFFORT");
    opts.effort = effort ? q27::effort_string_level(effort) : 2;
    RenderedPrompt result;
    result.text = q27::chatml_prompt(messages, q27::json::array(), think,
                                    &result.stable_offset, nullptr, {}, nullptr,
                                    &opts, xml_dialect);
    result.thinking_open = q27::scan_think_toggles(messages, think, opts.effort).thinking;
    return result;
}

} // namespace q27::native_agent
