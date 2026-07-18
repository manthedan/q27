// Shared ChatML-injection sanitizer (Security #7; review 2026-07-09 P1 #5).
// Extracted from tool_preamble.h so BOTH call paths — api_common.h's
// chatml_prompt and tokenizer.cpp's apply_chat_template — strip with one
// implementation. They were byte-identical duplicates (strip_ctrl here,
// strip_chatml in tokenizer.cpp): a future marker added to one and not the
// other would silently reopen the forgery hole on one path. Kept free of
// the nlohmann/json dependency so tokenizer.cpp need not pull it in.
#pragma once

#include <string>

namespace q27 {

// Strip ChatML role delimiters from untrusted content/roles so they can't forge
// prompt structure: the tokenizer matches <|im_start|>/<|im_end|> as control
// tokens anywhere, so a document or tool result containing them would otherwise
// become real role boundaries. Operator content that legitimately includes the
// literal markers loses them -- the safe tradeoff vs injection.
inline std::string strip_ctrl(std::string s) {
    for (const std::string& m : {std::string("<|im_start|>"), std::string("<|im_end|>")})
        for (size_t p; (p = s.find(m)) != std::string::npos;) s.erase(p, m.size());
    return s;
}

} // namespace q27
