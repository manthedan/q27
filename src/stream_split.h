// Streaming splitter for Qwopus output markers: <think>...</think> reasoning
// and <tool_call>...</tool_call> function calls, both emitted as plain-text
// tokens. Feeds token-by-token decoded text and routes segments into THINK /
// TEXT / TOOL channels, holding back any tail that could be a partial marker.
// Markers do not nest; tool_calls can appear only outside think blocks in
// well-formed output, but we tolerate them inside by scanning TEXT only.
// Adjacent calls (</tool_call><tool_call>, the multi-call batch shape) emit
// an empty {TEXT,""} boundary segment between them: consumers buffer one
// TOOL segment at a time and flush on any non-TOOL segment, so without the
// boundary two calls fold into one buffer, the combined parse fails, and a
// well-formed call followed by a malformed one silently loses the malformed
// raw (codex P2, 2026-07-17). Empty TEXT segments are no-ops for consumers
// that do not buffer tools.
#pragma once
#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace q27 {

struct StreamSplitter {
    enum Chan { TEXT, THINK, TOOL };
    Chan chan = TEXT;
    std::string hold;
    // Set when a TOOL closer returned us to TEXT and nothing has been emitted
    // since; a <tool_call> opener at position 0 then means ADJACENT calls and
    // gets an empty {TEXT,""} boundary segment (file-header comment).
    bool tool_boundary = false;

    static constexpr const char* T_OPEN = "<think>";
    static constexpr const char* T_CLOSE = "</think>";
    static constexpr const char* C_OPEN = "<tool_call>";
    static constexpr const char* C_CLOSE = "</tool_call>";

    std::vector<std::pair<Chan, std::string>> feed(const std::string& piece) {
        hold += piece;
        std::vector<std::pair<Chan, std::string>> out;
        for (;;) {
            if (chan == TEXT) {
                // whichever of <think> / <tool_call> comes first
                size_t pt = hold.find(T_OPEN), pc = hold.find(C_OPEN);
                if (pt != std::string::npos && (pc == std::string::npos || pt < pc)) {
                    if (pt > 0) out.push_back({TEXT, hold.substr(0, pt)});
                    tool_boundary = false; // think content flushes a pending tool
                    hold.erase(0, pt + strlen(T_OPEN));
                    chan = THINK;
                    continue;
                }
                if (pc != std::string::npos) {
                    if (pc == 0 && tool_boundary) out.push_back({TEXT, ""}); // adjacent-call boundary
                    tool_boundary = false;
                    if (pc > 0) out.push_back({TEXT, hold.substr(0, pc)});
                    hold.erase(0, pc + strlen(C_OPEN));
                    chan = TOOL;
                    continue;
                }
                // hold back the longest suffix that prefixes either opener
                size_t keep = tail_keep(T_OPEN);
                keep = std::max(keep, tail_keep(C_OPEN));
                if (emit_head(out, keep)) tool_boundary = false;
                break;
            }
            const char* closer = chan == THINK ? T_CLOSE : C_CLOSE;
            size_t p = hold.find(closer);
            if (p != std::string::npos) {
                if (p > 0) out.push_back({chan, hold.substr(0, p)});
                hold.erase(0, p + strlen(closer));
                tool_boundary = (chan == TOOL);
                chan = TEXT;
                continue;
            }
            emit_head(out, tail_keep(closer));
            break;
        }
        return out;
    }

    std::vector<std::pair<Chan, std::string>> flush() {
        std::vector<std::pair<Chan, std::string>> out;
        if (!hold.empty()) { out.push_back({chan, hold}); hold.clear(); }
        return out;
    }

  private:
    size_t tail_keep(const char* marker) const {
        size_t mlen = strlen(marker);
        size_t maxk = std::min(hold.size(), mlen - 1);
        for (size_t k = maxk; k > 0; k--)
            if (hold.compare(hold.size() - k, k, marker, k) == 0) return k;
        return 0;
    }
    bool emit_head(std::vector<std::pair<Chan, std::string>>& out, size_t keep) {
        if (hold.size() > keep) {
            out.push_back({chan, hold.substr(0, hold.size() - keep)});
            hold.erase(0, hold.size() - keep);
            return true;
        }
        return false;
    }
};

} // namespace q27
