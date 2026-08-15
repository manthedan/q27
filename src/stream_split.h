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
    bool tool_boundary = false;
    bool native_xml = false;
    bool native_tool = false;

    static constexpr const char* T_OPEN = "<think>";
    static constexpr const char* T_CLOSE = "</think>";
    static constexpr const char* C_OPEN = "<tool_call>";
    static constexpr const char* C_CLOSE = "</tool_call>";
    static constexpr const char* N_OPEN = "<function=";
    static constexpr const char* N_CLOSE = "</function>";

    std::vector<std::pair<Chan, std::string>> feed(const std::string& piece) {
        hold += piece;
        std::vector<std::pair<Chan, std::string>> out;
        for (;;) {
            if (chan == TEXT) {
                size_t pt = hold.find(T_OPEN), pc = hold.find(C_OPEN),
                       sc = hold.find(C_CLOSE);
                size_t pn = native_xml && native_possible_
                                ? hold.find(N_OPEN)
                                : std::string::npos;
                size_t e = std::min(std::min(pt, pc), std::min(sc, pn));
                if (e != std::string::npos) {
                    if (e == pn) {
                        if (pn == 0 && tool_boundary) out.push_back({TEXT, ""});
                        if (!native_boundary(pn)) {
                            const std::string visible = hold.substr(0, pn + 1);
                            out.push_back({TEXT, visible});
                            note_text(visible);
                            hold.erase(0, pn + 1);
                            tool_boundary = false;
                            continue;
                        }
                        if (pn > 0) {
                            const std::string visible = hold.substr(0, pn);
                            out.push_back({TEXT, visible});
                            note_text(visible);
                        }
                        hold.erase(0, pn);
                        tool_boundary = false;
                        native_tool = true;
                        chan = TOOL;
                        continue;
                    }
                    if (e == pt) {
                        if (pt > 0) {
                            const std::string visible = hold.substr(0, pt);
                            out.push_back({TEXT, visible});
                            note_text(visible);
                        }
                        tool_boundary = false;
                        hold.erase(0, pt + strlen(T_OPEN));
                        chan = THINK;
                        continue;
                    }
                    if (e == pc) {
                        if (pc == 0 && tool_boundary) out.push_back({TEXT, ""});
                        tool_boundary = false;
                        if (pc > 0) {
                            const std::string visible = hold.substr(0, pc);
                            out.push_back({TEXT, visible});
                            note_text(visible);
                        }
                        hold.erase(0, pc + strlen(C_OPEN));
                        native_tool = false;
                        chan = TOOL;
                        continue;
                    }
                    if (sc > 0) {
                        const std::string visible = hold.substr(0, sc);
                        out.push_back({TEXT, visible});
                        note_text(visible);
                    }
                    hold.erase(0, sc + strlen(C_CLOSE));
                    continue;
                }
                size_t keep = tail_keep(T_OPEN);
                keep = std::max(keep, tail_keep(C_OPEN));
                keep = std::max(keep, tail_keep(C_CLOSE));
                if (native_xml && native_possible_)
                    keep = std::max(keep, tail_keep(N_OPEN));
                if (hold.size() > keep) {
                    const std::string visible = hold.substr(0, hold.size() - keep);
                    out.push_back({TEXT, visible});
                    note_text(visible);
                    hold.erase(0, hold.size() - keep);
                    tool_boundary = false;
                }
                break;
            }
            const char* closer = chan == THINK ? T_CLOSE
                                                : (native_tool ? N_CLOSE : C_CLOSE);
            size_t p = hold.find(closer);
            if (p != std::string::npos) {
                const size_t take = p + (native_tool ? strlen(closer) : 0);
                if (take > 0) out.push_back({chan, hold.substr(0, take)});
                hold.erase(0, p + strlen(closer));
                tool_boundary = (chan == TOOL);
                if (chan == TOOL) native_tool = false;
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
    bool native_possible_ = true;
    size_t native_line_spaces_ = 0;
    bool native_line_invalid_ = false;

    void note_text(const std::string& visible) {
        for (char c : visible) {
            if (!native_possible_) return;
            if (c == '\n') {
                native_line_spaces_ = 0;
                native_line_invalid_ = false;
            } else if (c == ' ') {
                native_line_spaces_++;
            } else if (c == '\t' || c == '\r') {
                native_line_invalid_ = true;
            } else {
                native_possible_ = false;
            }
        }
    }

    bool native_boundary(size_t pos) const {
        if (!native_possible_) return false;
        size_t spaces = native_line_spaces_;
        bool invalid = native_line_invalid_;
        for (size_t i = 0; i < pos; ++i) {
            const char c = hold[i];
            if (c == '\n') { spaces = 0; invalid = false; }
            else if (c == ' ') spaces++;
            else if (c == '\t' || c == '\r') invalid = true;
            else return false;
        }
        return !invalid && spaces <= 3;
    }

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
