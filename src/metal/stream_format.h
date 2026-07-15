// Model-free streaming helpers for the Metal server: UTF-8 boundary gating,
// stop-sequence holdback, and SSE event framing. Kept separate from the
// engine so the SSE shapes and stop logic have unit coverage that runs
// without the model artifact (build/test_metal_stream). The event shapes and
// the UTF-8 gate mirror the CUDA reference server (src/server.cu,
// src/api_common.h) so Metal and CUDA are wire-compatible.
#pragma once

#include "../../third_party/json.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace q27 {

// Incremental UTF-8 boundary gate for streaming token pieces. Copied verbatim
// from src/api_common.h (the CUDA server's Utf8Gate) so both servers split
// multi-byte characters identically. BPE token boundaries can split a
// multi-byte character; the raw piece is then invalid UTF-8 and nlohmann
// json::dump throws type_error.316 on it. feed() returns the longest valid
// prefix, holding an incomplete trailing sequence until its continuation
// bytes arrive; flush() ends the stream, turning a dangling partial into
// U+FFFD.
struct Utf8Gate {
    std::string pend;
    static int seq_len(unsigned char b) {
        if (b < 0x80) return 1;
        if ((b & 0xE0) == 0xC0) return 2;
        if ((b & 0xF0) == 0xE0) return 3;
        if ((b & 0xF8) == 0xF0) return 4;
        return -1; // continuation or invalid lead byte
    }
    std::string feed(const std::string& piece) {
        pend += piece;
        size_t n = pend.size(), i = n;
        int back = 0;
        while (i > 0 && back < 4) {
            unsigned char b = (unsigned char)pend[i - 1];
            if ((b & 0xC0) != 0x80) { i--; break; }
            i--;
            back++;
        }
        size_t cut = n;
        if (i < n) {
            int L = seq_len((unsigned char)pend[i]);
            if (L > 0 && i + (size_t)L > n) cut = i; // incomplete tail: hold back
        }
        std::string out = pend.substr(0, cut);
        pend.erase(0, cut);
        return out;
    }
    std::string flush() {
        std::string out = pend.empty() ? std::string() : std::string("\xEF\xBF\xBD");
        pend.clear();
        return out;
    }
};

// Stop-sequence holdback for streaming text. Implements the OpenAI `stop` /
// Anthropic `stop_sequences` contract for an incremental byte stream: text up
// to (but not including) the first stop occurrence is emitted, the stop
// sequence itself and everything after it is suppressed, and generation stops.
// Because pieces arrive incrementally, a trailing substring of the emitted
// text that could still grow into a stop sequence is held back until it is
// resolved (completed -> stop, or diverged -> released). This is the standard
// partial-prefix holdback; without it a stop sequence split across two tokens
// would leak its first half to the client before the match fires.
struct StopBuffer {
    std::vector<std::string> stops;
    std::string pend;
    int matched = -1; // index into `stops` once a full match fires, else -1

    explicit StopBuffer(std::vector<std::string> seqs = {}) : stops(std::move(seqs)) {
        // Drop empty stop strings: they would "match" at every position.
        stops.erase(std::remove(stops.begin(), stops.end(), std::string()), stops.end());
    }

    bool active() const { return !stops.empty(); }

    // Feed one decoded (already UTF-8-gated) piece. Returns text safe to emit
    // now. Sets `stopped` true (and `matched`) when a stop sequence completed;
    // when stopped, the returned text is the content strictly before the stop.
    std::string feed(const std::string& piece, bool& stopped) {
        stopped = false;
        if (stops.empty()) return piece; // fast path: no stop sequences
        pend += piece;
        // 1. Earliest full match anywhere in the buffer ends the stream.
        size_t best_pos = std::string::npos;
        int best_idx = -1;
        for (size_t s = 0; s < stops.size(); s++) {
            size_t p = pend.find(stops[s]);
            if (p != std::string::npos && (best_pos == std::string::npos || p < best_pos)) {
                best_pos = p;
                best_idx = (int)s;
            }
        }
        if (best_pos != std::string::npos) {
            stopped = true;
            matched = best_idx;
            std::string out = pend.substr(0, best_pos);
            pend.clear();
            return out;
        }
        // 2. No full match: hold back the longest suffix of `pend` that is a
        // proper prefix of some stop sequence (it may complete next feed).
        size_t hold = 0;
        for (const std::string& s : stops) {
            size_t kmax = std::min(pend.size(), s.size() - 1);
            for (size_t k = kmax; k >= 1; k--) {
                if (pend.compare(pend.size() - k, k, s, 0, k) == 0) {
                    if (k > hold) hold = k;
                    break;
                }
            }
        }
        size_t emit_len = pend.size() - hold;
        std::string out = pend.substr(0, emit_len);
        pend.erase(0, emit_len);
        return out;
    }

    // End of stream without a stop match: held-back text is real output.
    std::string flush() {
        std::string out = pend;
        pend.clear();
        return out;
    }
};

// ---- SSE framing (mirrors src/server.cu) ----

// Invalid-UTF-8-tolerant serialize: json::dump's strict default throws
// type_error.316, and an uncaught throw inside a streaming provider is
// std::terminate. The Utf8Gate keeps split characters intact; this is the
// backstop for everything else. Same as server.cu's file-scope jdump.
inline std::string sse_dump(const nlohmann::json& j) {
    return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

// OpenAI framing: "data: {json}\n\n".
inline std::string sse_data(const nlohmann::json& j) {
    return "data: " + sse_dump(j) + "\n\n";
}

// Anthropic / Responses framing: "event: NAME\ndata: {json}\n\n".
inline std::string sse_event(const std::string& name, const nlohmann::json& j) {
    return "event: " + name + "\ndata: " + sse_dump(j) + "\n\n";
}

// OpenAI stream terminator.
inline std::string sse_done() { return "data: [DONE]\n\n"; }

// One OpenAI streaming delta chunk. `chat` selects chat.completion.chunk
// (delta.content) vs text_completion (text); finish_reason is always null in
// stream chunks, exactly as the CUDA server emits them.
inline nlohmann::json openai_stream_chunk(bool chat, const std::string& id, const char* object,
                                          long created, const std::string& model,
                                          const std::string& piece) {
    using nlohmann::json;
    json choice = chat
        ? json{{"index", 0}, {"delta", {{"content", piece}}}, {"finish_reason", nullptr}}
        : json{{"index", 0}, {"text", piece}, {"finish_reason", nullptr}};
    return json{{"id", id}, {"object", object}, {"created", created},
                {"model", model}, {"choices", json::array({choice})}};
}

} // namespace q27
