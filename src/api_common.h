// Shared prompt construction + tool-call parsing for the q27 API endpoints.
// Qwopus (qwen35) tool protocol, from the GGUF chat template:
//   system preamble lists tools as JSON inside <tools>...</tools>
//   model emits  <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
//   results go back as user content wrapped in <tool_response>...</tool_response>
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "../third_party/json.hpp"
#include "tool_preamble.h"
#include "stream_split.h"

namespace q27 {
using json = nlohmann::json;

// nlohmann::json::value() does not use its default when a key exists with
// JSON null; get<T>() throws instead. API clients (notably pi) send
// max_tokens:null to mean "unspecified", so nullable scalar options need an
// explicit helper. Non-null type errors remain loud.
inline int64_t json_i64_or(const json& body, const char* key, int64_t dflt) {
    auto it = body.find(key);
    return it == body.end() || it->is_null() ? dflt : it->get<int64_t>();
}

// ---- tolerant request-field readers ---------------------------------------
// json::value() THROWS type_error.302 when a key is PRESENT but null or
// wrong-typed, and httplib turns any throw out of a handler into a 500 with an
// EXCEPTION_WHAT header (the routing() try/catch in third_party/httplib.h).
// "Present and null" is how a large share of OpenAI-compatible clients spell
// "unset" -- {"max_tokens": null, "temperature": null, "stream": null} comes
// straight out of LangChain/LiteLLM-class request builders. Answering those
// with a 500 is wrong twice over: wrong actor (the request was fine), and
// wrong status class (500 is retryable, so the client loops the same body).
// These read null / wrong-typed exactly like ABSENT, which is what the field
// means on the wire. Numbers are read through double so an integer field sent
// as 8192.0 still works, and clamped to int range so a nonsense magnitude
// can't overflow the int the callers assign into.
inline double jnum(const json& b, const char* key, double dflt) {
    if (!b.is_object()) return dflt;
    const auto it = b.find(key);
    return (it != b.end() && it->is_number()) ? it->get<double>() : dflt;
}
inline long jint(const json& b, const char* key, long dflt) {
    if (!b.is_object()) return dflt;
    const auto it = b.find(key);
    if (it == b.end() || !it->is_number()) return dflt;
    const double v = it->get<double>();
    if (v >= 2147483647.0) return 2147483647L;
    if (v <= -2147483648.0) return -2147483648L;
    return (long)v;
}
inline bool jbool(const json& b, const char* key, bool dflt) {
    if (!b.is_object()) return dflt;
    const auto it = b.find(key);
    return (it != b.end() && it->is_boolean()) ? it->get<bool>() : dflt;
}
inline std::string jstr(const json& b, const char* key,
                        const std::string& dflt = std::string()) {
    if (!b.is_object()) return dflt;
    const auto it = b.find(key);
    return (it != b.end() && it->is_string()) ? it->get<std::string>() : dflt;
}

// Incremental UTF-8 boundary gate for streaming token pieces. BPE token
// boundaries can split a multi-byte character (em dash E2 80 94 is a Qwopus
// favorite), the raw piece is then invalid UTF-8, and nlohmann json::dump
// throws type_error.316 on it -- which took q27-server down mid-generation
// under Claude Code (R0, 2026-07-04). feed() returns the longest valid
// prefix, holding back an incomplete trailing sequence until its
// continuation bytes arrive; flush() ends the stream, turning a dangling
// partial into U+FFFD. Invalid leads/continuations pass through -- the
// dump-time replace error handler is the backstop for those.
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

// R1b: FIFO ticket lock time-slicing the GPU across concurrent generations.
// Replaces the server's whole-generation mutex. The holder calls
// maybe_yield() at round/chunk boundaries: if anyone is queued it releases
// and re-acquires -- the fresh ticket lands at the TAIL, so contended
// requests round-robin at round granularity instead of head-of-line
// blocking for a whole generation. Solo path: one relaxed atomic load per
// call, no syscalls. contended() can miss a waiter arriving in the same
// instant (nwait is read unlocked); it is caught one round (~27ms) later.
struct GpuGate {
    void acquire() {
        std::unique_lock<std::mutex> lk(m);
        uint64_t t = next++;
        if (t != serving) {
            nwait.fetch_add(1, std::memory_order_relaxed);
            cv.wait(lk, [&] { return serving == t; });
            nwait.fetch_sub(1, std::memory_order_relaxed);
        }
    }
    void release() {
        { std::lock_guard<std::mutex> lk(m); serving++; }
        cv.notify_all();
    }
    int contended() const { return nwait.load(std::memory_order_relaxed); }
    // RAII whole-hold (the R1-equivalent region): exception-safe release,
    // same role the old lock_guard played at the server call sites.
    // Exemption to the drained-handover invariant: microsecond-scale async
    // copies queued AFTER the last yield point (tool-constraint clears,
    // n_max==0 tails) may still be in flight at ~Lease. All target
    // per-engine buffers and are stream-ordered ahead of that engine's next
    // work, so no cross-engine hazard exists; the GPU is "idle" at release
    // only up to those copies.
    struct Lease {
        explicit Lease(GpuGate& gg) : g(gg) { g.acquire(); }
        ~Lease() { g.release(); }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        GpuGate& g;
    };
    // Yield the GPU to queued waiters; true if a handover actually happened.
    // The new ticket is taken in the SAME critical section as the handover:
    // release();acquire() would let a descheduled yielder lose its queue
    // position to the next yielder (caught by the C1 self-test), breaking
    // strict rotation.
    bool maybe_yield() {
        if (!contended()) return false;
        std::unique_lock<std::mutex> lk(m);
        if (next - serving <= 1) return false; // raced: waiter already gone
        uint64_t t = next++;
        serving++;
        cv.notify_all();
        nwait.fetch_add(1, std::memory_order_relaxed);
        cv.wait(lk, [&] { return serving == t; });
        nwait.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }
private:
    std::mutex m;
    std::condition_variable cv;
    uint64_t next = 0, serving = 0;
    std::atomic<int> nwait{0};
};

struct Msg {
    std::string role;     // system | user | assistant
    std::string content;  // flattened text (think blocks already reconstructed)
};

// Tools preamble, verbatim structure from the chat template. `tools` entries
// must already be in {"type":"function","function":{...}} shape.
// Claude Code (<= 2.1.1xx era) prefixes its system prompt with
//   x-anthropic-billing-header: cc_version=...; cc_entrypoint=cli; cch=a5145;You are...
// The cch stamp is an integrity hint that CHANGES ON EVERY REQUEST, so the
// first bytes of the prompt mutate per turn -- which voids the P8 stable-prefix
// snapshot and P9 checkpoint routing for the entire conversation (measured
// under Claude Code: 126K-token full re-prefill, ~72s, on every turn). Pin the
// stamp to 'f's, mirroring llama.cpp's normalize_anthropic_billing_header
// (ggml-org/llama.cpp#21793), so both engines canonicalize to the same bytes.
// Only a header at the very start of the system text is touched, and the stamp
// is only looked for inside the short header segment.
//
// FORMAT DRIFT (measured 2026-07-24 against Claude Code 2.1.220): the `cch=`
// field is GONE, and the volatile stamp moved onto the version itself --
//   x-anthropic-billing-header: cc_version=2.1.220.473; cc_entrypoint=sdk-cli;
//                                                  ^^^ changes between conversations
// The old `cch=`-only normalizer returns early on that shape and does nothing,
// so the first ~15 tokens of every CC system prompt differ between sessions.
// That silently voids CROSS-SESSION reuse for all three tiers at once (P8
// snapshot, P9 ring, P16 disk) -- captured live: two sessions differing ONLY in
// their task text produced two distinct 21,504-token cache entries and never
// hit each other. Within one conversation the stamp is stable, which is why
// same-session warm turns kept working and this hid.
// Both stamp forms are now pinned: the legacy `cch=` value, and any 4th+
// dot-component of `cc_version=` (2.1.220 = the real version; .473 = the
// volatile tail). Pinning is safe -- this is prompt text an engine only needs
// to canonicalize, and llama.cpp does the same thing for the same reason.
inline void normalize_cc_billing_header(std::string& sys) {
    static const char* PFX = "x-anthropic-billing-header:";
    if (sys.rfind(PFX, 0) != 0) return;
    // legacy: cch=<stamp>;
    size_t cch = sys.find("cch=", 27);
    if (cch != std::string::npos && cch <= 160) {
        size_t v = cch + 4, end = sys.find(';', v);
        if (end != std::string::npos && end != v && end - v <= 16)
            for (size_t i = v; i < end; ++i) sys[i] = 'f';
    }
    // 2.1.220+: cc_version=<a.b.c>.<volatile>;  -- pin everything past the 3rd dot
    size_t cv = sys.find("cc_version=", 0);
    if (cv == std::string::npos || cv > 160) return;
    size_t v = cv + 11, end = sys.find(';', v);
    if (end == std::string::npos || end <= v || end - v > 64) return;
    int dots = 0;
    for (size_t i = v; i < end; ++i) {
        if (sys[i] == '.' && ++dots == 3) {
            for (size_t j = i + 1; j < end; ++j) sys[j] = 'f';
            return;
        }
    }
}

// strip_ctrl + tools_preamble moved to tool_preamble.h (shared with the
// Metal server so both backends render byte-identical tool schemas).

// Build the full ChatML prompt string. If tools are present they are merged
// into the (first) system message per the template's merged_system behavior.
// think=false appends the empty think block (enable_thinking=false
// convention); the tokenizer matches <think>/</think> as single added tokens.
// stable_off (P8): char offset where the trailing assistant-open begins.
// Everything before it re-renders identically next turn (snapshot-safe);
// everything after (assistant open + think prefill) is per-turn volatile.
// sys_off (P16b): char offset just past the system+tools block, or 0 when the
// request has neither. That block is the part MULTIPLE conversations share --
// Claude Code re-sends the same 20-25K-token system + tool definitions every
// session -- so it is the only boundary a cross-conversation cache entry can
// usefully be cut at. Like stable_off it abuts an <|im_start|>, so the same
// split-invariance argument applies.
inline std::string chatml_prompt(const std::vector<Msg>& msgs, const json& tools,
                                 bool think = true, size_t* stable_off = nullptr,
                                 size_t* sys_off = nullptr) {
    std::string p;
    size_t start = 0;
    std::string sys;
    if (!msgs.empty() && msgs[0].role == "system") { sys = strip_ctrl(msgs[0].content); start = 1; }
    // Over-refusal fix (2026-07-13, external review): under the no-think
    // serving default a bare request WITH NO SYSTEM PROMPT gives the model
    // zero context AND zero reasoning budget, so it falls to a defensive
    // refusal prior on borderline-legitimate requests (measured: a
    // signed-authorization pentest command). A minimal neutral default --
    // only when the client supplied none -- fully recovers compliance at
    // zero reasoning cost (real Claude Code always sends a system prompt, so
    // this never fires there). Q27_BARE=1 restores the no-default behavior.
    if (sys.empty() && !getenv("Q27_BARE")) sys = "You are a helpful assistant.";
    if (tools.is_array() && !tools.empty()) {
        p += "<|im_start|>system\n" + tools_preamble(tools);
        if (!sys.empty()) p += "\n\n" + sys;
        p += "<|im_end|>\n";
    } else if (!sys.empty()) {
        p += "<|im_start|>system\n" + sys + "<|im_end|>\n";
    }
    if (sys_off) *sys_off = p.size();  // 0 when no system block was emitted
    for (size_t i = start; i < msgs.size(); i++)
        p += "<|im_start|>" + strip_ctrl(msgs[i].role) + "\n" + strip_ctrl(msgs[i].content) +
             "<|im_end|>\n";
    if (stable_off) *stable_off = p.size();
    p += "<|im_start|>assistant\n";
    // think=false: the empty CLOSED block signals "reasoning done" so the model
    // answers directly. think=true: prefill the OPEN <think> tag so the model
    // enters a real thinking block -- this checkpoint reasons inline and never
    // opens <think> on its own, but given the opener it fills a trace and closes
    // with </think> before answering. Both sit in the volatile tail (past
    // stable_off), so P8 prefix reuse is untouched. Generation paths pre-seed the
    // StreamSplitter to THINK when think=true so the model's first generated
    // token (already inside the block) routes to the think channel -- mirrors the
    // FORCED tool_choice TOOL pre-seed.
    if (!think) p += "<think>\n\n</think>\n\n";
    else p += "<think>\n";
    return p;
}

// Experimental harness-prefix prewarming accepts only an initial request:
// zero or more merged system messages followed by exactly one live user
// message. Return the closed static prompt and optionally the complete
// ordinary generation prompt. Token-level callers still verify that encoding
// the former is an exact prefix of encoding the latter before any side effect.
inline std::string initial_harness_prefix(const std::vector<Msg>& messages,
                                          const json& tools, bool think,
                                          std::string* full_prompt = nullptr) {
    if (messages.empty() || messages.back().role != "user")
        throw std::runtime_error(
            "prewarm requires an initial request ending in one user message");
    std::vector<Msg> prefix_messages(messages.begin(), messages.end() - 1);
    for (const auto& message : prefix_messages)
        if (message.role != "system")
            throw std::runtime_error(
                "prewarm accepts initial requests only (system plus final user)");
    if (full_prompt) *full_prompt = chatml_prompt(messages, tools, think);
    size_t stable_bytes = 0;
    std::string prefix = chatml_prompt(prefix_messages, tools, think, &stable_bytes);
    prefix.resize(stable_bytes);
    return prefix;
}

inline std::string tool_call_text(const std::string& name, const json& args) {
    return "<tool_call>\n{\"name\": \"" + name + "\", \"arguments\": " + args.dump() +
           "}\n</tool_call>";
}

inline std::string tool_response_text(const std::string& out) {
    return "<tool_response>\n" + out + "\n</tool_response>";
}

// Per-request thinking resolution, GATED behind the server's --request-think
// flag (`allow_request`). `server_default` is the server profile's stance
// (!no_think_srv): no-think serving passes false, --think / the ref profile
// pass true.
//
// Without --request-think (the default), the request's thinking fields are
// IGNORED and the server default stands -- so a benchmark or client that sends
// enable_thinking:True (many do) can't silently flip a no-think server into
// thinking mode. Thinking is then purely a boot decision (--think).
//
// With --request-think, an explicit request field OVERRIDES the default in
// either direction (a no-think server serves thinking on request; a --think
// server suppresses it). Three client conventions, all honored -- a given
// client sends exactly one:
//   OpenAI / Qwen   : top-level  "enable_thinking": <bool>
//   llama.cpp / GLM : "chat_template_kwargs": {"enable_thinking": <bool>}
//   Anthropic       : "thinking": {"type": "enabled"|"disabled"}   (Claude
//                     Code's own thinking toggle; budget_tokens ignored)
// Malformed/wrong-typed fields are ignored rather than thrown (Security #1).
// Later checks win over earlier ones; the conventions never co-occur in practice.
inline bool resolve_think(const json& body, bool server_default, bool allow_request) {
    if (!allow_request) return server_default;
    bool think = server_default;
    if (body.contains("enable_thinking") && body["enable_thinking"].is_boolean())
        think = body["enable_thinking"].get<bool>();
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
        const auto& k = body["chat_template_kwargs"];
        if (k.contains("enable_thinking") && k["enable_thinking"].is_boolean())
            think = k["enable_thinking"].get<bool>();
    }
    if (body.contains("thinking") && body["thinking"].is_object()) {
        const auto& t = body["thinking"];
        if (t.contains("type") && t["type"].is_string()) {
            const std::string ty = t["type"].get<std::string>();
            if (ty == "enabled") think = true;
            else if (ty == "disabled") think = false;
        }
    }
    return think;
}

// Anthropic error envelope, exactly the real API's shape: the SDK inside
// Claude Code reads error.message from it, and CC's compact-vs-retry
// decision substring-matches that message.
inline std::string anthropic_error_json(const std::string& err_type,
                                        const std::string& message) {
    json e = {{"type", "error"},
              {"error", {{"type", err_type}, {"message", message}}}};
    return e.dump(-1, ' ', false, json::error_handler_t::replace);
}

// The real API's context-limit message, byte-for-byte format. CC (2.1.x)
// treats "prompt is too long" as compact-now; anything else (including our
// old end=refused empty 200) is retried verbatim and loops.
inline std::string ctx_limit_error_message(int n_prompt, int n_max_prompt) {
    return "prompt is too long: " + std::to_string(n_prompt) + " tokens > " +
           std::to_string(n_max_prompt) + " maximum";
}

// Anthropic tools -> qwen tools json for the system preamble (the
// /v1/messages request mapping; count_tokens must count the same bytes).
inline json anthropic_tools_json(const json& body) {
    json out = json::array();
    if (body.contains("tools") && body["tools"].is_array())
        for (auto& t : body["tools"]) {
            if (!t.is_object() || !t.contains("name")) continue;
            out.push_back({{"type", "function"},
                           {"function", {{"name", t["name"]},
                                         {"description", t.value("description", "")},
                                         {"parameters", t.contains("input_schema")
                                                            ? t["input_schema"]
                                                            : json::object()}}}});
        }
    return out;
}

// Anthropic messages -> Msg list (thinking + tool_use reconstructed to
// model markers, tool_result wrapped in <tool_response>)
inline std::vector<Msg> anthropic_msgs(const json& body) {
    std::vector<Msg> msgs;
    if (body.contains("system")) {
        std::string sys;
        if (body["system"].is_string()) sys = body["system"];
        else if (body["system"].is_array())
            for (auto& b : body["system"])
                // is_object() FIRST: value() on a non-object throws 306, which
                // httplib reports as a 500 -- a bare string in the array is a
                // client-side shape error, not a server fault (same guard
                // openai_msgs has always had on its content parts).
                if (b.is_object() && b.value("type", "") == "text") sys += b.value("text", "");
        if (!sys.empty()) {
            normalize_cc_billing_header(sys);
            msgs.push_back({"system", sys});
        }
    }
    if (!body.contains("messages")) return msgs;
    for (auto& m : body["messages"]) {
        // is_object() BEFORE the first value() call: value() on a non-object
        // (messages:["hi"]) throws 306 -- which the old ordering did one line
        // ahead of the guard meant to prevent exactly that. A non-object
        // element is skipped, matching openai_msgs.
        if (!m.is_object()) continue;
        std::string role = m.value("role", "user"), think, content;
        // guard: const operator[] on a missing key is an abort (json.hpp
        // assertion) -- a content-less message must not kill the server
        if (!m.contains("content")) { msgs.push_back({role, content}); continue; }
        if (m["content"].is_string()) content = m["content"];
        else if (m["content"].is_array())
            for (auto& part : m["content"]) {
                if (!part.is_object()) continue; // bare string in a content array
                std::string ty = part.value("type", "");
                if (ty == "text") content += part.value("text", "");
                else if (ty == "thinking") think += part.value("thinking", "");
                else if (ty == "tool_use") {
                    if (!content.empty() && content.back() != '\n') content += "\n";
                    content += tool_call_text(part.value("name", ""),
                                              part.contains("input") ? part["input"]
                                                                     : json::object());
                } else if (ty == "tool_result") {
                    std::string rc;
                    if (part.contains("content")) {
                        if (part["content"].is_string()) rc = part["content"];
                        else if (part["content"].is_array())
                            for (auto& b : part["content"])
                                if (b.is_object() && b.value("type", "") == "text")
                                    rc += b.value("text", "");
                    }
                    if (!content.empty() && content.back() != '\n') content += "\n";
                    content += tool_response_text(rc);
                }
            }
        if (role == "assistant" && !think.empty())
            content = "<think>\n" + think + "\n</think>\n" + content;
        msgs.push_back({role, content});
    }
    return msgs;
}

// OpenAI chat/completions tools -> the same {"type":"function","function":{...}}
// shape tools_preamble/chatml_prompt expect. Unlike anthropic_tools_json this
// is nearly a pass-through (the wire shape already matches); entries missing
// "type":"function" or a function.name are dropped rather than failing the
// whole request, mirroring the Responses bridge's tolerance of hosted tool
// types it doesn't model (review parity: a malformed ONE tool must not take
// down an otherwise-valid request).
inline json openai_tools_json(const json& body) {
    json out = json::array();
    if (body.contains("tools") && body["tools"].is_array())
        for (const auto& t : body["tools"]) {
            if (!t.is_object() || !t.contains("type") || !t["type"].is_string() ||
                t["type"] != "function") continue;
            if (!t.contains("function") || !t["function"].is_object()) continue;
            const json& fn = t["function"];
            if (!fn.contains("name") || !fn["name"].is_string() ||
                fn["name"].get_ref<const std::string&>().empty()) continue;
            if (fn.contains("description") && !fn["description"].is_string()) continue;
            if (fn.contains("parameters") && !fn["parameters"].is_object()) continue;
            out.push_back({{"type", "function"},
                           {"function", {{"name", fn["name"]},
                                         {"description", fn.contains("description")
                                                             ? fn["description"] : json("")},
                                         {"parameters", fn.contains("parameters")
                                                            ? fn["parameters"]
                                                            : json::object()}}}});
        }
    return out;
}

// OpenAI chat/completions messages -> Msg list, the /v1/chat/completions twin
// of anthropic_msgs. Two bridges the flat "content is a string" reading
// misses entirely (silently dropping the model's own tool use from history,
// which breaks any multi-turn agentic loop after the first call):
//   - assistant.tool_calls[] (OpenAI shape: function.arguments is a JSON
//     STRING) -> reconstructed <tool_call> marker(s), appended after any
//     sibling content text (order matches anthropic_msgs' text-then-tool_use
//     handling).
//   - role:"tool" (tool_call_id + content) -> folded into a <tool_response>-
//     wrapped USER turn, same as anthropic_msgs' tool_result bridge. The
//     call_id is intentionally not echoed into the prompt text: the chat
//     template's <tool_response> carries no id, and the fine-tune associates
//     a result with the immediately preceding call by POSITION.
//   - role:"developer" (the newer OpenAI system-role alias) -> "system",
//     matching the /v1/responses bridge.
inline std::vector<Msg> openai_msgs(const json& body) {
    std::vector<Msg> msgs;
    if (!body.contains("messages") || !body["messages"].is_array()) return msgs;
    for (auto& m : body["messages"]) {
        if (!m.is_object()) continue;
        std::string role = m.value("role", "user");
        if (role == "developer") role = "system";
        std::string content;
        if (m.contains("content")) {
            if (m["content"].is_string()) content = m["content"];
            else if (m["content"].is_array())
                for (auto& part : m["content"])
                    if (part.is_object() && part.value("type", "") == "text")
                        content += part.value("text", "");
        }
        if (role == "tool") {
            msgs.push_back({"user", tool_response_text(content)});
            continue;
        }
        if (role == "assistant" && m.contains("tool_calls") && m["tool_calls"].is_array()) {
            for (auto& tc : m["tool_calls"]) {
                if (!tc.is_object() || !tc.contains("function") || !tc["function"].is_object())
                    continue;
                const json& fn = tc["function"];
                std::string name = fn.value("name", std::string());
                json args = json::object();
                if (fn.contains("arguments")) {
                    if (fn["arguments"].is_string()) {
                        // OpenAI wire shape: a JSON-encoded string. Keep the
                        // raw string (rather than dropping the call) if it
                        // fails to parse -- same "never lose a turn" stance
                        // as parse_tool_call's double-encode tolerance.
                        try { args = json::parse(fn["arguments"].get<std::string>()); }
                        catch (...) { args = fn["arguments"]; }
                    } else args = fn["arguments"];
                }
                if (!content.empty() && content.back() != '\n') content += "\n";
                content += tool_call_text(name, args);
            }
        }
        msgs.push_back({role, content});
    }
    return msgs;
}

// tool_choice (OpenAI shape): "auto"/absent -> AUTO; "none" -> NONE;
// "required", a named function, or allowed_tools mode:"required" -> FORCED.
// allowed_tools mode:"auto" keeps AUTO while narrowing the eligible registry.
struct ToolChoice {
    enum Mode { AUTO, NONE, FORCED } mode = AUTO;
    std::string forced_name; // non-empty only for a named function choice
    std::vector<std::string> allowed_names; // empty = every declared tool
    bool invalid = false;
};
inline ToolChoice parse_tool_choice(const json& body) {
    ToolChoice tc;
    if (!body.contains("tool_choice")) return tc;
    const json& v = body["tool_choice"];
    if (v.is_string()) {
        if (v == "none") tc.mode = ToolChoice::NONE;
        else if (v == "required") tc.mode = ToolChoice::FORCED;
        // "auto" or any other/unknown string: default AUTO
        return tc;
    }
    if (v.is_null()) return tc;
    if (!v.is_object() || !v.contains("type") || !v["type"].is_string()) {
        tc.invalid = true;
        return tc;
    }
    if (v["type"] == "function") {
        if (!v.contains("function") || !v["function"].is_object() ||
            !v["function"].contains("name") || !v["function"]["name"].is_string() ||
            v["function"]["name"].get_ref<const std::string&>().empty()) {
            tc.invalid = true;
            return tc;
        }
        tc.mode = ToolChoice::FORCED;
        tc.forced_name = v["function"]["name"].get<std::string>();
        tc.allowed_names.push_back(tc.forced_name);
        return tc;
    }
    if (v["type"] == "allowed_tools") {
        if (!v.contains("allowed_tools") || !v["allowed_tools"].is_object()) {
            tc.invalid = true;
            return tc;
        }
        const json& allowed = v["allowed_tools"];
        if (!allowed.contains("mode") || !allowed["mode"].is_string() ||
            (allowed["mode"] != "auto" && allowed["mode"] != "required") ||
            !allowed.contains("tools") || !allowed["tools"].is_array()) {
            tc.invalid = true;
            return tc;
        }
        tc.mode = allowed["mode"] == "required" ? ToolChoice::FORCED : ToolChoice::AUTO;
        for (const auto& tool : allowed["tools"]) {
            if (!tool.is_object() || !tool.contains("type") || !tool["type"].is_string() ||
                tool["type"] != "function" || !tool.contains("function") ||
                !tool["function"].is_object() || !tool["function"].contains("name") ||
                !tool["function"]["name"].is_string() ||
                tool["function"]["name"].get_ref<const std::string&>().empty()) {
                tc.invalid = true;
                return tc;
            }
            const std::string name=tool["function"]["name"].get<std::string>();
            if (std::find(tc.allowed_names.begin(),tc.allowed_names.end(),name)==tc.allowed_names.end())
                tc.allowed_names.push_back(name);
        }
        if (tc.allowed_names.empty()) tc.invalid = true;
        return tc;
    }
    tc.invalid = true;
    return tc;
}

struct OpenAIToolSelection {
    json tools=json::array();
    std::vector<std::string> names;
};

// Normalize the declared registry once, then apply tool_choice's eligible
// subset. Both backends use this helper so prompt injection, grammar names,
// fallback parsing, and output validation all share the same registry.
inline OpenAIToolSelection select_openai_tools(const json& body,const ToolChoice& choice) {
    if (choice.invalid) throw std::runtime_error("invalid object-form tool_choice");
    OpenAIToolSelection selected;
    if (choice.mode == ToolChoice::NONE) return selected;
    selected.tools=openai_tools_json(body);
    for (const auto& tool : selected.tools)
        selected.names.push_back(tool["function"]["name"].get<std::string>());
    if (!choice.allowed_names.empty()) {
        const std::set<std::string> declared(selected.names.begin(),selected.names.end());
        for (const auto& name : choice.allowed_names)
            if (!declared.count(name))
                throw std::runtime_error("tool_choice names a function not present in tools");
        const std::set<std::string> allowed(choice.allowed_names.begin(),choice.allowed_names.end());
        json filtered=json::array();
        for (auto& tool : selected.tools)
            if (allowed.count(tool["function"]["name"].get<std::string>()))
                filtered.push_back(std::move(tool));
        selected.tools=std::move(filtered);
        selected.names=choice.allowed_names;
    }
    if (choice.mode == ToolChoice::FORCED && selected.names.empty())
        throw std::runtime_error("tool_choice requires at least one valid function tool");
    return selected;
}

// Parsed model tool call. `ok` false if the JSON was malformed (raw kept).
struct ToolCall {
    bool ok = false;
    std::string name;
    json arguments;
    std::string raw;
};

inline std::string escape_content_tags(const std::string& text);

// Q27_TOOL_STRICT=1: disable EVERY tolerant-parser rescue (the strict-parser
// A/B knob). Wrapped calls must be plain valid JSON (no <content>-tag rewrite,
// no double-encode unwrap); the wrapper-less bare-scan recovery is suppressed
// entirely. Suppressed rescues are logged ([q27-strict]) so a campaign can
// count what the tolerant chain WOULD have carried. Read once (server-lifetime
// knob, one leg per server run).
inline bool tool_strict() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("Q27_TOOL_STRICT"); v = e ? atoi(e) : 0; }
    return v == 1;
}

inline ToolCall parse_tool_call(const std::string& seg) {
    ToolCall tc;
    tc.raw = seg;
    if (tool_strict()) {
        // strict: the wrapped segment must parse as-is, with a JSON-object
        // arguments value. Anything else stays text (rescue suppressed).
        try {
            json j = json::parse(seg);
            tc.name = j.value("name", std::string());
            tc.arguments = j.contains("arguments") ? j["arguments"] : json::object();
            if (tc.arguments.is_string()) {
                fprintf(stderr, "[q27-strict] rejected double-encoded arguments (tool=%s)\n",
                        tc.name.c_str());
                tc.ok = false;
                return tc;
            }
            tc.ok = !tc.name.empty();
        } catch (...) {
            tc.ok = false;
            if (seg.find("<content>") != std::string::npos ||
                seg.find("</content>") != std::string::npos)
                fprintf(stderr, "[q27-strict] rejected <content>-tagged call (mode 3)\n");
        }
        return tc;
    }
    try {
        json j = json::parse(escape_content_tags(seg));
        tc.name = j.value("name", std::string());
        tc.arguments = j.contains("arguments") ? j["arguments"] : json::object();
        if (tc.arguments.is_string()) // some models double-encode
            tc.arguments = json::parse(tc.arguments.get<std::string>());
        tc.ok = !tc.name.empty();
    } catch (...) { tc.ok = false; }
    return tc;
}

inline std::string strip_ws2(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Minimal JSON structural context for mode-10 quote repair. A colon can
// terminate a quoted OBJECT KEY, but inside a string VALUE it is ordinary
// content (for example the raw shell fragment `echo "key": value`).
struct JsonQuoteContext {
    struct Frame { char kind; bool expect_key; };
    std::vector<Frame> stack;
    bool opening_string_is_key() const {
        return !stack.empty() && stack.back().kind=='{' && stack.back().expect_key;
    }
    void structural(char c) {
        if(c=='{') stack.push_back({'{',true});
        else if(c=='[') stack.push_back({'[',false});
        else if(c=='}') { if(!stack.empty() && stack.back().kind=='{') stack.pop_back(); }
        else if(c==']') { if(!stack.empty() && stack.back().kind=='[') stack.pop_back(); }
        else if(c==':' && !stack.empty() && stack.back().kind=='{') stack.back().expect_key=false;
        else if(c==',' && !stack.empty() && stack.back().kind=='{') stack.back().expect_key=true;
    }
};

inline bool quote_terminates_string(const std::string& s, size_t j, bool string_is_key) {
    for (size_t k = j + 1; k < s.size(); k++) {
        char c = s[k];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        return string_is_key ? c == ':' : (c == ',' || c == '}' || c == ']');
    }
    return true; // preserve the end-of-buffer truncation rule
}

// Shared JSON drift-sanitizer stepper (the "sanitizer state machine in 3
// places / 2 idioms" registered residue, 2026-07-19). The buffered recovery
// scan (parse_bare_tool_calls' scan_seg), the name-dropped batch scan
// (scan_namedropped), and the incremental ToolCallStreamer all walk a JSON
// byte stream with the SAME core loop: track in_str / escape / key-vs-value
// (JsonQuoteContext), escape raw in-string control chars (mode 5), and
// re-escape in-string quotes the lookahead proves literal (mode 10). A bug
// fixed in one copy and not the others silently diverges the drift behavior
// between the buffered and streaming paths (the exact class the strip_ctrl
// extraction closed for injection). This struct holds the state and one
// step(); per-call-site behavior is governed by flags.
//
// Usage: construct with the mode flags, then drive step() byte-by-byte with
// the full source string and current index (the lookahead needs the whole
// buffer, so index-based like quote_terminates_string). step() returns the
// sanitized output for this byte ("" when the byte was consumed into
// pending state) and advances depth_. Callers handle structure beyond
// string repair (depth-0 = end of segment, tag capture, etc.) via the
// exposed fields.
struct JsonSanitizerStepper {
    bool m10_lookahead = true;  // mode 10: re-escape literal in-string quotes
    bool m5_control   = true;   // mode 5: escape raw \n \r \t inside strings

    int  depth = 0;
    bool in_str = false, esc = false, string_is_key = false;
    bool m10_used = false;      // a quote was re-escaped this scan (gate flag)
    bool m5_used  = false;      // a control char was escaped this scan
    JsonQuoteContext json_ctx;

    // Process source[i]; append the sanitized bytes to `out`. Returns nothing;
    // depth/is_str/etc. are readable after each call. The caller advances i.
    void step(const std::string& src, size_t i, std::string& out) {
        const char c = src[i];
        if (esc) { esc = false; out += c; return; }
        if (in_str) {
            if (c == '\\') { esc = true; out += c; return; }
            if (c == '"') {
                // mode 10: a quote not followed by a JSON delimiter is
                // literal content, not a terminator -- re-escape it.
                if (!m10_lookahead || quote_terminates_string(src, i, string_is_key)) {
                    in_str = false; out += c;
                } else { out += "\\\""; m10_used = true; }
                return;
            }
            if (m5_control && c == '\n') { out += "\\n"; m5_used = true; return; }
            if (m5_control && c == '\r') { out += "\\r"; m5_used = true; return; }
            if (m5_control && c == '\t') { out += "\\t"; m5_used = true; return; }
            out += c;
            return;
        }
        out += c;
        if (c == '"') {
            string_is_key = json_ctx.opening_string_is_key();
            in_str = true;
        } else {
            json_ctx.structural(c);
            if (c == '{') depth++;
            else if (c == '}') depth--;
        }
    }
};

// Incremental tool-call argument streamer (pre-registered:
// docs/metal/plans/2026-07-17-incremental-tool-call-streaming.md). Shares the
// mode-5/10 drift semantics of JsonSanitizerStepper but deliberately keeps
// its OWN loop: it is STREAMING (no full buffer for quote_terminates_string
// lookahead — it holds a quote for a one-byte lookahead instead) and adds
// mode-3 <content>-tag capture, neither of which fits the stepper's
// buffer+index contract. Keep the escape/repair RULES in lockstep with the
// stepper; the mechanics differ by necessity.
// Feeds the
// splitter's TOOL-channel bytes as they decode; once the call head parses
// ({"name": "X", "arguments": {  — tolerating the mode-9 missing opening
// quote on the arguments key), the arguments object streams out SANITIZED
// (mode-5 control-char escaping, mode-10 in-string quote re-escaping, and
// mode-3 <content>-tag repair applied inline; a quote or </content> holds
// for exactly one non-whitespace byte of lookahead) as production-shape
// argument fragments. Heads that deviate
// (mode 6/7/8 shapes) or exceed the bound never stream: the raw body is
// preserved byte-exact for the buffered parse+recovery path. The trade is
// pre-registered: a streamed call whose body ends unbalanced reaches the
// client unbalanced (production semantics — the model's bytes are the
// model's bytes); end-of-call repair applies only to un-streamed calls.
struct ToolCallStreamer {
    enum State { HEAD, ARGS, DONE, INVALID_DONE, FALLBACK };
    State state = HEAD;
    std::string raw;      // entire body verbatim (fallback + logging)
    std::string name;     // valid once opened
    bool opened = false;  // head parsed; an opener chunk belongs on the wire

    bool active() const { return !raw.empty(); }
    bool invalid() const { return invalid_done_; }
    void reset() { *this = ToolCallStreamer(); }

    // Feed body bytes; returns the sanitized argument fragment to stream
    // (often empty). *opened_now fires on the feed that completes the head.
    std::string feed(const std::string& t, bool* opened_now) {
        if (opened_now) *opened_now = false;
        raw += t;
        if (state == DONE || state == INVALID_DONE) { add_trail(t); return ""; }
        if (state == FALLBACK) return "";
        std::string out;
        if (state == HEAD) {
            head_ += t;
            size_t consumed = 0;
            int m = match_head(head_, name, consumed);
            if (m == 0) {
                if (head_.size() > 512) state = FALLBACK;
                return "";
            }
            if (m < 0) { state = FALLBACK; return ""; }
            state = ARGS;
            opened = true;
            if (opened_now) *opened_now = true;
            std::string rest = head_.substr(consumed);
            head_.clear();
            scan(rest, out);
            sanitized_ += out;
            validate_done();
            return out;
        }
        scan(t, out);
        sanitized_ += out;
        validate_done();
        return out;
    }

    // Wrapper closed (or turn flushed). True = the args object completed
    // cleanly (DONE). Mid-ARGS: resolves a pending quote per the EOF rule
    // (terminator), appends the final bytes to *tail, returns false — the
    // handler closes the call as-is. HEAD/FALLBACK: false, nothing streamed.
    bool finalize(std::string* tail) {
        if (state == DONE) return true;
        if (state == ARGS) {
            if (!tag_.empty()) { *tail += tag_; tag_.clear(); } // partial tag: literal
            if (pend_q_ || pend_tag_) {          // EOF rule: terminator
                *tail += '"';
                *tail += pend_ws_;
                pend_q_ = pend_tag_ = false;
            }
        }
        return false;
    }

    // Bytes seen after the streamed call's arguments object closed. A
    // wrapper can pack more than one call; these are NOT framing — the
    // handler runs them through the bare-call recovery chain (review
    // 2026-07-17: the DONE-state byte drop silently lost every call after
    // the first, a regression vs the buffered path).
    const std::string& trail() const { return trail_; }

  private:
    std::string head_;
    std::string trail_;
    std::string sanitized_;
    bool invalid_done_ = false;
    int depth_ = 0;
    bool in_str_ = false, esc_ = false, string_is_key_ = false;
    JsonQuoteContext json_ctx_;
    bool pend_q_ = false;   // in-string quote awaiting one-byte lookahead
    bool pend_tag_ = false; // full </content> matched, awaiting the same lookahead
    bool outer_seen_ = false; // the call object's OWN closing } consumed from the trail
    std::string pend_ws_;   // whitespace held behind the pending quote/tag
    std::string tag_;       // partial <content>/</content> capture (mode 3)

    // Post-DONE bytes: the head consumed the call object's opening { without
    // counting it, so exactly one closing } after the args object is the
    // call's own framing — swallow it once; everything else is trail.
    void add_trail(const std::string& s) {
        size_t k = 0;
        if (!outer_seen_) {
            while (k < s.size() && is_ws(s[k])) k++;
            if (k == s.size()) return;      // only ws so far: keep waiting
            outer_seen_ = true;
            if (s[k] == '}') k++;
        }
        trail_ += s.substr(k);
    }

    static bool is_ws(char c) { return c==' '||c=='\t'||c=='\r'||c=='\n'; }

    void validate_done() {
        if(state!=DONE) return;
        try {
            json parsed=json::parse(sanitized_);
            if(!parsed.is_object()) throw std::runtime_error("tool arguments are not an object");
        } catch(...) {
            invalid_done_=true;
            // The opener is already on wire, so this is not an ordinary
            // buffered fallback. Preserve all later feeds as recovery trail
            // for packed calls that start in a subsequent token.
            state=INVALID_DONE;
        }
    }

    void scan(const std::string& in, std::string& out) {
        for (size_t i = 0; i < in.size() && state == ARGS; ) {
            const char c = in[i];
            if (!tag_.empty()) {
                // Mode-3 tag capture: held bytes that may still complete
                // <content> (value position — the tag opens the string the
                // model forgot) or </content> (in-string — maybe the close
                // the model wrote instead of a quote). The streamer applies
                // the same repair the buffered escape_content_tags path did;
                // a mismatch flushes the held bytes as ordinary input and
                // reprocesses c.
                const char* want = in_str_ ? "</content>" : "<content>";
                const size_t wl = in_str_ ? 10 : 9;
                if (c == want[tag_.size()]) {
                    tag_ += c; i++;
                    if (tag_.size() == wl) {
                        tag_.clear();
                        if (in_str_) pend_tag_ = true;       // shape 2: lookahead decides
                        else { out += '"'; in_str_ = true; string_is_key_ = false; } // shape 1: value opens as string
                    }
                    continue;
                }
                out += tag_;   // literal bytes (in-string they need no escaping)
                tag_.clear();
                continue;      // reprocess c
            }
            if (pend_q_ || pend_tag_) {
                if (is_ws(c)) { pend_ws_ += c; i++; continue; }
                const bool terminates = string_is_key_ ? c == ':'
                    : (c == ',' || c == '}' || c == ']');
                if (terminates) {
                    out += '"'; out += pend_ws_;   // terminator: ws is framing
                    in_str_ = false;
                } else {
                    // literal: ws is content, escaped in-string
                    out += pend_tag_ ? "</content>" : "\\\"";
                    for (char w : pend_ws_)
                        out += w=='\n' ? "\\n" : w=='\r' ? "\\r"
                             : w=='\t' ? "\\t" : std::string(1, w);
                }
                pend_ws_.clear();
                pend_q_ = pend_tag_ = false;
                // fall through: c processes normally in its resolved context
            }
            if (esc_) { esc_ = false; out += c; i++; continue; }
            if (in_str_) {
                if (c == '\\') { esc_ = true; out += c; i++; continue; }
                if (c == '"') { pend_q_ = true; i++; continue; }
                if (c == '<') { tag_ += c; i++; continue; }
                if (c == '\n') { out += "\\n"; i++; continue; }
                if (c == '\r') { out += "\\r"; i++; continue; }
                if (c == '\t') { out += "\\t"; i++; continue; }
                out += c; i++;
                continue;
            }
            if (c == '<') { tag_ += c; i++; continue; } // never valid JSON here
            out += c; i++;
            if (c == '"') {
                string_is_key_ = json_ctx_.opening_string_is_key();
                in_str_ = true;
            } else {
                json_ctx_.structural(c);
                if (c == '{') depth_++;
                else if (c == '}' && --depth_ == 0) {
                    state = DONE;
                    add_trail(in.substr(i)); // possible packed second call, not framing
                }
            }
        }
    }

    // 1 = matched (name_out set, consumed = index OF the args '{'),
    // 0 = undecided (need more bytes), -1 = not this shape (fallback).
    static int match_head(const std::string& b, std::string& name_out,
                          size_t& consumed) {
        size_t i = 0;
        auto skip = [&]() { while (i < b.size() && is_ws(b[i])) i++; return i < b.size(); };
        auto lit = [&](const char* s) -> int {
            for (size_t k = 0; s[k]; k++, i++) {
                if (i >= b.size()) return 0;
                if (b[i] != s[k]) return -1;
            }
            return 1;
        };
        int r;
        if (!skip()) return 0;
        if ((r = lit("{")) <= 0) return r;
        if (!skip()) return 0;
        if ((r = lit("\"name\"")) <= 0) return r;
        if (!skip()) return 0;
        if ((r = lit(":")) <= 0) return r;
        if (!skip()) return 0;
        if ((r = lit("\"")) <= 0) return r;
        std::string nm;
        for (;; i++) {
            if (i >= b.size()) return 0;
            if (b[i] == '\\') return -1;   // escaped names: buffered path
            if (b[i] == '"') { i++; break; }
            nm += b[i];
        }
        if (nm.empty()) return -1;
        if (!skip()) return 0;
        if ((r = lit(",")) <= 0) return r;
        if (!skip()) return 0;
        if (b[i] == '"') i++;              // mode-9: opening quote optional
        if ((r = lit("arguments\"")) <= 0) return r;
        if (!skip()) return 0;
        if ((r = lit(":")) <= 0) return r;
        if (!skip()) return 0;
        if (b[i] != '{') return -1;        // non-object args: buffered path
        name_out = nm;
        consumed = i;
        return 1;
    }
};

// Fallback for models that drop the <tool_call> wrapper and emit the call
// JSON as plain text (observed on long write calls under no-think greedy;
// llama.cpp's chat parser has the same class of tolerance). Scans for the
// first balanced {...} that parses as {"name":..., "arguments":...}. On
// success: prefix = text before the JSON, suffix = text after it (typically
// junk like "</file>" -- caller decides to drop it).
// Third observed drift mode: JSON framing with a raw-code value inside
// <content>...</content> tags (the fine-tune's SFT file format leaking into
// arguments). Rewrite `: <content>RAW</content>` spans into proper JSON
// strings so the call parses. Returns the input unchanged if no tag pair.
inline std::string escape_json_interior(const std::string& s) {
    std::string esc;
    for (char c : s) {
        switch (c) {
            case '"': esc += "\\\""; break;
            case '\\': esc += "\\\\"; break;
            case '\n': esc += "\\n"; break;
            case '\r': esc += "\\r"; break;
            case '\t': esc += "\\t"; break;
            default: esc += c;
        }
    }
    return esc;
}

// Rewrite a raw code/text value the fine-tune delimited with <content> tags into a
// proper JSON string so the call parses. Two observed shapes:
//   (1) "key": <content>RAW</content>   -- both angle tags in value position.
//   (2) "content": "RAW</content>       -- JSON-quote OPEN, tag CLOSE. The file-write
//       drift: RAW is a multi-line file body with unescaped quotes/newlines/braces
//       that break the JSON string; the model terminates it with a stray </content>
//       (SFT format leak) instead of a closing quote, then continues ", "file_path":...".
//       Verified 6/6 on the failing writes in the 2026-07-06 CRUSH A/B batch.
inline std::string escape_content_tags(const std::string& text) {
    size_t a = text.find("<content>");
    if (a != std::string::npos) {                    // shape 1
        size_t v = a + 9;
        size_t b = text.rfind("</content>");
        if (b == std::string::npos || b < v) return text;
        size_t k = text.find_last_not_of(" \t\r\n", a - 1);
        if (k == std::string::npos || text[k] != ':') return text;
        return text.substr(0, a) + "\"" + escape_json_interior(text.substr(v, b - v)) +
               "\"" + text.substr(b + 10);
    }
    // shape 2: "content": "RAW</content>  (no opening <content> tag)
    size_t close = text.find("</content>");
    if (close == std::string::npos) return text;
    size_t key = text.rfind("\"content\":", close); // nearest content key before the tag
    if (key == std::string::npos) return text;
    size_t q = text.find('"', key + 10);            // value-open quote after "content":
    if (q == std::string::npos || q > close) return text;
    return text.substr(0, q) + "\"" +
           escape_json_interior(text.substr(q + 1, close - q - 1)) + "\"" +
           text.substr(close + 10);
}

// Infer a tool name from an orphaned arguments object (drift mode 6: the model emits
// {"name": {ARGS}} with the name STRING value and the "arguments" key both absent from
// the bytes -- observed on the Claude Code tool schema). Match ARGS's keys against each
// tool's schema: an exact required-set match wins outright; else the highest
// (2*overlap - foreign) score, refusing on a tie (a wrong tool is worse than leaving it
// un-rescued). Needs the request's anthropic_tools_json; nullptr/no-match -> "".
inline std::string infer_tool_name(const json& tools, const json& args) {
    if (!tools.is_array() || !args.is_object()) return "";
    std::set<std::string> ak;
    for (auto it = args.begin(); it != args.end(); ++it) ak.insert(it.key());
    if (ak.empty()) return "";
    // Tie-break (2026-07-11, thunderdome T8 one-shot-quit root cause): the
    // modern CC registry has property-set near-twins (Bash and Monitor both
    // carry {command, description}), so orphaned Bash args scored a 4-4 tie
    // and the rescue refused. A tied candidate whose REQUIRED params are not
    // all present in ARGS could never validate as a call -- eliminate those;
    // a UNIQUE survivor wins. Both-satisfied ties still refuse (a wrong tool
    // remains worse than un-rescued).
    struct Cand { std::string name; bool req_ok; };
    std::vector<Cand> best;
    int best_score = 0;
    for (const auto& t : tools) {
        if (!t.contains("function")) continue;
        const json& fn = t["function"];
        std::string name = fn.value("name", std::string());
        if (name.empty()) continue;
        std::set<std::string> props, req;
        if (fn.contains("parameters") && fn["parameters"].is_object()) {
            const json& p = fn["parameters"];
            if (p.contains("properties") && p["properties"].is_object())
                for (auto it = p["properties"].begin(); it != p["properties"].end(); ++it) props.insert(it.key());
            if (p.contains("required") && p["required"].is_array())
                for (const auto& r : p["required"]) if (r.is_string()) req.insert(r.get<std::string>());
        }
        if (!req.empty() && req == ak) return name;   // exact required-set match: decisive
        int overlap = 0, foreign = 0;
        for (const auto& k : ak) (props.count(k) ? overlap : foreign)++;
        int score = 2 * overlap - foreign;
        if (overlap > 0 && score > 0) { // score-0 candidates never won before either
            bool rok = true;
            for (const auto& r : req)
                if (!ak.count(r)) { rok = false; break; }
            if (score > best_score) {
                best_score = score;
                best.clear();
                best.push_back({name, rok});
            } else if (score == best_score) {
                best.push_back({name, rok});
            }
        }
    }
    if (best.size() == 1) return best[0].name;
    std::string pick;
    for (const auto& c : best)
        if (c.req_ok) {
            if (!pick.empty()) return ""; // >1 required-satisfied: genuine ambiguity
            pick = c.name;
        }
    return pick;
}

// Drift mode 7 (2026-07-08, base Qwen3.6-27B-MTP on the CC schema, first turn,
// deterministic at greedy): the mode-6 orphaned args arrive nested one level
// deeper under a lone shell key -- {"name":\n{"function":{ARGS}}}. Inference on
// the raw object first (never disturbs a working mode-6 rescue); only when that
// fails, peel single-key object shells (bounded) and retry. On success the
// caller gets the INNER args -- the shell must not reach the tool.
// Drift mode 8 (2026-07-09, base Qwen3.6-27B-MTP, T10 ecommerce first turn,
// stable at greedy): the call object is well-formed but names the tool under
// a string-valued ALIAS key -- {"function": "Read", "arguments": {...}} --
// typically batched, behind a dangling {"name": prefix line, with a stray
// </tool_call> closer. Resolve the alias against the REGISTERED tool names
// only (a prose JSON example with an unregistered value must not become a
// call); args = the sibling arguments/parameters/input object, else the
// remainder minus the alias key.
inline bool resolve_aliased_call(const json& tools, const json& obj, std::string& name,
                                 json& args) {
    if (!tools.is_array() || !obj.is_object()) return false;
    static const char* aliases[] = {"function", "tool", "tool_name"};
    const char* akey = nullptr;
    std::string cand;
    for (const char* a : aliases)
        if (obj.contains(a) && obj[a].is_string()) { cand = obj[a]; akey = a; break; }
    if (!akey || cand.empty()) return false;
    bool registered = false;
    for (const auto& t : tools) {
        std::string nm;
        if (t.is_object() && t.contains("function") && t["function"].is_object())
            nm = t["function"].value("name", std::string());
        if (nm.empty() && t.is_object()) nm = t.value("name", std::string());
        if (nm == cand) { registered = true; break; }
    }
    if (!registered) return false;
    static const char* argkeys[] = {"arguments", "parameters", "input"};
    for (const char* k : argkeys)
        if (obj.contains(k) && obj[k].is_object()) {
            name = cand;
            args = obj[k];
            return true;
        }
    json rest = obj;
    rest.erase(akey);
    if (!rest.is_object()) return false;
    name = cand;
    args = std::move(rest);
    return true;
}

inline std::string infer_tool_name_unwrapped(const json& tools, json& args) {
    std::string nm = infer_tool_name(tools, args);
    if (!nm.empty()) return nm;
    { // mode 8: the extracted object itself is an alias-named call
        std::string an;
        json aa;
        if (resolve_aliased_call(tools, args, an, aa)) {
            args = std::move(aa);
            return an;
        }
    }
    static const char* shells[] = {"function", "arguments", "parameters", "input", "tool_call"};
    json u = args;
    for (int hop = 0; hop < 3 && u.is_object() && u.size() == 1; hop++) {
        bool peeled = false;
        for (const char* s : shells)
            if (u.contains(s) && u[s].is_object()) {
                json inner = u[s];
                u = std::move(inner);
                peeled = true;
                break;
            }
        if (!peeled) break;
        nm = infer_tool_name(tools, u);
        if (!nm.empty()) { args = std::move(u); return nm; }
    }
    return "";
}

// Drift mode 10 (2026-07-17, pi live traffic, T2 tier): unescaped double
// quotes inside a string value — shell quoting written verbatim into the
// command string (`... || echo "empty dir"`). JsonQuoteContext distinguishes
// key strings (only ':' may follow) from value strings (only , } ] may
// follow), so punctuation inside shell content is not mistaken for framing.

// Recover a name-dropped mode-6 BATCH: {"name":<ws>{ARGS}[ {"name":<ws>{ARGS}]... where
// each outer {"name": never closes (net +1 depth per unit) so the main balanced scan
// misses the whole run (observed on CC greedy: six {"name":\n{"file_path":...} Read calls).
// For each unit, extract the balanced ARGS object (control-chars sanitized) and infer the
// tool from its key signature. Appends recovered calls; sets *first to the earliest hit.
inline void scan_namedropped(const std::string& text, const json* tools,
                             std::vector<ToolCall>& out, size_t* first) {
    if (!tools) return;
    size_t p = 0;
    while ((p = text.find("{\"name\":", p)) != std::string::npos) {
        size_t q = p + 8;
        while (q < text.size() && (text[q]==' '||text[q]=='\t'||text[q]=='\r'||text[q]=='\n')) q++;
        if (q >= text.size() || text[q] != '{') { p += 8; continue; }
        JsonSanitizerStepper st;
        size_t e = std::string::npos;
        std::string san;
        for (size_t j = q; j < text.size(); j++) {
            st.step(text, j, san);
            if (!st.in_str && st.depth == 0 && text[j] == '}') { e = j; break; }
        }
        if (e == std::string::npos) break;   // truncated final unit
        try {
            json args = json::parse(san);
            if (args.is_object()) {
                std::string nm = infer_tool_name_unwrapped(*tools, args);
                if (!nm.empty()) {
                    ToolCall tc; tc.ok = true; tc.name = nm; tc.arguments = std::move(args);
                    if (*first == std::string::npos) *first = p;
                    out.push_back(std::move(tc));
                }
            }
        } catch (...) {}
        p = e + 1;
    }
}

// Scan for ALL recoverable bare calls. Balanced {"name":...,"arguments":...}
// objects anywhere in the text are collected (skipping unbalanced wrappers
// like the literal {"tool_call": opener, which nets +1 depth per blob and
// never closes); a trailing truncated {"name" candidate gets framing repair
// ONLY when allow_trunc_repair (the default) — repair is semantically an
// end-of-turn rescue, so callers scanning mid-turn segments (responses
// per-segment recovery) pass false: a segment boundary is not a truncation
// and inventing framing there false-positives prose fragments into calls
// (codex P2, 2026-07-17). prefix = text before the first recovered call.
// `tools` (optional) enables mode-6 name inference. allow_o10 (default on)
// gates the mode-10 dropped-opener re-parse recursion (2026-07-20, ported
// from upstream/master: a still-broken splice passes false so it cannot loop).
//
// The helpers below (minimal_escape_body / first_balanced_object /
// inside_fence / recover_raw_value_call) are upstream's drift mode-11 raw
// code-body rescue; the function body that uses them follows.
// (close open string, strip junk tags, close braces). prefix = text before
// the first recovered call. `tools` (optional) enables mode-6 name inference.
// Drift mode 11 (2026-07-19, issue #4): a tool call whose big string argument
// is a raw code body with unescaped inner quotes / newlines / braces
// ({"name":"Write","arguments":{"content":"<raw source, to end>"}}). Normal
// JSON parsing dies on the first inner `"`, and no local escape heuristic is
// safe (code has `",` `"}` `[]string{"a","b"}` everywhere). Recover
// positionally: the raw value runs to the end of the object, so its terminator
// is the last `"` before the object's closing braces. Extract that span
// literally, then parse the object with the big value blanked to pick up name
// + any other scalar args. Registered-tool + shell-parses gating keeps prose
// out. Handles content-last and scalar-args-before-content; a scalar AFTER the
// big value (rare ordering) over-captures -- accepted vs the current total
// failure (the UN-RESCUED session death). Only runs when nothing else parsed.
// Escape ONLY what's actually unescaped in a nearly-valid JSON string body:
// a valid \-escape is kept verbatim, a bare `"` or raw control char is
// escaped, a lone `\` is doubled. Unlike json(s).dump() this does NOT double-
// escape content the model already escaped correctly (\n \t \" ...), which is
// the mostly-escaped-with-sparse-errors case (issue #4, 2026-07-20:
// {"content":"...\"fmt\"...\"strings"..."} -- most quotes escaped, one not).
inline std::string minimal_escape_body(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (size_t i = 0; i < s.size(); i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\\') {
            if (i + 1 < s.size()) {
                char n = s[i + 1];
                if (n == '"' || n == '\\' || n == '/' || n == 'b' || n == 'f' || n == 'n' ||
                    n == 'r' || n == 't' || n == 'u') {
                    out += '\\';
                    out += n;
                    i++;
                    continue;
                }
            }
            out += "\\\\";
            continue;
        }
        if (c == '"') {
            out += "\\\"";
            continue;
        }
        if (c < 0x20) {
            switch (c) {
                case '\n': out += "\\n"; break;
                case '\t': out += "\\t"; break;
                case '\r': out += "\\r"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                default: {
                    char buf[8];
                    snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                }
            }
            continue;
        }
        out += (char)c;
    }
    return out;
}

// Return the first top-level balanced {...} (JSON string/escape aware), so
// trailing junk after the call object -- </tool_call>, prose, a second blob --
// doesn't make json::parse reject an otherwise-valid reconstruction.
inline std::string first_balanced_object(const std::string& s) {
    size_t start = s.find('{');
    if (start == std::string::npos) return "";
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t i = start; i < s.size(); i++) {
        char ch = s[i];
        if (esc) { esc = false; continue; }
        if (in_str) {
            if (ch == '\\') esc = true;
            else if (ch == '"') in_str = false;
            continue;
        }
        if (ch == '"') in_str = true;
        else if (ch == '{') depth++;
        else if (ch == '}') {
            if (--depth == 0) return s.substr(start, i - start + 1);
        }
    }
    return "";
}

// True if `pos` sits inside an open ```...``` fenced code block (an odd number
// of ``` precede it). A tool call the model INTENDS to emit is never markdown-
// fenced; a call shown as an example, or echoed from an injected file/page, is.
// We look ONLY before `pos`, so a write whose VALUE contains fences (its ``` are
// after the call's opener) is not mistaken for a fenced call.
inline bool inside_fence(const std::string& s, size_t pos) {
    size_t f = 0, count = 0;
    while ((f = s.find("```", f)) != std::string::npos && f < pos) {
        count++;
        f += 3;
    }
    return (count & 1) != 0;
}

inline bool recover_raw_value_call(const std::string& text, const json& tools,
                                   std::vector<ToolCall>& out) {
    size_t mo = text.rfind("{\"name\"");
    if (mo == std::string::npos) return false;
    if (inside_fence(text, mo)) return false; // fenced example, not a real call
    size_t colon = text.find(':', mo + 6);
    if (colon == std::string::npos) return false;
    size_t q1 = text.find('"', colon + 1);
    if (q1 == std::string::npos) return false;
    size_t q2 = text.find('"', q1 + 1);
    if (q2 == std::string::npos) return false;
    const std::string nm = text.substr(q1 + 1, q2 - q1 - 1);
    const json* fn = nullptr;
    for (const auto& t : tools)
        if (t.contains("function") && t["function"].value("name", std::string()) == nm) {
            fn = &t["function"];
            break;
        }
    if (!fn) return false;
    std::vector<std::string> strkeys;
    if (fn->contains("parameters") && (*fn)["parameters"].is_object()) {
        const json& pr = (*fn)["parameters"];
        if (pr.contains("properties") && pr["properties"].is_object())
            for (auto it = pr["properties"].begin(); it != pr["properties"].end(); ++it)
                if (it.value().is_object() &&
                    it.value().value("type", std::string()) == "string")
                    strkeys.push_back(it.key());
    }
    if (strkeys.empty()) return false;
    // For each string param, forward-scan candidate terminators of ITS value:
    // escape the span [opener+1, cand), keep the tail after cand literal, and
    // parse the reconstructed object. The FIRST candidate that parses is the
    // real terminator -- inner quotes leave the tail as un-parseable raw code,
    // and a scalar arg after the value forces the correct earlier terminator
    // (making that arg a valid sibling). Ordering-independent. The call must
    // be at the end of the model output, which is the UN-RESCUED reality.
    for (const auto& k : strkeys) {
        size_t kp = text.find("\"" + k + "\"", mo);
        if (kp == std::string::npos) continue;
        size_t kc = text.find(':', kp + k.size() + 2);
        if (kc == std::string::npos) continue;
        size_t opener = text.find('"', kc + 1);
        if (opener == std::string::npos) continue;
        for (size_t cand = text.find('"', opener + 1); cand != std::string::npos;
             cand = text.find('"', cand + 1)) {
            // Minimal-escape the value body (don't double-escape already-valid
            // \-escapes) and trim trailing junk by taking the first balanced
            // object, so both fully-raw and mostly-escaped content recover.
            const std::string body = minimal_escape_body(text.substr(opener + 1, cand - opener - 1));
            const std::string recon = first_balanced_object(
                text.substr(mo, opener - mo) + "\"" + body + "\"" + text.substr(cand + 1));
            if (recon.empty()) continue;
            json obj;
            try { obj = json::parse(recon); } catch (...) { continue; }
            if (!obj.is_object() || obj.value("name", std::string()) != nm) continue;
            json args = obj.contains("arguments") && obj["arguments"].is_object()
                            ? obj["arguments"]
                            : json::object();
            ToolCall tc;
            tc.ok = true;
            tc.name = nm;
            tc.arguments = std::move(args);
            fprintf(stderr, "[drift] mode-11 raw-value rescue: %s.%s (%zu bytes)\n", nm.c_str(),
                    k.c_str(), cand - opener - 1);
            out.push_back(std::move(tc));
            return true;
        }
    }
    return false;
}

inline std::vector<ToolCall> parse_bare_tool_calls(const std::string& text_in,
                                                   std::string* prefix,
                                                   const json* tools = nullptr,
                                                   bool allow_trunc_repair = true,
                                                   bool allow_o10 = true) {
    std::vector<ToolCall> out;
    if (tool_strict()) {
        // strict-parser A/B: the wrapper-less recovery chain (drift modes 1-6)
        // is OFF. Log when the text plausibly contained an intended call so the
        // campaign can count suppressed rescues against the tolerant leg.
        if (prefix) *prefix = "";
        if (text_in.find("{\"name\"") != std::string::npos ||
            text_in.find("{\"tool_call\"") != std::string::npos ||
            text_in.find("</content>") != std::string::npos)
            fprintf(stderr, "[q27-strict] SUPPRESSED bare-call rescue: %.200s\n",
                    text_in.c_str());
        return out;
    }
    bool m2 = false, m5 = false, m6 = false, m8 = false; // drift-mode flags (exit-gate catalog)
    bool m10 = false;                                    // mode 10: in-string quote re-escaped
    // drift mode 9 (2026-07-11, codex-harnessed traffic): the model drops the
    // OPENING quote of the "arguments" key ({"name":"X",\narguments":{...}}).
    // "arguments" is the tool-call schema key, so quoting a bare `arguments":`
    // is unambiguous inside these segments. Applied before segmentation so
    // both the whole-object and truncated-tail parse paths see valid JSON.
    auto fix_arg_quote = [](std::string s) {
        size_t p = 0;
        while ((p = s.find("arguments\"", p)) != std::string::npos) {
            if (p == 0 || s[p - 1] != '"') { s.insert(p, "\""); p += 11; }
            else p += 10;
        }
        return s;
    };
    // drift mode 12 (2026-07-19, club-3090 cli-40 agent): the model drops the
    // QUOTES around the tool-NAME value -- {"name": bash, "arguments": {...}}.
    // Invalid JSON, so the whole call went UN-RESCUED and the agent's turn
    // stopped (turnsUsed=0). The arguments are valid JSON, so once the bare
    // name is quoted the object parses on the normal path. SAFE because we only
    // quote a bareword that EXACTLY matches a registered tool name -- prose,
    // non-tool JSON, null/numbers, and unknown names are left untouched.
    auto fix_unquoted_name = [](std::string s, const json* tools) {
        if (!tools || !tools->is_array()) return s;
        size_t p = 0;
        while ((p = s.find("\"name\":", p)) != std::string::npos) {
            size_t c = p + 7;
            while (c < s.size() && (s[c] == ' ' || s[c] == '\t' || s[c] == '\n' || s[c] == '\r'))
                c++;
            if (c < s.size() && s[c] != '"' &&
                (isalpha((unsigned char)s[c]) || s[c] == '_')) {
                size_t e = c;
                while (e < s.size() && (isalnum((unsigned char)s[e]) || s[e] == '_' ||
                                        s[e] == '-' || s[e] == '.'))
                    e++;
                const std::string word = s.substr(c, e - c);
                bool match = false;
                for (const auto& t : *tools)
                    if (t.contains("function") &&
                        t["function"].value("name", std::string()) == word) {
                        match = true;
                        break;
                    }
                if (match) {
                    // Add the closing quote only if one isn't already there:
                    // {"name": bash,   -> both quotes; {"name": read"  (dropped
                    // OPENING quote, stray close) -> opening only (thunderdome).
                    if (!(e < s.size() && s[e] == '"')) s.insert(e, "\"");
                    s.insert(c, "\"");
                    p = e + 2;
                    continue;
                }
            }
            p = c;
        }
        return s;
    };
    const std::string text =
        fix_unquoted_name(fix_arg_quote(escape_content_tags(text_in)), tools);
    const bool m3 = (text != text_in);              // mode 3: <content>-tagged value rewritten
    const bool m4 = text.find("{\"tool_call\":") != std::string::npos; // mode 4: JSON-keyed opener
    size_t first = std::string::npos;
    size_t i = text.find('{');
    while (i != std::string::npos) {
        // Segment scan, two-pass on the mode-10 lookahead: the lookahead's
        // premise — valid JSON framing follows every real closing quote —
        // only holds when the segment reaches balance. On an unbalanced-to-
        // EOF segment the re-escape can swallow trailing prose (or a whole
        // second call) into an open string that the truncation repair then
        // "successfully" closes into one merged garbage command (review
        // 2026-07-17). Unbalanced-with-m10 segments rescan with the
        // unconditional terminator and take the pre-mode-10 repair path.
        // A {...} sitting inside a ```fenced``` block is a displayed example /
        // echoed injection, not a call the model is making -- don't recover it
        // (upstream inside_fence guard, 2026-07-19).
        if (inside_fence(text, i)) {
            i = text.find('{', i + 1);
            continue;
        }
        bool m5_here = false, m10_here = false;
        auto scan_seg = [&](bool m10_look, std::string& san) -> size_t {
            san.clear();
            m5_here = m10_here = false;
            JsonSanitizerStepper st;
            st.m10_lookahead = m10_look;
            for (size_t j = i; j < text.size(); j++) {
                st.step(text, j, san);
                m5_here = m5_here || st.m5_used;
                m10_here = m10_here || st.m10_used;
                if (!st.in_str && st.depth == 0 && text[j] == '}') return j;
            }
            return std::string::npos;
        };
        std::string san;  // segment with raw in-string control chars escaped
        size_t end = scan_seg(true, san);
        if (end == std::string::npos && m10_here) end = scan_seg(false, san);
        if (m5_here) m5 = true;
        if (m10_here) m10 = true;
        if (end == std::string::npos) {
            // unbalanced to EOF: repair only a {"name" candidate (truncated
            // final call), and only when the caller guarantees the buffer is
            // end-of-turn; otherwise keep scanning inner objects. `san` holds
            // the sanitized remainder (scan ran to EOF).
            if (!allow_trunc_repair || san.rfind("{\"name\"", 0) != 0) { i = text.find('{', i + 1); continue; }
            std::string r = san;
            while (true) {
                size_t e2 = r.find_last_not_of(" \t\r\n");
                if (e2 == std::string::npos) break;
                r.resize(e2 + 1);
                if (r.size() >= 7 && r.compare(r.size() - 7, 7, "</file>") == 0)
                    r.resize(r.size() - 7);
                else break;
            }
            int d2 = 0;
            bool s2 = false, e2f = false;
            for (char ch : r) {
                if (e2f) { e2f = false; continue; }
                if (s2) {
                    if (ch == '\\') e2f = true;
                    else if (ch == '"') s2 = false;
                    continue;
                }
                if (ch == '"') s2 = true;
                else if (ch == '{') d2++;
                else if (ch == '}') d2--;
            }
            // Drift mode 13 (2026-07-24): the truncation can land INSIDE an
            // escape sequence. A trailing "\\" swallows the quote we are about
            // to append -- the string stays open, the object never parses, and
            // the whole call goes UN-RESCUED. Observed live: a wrapper-less
            // Write whose markdown `content` was cut mid-`\\"` while writing an
            // escaped JSON example. Same for a partial \\uXXXX. Trim back to the
            // last safe byte before closing.
            if (s2) {
                if (e2f) {
                    r.pop_back();  // dangling backslash
                } else {
                    const size_t u = r.find_last_of('\\');
                    // a COMPLETE \uXXXX is 6 bytes; anything shorter at the very
                    // end is a partial one. Leading-backslash check avoids
                    // trimming an escaped backslash that merely precedes a 'u'.
                    if (u != std::string::npos && u + 1 < r.size() && r[u + 1] == 'u' &&
                        r.size() - u < 6 && !(u > 0 && r[u - 1] == '\\'))
                        r.resize(u);
                }
                r += '"';
            }
            for (; d2 > 0; d2--) r += '}';
            bool shaped = false;
            try {
                json j = json::parse(r);
                shaped = j.is_object() && j.contains("name") && j.contains("arguments");
            } catch (...) {}
            bool recovered_here = false;
            if (shaped) {
                ToolCall tc = parse_tool_call(r);
                if (tc.ok) {
                    if (first == std::string::npos) first = i;
                    out.push_back(tc);
                    m2 = true;   // mode 2: truncated/unterminated JSON repaired
                    recovered_here = true;
                }
            }
            if (recovered_here) break;   // genuine truncated FINAL call consumed the rest
            // else: a dangling/failed {"name": opener -- e.g. the mode-6 HYBRID where the
            // model prepends a bare {"name": before a batch of VALID calls ("read all files
            // in parallel"). Don't discard the rest: advance past the opener and keep scanning
            // so the real calls after it recover normally.
            i = text.find('{', i + 1);
            continue;
        }
        const std::string& seg = san;
        bool shaped = false, m6cand = false, m8cand = false;
        json j6, j8;
        try {
            json j = json::parse(seg);
            shaped = j.is_object() && j.contains("name") && j.contains("arguments");
            if (!shaped && j.is_object() && j.contains("name") &&
                j["name"].is_object() && !j.contains("arguments")) {
                m6cand = true; j6 = std::move(j);
            } else if (!shaped && j.is_object()) {
                m8cand = true; j8 = std::move(j);
            }
        } catch (...) {}
        if (shaped) {
            ToolCall tc = parse_tool_call(seg);
            if (tc.ok) {
                if (first == std::string::npos) first = i;
                out.push_back(tc);
                i = text.find('{', end + 1);
                continue;
            }
        } else if (m6cand && tools) {
            // mode 6: {"name": {ARGS}} -- name string + "arguments" key both dropped.
            // Infer the tool from the orphaned args' key signature (mode 7:
            // unwrap a lone shell key first when the raw keys match nothing).
            json m6args = j6["name"];
            std::string nm = infer_tool_name_unwrapped(*tools, m6args);
            if (!nm.empty()) {
                ToolCall tc; tc.ok = true; tc.name = nm; tc.arguments = std::move(m6args);
                if (first == std::string::npos) first = i;
                out.push_back(std::move(tc));
                m6 = true;
                i = text.find('{', end + 1);
                continue;
            }
        } else if (m8cand && tools) {
            // mode 10 tail (2026-07-18): flat name+args -- a STRING "name"
            // matching a registered tool, with the arguments as SIBLING keys
            // instead of nested under "arguments" ({"name":"Read","file_path":
            // ...}). This is what the mode-10 opener-splice produces, and
            // the model also emits it wrapper-less. Validate the name against
            // the registry so prose JSON with a "name" field can't match.
            if (j8.contains("name") && j8["name"].is_string()) {
                const std::string cand = j8["name"].get<std::string>();
                bool known = false;
                for (const auto& t : *tools)
                    if (t.contains("function") &&
                        t["function"].value("name", std::string()) == cand) { known = true; break; }
                if (known && !cand.empty()) {
                    json args = json::object();
                    for (auto it = j8.begin(); it != j8.end(); ++it)
                        if (it.key() != "name") args[it.key()] = it.value();
                    ToolCall tc; tc.ok = true; tc.name = cand; tc.arguments = std::move(args);
                    if (first == std::string::npos) first = i;
                    out.push_back(std::move(tc));
                    m8 = true;
                    i = text.find('{', end + 1);
                    continue;
                }
            }
            // mode 8: alias-named call object ({"function": "Read", ...});
            // registered-name validation inside keeps prose JSON out
            std::string an;
            json aa;
            if (resolve_aliased_call(*tools, j8, an, aa)) {
                ToolCall tc; tc.ok = true; tc.name = std::move(an); tc.arguments = std::move(aa);
                if (first == std::string::npos) first = i;
                out.push_back(std::move(tc));
                m8 = true;
                i = text.find('{', end + 1);
                continue;
            }
        }
        i = text.find('{', i + 1);
    }
    // Fallback: name-dropped mode-6 BATCH (the main scan can't segment it). Only when the
    // standard scan found nothing, so normal calls are never double-counted.
    if (out.empty() && tools && text.find("{\"name\":") != std::string::npos) {
        scan_namedropped(text, tools, out, &first);
        if (!out.empty()) m6 = true;
    }
    if (prefix) {
        std::string p = first == std::string::npos ? "" : strip_ws2(text.substr(0, first));
        // drop a dangling {"tool_call": opener fragment (mode-4 wrapper junk)
        if (p.size() >= 13 && p.compare(p.size() - 13, 13, "{\"tool_call\":") == 0)
            p = strip_ws2(p.substr(0, p.size() - 13));
        // drop a dangling {"name": opener fragment (mode-6/8 hybrid prefix line)
        if (p.size() >= 8 && p.compare(p.size() - 8, 8, "{\"name\":") == 0)
            p = strip_ws2(p.substr(0, p.size() - 8));
        *prefix = p;
    }
    // Drift mode 10 (2026-07-18, SWE-bench flask-5014 first-tool-call
    // rescue miss): the model drops the ENTIRE `{"name": "` opener, emitting
    // `NAME", "key": val ...}` with no brace at all -- so the {-scanner above
    // finds no candidate. When nothing else rescued and a KNOWN tool name is
    // followed by the `", "` argument-separator signature (and is not already
    // properly quoted), splice the opener back and re-parse ONCE (allow_o10
    // false in the recursion so a still-broken splice cannot loop). This is
    // the deterministic early-quit the n=3 seal caught: a missed first call
    // ends the agent turn with a leaked-JSON text response.
    if (out.empty() && allow_o10 && tools && tools->is_array()) {
        for (const auto& t : *tools) {
            std::string nm = t.contains("function")
                                 ? t["function"].value("name", std::string())
                                 : std::string();
            if (nm.empty()) continue;
            const std::string sig = nm + "\", \"";
            size_t p = text.find(sig);
            if (p == std::string::npos) continue;
            if (p > 0 && text[p - 1] == '"') continue; // already `"name": "NAME"`
            std::string synth = text.substr(0, p) + "{\"name\": \"" + text.substr(p);
            if (synth.find('}') == std::string::npos) synth += "}";
            std::string pre2;
            auto rec = parse_bare_tool_calls(synth, &pre2, tools,
                                             allow_trunc_repair, /*allow_o10=*/false);
            if (!rec.empty()) {
                out = std::move(rec);
                if (prefix) *prefix = pre2;
                fprintf(stderr, "[drift] mode-10 dropped-opener rescue: %s\n", nm.c_str());
                break;
            }
        }
    }
    // mode 11: raw code-body string value (unescaped inner quotes/newlines).
    // The positional recovery restores the FULL string value, so it is the
    // right answer whenever the value genuinely contains unescaped inner
    // quotes. Our stepper's mode-10 lookahead can reach a *parseable but
    // wrong* segment on exactly that input (it re-escapes the first inner
    // quote, truncating the value and swallowing any sibling scalar -- the
    // "scalar-after-content" case, wrong-ok). Upstream's plain in-string scan
    // cannot parse that input at all, so it always falls through to mode 11.
    // To match upstream's recovery on this overlap: prefer mode 11's positional
    // result over the scan's mode-10-re-escaped candidate; if mode 11 also
    // declines, keep the scan's (it is still the best available read of a
    // genuinely-mode-10 string). Runs as last resort too. Multi-call: mode 11
    // only returns the LAST `{"name"` call, so preserve any earlier calls the
    // scan recovered normally (only the trailing candidate is m10-suspect) --
    // replacing the whole vector would silently drop them (codex P2).
    if (tools && tools->is_array() && (out.empty() || (m10 && !out.empty()))) {
        std::vector<ToolCall> raw;
        if (recover_raw_value_call(text, *tools, raw) && !raw.empty()) {
            if (m10 && out.size() > 1) {
                out.back() = std::move(raw.back()); // replace only the m10-wrong tail
            } else {
                out = std::move(raw);
            }
            if (prefix) {
                size_t mo = text.rfind("{\"name\"");
                *prefix = mo != std::string::npos ? strip_ws2(text.substr(0, mo))
                                                  : std::string();
            }
        }
    }
    // Drift catalog (exit gate, docs/sampling-exit-gate.md): tag which tool-format
    // drift mode(s) the fallback chain rescued, or flag an intended call it could
    // NOT recover. Log-only; the parse result is unchanged.
    if (!out.empty()) {
        char modes[10]; int mi = 0;
        modes[mi++] = '1';               // baseline: dropped-<tool_call>-wrapper recovery
        if (m2) modes[mi++] = '2';
        if (m3) modes[mi++] = '3';
        if (m4) modes[mi++] = '4';
        if (m5) modes[mi++] = '5';
        if (m6) modes[mi++] = '6';
        if (m8) modes[mi++] = '8';
        if (m10) modes[mi++] = 'a';      // mode 10 ('a': single-char catalog)
        modes[mi] = 0;
        fprintf(stderr, "[drift] recovered=%zu modes=%s\n", out.size(), modes);
    } else if (text_in.find("{\"name\"") != std::string::npos ||
               text_in.find("{\"tool_call\"") != std::string::npos) {
        // ntools distinguishes plumbing (-1/0) from a schema/inference miss (>0 =
        // mode-6 args didn't confidently match any tool). Longer window so the call
        // (not just a long preamble) is visible for post-hoc arg-shape diagnosis.
        fprintf(stderr, "[drift] UN-RESCUED (ntools=%d) intended tool call: %.400s\n",
                tools ? (int)tools->size() : -1, text_in.c_str());
        // Corpus capture: Q27_DRIFT_CORPUS=<file> appends the FULL untruncated
        // miss (the stderr line caps at 400 chars, which is why issue #4's
        // payload wasn't visible). Each miss is a replayable fixture for
        // tools/test_tool_drift_corpus.cpp -- turns an unknown drift mode into
        // a permanent regression the next time it recurs. NUL-separated
        // records so embedded newlines don't confuse the reader.
        if (const char* cp = getenv("Q27_DRIFT_CORPUS")) {
            if (FILE* f = fopen(cp, "ab")) {
                fwrite(text_in.data(), 1, text_in.size(), f);
                fputc('\0', f);
                fclose(f);
            }
        }
    }
    return out;
}

// Single-call convenience wrapper (first recovered call). suffix retained for
// callers that trim trailing junk; multi-call callers use the vector form.
inline ToolCall parse_bare_tool_call(const std::string& text_in, std::string* prefix,
                                     std::string* suffix, const json* tools = nullptr) {
    auto v = parse_bare_tool_calls(text_in, prefix, tools);
    if (v.empty()) return ToolCall{};
    if (suffix) *suffix = "";
    return v.front();
}

// ---- OpenAI /v1/chat/completions response shaping -------------------------
// Pulled out of server.cu (same rationale as the Anthropic helpers above):
// pure JSON assembly, unit-tested without CUDA. server.cu's job is only to
// wire the engine callbacks that feed `calls`/`text` -- an exact mechanical
// twin of the already-shipped /v1/messages plumbing.

inline json openai_tool_call_json(const std::string& id, const ToolCall& c) {
    return {{"id", id}, {"type", "function"},
            {"function", {{"name", c.name}, {"arguments", c.arguments.dump()}}}};
}

// Non-streaming choices[0].message. content is null ONLY when there is at
// least one tool call and no leftover text (matches real OpenAI's
// convention); otherwise content is always the string (possibly empty),
// never null, so a plain content-only turn never confuses a strict client.
// `calls` may include ok==false entries (malformed calls) -- the caller is
// expected to have already folded those into `text` (matching the
// /v1/messages precedent) before calling this. `reasoning` (optional):
// non-empty adds a `reasoning_content` field -- not part of the official
// OpenAI schema, but the de facto convention vLLM/SGLang/llama.cpp's server
// all converged on for surfacing a reasoning model's thinking trace over the
// chat/completions wire; unknown fields are inert to clients that don't
// look for it.
inline json openai_chat_message_json(const std::string& text, const std::vector<ToolCall>& calls,
                                     long rid, const std::string& reasoning = std::string()) {
    json msg = {{"role", "assistant"}};
    json tool_calls = json::array();
    int i = 0;
    for (auto& c : calls)
        if (c.ok)
            tool_calls.push_back(openai_tool_call_json(
                "call_q27_" + std::to_string(rid) + "_" + std::to_string(i++), c));
    bool any_call = !tool_calls.empty();
    msg["content"] = (any_call && text.empty()) ? json(nullptr) : json(text);
    if (any_call) msg["tool_calls"] = tool_calls;
    if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
    return msg;
}

// Streamed reasoning_content delta (see openai_chat_message_json's comment
// for the convention this matches).
inline json openai_reasoning_delta(const std::string& t) {
    return {{"reasoning_content", t}};
}

// One SSE chunk envelope (chat.completion.chunk), shared by every delta this
// endpoint emits (content, tool_calls, or the terminal empty-delta chunk).
inline json openai_stream_chunk(const std::string& id, const std::string& obj, long created,
                                const std::string& model, const json& delta,
                                const char* finish_reason = nullptr) {
    json choice = {{"index", 0}, {"delta", delta},
                   {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)}};
    return {{"id", id}, {"object", obj}, {"created", created}, {"model", model},
            {"choices", json::array({choice})}};
}

// One streamed tool_calls[] delta entry. Whole-shot (id+name+full arguments
// in a single chunk) rather than incremental-argument streaming -- matches
// the existing /v1/messages input_json_delta precedent (one full
// partial_json chunk per call, not char-by-char) and is spec-valid: a client
// that expects incremental fragments just accumulates a single fragment.
inline json openai_tool_call_delta(int index, const std::string& id, const ToolCall& c) {
    return {{"tool_calls", json::array({{{"index", index},
                                         {"id", id},
                                         {"type", "function"},
                                         {"function", {{"name", c.name},
                                                       {"arguments", c.arguments.dump()}}}}})}};
}

// ---- API key authentication -----------------------------------------------
// q27 has no auth by default (loopback-only is the safety net) -- these are
// the pure, testable pieces of an opt-in bearer/x-api-key check, wired into
// a pre-routing handler in server.cu when --api-key/--api-key-file/
// Q27_API_KEY configure at least one key.

// Constant-time string comparison: prevents a timing side-channel where an
// early-exit compare leaks how many leading bytes of a guessed key matched
// via response-time variance. Deliberately does not early-exit on length
// mismatch either (still walks max(a,b) bytes) or on the first bit
// difference (accumulates via OR instead of returning).
inline bool secure_compare(const std::string& a, const std::string& b) {
    size_t n = std::max(a.size(), b.size());
    unsigned char diff = (unsigned char)(a.size() != b.size());
    for (size_t i = 0; i < n; i++) {
        unsigned char ca = i < a.size() ? (unsigned char)a[i] : 0;
        unsigned char cb = i < b.size() ? (unsigned char)b[i] : 0;
        diff |= (unsigned char)(ca ^ cb);
    }
    return diff == 0;
}

// q27 serves two API families with two different native auth header
// conventions -- support both, so neither client population needs special
// configuration:
//   Authorization: Bearer <key>   -- OpenAI / llama.cpp convention (Kilocode
//                                    and other OpenAI-compatible clients)
//   x-api-key: <key>              -- Anthropic convention (what Claude Code
//                                    actually sends to /v1/messages)
// x-api-key wins if a request somehow sends both (arbitrary but
// deterministic; no client sends both in practice). Returns "" if neither
// header is present/well-formed.
inline std::string extract_api_key(const std::string& authorization_header,
                                   const std::string& x_api_key_header) {
    if (!x_api_key_header.empty()) return x_api_key_header;
    static const std::string prefix = "Bearer ";
    if (authorization_header.size() > prefix.size() &&
        authorization_header.compare(0, prefix.size(), prefix) == 0)
        return authorization_header.substr(prefix.size());
    return "";
}

// True if `provided` matches ANY configured key. Always scans every key
// (no early return on the first match) so total compare time depends only
// on key count/length, not on which key -- if any -- matched, or how far
// into it a wrong guess got. An empty `provided` is always rejected without
// comparing; an empty KEY is never configured either (load_api_key_file drops
// blank lines, and server.cu's arg parser refuses an empty --api-key /
// Q27_API_KEY at boot), so this is a fast path, not a security-relevant branch.
inline bool api_key_valid(const std::string& provided, const std::vector<std::string>& keys) {
    if (provided.empty()) return false;
    bool any = false;
    for (auto& k : keys) any |= secure_compare(provided, k);

    return any;
}

// 401 body shaped per API family, matching the existing anthropic_error_json
// / OpenAI-error-shape split already used elsewhere in this file for 400s --
// each client SDK gets the error shape it actually parses.
inline std::string auth_error_json(bool anthropic_shape) {
    if (anthropic_shape) return anthropic_error_json("authentication_error", "invalid x-api-key");
    json e = {{"error", {{"message", "Incorrect API key provided"},
                         {"type", "invalid_request_error"},
                         {"code", "invalid_api_key"}}}};
    return e.dump();
}

// Load newline-separated keys from a file (llama.cpp --api-key-file
// convention: one key per line, blank lines and lines starting with '#'
// ignored, surrounding whitespace trimmed). Returns false (and leaves `out`
// untouched) if the file can't be opened, so the caller can fail loudly
// with the actual path rather than silently starting with no auth.
// A file that OPENS but yields no keys (all blank/comment) also returns true
// with `out` unchanged -- the caller must compare sizes and refuse, or an
// operator who asked for auth gets a server with none (server.cu does).
inline bool load_api_key_file(const std::string& path, std::vector<std::string>* out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::vector<std::string> keys;
    std::string line;
    while (std::getline(f, line)) {
        size_t a = line.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        size_t b = line.find_last_not_of(" \t\r\n");
        line = line.substr(a, b - a + 1);
        if (line.empty() || line[0] == '#') continue;
        keys.push_back(line);
    }
    for (auto& k : keys) out->push_back(std::move(k));
    return true;
}

} // namespace q27
