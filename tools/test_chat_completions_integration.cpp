// Integration-level compile+run check for the new /v1/chat/completions
// tool-calling code in server.cu. server.cu needs nvcc (CUDA kernels) to
// build, unavailable in this CPU-only review environment -- so this harness
// fakes ONLY the CUDA-touching surface (Engine, Slot, conductor, httplib)
// with the exact interface the new code calls, and otherwise uses the REAL
// api_common.h, toolgram.h, toolconstrain.h, stream_split.h, and
// tokenizer.{h,cpp} unmodified. The build_prompt/handle block below this
// preamble is a byte-for-byte extraction of src/server.cu's new code
// (see tools/extract_check.sh) -- this is a genuine compile+run check of
// the shipped logic, not a reimplementation of it.
//
// Build+run:
//   bash tools/build_chat_completions_integration.sh
#include "api_common.h"
#include "toolconstrain.h"
#include "tokenizer.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

using json = nlohmann::json;
using q27::Msg;
using q27::StreamSplitter;

namespace q27k {
struct SampleParams { float inv_temp = 0.f; float top_p = 1.f; unsigned long long seed = 0; };
}

// ---- fake Tokenizer ---------------------------------------------------
// decode_one() replays a pre-scripted piece-table; encode() hands back one
// synthetic id per call (only the resulting vector SIZE matters to the code
// under test -- ids are never decoded back).
struct FakeTok {
    std::vector<std::string> pieces;
    int eos_id = 0;
    int next_encode_id = 1000;
    std::string decode_one(int id) const {
        return (id >= 0 && (size_t)id < pieces.size()) ? pieces[(size_t)id] : std::string();
    }
    std::vector<int> encode(const std::string&) { return {next_encode_id++}; }
    int eos() const { return eos_id; }
    int token_id(const std::string&) const { return -1; }
    std::vector<std::string> vocab_bytes() const { return pieces; }
    // Only used by the UNCHANGED build_prompt() fallback path (non-routed_chat
    // requests); real behavior is irrelevant here, tests for that path only
    // check the resulting prompt-size bookkeeping via encode()'s id count.
    std::vector<int> apply_chat_template(const std::vector<std::pair<std::string, std::string>>&,
                                         bool) {
        return {next_encode_id++};
    }
};

// ---- fake Engine --------------------------------------------------------
struct FakeEngine {
    struct DecodeTask {
        int rounds = 0, bat_members = 0; long bat_r2 = 0; int emitted = 0;
        std::atomic<bool> cancel{false};
    };
    q27k::SampleParams samp;
    int max_ctx = 100000;
    int ctx_round_reserve() const { return 8; }
    std::function<bool()> on_round_gap;
    std::function<void(int)> on_pending;
    std::function<void(const int*)> on_drafts;
    std::function<int(const int*, int)> on_round;

    int mask_pool_used = 0;
    int mask_pool_cap = 512;
    int mask_pool_add(const void*) { return mask_pool_used >= mask_pool_cap ? -1 : mask_pool_used++; }
    void set_tool_constraint(int) {}
    void set_tool_masks5(const int[5]) {}

    std::vector<int> script; // token ids the test wants "generated"
    template <typename F>
    int generate(const std::vector<int>&, int n_max, int /*eos*/, F&& on_token, int = -1) {
        int n = 0;
        for (int id : script) {
            if (n >= n_max) break;
            if (!on_token(id)) break;
            n++;
        }
        return n;
    }
};
using Engine = FakeEngine;

struct Slot {
    std::unique_ptr<Engine> eng;
    int id = 0;
    bool busy = false;
    long last_used = 0;
    std::vector<int> tool_mask_host2dev;
};

using ToolConstrainer = q27::BasicToolConstrainer<Engine, FakeTok>;

struct HookGuard {
    Engine& e;
    ~HookGuard() { e.on_pending = nullptr; e.on_drafts = nullptr; e.on_round = nullptr; }
};

// ---- fake httplib -------------------------------------------------------
namespace httplib {
struct Request { std::string body; };
struct DataSink {
    std::string* out;
    bool write(const char* d, size_t n) { out->append(d, n); return true; }
    bool done() { return true; }
};
struct Response {
    int status = 200;
    std::string content, content_type;
    std::function<bool(size_t, DataSink&)> provider;
    void set_content(const std::string& s, const std::string& ct) { content = s; content_type = ct; }
    void set_header(const char*, const char*) {}
    void set_chunked_content_provider(const char*, std::function<bool(size_t, DataSink&)> p) {
        provider = std::move(p);
    }
};
} // namespace httplib

static std::string jdump(const json& j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

// ---- global test scaffolding (populated fresh per test in main()) ------
json g_last_response;
std::vector<json> g_sse_events; // parsed "data: {...}" payloads, [DONE] excluded

static void run_request(FakeTok& tok, std::string served_name, bool no_think_srv,
                        bool constrain_tools, bool sampled_on, int max_prompt,
                        int max_slot_ctx, std::atomic<long>& req_counter,
                        q27::ToolMaskCache& tool_mask_cache, std::vector<Slot>& slots,
                        const json& body, bool chat) {
    const int EOS = tok.eos();
    std::mutex route_m;
    void* conductor = nullptr; // always null: only the non-conductor branches execute
    q27::GpuGate gpu_gate; // real type (api_common.h, CUDA-free) -- Lease is RAII-only here

    auto ms_since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t)
            .count();
    };
    auto conv_fp = [&](const json&) -> unsigned long long { return 0; };
    struct ReqTrace {
        long rid; const char* api; unsigned long long conv;
        std::chrono::steady_clock::time_point t0; double tok_ms;
    };
    auto claim_slot = [&](const std::vector<int>&) -> Slot& { slots[0].busy = true; return slots[0]; };
    auto slot_guard = [&](Slot& s) {
        return std::shared_ptr<Slot>(&s, [](Slot* p) { p->busy = false; });
    };
    auto make_yield = [&](Engine&) -> std::function<bool()> { return nullptr; };
    auto tg_stats = [&](const ToolConstrainer&) -> std::string { return ""; };
    auto bat_stats = [&](const FakeEngine::DecodeTask&) -> std::string { return ""; };
    auto req_log = [&](const ReqTrace&, double, const Engine&, int,
                       const std::string& = std::string()) {};
    auto parse_sample = [&](const json& b) -> q27k::SampleParams {
        q27k::SampleParams s;
        double temp = b.value("temperature", 0.0);
        if (temp > 0.0) s.inv_temp = (float)(1.0 / temp);
        return s;
    };
    auto batch_generate = [&](Engine& eng, const std::vector<int>&, int nm,
                              std::function<bool(int)> on_token,
                              std::function<void(int)> on_emit, int /*stable_len*/, double&,
                              const ReqTrace&, FakeEngine::DecodeTask& t,
                              std::string*) -> int {
        int n = 0;
        for (int id : eng.script) {
            if (n >= nm) break;
            if (on_emit) on_emit(id);
            if (!on_token(id)) break;
            n++;
        }
        t.emitted = n;
        return n;
    };

    httplib::Request req;
    req.body = body.dump();
    httplib::Response res;

    auto build_prompt = [&](const json& body) -> std::vector<int> {
        if (body.contains("messages")) {
            std::vector<std::pair<std::string, std::string>> msgs;
            for (auto& m : body["messages"]) {
                std::string role = m.value("role", "user");
                std::string content;
                // const operator[] on a missing key aborts (json.hpp assertion) --
                // a content-less message must not kill the server (Security #1;
                // mirrors the Anthropic-path guard in api_common.h).
                if (m.is_object() && m.contains("content")) {
                    if (m["content"].is_string()) content = m["content"];
                    else if (m["content"].is_array())
                        for (auto& part : m["content"])
                            if (part.value("type", "") == "text")
                                content += part.value("text", "");
                }
                msgs.push_back({role, content});
            }
            // enable_thinking=false: top-level (Qwen-style clients) or nested
            // chat_template_kwargs (llama.cpp/GLM-style) -> empty-think prefill
            bool think = body.value("enable_thinking", true);
            if (body.contains("chat_template_kwargs"))
                think = body["chat_template_kwargs"].value("enable_thinking", think);
            if (no_think_srv) think = false;
            return tok.apply_chat_template(msgs, think);
        }
        return tok.encode(body.value("prompt", std::string()));
    };

    auto handle = [&](const httplib::Request& req, httplib::Response& res, bool chat) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return; }
        int n_max = body.value("max_tokens", 256);
        bool stream = body.value("stream", false);
        // stream_options.include_usage (OpenAI streaming spec, both API
        // shapes): when true, one extra SSE chunk -- empty choices + the
        // usage totals -- goes out after the finish_reason chunk, before
        // [DONE]. Tolerant parse: a non-object stream_options or non-bool
        // include_usage reads as false (a malformed option must not throw
        // out of the handler). Absent/false -> zero framing change.
        bool inc_usage = false;
        if (stream && body.contains("stream_options") && body["stream_options"].is_object()) {
            const auto& so = body["stream_options"];
            inc_usage = so.contains("include_usage") && so["include_usage"].is_boolean() &&
                        so["include_usage"].get<bool>();
        }
        // Tool-calling admission: gate strictly on chat==true AND an actual
        // "messages" array, so /v1/completions and the raw-"prompt" chat
        // fallback are provably untouched by anything below (they never
        // enter any of the branches this flag guards).
        const bool routed_chat = chat && body.contains("messages") && body["messages"].is_array();
        json tools = json::array();
        q27::ToolChoice tchoice;
        std::vector<std::string> tool_names_v;
        if (routed_chat) {
            tchoice = q27::parse_tool_choice(body);
            tools = tchoice.mode == q27::ToolChoice::NONE ? json::array() : q27::openai_tools_json(body);
            if (constrain_tools && tools.is_array())
                for (auto& t : tools)
                    if (t.contains("function") && t["function"].contains("name"))
                        tool_names_v.push_back(t["function"]["name"].get<std::string>());
            // named-forced tool_choice restricts the grammar to that one name;
            // "required" (no name) leaves every registered tool eligible.
            if (tchoice.mode == q27::ToolChoice::FORCED && !tchoice.forced_name.empty())
                tool_names_v = {tchoice.forced_name};
        }
        long rid = req_counter++;
        auto tk0 = std::chrono::steady_clock::now();
        std::vector<int> prompt;
        int stable_len = -1; // -1 = legacy tail snapshot (build_prompt's fallback path)
        if (routed_chat) {
            bool think = body.value("enable_thinking", true);
            if (body.contains("chat_template_kwargs"))
                think = body["chat_template_kwargs"].value("enable_thinking", think);
            if (no_think_srv) think = false;
            size_t stable_off = 0;
            std::string rendered =
                q27::chatml_prompt(q27::openai_msgs(body), tools, think, &stable_off);
            // FORCED tool_choice: inject the opener into the volatile tail
            // (past stable_off, alongside the assistant-open/think-prefill --
            // P8 prefix-cache reuse is unaffected). The stream router below
            // is pre-seeded straight into the TOOL channel since the marker
            // itself never appears in the GENERATED text this way.
            if (tchoice.mode == q27::ToolChoice::FORCED) rendered += "<tool_call>\n";
            prompt = tok.encode(rendered.substr(0, stable_off));
            stable_len = (int)prompt.size();
            std::vector<int> tailv = tok.encode(rendered.substr(stable_off));
            prompt.insert(prompt.end(), tailv.begin(), tailv.end());
        } else {
            prompt = build_prompt(body);
        }
        ReqTrace rt{rid, chat ? "oai" : "cmpl", conv_fp(body),
                    std::chrono::steady_clock::now(), ms_since(tk0)};
        // Reject an empty prompt before slot selection: reuse_len() would run
        // ckpt_best() over an empty vector, and (pre-fix) a zero-token prompt
        // decodes from stale recurrent state and echoes the prior request's
        // pending token. An empty /v1/completions prompt is nonsensical anyway;
        // chat/messages always tokenize non-empty (template structure).
        if (prompt.empty()) {
            res.status = 400;
            res.set_content(json{{"error", {{"message", "empty prompt"},
                                            {"type", "invalid_request_error"},
                                            {"code", "empty_prompt"}}}}
                                .dump(),
                            "application/json");
            return;
        }
        // context-limit preflight BEFORE slot claim / SSE commit (review
        // follow-up 2026-07-09 #3): past this bound the routed slot's
        // n_max clamp floors at 0 -> empty 200
        if ((int)prompt.size() > max_prompt) {
            res.status = 400;
            res.set_content(json{{"error",
                                  {{"message", q27::ctx_limit_error_message(
                                                   (int)prompt.size(), max_prompt)},
                                   {"type", "invalid_request_error"},
                                   {"code", "context_length_exceeded"}}}}
                                .dump(),
                            "application/json");
            return;
        }
        if ((int)prompt.size() + n_max > max_slot_ctx)
            n_max = max_slot_ctx - (int)prompt.size();
        // Q27_SAMPLED=0 preflight: the sampled graphs were never captured.
        // (Q27_FORCE_TEMP>0 is a boot-time FATAL on such boots, so the
        // request-absent default here is genuinely greedy.)
        if (!sampled_on && body.value("temperature", 0.0) > 0.0) {
            res.status = 400;
            res.set_content(json{{"error",
                                  {{"message", "sampling disabled: server booted with "
                                               "Q27_SAMPLED=0 (greedy-only)"},
                                   {"type", "invalid_request_error"},
                                   {"code", "sampling_disabled"}}}}
                                .dump(),
                            "application/json");
            return;
        }
        long created = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();

        const char* obj = chat ? "chat.completion" : "text_completion";
        const char* objd = chat ? "chat.completion.chunk" : "text_completion";

        if (!stream) {
            Slot& sl = claim_slot(prompt); // may wait for a free engine
            auto sl_lease = slot_guard(sl);
            Engine& eng = *sl.eng;
            HookGuard hooks{eng}; // safe even when routed_chat is false: hooks
                                  // are never set on that path, so the clear
                                  // on scope-exit is a no-op (P15 M1 pattern)
            eng.samp = parse_sample(body);
            // Q27_BATCH: solo keeps the whole-call lease; batch mode scopes
            // its prefill lease inside batch_generate (A7) and re-stamps qw.
            std::optional<q27::GpuGate::Lease> lk;
            if (!conductor) lk.emplace(gpu_gate);
            double qw = ms_since(rt.t0);
            eng.on_round_gap = make_yield(eng);
            // re-clamp to the routed slot (rows P+1..P+gate_maxd+1 must stay
            // in ctx; reserve derived from the engine's active max depth)
            n_max = std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - (eng.ctx_round_reserve() - 1)));

            if (!routed_chat) {
                // ORIGINAL text-only behavior, byte-for-byte unchanged.
                std::string text;
                q27::Utf8Gate ugate;
                auto on_tok = [&](int id) {
                    text += ugate.feed(tok.decode_one(id));
                    return true;
                };
                Engine::DecodeTask bt;
                std::string berr;
                int n = conductor ? batch_generate(eng, prompt, n_max, on_tok, nullptr, -1,
                                                   qw, rt, bt, &berr)
                                  : eng.generate(prompt, n_max, EOS, on_tok);
                eng.on_round_gap = nullptr;
                text += ugate.flush();
                req_log(rt, qw, eng, sl.id, bat_stats(bt));
                // batch error surfacing (review pass 2): nothing emitted = an
                // honest 500 in the OpenAI error envelope; if tokens WERE
                // produced, keep the 200 with the partial text -- end=error is
                // already in the [req] line either way.
                if (!berr.empty() && n == 0) {
                    res.status = 500;
                    res.set_content(json{{"error", {{"message", berr},
                                                    {"type", "api_error"}}}}
                                        .dump(),
                                    "application/json");
                    return;
                }
                json choice;
                if (chat)
                    choice = {{"index", 0}, {"finish_reason", n >= n_max ? "length" : "stop"},
                              {"message", {{"role", "assistant"}, {"content", text}}}};
                else
                    choice = {{"index", 0}, {"finish_reason", n >= n_max ? "length" : "stop"},
                              {"text", text}};
                json out = {{"id", "q27-0"}, {"object", obj}, {"created", created},
                            {"model", served_name}, {"choices", json::array({choice})},
                            {"usage", {{"prompt_tokens", (int)prompt.size()},
                                       {"completion_tokens", n},
                                       {"total_tokens", (int)prompt.size() + n}}}};
                res.set_content(jdump(out), "application/json");
                return;
            }

            // routed_chat: think/tool-aware path, an exact mechanical twin of
            // the /v1/messages non-stream handler above, OpenAI-shaped output.
            StreamSplitter sp;
            q27::Utf8Gate ugate;
            std::string think, text, tool_buf;
            std::vector<q27::ToolCall> calls;
            auto route = [&](StreamSplitter::Chan ch, const std::string& t) {
                if (ch == StreamSplitter::TOOL) { tool_buf += t; return; }
                if (!tool_buf.empty()) { // tool segment closed
                    calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));
                    tool_buf.clear();
                }
                (ch == StreamSplitter::THINK ? think : text) += t;
            };
            ToolConstrainer tc;
            tc.eng = &eng; tc.tok = &tok; tc.cache = &tool_mask_cache;
            tc.host2dev = &sl.tool_mask_host2dev;
            // FORCED requests are prompt-injected past the <tool_call> marker
            // (above) -- scan_round's engage trigger scans GENERATED text for
            // that marker and will never fire, so grammar masking is skipped
            // for those (documented limitation, api_common.h ToolChoice
            // comment); AUTO (the default) and NONE are unaffected.
            tc.enabled = constrain_tools && tchoice.mode != q27::ToolChoice::FORCED &&
                        eng.samp.inv_temp <= 0.f; // constrained+sampled is Phase 3
            tc.begin(tool_names_v);
            // FORCED: the opener was injected into the PROMPT, not generated,
            // so the splitter must start already inside the TOOL channel or
            // the call body would be read back as ordinary text.
            if (tchoice.mode == q27::ToolChoice::FORCED) sp.chan = StreamSplitter::TOOL;
            eng.on_pending = [&](int id) { tc.on_pending(id); };
            eng.on_drafts = [&](const int* dr) { tc.on_drafts(dr); };
            if (tc.enabled)
                eng.on_round = [&](const int* em, int nr) { return tc.scan_round(em, nr); };
            auto on_tok = [&](int id) {
                for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) route(ch, t);
                return true;
            };
            Engine::DecodeTask bt;
            std::string berr;
            int n = conductor
                        ? batch_generate(eng, prompt, n_max, on_tok,
                                         [&](int id) { tc.on_id(id); }, stable_len, qw,
                                         rt, bt, &berr)
                        : eng.generate(prompt, n_max, EOS, [&](int id) {
                              tc.on_id(id);
                              return on_tok(id);
                          }, stable_len);
            tc.end();
            eng.on_pending = nullptr;
            eng.on_drafts = nullptr;
            eng.on_round = nullptr;
            eng.on_round_gap = nullptr;
            req_log(rt, qw, eng, sl.id, tg_stats(tc) + bat_stats(bt));
            if (!berr.empty() && n == 0) {
                res.status = 500;
                res.set_content(json{{"error", {{"message", berr}, {"type", "api_error"}}}}
                                    .dump(),
                                "application/json");
                return;
            }
            for (auto& [ch, t] : sp.feed(ugate.flush())) route(ch, t);
            for (auto& [ch, t] : sp.flush()) route(ch, t);
            if (!tool_buf.empty())
                calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));

            std::string tx = q27::strip_ws2(text);
            for (auto& c : calls)
                if (!c.ok) tx += (tx.empty() ? "" : "\n") + c.raw;
            if (tools.is_array() && !tools.empty()) {
                // wrapper-less call recovery (see parse_bare_tool_calls)
                std::string pre;
                auto bcs = q27::parse_bare_tool_calls(tx, &pre, &tools);
                if (!bcs.empty()) {
                    fprintf(stderr,
                            "[tool-fallback] %zu bare call(s) recovered (oai-nonstream)\n",
                            bcs.size());
                    tx = pre;
                    for (auto& bc : bcs) calls.push_back(bc);
                }
            }
            bool any_call = false;
            for (auto& c : calls)
                if (c.ok) any_call = true;
            json msg = q27::openai_chat_message_json(tx, calls, rid, q27::strip_ws2(think));
            json choice = {{"index", 0},
                          {"finish_reason", any_call ? "tool_calls" : (n >= n_max ? "length" : "stop")},
                          {"message", msg}};
            json out = {{"id", "chatcmpl-q27-" + std::to_string(rid)}, {"object", obj},
                        {"created", created}, {"model", served_name},
                        {"choices", json::array({choice})},
                        {"usage", {{"prompt_tokens", (int)prompt.size()},
                                   {"completion_tokens", n},
                                   {"total_tokens", (int)prompt.size() + n}}}};
            res.set_content(jdump(out), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        const bool has_tools = tools.is_array() && !tools.empty();
        q27k::SampleParams samp = parse_sample(body);
        res.set_chunked_content_provider(
            "text/event-stream",
            [&, samp, prompt, n_max, created, chat, obj, objd, rt, inc_usage, routed_chat,
             tools, tool_names_v, tchoice, stable_len, has_tools, rid](size_t, httplib::DataSink& sink) {
                Slot& sl = claim_slot(prompt);
                auto sl_lease = slot_guard(sl);
                Engine& eng = *sl.eng;
                HookGuard hooks{eng}; // see the non-stream twin
                eng.samp = samp;
                std::optional<q27::GpuGate::Lease> lk; // see the non-stream twin
                if (!conductor) lk.emplace(gpu_gate);
                double qw = ms_since(rt.t0);
                eng.on_round_gap = make_yield(eng);
                const int nm =
                    std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - (eng.ctx_round_reserve() - 1)));
                auto send = [&](const json& j) {
                    std::string s = "data: " + jdump(j) + "\n\n";
                    return sink.write(s.data(), s.size());
                };

                if (!routed_chat) {
                    // ORIGINAL text-only streaming behavior, byte-for-byte unchanged.
                    q27::Utf8Gate ugate;
                    auto piece_chunk = [&](const std::string& piece) {
                        json delta = chat ? json{{"content", piece}} : json{};
                        json choice = chat
                            ? json{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}
                            : json{{"index", 0}, {"text", piece}, {"finish_reason", nullptr}};
                        return json{{"id", "q27-0"}, {"object", objd}, {"created", created},
                                    {"model", served_name}, {"choices", json::array({choice})}};
                    };
                    auto on_tok = [&](int id) {
                        // empty pieces (control tokens, gate holdbacks) still probe
                        // the socket so a disconnected client stops generation
                        return send(piece_chunk(ugate.feed(tok.decode_one(id))));
                    };
                    Engine::DecodeTask bt;
                    // TODO(batch error surfacing): on a failed queue (A2) this
                    // stream just ends with a normal finish_reason -- the OpenAI
                    // SSE shape has no standard mid-stream error event, so none
                    // is invented; end=error lands in the [req] line and
                    // [req-error] carries the what().
                    int produced = conductor ? batch_generate(eng, prompt, nm, on_tok, nullptr,
                                                              -1, qw, rt, bt, nullptr)
                                             : eng.generate(prompt, nm, EOS, on_tok);
                    eng.on_round_gap = nullptr;
                    std::string tailp = ugate.flush();
                    if (!tailp.empty()) send(piece_chunk(tailp));
                    // Terminal chunk with a real finish_reason (OpenAI streaming spec):
                    // clients otherwise never learn whether generation hit EOS or the
                    // token cap. produced >= nm == the length cap; else a stop.
                    {
                        const char* fr = produced >= nm ? "length" : "stop";
                        json fchoice = chat ? json{{"index", 0}, {"delta", json::object()},
                                                   {"finish_reason", fr}}
                                            : json{{"index", 0}, {"text", ""}, {"finish_reason", fr}};
                        send(json{{"id", "q27-0"}, {"object", objd}, {"created", created},
                                  {"model", served_name}, {"choices", json::array({fchoice})}});
                    }
                    // stream_options.include_usage: final usage chunk (empty
                    // choices) mirroring the non-stream usage body above.
                    if (inc_usage)
                        send(json{{"id", "q27-0"}, {"object", objd}, {"created", created},
                                  {"model", served_name}, {"choices", json::array()},
                                  {"usage", {{"prompt_tokens", (int)prompt.size()},
                                             {"completion_tokens", produced},
                                             {"total_tokens", (int)prompt.size() + produced}}}});
                    req_log(rt, qw, eng, sl.id, bat_stats(bt));
                    std::string done = "data: [DONE]\n\n";
                    sink.write(done.data(), done.size());
                    sink.done();
                    return true;
                }

                // routed_chat: think/tool-aware streaming path, an exact
                // mechanical twin of the /v1/messages SSE handler above.
                const std::string cid = "chatcmpl-q27-" + std::to_string(rid);
                ToolConstrainer tc;
                tc.eng = &eng; tc.tok = &tok; tc.cache = &tool_mask_cache;
                tc.host2dev = &sl.tool_mask_host2dev;
                tc.enabled = constrain_tools && tchoice.mode != q27::ToolChoice::FORCED &&
                            eng.samp.inv_temp <= 0.f; // constrained+sampled is Phase 3
                tc.begin(tool_names_v);
                StreamSplitter sp;
                if (tchoice.mode == q27::ToolChoice::FORCED) sp.chan = StreamSplitter::TOOL;
                q27::Utf8Gate ugate;
                bool alive = true; // cleared when a write fails (client disconnected)
                int tool_idx = 0;
                bool any_call = false;
                std::string tool_buf, text_accum;
                auto emit_tool = [&]() {
                    auto c = q27::parse_tool_call(q27::strip_ws2(tool_buf));
                    tool_buf.clear();
                    if (!c.ok) { // malformed: surface as text so nothing is lost
                        if (!send(q27::openai_stream_chunk(cid, objd, created, served_name,
                                                           json{{"content", c.raw}})))
                            alive = false;
                        return;
                    }
                    any_call = true;
                    std::string tid =
                        "call_q27_" + std::to_string(rid) + "_" + std::to_string(tool_idx);
                    bool ok = send(q27::openai_stream_chunk(
                        cid, objd, created, served_name,
                        q27::openai_tool_call_delta(tool_idx, tid, c)));
                    tool_idx++;
                    if (!ok) alive = false;
                };
                auto emit_seg = [&](StreamSplitter::Chan ch, const std::string& t) {
                    if (ch == StreamSplitter::TOOL) { tool_buf += t; return; }
                    if (!tool_buf.empty()) emit_tool();
                    if (t.empty()) return;
                    // reasoning_content (no official OpenAI field for this;
                    // matches the vLLM/SGLang/llama.cpp convention -- see
                    // openai_reasoning_delta) rather than leaking raw <think>
                    // tags into `content` (the bug this whole path also
                    // happens to fix).
                    if (ch == StreamSplitter::THINK) {
                        if (!send(q27::openai_stream_chunk(cid, objd, created, served_name,
                                                            q27::openai_reasoning_delta(t))))
                            alive = false;
                        return;
                    }
                    text_accum += t;
                    if (!send(q27::openai_stream_chunk(cid, objd, created, served_name,
                                                       json{{"content", t}})))
                        alive = false;
                };
                eng.on_pending = [&](int id) { tc.on_pending(id); };
                eng.on_drafts = [&](const int* dr) { tc.on_drafts(dr); };
                if (tc.enabled)
                    eng.on_round = [&](const int* em, int nr) { return tc.scan_round(em, nr); };
                auto on_tok = [&](int id) {
                    for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) emit_seg(ch, t);
                    return alive; // stop generating once the client has disconnected
                };
                Engine::DecodeTask bt;
                int produced = conductor
                                   ? batch_generate(eng, prompt, nm, on_tok,
                                                    [&](int id) { tc.on_id(id); },
                                                    stable_len, qw, rt, bt, nullptr)
                                   : eng.generate(prompt, nm, EOS, [&](int id) {
                                         tc.on_id(id);
                                         return on_tok(id);
                                     }, stable_len);
                tc.end();
                eng.on_pending = nullptr;
                eng.on_drafts = nullptr;
                eng.on_round = nullptr;
                eng.on_round_gap = nullptr;
                req_log(rt, qw, eng, sl.id, tg_stats(tc) + bat_stats(bt));
                for (auto& [ch, t] : sp.feed(ugate.flush())) emit_seg(ch, t);
                for (auto& [ch, t] : sp.flush()) emit_seg(ch, t);
                if (!tool_buf.empty()) emit_tool();
                if (has_tools) {
                    // wrapper-less call recovery: text already streamed as a
                    // content delta (cosmetic); the tool_calls delta still fires
                    std::string pre;
                    auto bcs = q27::parse_bare_tool_calls(text_accum, &pre, &tools);
                    if (!bcs.empty()) {
                        fprintf(stderr,
                                "[tool-fallback] %zu bare call(s) recovered (oai-stream)\n",
                                bcs.size());
                        any_call = true;
                        for (auto& bc : bcs) {
                            std::string tid = "call_q27_" + std::to_string(rid) + "_" +
                                              std::to_string(tool_idx);
                            bool ok = send(q27::openai_stream_chunk(
                                cid, objd, created, served_name,
                                q27::openai_tool_call_delta(tool_idx, tid, bc)));
                            tool_idx++;
                            if (!ok) alive = false;
                        }
                    }
                }
                // TODO(batch error surfacing): no standard OpenAI mid-stream
                // error chunk exists (matches the plain-text leg's TODO
                // above); end=error lands in the [req] line, [req-error]
                // carries the what() (batch_generate logs it unconditionally
                // when err_out is null, same as that leg's nullptr err_out).
                {
                    const char* fr = any_call ? "tool_calls" : (produced >= nm ? "length" : "stop");
                    send(q27::openai_stream_chunk(cid, objd, created, served_name,
                                                  json::object(), fr));
                }
                if (inc_usage)
                    send(json{{"id", cid}, {"object", objd}, {"created", created},
                              {"model", served_name}, {"choices", json::array()},
                              {"usage", {{"prompt_tokens", (int)prompt.size()},
                                         {"completion_tokens", produced},
                                         {"total_tokens", (int)prompt.size() + produced}}}});
                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    };

    handle(req, res, chat);

    g_last_response = json::object();
    g_sse_events.clear();
    if (!res.content.empty()) {
        g_last_response = json::parse(res.content);
        g_last_response["__status"] = res.status;
    }
    if (res.provider) {
        std::string buf;
        httplib::DataSink sink{&buf};
        res.provider(0, sink);
        size_t pos = 0;
        while (true) {
            size_t start = buf.find("data: ", pos);
            if (start == std::string::npos) break;
            size_t end = buf.find("\n\n", start);
            if (end == std::string::npos) break;
            std::string payload = buf.substr(start + 6, end - start - 6);
            if (payload != "[DONE]") g_sse_events.push_back(json::parse(payload));
            pos = end + 2;
        }
    }
}

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

static std::vector<Slot> fresh_slots() {
    std::vector<Slot> slots;
    Slot s; s.eng = std::make_unique<FakeEngine>(); s.id = 0;
    slots.push_back(std::move(s));
    return slots;
}

static q27::ToolMaskCache fresh_cache(std::vector<std::string>& vocab_bytes) {
    q27::ToolMaskCache c;
    c.init(&vocab_bytes, -1); // </tool_call> id unused (constrain_tools off in these tests)
    return c;
}

int main() {
    // ---- Test 1: plain chat, no tools -> plain content, no tool_calls ----
    {
        FakeTok tok;
        tok.pieces = {"<eos>", "Hello", ", ", "world!"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3};
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role","user"},{"content","hi"}}})}};
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        CHECK(g_last_response["choices"][0]["message"]["content"] == "Hello, world!");
        CHECK(!g_last_response["choices"][0]["message"].contains("tool_calls"));
        CHECK(g_last_response["choices"][0]["finish_reason"] == "stop");
        CHECK(g_last_response["object"] == "chat.completion");
    }

    // ---- Test 2: tool call, non-stream ----
    // model emits: "Sure, checking." <tool_call>\n{"name": "get_weather",
    // "arguments": {"location": "Tokyo"}}\n</tool_call>
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",
            /*1*/ "Sure, checking.",
            /*2*/ "<tool_call>\n",
            /*3*/ "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}\n",
            /*4*/ "</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3, 4};
        std::atomic<long> rc{0};
        json body = {
            {"tools", json::array({
                {{"type","function"},{"function",{{"name","get_weather"},
                    {"description","w"},{"parameters", json::object()}}}}
            })},
            {"messages", json::array({{{"role","user"},{"content","weather in tokyo?"}}})},
        };
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        auto& msg = g_last_response["choices"][0]["message"];
        CHECK(g_last_response["choices"][0]["finish_reason"] == "tool_calls");
        CHECK(msg["content"] == "Sure, checking.");
        CHECK(msg["tool_calls"].size() == 1);
        CHECK(msg["tool_calls"][0]["type"] == "function");
        CHECK(msg["tool_calls"][0]["function"]["name"] == "get_weather");
        json args = json::parse(msg["tool_calls"][0]["function"]["arguments"].get<std::string>());
        CHECK(args["location"] == "Tokyo");
    }

    // ---- Test 3: thinking + tool call, non-stream: <think> must never leak
    // into `content` verbatim (the original leak this path fixes), and must
    // not break tool-call parsing. Post-Patch-3, the think trace is
    // deliberately surfaced via `reasoning_content` (not silently dropped),
    // so it must land there specifically. ----
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",
            /*1*/ "<think>",
            /*2*/ "reasoning about tokyo weather",
            /*3*/ "</think>",
            /*4*/ "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}\n</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3, 4};
        std::atomic<long> rc{0};
        json body = {
            {"tools", json::array({
                {{"type","function"},{"function",{{"name","get_weather"},{"description","w"},
                    {"parameters", json::object()}}}}
            })},
            {"messages", json::array({{{"role","user"},{"content","weather in tokyo?"}}})},
            {"enable_thinking", true},
        };
        run_request(tok, "q27-test", /*no_think_srv=*/false, false, true, 100000, 100000, rc,
                   cache, slots, body, true);
        auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["content"].is_null()); // no leftover visible text, only the call
        CHECK(msg["tool_calls"].size() == 1);
        CHECK(msg["tool_calls"][0]["function"]["name"] == "get_weather");
        // the think trace must land in reasoning_content specifically...
        CHECK(msg["reasoning_content"] == "reasoning about tokyo weather");
        // ...and NOWHERE else (content, tool_calls) -- dump minus the
        // reasoning_content field's own value must not contain the phrase.
        json msg_no_reasoning = msg;
        msg_no_reasoning.erase("reasoning_content");
        CHECK(msg_no_reasoning.dump().find("reasoning about tokyo weather") == std::string::npos);
    }

    // ---- Test 4: multi-turn history round-trip -- a PRIOR assistant
    // tool_calls[] + a role:"tool" result message must both survive into the
    // model-visible prompt structure. We can't inspect the rendered prompt
    // string directly through this harness (FakeTok::encode discards it by
    // design), so this test instead exercises openai_msgs directly (already
    // covered end-to-end in test_openai_bridge.cpp) and confirms handle()
    // does not crash/error on a body containing that history shape end-to-end.
    {
        FakeTok tok;
        tok.pieces = {"<eos>", "72F and sunny tomorrow too."};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1};
        std::atomic<long> rc{0};
        json body = {
            {"messages", json::array({
                {{"role","user"},{"content","weather in Tokyo?"}},
                {{"role","assistant"},{"content", nullptr},
                 {"tool_calls", json::array({
                     {{"id","call_1"},{"type","function"},
                      {"function",{{"name","get_weather"},{"arguments","{\"location\":\"Tokyo\"}"}}}}
                 })}},
                {{"role","tool"},{"tool_call_id","call_1"},{"content","72F and sunny"}},
                {{"role","user"},{"content","and tomorrow?"}},
            })},
        };
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        CHECK(g_last_response["__status"] == 200);
        CHECK(g_last_response["choices"][0]["message"]["content"] == "72F and sunny tomorrow too.");
    }

    // ---- Test 5: tool_choice: "none" -> tools stripped, model can't call,
    // even if it emits <tool_call> markers they parse as a call but the
    // request must still render (tools array empty means chatml_prompt gets
    // no tools_preamble; we verify via the empty-tools bare-recovery gate:
    // parse_bare_tool_calls only runs when `tools` is non-empty in this path,
    // matching the /v1/messages precedent) ----
    {
        FakeTok tok;
        tok.pieces = {"<eos>", "plain answer, no tools mentioned"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1};
        std::atomic<long> rc{0};
        json body = {
            {"tool_choice", "none"},
            {"tools", json::array({
                {{"type","function"},{"function",{{"name","get_weather"},{"description","w"},
                    {"parameters", json::object()}}}}
            })},
            {"messages", json::array({{{"role","user"},{"content","weather in tokyo?"}}})},
        };
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["content"] == "plain answer, no tools mentioned");
        CHECK(!msg.contains("tool_calls"));
    }

    // ---- Test 6: tool_choice FORCED (named) -> the "<tool_call>\n" opener
    // is injected into the PROMPT (invisible to this harness's FakeTok, which
    // discards prompt text) and the stream splitter is pre-seeded into TOOL
    // channel, so generated text -- with NO literal opening marker this time
    // -- must still parse as a tool call. ----
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",
            /*1*/ "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n",
            /*2*/ "</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2}; // NOTE: no <tool_call> opener token at all
        std::atomic<long> rc{0};
        json body = {
            {"tool_choice", {{"type","function"},{"function",{{"name","get_weather"}}}}},
            {"tools", json::array({
                {{"type","function"},{"function",{{"name","get_weather"},{"description","w"},
                    {"parameters", json::object()}}}}
            })},
            {"messages", json::array({{{"role","user"},{"content","weather in paris?"}}})},
        };
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        auto& msg = g_last_response["choices"][0]["message"];
        CHECK(g_last_response["choices"][0]["finish_reason"] == "tool_calls");
        CHECK(msg["tool_calls"].size() == 1);
        CHECK(msg["tool_calls"][0]["function"]["name"] == "get_weather");
        json args = json::parse(msg["tool_calls"][0]["function"]["arguments"].get<std::string>());
        CHECK(args["location"] == "Paris");
    }

    // ---- Test 7: /v1/completions (chat=false) with a raw "prompt" field
    // must take the ORIGINAL text-only path untouched (no routed_chat). ----
    {
        FakeTok tok;
        tok.pieces = {"<eos>", "plain completion text"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1};
        std::atomic<long> rc{0};
        json body = {{"prompt", "continue: "}};
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, /*chat=*/false);
        CHECK(g_last_response["object"] == "text_completion");
        CHECK(g_last_response["choices"][0]["text"] == "plain completion text");
        CHECK(!g_last_response["choices"][0].contains("message"));
    }

    // ---- Test 8: streaming, tool call -- verify SSE delta shapes ----
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",
            /*1*/ "Checking now.",
            /*2*/ "<tool_call>\n",
            /*3*/ "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Rome\"}}\n",
            /*4*/ "</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3, 4};
        std::atomic<long> rc{0};
        json body = {
            {"stream", true},
            {"tools", json::array({
                {{"type","function"},{"function",{{"name","get_weather"},{"description","w"},
                    {"parameters", json::object()}}}}
            })},
            {"messages", json::array({{{"role","user"},{"content","weather in rome?"}}})},
        };
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        bool saw_content = false, saw_tool_call = false, saw_finish = false;
        for (auto& ev : g_sse_events) {
            auto& delta = ev["choices"][0]["delta"];
            if (delta.contains("content") && delta["content"] == "Checking now.") saw_content = true;
            if (delta.contains("tool_calls")) {
                saw_tool_call = true;
                CHECK(delta["tool_calls"][0]["index"] == 0);
                CHECK(delta["tool_calls"][0]["function"]["name"] == "get_weather");
                json args = json::parse(
                    delta["tool_calls"][0]["function"]["arguments"].get<std::string>());
                CHECK(args["location"] == "Rome");
            }
            if (ev["choices"][0]["finish_reason"] == "tool_calls") saw_finish = true;
        }
        CHECK(saw_content);
        CHECK(saw_tool_call);
        CHECK(saw_finish);
    }

    // ---- Test 8b: streaming with thinking enabled -- reasoning_content
    // deltas must appear, must carry the think text, and must never appear
    // under `content` instead. ----
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",
            /*1*/ "<think>",
            /*2*/ "pondering rome weather",
            /*3*/ "</think>",
            /*4*/ "Sunny in Rome.",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3, 4};
        std::atomic<long> rc{0};
        json body = {
            {"stream", true},
            {"enable_thinking", true},
            {"messages", json::array({{{"role","user"},{"content","weather in rome?"}}})},
        };
        run_request(tok, "q27-test", /*no_think_srv=*/false, false, true, 100000, 100000, rc,
                   cache, slots, body, true);
        bool saw_reasoning = false, saw_content = false;
        for (auto& ev : g_sse_events) {
            auto& delta = ev["choices"][0]["delta"];
            if (delta.contains("reasoning_content")) {
                saw_reasoning = true;
                CHECK(delta["reasoning_content"] == "pondering rome weather");
                CHECK(!delta.contains("content"));
            }
            if (delta.contains("content")) {
                saw_content = true;
                CHECK(delta["content"] == "Sunny in Rome.");
                CHECK(!delta.contains("reasoning_content"));
            }
        }
        CHECK(saw_reasoning);
        CHECK(saw_content);
    }

    // ---- Test 9: malformed tool call body -> surfaced as content, not
    // silently dropped, finish_reason falls back to stop/length. ----
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",
            /*1*/ "<tool_call>\n",
            /*2*/ "not valid json at all",
            /*3*/ "</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3};
        std::atomic<long> rc{0};
        json body = {
            {"tools", json::array({
                {{"type","function"},{"function",{{"name","get_weather"},{"description","w"},
                    {"parameters", json::object()}}}}
            })},
            {"messages", json::array({{{"role","user"},{"content","weather?"}}})},
        };
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        auto& msg = g_last_response["choices"][0]["message"];
        CHECK(!msg.contains("tool_calls"));
        CHECK(msg["content"].get<std::string>().find("not valid json at all") != std::string::npos);
        CHECK(g_last_response["choices"][0]["finish_reason"] == "stop");
    }

    fprintf(stderr, failures ? "%d FAILURE(S)\n" : "all integration tests passed\n", failures);
    return failures ? 1 : 0;
}
