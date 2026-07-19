#include "q27_agent_engine.h"
#include "q27_agent_session.h"

#include "../../src/metal/metal_engine.h"
#include "../../src/tokenizer.h"
#include "../../src/toolconstrain.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

struct q27_agent_engine {
    std::unique_ptr<q27::Tokenizer> tokenizer;
    std::shared_ptr<q27::MetalEngine::Shared> shared;
    std::unique_ptr<q27::MetalEngine> session;
    q27::agent::AgentSession agent_session;
    std::vector<std::string> tool_vocab;
    q27::ToolMaskCache tool_masks;
    std::vector<int> tool_host2dev;
    bool tool_masks_ready = false;
    bool poisoned = false;
    char poison_error[256] = {0};
    uint32_t context;

    q27_agent_engine(const char *model_path, const char *tokenizer_path, uint32_t ctx)
        : tokenizer(std::make_unique<q27::Tokenizer>(tokenizer_path)), context(ctx) {
        // Match the CLI/server fail-fast ordering: validate the small tokenizer
        // before mapping or allocating the model.
        if (tokenizer->vocab_size() != q27::MetalEngine::vocabulary_size())
            throw std::runtime_error("tokenizer/model vocabulary mismatch");
        shared = q27::MetalEngine::open_shared(model_path);
        session = std::make_unique<q27::MetalEngine>(shared, ctx, false);
    }
};

static void set_error(char *out, size_t cap, const char *message) noexcept {
    if (!out || cap == 0) return;
    std::snprintf(out, cap, "%s", message ? message : "unknown error");
}

extern "C" q27_agent_engine *q27_agent_engine_open(
    const char *model_path, const char *tokenizer_path, uint32_t context,
    char *error, size_t error_cap) {
    if (!model_path || !*model_path || !tokenizer_path || !*tokenizer_path) {
        set_error(error, error_cap, "model and tokenizer paths are required");
        return nullptr;
    }
    if (context < 2) {
        set_error(error, error_cap, "context must be at least 2 tokens");
        return nullptr;
    }
    try {
        return new q27_agent_engine(model_path, tokenizer_path, context);
    } catch (const std::exception& e) {
        set_error(error, error_cap, e.what());
    } catch (...) {
        set_error(error, error_cap, "unknown engine-open failure");
    }
    return nullptr;
}

extern "C" void q27_agent_engine_close(q27_agent_engine *engine) {
    try {
        delete engine;
    } catch (...) {
        // Destruction is a C ABI boundary and must remain noexcept.
    }
}

extern "C" q27_agent_status q27_agent_generate(
    q27_agent_engine *engine, const q27_agent_message *messages,
    size_t message_count, int enable_thinking, int enable_tools,
    uint32_t max_tokens, q27_agent_text_sink sink,
    q27_agent_alive_check alive, void *opaque,
    uint32_t *prompt_tokens, uint32_t *cached_tokens,
    uint32_t *prefill_tokens, uint32_t *output_tokens,
    int *tool_call_complete, char *error, size_t error_cap) {
    if (prompt_tokens) *prompt_tokens = 0;
    if (cached_tokens) *cached_tokens = 0;
    if (prefill_tokens) *prefill_tokens = 0;
    if (output_tokens) *output_tokens = 0;
    if (tool_call_complete) *tool_call_complete = 0;
    if (!engine || !messages || message_count == 0 || !sink || !alive ||
        max_tokens == 0) {
        set_error(error, error_cap, "invalid generation arguments");
        return Q27_AGENT_REJECTED;
    }
    if (engine->poisoned) {
        set_error(error, error_cap, engine->poison_error[0] ?
                  engine->poison_error : "engine was poisoned after final-token ingestion");
        return Q27_AGENT_ERROR;
    }

    try {
        std::vector<std::pair<std::string,std::string>> chat;
        chat.reserve(message_count);
        for (size_t i = 0; i < message_count; ++i) {
            if (!messages[i].role || !messages[i].content) {
                set_error(error, error_cap, "message role/content must not be null");
                return Q27_AGENT_REJECTED;
            }
            const std::string role(messages[i].role);
            if (role != "system" && role != "user" && role != "assistant" &&
                role != "tool") {
                set_error(error, error_cap, "unsupported message role");
                return Q27_AGENT_REJECTED;
            }
            chat.emplace_back(role,
                              std::string(messages[i].content,
                                          messages[i].content_len));
        }

        const std::vector<int> encoded =
            engine->tokenizer->apply_chat_template(chat, enable_thinking != 0);
        if (encoded.empty()) {
            set_error(error, error_cap, "rendered prompt is empty");
            return Q27_AGENT_REJECTED;
        }
        if (encoded.size() > engine->context ||
            static_cast<uint64_t>(encoded.size()) + max_tokens >
                static_cast<uint64_t>(engine->context) + 1) {
            set_error(error, error_cap, "prompt plus max_tokens exceeds context");
            return Q27_AGENT_REJECTED;
        }

        std::vector<uint32_t> ids;
        ids.reserve(encoded.size());
        for (int token : encoded) {
            if (token < 0)
                throw std::runtime_error("tokenizer returned a negative id");
            ids.push_back(static_cast<uint32_t>(token));
        }
        if (prompt_tokens) *prompt_tokens = static_cast<uint32_t>(ids.size());

        const q27::agent::AgentSession::Plan plan =
            engine->agent_session.plan(ids, engine->session->position());
        if (cached_tokens)
            *cached_tokens = static_cast<uint32_t>(plan.cached_tokens);

        auto cancelled = [&]() {
            engine->agent_session.invalidate();
            return Q27_AGENT_CANCELLED;
        };
        if (!alive(opaque)) return cancelled();

        size_t offset = 0;
        uint32_t current = 0;
        if (plan.reset) {
            // Prefix or position mismatch: discard both ledgers before the
            // first GPU mutation, then establish a fresh exact prompt.
            engine->agent_session.invalidate();
            engine->session->reset();
        } else {
            offset = plan.append_offset;
            if (plan.finalize_pending) {
                current = engine->session->step(ids[offset - 1]);
                if (prefill_tokens) ++*prefill_tokens;
                if (!engine->agent_session.mark_pending_encoded())
                    throw std::runtime_error("agent session pending-token mismatch");
                if (!alive(opaque)) return cancelled();
            }
        }

        // Ingest only the unstable suffix. As in MetalEngine::prefill, leave
        // the final prompt token on step() so resident logits/pending are
        // exact. Every GPU call is a bounded cancellation quantum.
        const size_t chunkable = engine->session->chunked_prefill() ?
                                 ids.size() - 1 : 0;
        while (offset < chunkable && chunkable - offset >= 2) {
            const uint32_t count = static_cast<uint32_t>(std::min<size_t>(
                q27::MetalEngine::prefill_chunk_max(), chunkable - offset));
            engine->session->prefill_chunk(ids.data() + offset, count);
            offset += count;
            if (prefill_tokens) *prefill_tokens += count;
            if (!alive(opaque)) return cancelled();
        }
        while (offset < ids.size()) {
            current = engine->session->step(ids[offset++]);
            if (prefill_tokens) ++*prefill_tokens;
            if (!alive(opaque)) return cancelled();
        }
        if (offset == plan.append_offset && !plan.finalize_pending)
            current = engine->session->pending_from_logits();

        // The GPU and ledger now agree at the complete rendered prompt.
        engine->agent_session.commit_prompt(ids);

        using ToolConstrainer =
            q27::BasicToolConstrainer<q27::MetalEngine, q27::Tokenizer>;
        std::optional<ToolConstrainer> constrainer;
        if (enable_tools) {
            if (!engine->tool_masks_ready) {
                const int closer = engine->tokenizer->token_id("</tool_call>");
                if (closer < 0)
                    throw std::runtime_error("tokenizer lacks </tool_call>");
                engine->tool_vocab = engine->tokenizer->vocab_bytes();
                engine->tool_masks.init(&engine->tool_vocab, closer);
                engine->tool_masks_ready = true;
            }
            constrainer.emplace();
            constrainer->eng = engine->session.get();
            constrainer->tok = engine->tokenizer.get();
            constrainer->cache = &engine->tool_masks;
            constrainer->host2dev = &engine->tool_host2dev;
            constrainer->enabled = true;
            constrainer->begin({"read", "search", "edit", "shell"});
        }
        struct ConstraintCleanup {
            q27_agent_engine *engine;
            std::optional<ToolConstrainer> *constrainer;
            ~ConstraintCleanup() {
                try {
                    if (constrainer->has_value()) (*constrainer)->end();
                    engine->session->set_tool_constraint(-1);
                } catch (...) {}
            }
        } constraint_cleanup{engine, &constrainer};

        const uint32_t eos = static_cast<uint32_t>(engine->tokenizer->eos());
        uint32_t produced = 0;
        while (produced < max_tokens && current != eos) {
            if (!alive(opaque)) {
                if (output_tokens) *output_tokens = produced;
                return cancelled();
            }

            bool call_closed = false;
            if (constrainer) {
                const long engaged_before = constrainer->engaged;
                const bool active_before = constrainer->active;
                const int token = static_cast<int>(current);
                (void)constrainer->scan_round(&token, 1);
                constrainer->on_id(token);
                const bool newly_engaged =
                    constrainer->engaged != engaged_before;
                if (constrainer->pool_dead ||
                    (newly_engaged && !constrainer->active &&
                     !constrainer->tg.closed()))
                    throw std::runtime_error(
                        "tool grammar could not remain fail-closed");
                if (constrainer->active) {
                    constrainer->apply(constrainer->tg);
                    if (constrainer->pool_dead || !constrainer->active)
                        throw std::runtime_error(
                            "tool grammar mask pool exhausted");
                }
                call_closed = !constrainer->active &&
                              constrainer->tg.closed() &&
                              (active_before || newly_engaged);
            }

            const std::string bytes =
                engine->tokenizer->decode_one(static_cast<int>(current));
            if (!bytes.empty() && !sink(bytes.data(), bytes.size(), opaque)) {
                if (output_tokens) *output_tokens = produced;
                return cancelled();
            }
            ++produced;
            // The callback is irreversible: publish accounting before any
            // allocation, Metal step, or other bookkeeping can fail.
            if (output_tokens) *output_tokens = produced;
            if (call_closed && tool_call_complete) *tool_call_complete = 1;

            // A closed tool call is a semantic terminal: never generate prose
            // after </tool_call>. Like max_tokens, the visible final token is
            // irreversible; resident finalization is best-effort bookkeeping.
            if (produced == max_tokens || call_closed) {
                try {
                    if (!engine->agent_session.record_emitted(current))
                        throw std::runtime_error(
                            "agent session emitted-token mismatch");
                } catch (...) {
                    engine->agent_session.invalidate();
                    break;
                }
                // Final-token Metal ingestion is also bookkeeping:
                // cancellation invalidates reuse, while a runtime failure
                // poisons the next command, but neither reverses this result.
                if (engine->session->position() < engine->context) {
                    if (!alive(opaque)) {
                        engine->agent_session.invalidate();
                        break;
                    }
                    try {
                        current = engine->session->step(current);
                        if (!engine->agent_session.mark_pending_encoded())
                            throw std::runtime_error(
                                "agent session final-token mismatch");
                    } catch (const std::exception& e) {
                        engine->agent_session.invalidate();
                        engine->poisoned = true;
                        set_error(engine->poison_error,
                                  sizeof(engine->poison_error), e.what());
                        break;
                    } catch (...) {
                        engine->agent_session.invalidate();
                        engine->poisoned = true;
                        set_error(engine->poison_error,
                                  sizeof(engine->poison_error),
                                  "unknown final-token ingestion failure");
                        break;
                    }
                    if (!alive(opaque)) engine->agent_session.invalidate();
                }
                break;
            }
            if (!engine->agent_session.record_emitted(current))
                throw std::runtime_error("agent session emitted-token mismatch");
            if (!alive(opaque)) {
                if (output_tokens) *output_tokens = produced;
                return cancelled();
            }
            current = engine->session->step(current);
            if (!engine->agent_session.mark_pending_encoded())
                throw std::runtime_error("agent session token-step mismatch");
            if (!alive(opaque)) {
                if (output_tokens) *output_tokens = produced;
                return cancelled();
            }
        }
        if (output_tokens) *output_tokens = produced;
        return Q27_AGENT_OK;
    } catch (const std::exception& e) {
        engine->agent_session.invalidate();
        set_error(error, error_cap, e.what());
    } catch (...) {
        engine->agent_session.invalidate();
        set_error(error, error_cap, "unknown generation failure");
    }
    return Q27_AGENT_ERROR;
}
