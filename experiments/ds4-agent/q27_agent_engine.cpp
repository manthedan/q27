#include "q27_agent_engine.h"

#include "../../src/metal/metal_engine.h"
#include "../../src/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

struct q27_agent_engine {
    std::unique_ptr<q27::Tokenizer> tokenizer;
    std::shared_ptr<q27::MetalEngine::Shared> shared;
    std::unique_ptr<q27::MetalEngine> session;
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
    size_t message_count, int enable_thinking, uint32_t max_tokens,
    q27_agent_text_sink sink, q27_agent_alive_check alive, void *opaque,
    uint32_t *prompt_tokens, uint32_t *output_tokens,
    char *error, size_t error_cap) {
    if (prompt_tokens) *prompt_tokens = 0;
    if (output_tokens) *output_tokens = 0;
    if (!engine || !messages || message_count == 0 || !sink || !alive || max_tokens == 0) {
        set_error(error, error_cap, "invalid generation arguments");
        return Q27_AGENT_REJECTED;
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
            if (role != "system" && role != "user" && role != "assistant" && role != "tool") {
                set_error(error, error_cap, "unsupported message role");
                return Q27_AGENT_REJECTED;
            }
            chat.emplace_back(role,
                              std::string(messages[i].content, messages[i].content_len));
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
            if (token < 0) throw std::runtime_error("tokenizer returned a negative id");
            ids.push_back(static_cast<uint32_t>(token));
        }
        if (prompt_tokens) *prompt_tokens = static_cast<uint32_t>(ids.size());

        // Phase 0 reset/re-prefill contract. Drive bounded GPU quanta here
        // instead of calling ingest_prompt(), whose whole-prompt operation has
        // no cancellation hook. The final token goes through step() to leave
        // the same resident logits/pending token as ordinary prefill.
        if (!alive(opaque)) return Q27_AGENT_CANCELLED;
        engine->session->reset();
        size_t offset = 0;
        const size_t chunkable = engine->session->chunked_prefill() ? ids.size() - 1 : 0;
        while (chunkable - offset >= 2) {
            const uint32_t count = static_cast<uint32_t>(std::min<size_t>(
                q27::MetalEngine::prefill_chunk_max(), chunkable - offset));
            engine->session->prefill_chunk(ids.data() + offset, count);
            offset += count;
            if (!alive(opaque)) return Q27_AGENT_CANCELLED;
        }
        uint32_t current = 0;
        while (offset < ids.size()) {
            current = engine->session->step(ids[offset++]);
            if (!alive(opaque)) return Q27_AGENT_CANCELLED;
        }

        const uint32_t eos = static_cast<uint32_t>(engine->tokenizer->eos());
        uint32_t produced = 0;
        while (produced < max_tokens && current != eos) {
            const std::string bytes = engine->tokenizer->decode_one(static_cast<int>(current));
            if (!bytes.empty() && !sink(bytes.data(), bytes.size(), opaque)) {
                if (output_tokens) *output_tokens = produced;
                return Q27_AGENT_CANCELLED;
            }
            ++produced;
            if (produced == max_tokens) break;
            if (!alive(opaque)) {
                if (output_tokens) *output_tokens = produced;
                return Q27_AGENT_CANCELLED;
            }
            current = engine->session->step(current);
        }
        if (output_tokens) *output_tokens = produced;
        if (!alive(opaque)) return Q27_AGENT_CANCELLED;
        return Q27_AGENT_OK;
    } catch (const std::exception& e) {
        set_error(error, error_cap, e.what());
    } catch (...) {
        set_error(error, error_cap, "unknown generation failure");
    }
    return Q27_AGENT_ERROR;
}
