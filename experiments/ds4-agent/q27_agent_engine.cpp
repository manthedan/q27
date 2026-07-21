#include "q27_agent_engine.h"
#include "q27_agent_protocol.h"
#include "q27_agent_session.h"
#include "q27_agent_stall.h"

#include "../../src/metal/metal_engine.h"
#include "../../src/sampling.h"
#include "../../src/tokenizer.h"
#include "../../src/toolconstrain.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <stdexcept>
#include <utility>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <CommonCrypto/CommonDigest.h>

static bool sha1_fd(int fd, unsigned char digest[20]) {
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0)
        return false;
    CC_SHA1_CTX sha;
    CC_SHA1_Init(&sha);
    std::vector<unsigned char> chunk(1024 * 1024);
    off_t offset = 0;
    while (offset < st.st_size) {
        const size_t wanted = static_cast<size_t>(std::min<off_t>(
            static_cast<off_t>(chunk.size()), st.st_size - offset));
        ssize_t got;
        do { got = pread(fd, chunk.data(), wanted, offset); }
        while (got < 0 && errno == EINTR);
        if (got <= 0 || static_cast<size_t>(got) != wanted) return false;
        CC_SHA1_Update(&sha, chunk.data(), static_cast<CC_LONG>(got));
        offset += got;
    }
    CC_SHA1_Final(digest, &sha);
    return true;
}

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
    unsigned char tokenizer_sha1[20] = {0};
    uint32_t context;
    uint32_t mtp_width = 0; // 0 = serial; 2..12 = free-decode MTP quanta

    q27_agent_engine(const char *model_path, const char *tokenizer_path, uint32_t ctx)
        : context(ctx) {
        // Pin one tokenizer inode across identity hashing and parsing. The
        // /dev/fd open gives Tokenizer its own stream over these exact bytes,
        // so an atomic pathname replacement cannot split manifest identity
        // from the tokenizer actually used by this engine.
        const int tokenizer_fd = open(tokenizer_path, O_RDONLY | O_CLOEXEC);
        if (tokenizer_fd < 0 || !sha1_fd(tokenizer_fd, tokenizer_sha1)) {
            if (tokenizer_fd >= 0) close(tokenizer_fd);
            throw std::runtime_error("cannot pin/hash tokenizer artifact");
        }
        char pinned_path[64];
        std::snprintf(pinned_path, sizeof(pinned_path), "/dev/fd/%d", tokenizer_fd);
        try {
            tokenizer = std::make_unique<q27::Tokenizer>(pinned_path);
            close(tokenizer_fd);
        } catch (...) {
            close(tokenizer_fd);
            throw;
        }
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

static void reset_agent_state(q27_agent_engine *engine) noexcept {
    if (!engine) return;
    engine->agent_session.invalidate();
    try { engine->session->reset(); } catch (...) {}
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

extern "C" void q27_agent_engine_set_mtp_width(q27_agent_engine *engine,
                                               uint32_t width) {
    if (!engine) return;
    // 0 disables; 1 is not a valid MTP width (needs pending+draft). Clamp.
    if (width == 1) width = 0;
    if (width > 12) width = 12;
    engine->mtp_width = width;
}

extern "C" uint32_t q27_agent_engine_mtp_width(const q27_agent_engine *engine) {
    return engine ? engine->mtp_width : 0;
}

extern "C" void q27_agent_engine_close(q27_agent_engine *engine) {
    try {
        delete engine;
    } catch (...) {
        // Destruction is a C ABI boundary and must remain noexcept.
    }
}

extern "C" q27_agent_status q27_agent_engine_tokenizer_sha1(
    q27_agent_engine *engine, unsigned char out_sha1[20],
    char *error, size_t error_cap) {
    if (!engine || !out_sha1) {
        set_error(error, error_cap, "invalid tokenizer identity request");
        return Q27_AGENT_REJECTED;
    }
    std::memcpy(out_sha1, engine->tokenizer_sha1, 20);
    return Q27_AGENT_OK;
}

static bool render_messages(q27_agent_engine *engine,
                            const q27_agent_message *messages,
                            size_t message_count, int enable_thinking,
                            std::vector<uint32_t>& ids,
                            char *error, size_t error_cap) {
    if (!engine || !messages || !message_count) {
        set_error(error, error_cap, "invalid transcript arguments");
        return false;
    }
    std::vector<std::pair<std::string,std::string>> chat;
    chat.reserve(message_count);
    for (size_t i = 0; i < message_count; ++i) {
        if (!messages[i].role || !messages[i].content) {
            set_error(error, error_cap, "message role/content must not be null");
            return false;
        }
        const std::string role(messages[i].role);
        if (role != "system" && role != "user" && role != "assistant" &&
            role != "tool") {
            set_error(error, error_cap, "unsupported message role");
            return false;
        }
        chat.emplace_back(role, std::string(messages[i].content,
                                            messages[i].content_len));
    }
    const std::vector<int> encoded =
        engine->tokenizer->apply_chat_template(chat, enable_thinking != 0);
    if (encoded.empty() || encoded.size() > UINT32_MAX) {
        set_error(error, error_cap, "rendered prompt is empty or too large");
        return false;
    }
    ids.clear();
    ids.reserve(encoded.size());
    for (int token : encoded) {
        if (token < 0)
            throw std::runtime_error("tokenizer returned a negative id");
        ids.push_back(static_cast<uint32_t>(token));
    }
    return true;
}

static bool render_closed_messages(q27_agent_engine *engine,
                                   const q27_agent_message *messages,
                                   size_t message_count,
                                   std::vector<uint32_t>& ids,
                                   char *error, size_t error_cap) {
    if (!engine || !messages || !message_count) {
        set_error(error, error_cap, "invalid transcript arguments");
        return false;
    }
    std::vector<std::pair<std::string,std::string>> chat;
    chat.reserve(message_count);
    for (size_t i = 0; i < message_count; ++i) {
        if (!messages[i].role || !messages[i].content) {
            set_error(error, error_cap, "message role/content must not be null");
            return false;
        }
        const std::string role(messages[i].role);
        if (role != "system" && role != "user" && role != "assistant" &&
            role != "tool") {
            set_error(error, error_cap, "unsupported message role");
            return false;
        }
        chat.emplace_back(role, std::string(messages[i].content,
                                            messages[i].content_len));
    }
    const std::vector<int> encoded = engine->tokenizer->apply_chat_prefix(chat);
    if (encoded.empty() || encoded.size() > UINT32_MAX) {
        set_error(error, error_cap, "closed transcript prefix is empty or too large");
        return false;
    }
    ids.clear(); ids.reserve(encoded.size());
    for (int token : encoded) {
        if (token < 0)
            throw std::runtime_error("tokenizer returned a negative id");
        ids.push_back(static_cast<uint32_t>(token));
    }
    return true;
}

static void prefill_canonical(q27_agent_engine *engine,
                              const std::vector<uint32_t>& ids) {
    if (ids.empty() || ids.size() > engine->context)
        throw std::runtime_error("closed transcript prefix exceeds context");
    engine->agent_session.invalidate();
    engine->session->reset();
    size_t offset = 0;
    const size_t chunkable = engine->session->chunked_prefill() ?
                             ids.size() - 1 : 0;
    while (offset < chunkable && chunkable - offset >= 2) {
        const uint32_t count = static_cast<uint32_t>(std::min<size_t>(
            q27::MetalEngine::prefill_chunk_max(), chunkable - offset));
        engine->session->prefill_chunk(ids.data() + offset, count);
        offset += count;
    }
    while (offset < ids.size())
        (void)engine->session->step(ids[offset++]);
    engine->agent_session.commit_prompt(ids);
}

extern "C" q27_agent_status q27_agent_engine_count_prompt(
    q27_agent_engine *engine, const q27_agent_message *messages,
    size_t message_count, int enable_thinking, uint32_t *prompt_tokens,
    char *error, size_t error_cap) {
    if (prompt_tokens) *prompt_tokens = 0;
    if (!prompt_tokens) {
        set_error(error, error_cap, "prompt token output is required");
        return Q27_AGENT_REJECTED;
    }
    try {
        std::vector<uint32_t> ids;
        if (!render_messages(engine, messages, message_count, enable_thinking,
                             ids, error, error_cap))
            return Q27_AGENT_REJECTED;
        *prompt_tokens = static_cast<uint32_t>(ids.size());
        return Q27_AGENT_OK;
    } catch (const std::exception& e) {
        set_error(error, error_cap, e.what());
    } catch (...) {
        set_error(error, error_cap, "unknown prompt-count failure");
    }
    return Q27_AGENT_ERROR;
}

extern "C" q27_agent_status q27_agent_engine_save_session(
    q27_agent_engine *engine, const char *snapshot_path,
    const q27_agent_message *messages, size_t message_count,
    int enable_thinking, char *error, size_t error_cap) {
    if (!engine || !snapshot_path || !*snapshot_path || !messages ||
        !message_count) {
        set_error(error, error_cap, "invalid session-save arguments");
        return Q27_AGENT_REJECTED;
    }
    if (engine->poisoned) {
        set_error(error, error_cap, "poisoned engine cannot publish a session");
        return Q27_AGENT_ERROR;
    }
    try {
        std::vector<uint32_t> rendered;
        if (!render_messages(engine, messages, message_count, enable_thinking,
                             rendered, error, error_cap))
            return Q27_AGENT_REJECTED;
        const auto ledger_matches_render = [&]() {
            const auto& ledger = engine->agent_session.tokens();
            return engine->agent_session.valid() && !ledger.empty() &&
                ledger.size() <= rendered.size() &&
                engine->agent_session.encoded_count() ==
                    engine->session->position() &&
                std::equal(ledger.begin(), ledger.end(), rendered.begin());
        };
        if (!ledger_matches_render()) {
            // Decoded generated bytes can retokenize differently. Never
            // publish such a ledger: prefill the canonical closed-message
            // boundary, which remains an exact prefix after any future turn.
            std::vector<uint32_t> closed;
            if (!render_closed_messages(engine, messages, message_count,
                                        closed, error, error_cap))
                return Q27_AGENT_REJECTED;
            prefill_canonical(engine, closed);
            if (!ledger_matches_render())
                throw std::runtime_error(
                    "canonical transcript prefix did not match full render");
        }
        const auto& ledger = engine->agent_session.tokens();
        engine->session->save_state(snapshot_path, ledger.data(),
                                    static_cast<uint32_t>(ledger.size()), true);
        return Q27_AGENT_OK;
    } catch (const std::exception& e) {
        set_error(error, error_cap, e.what());
    } catch (...) {
        set_error(error, error_cap, "unknown session-save failure");
    }
    return Q27_AGENT_ERROR;
}

static bool snapshot_sha256(int fd, unsigned char digest[32]) {
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0)
        return false;
    CC_SHA256_CTX sha;
    CC_SHA256_Init(&sha);
    std::vector<unsigned char> chunk(1024 * 1024);
    off_t offset = 0;
    while (offset < st.st_size) {
        const size_t wanted = static_cast<size_t>(std::min<off_t>(
            static_cast<off_t>(chunk.size()), st.st_size - offset));
        ssize_t got;
        do { got = pread(fd, chunk.data(), wanted, offset); }
        while (got < 0 && errno == EINTR);
        if (got <= 0 || static_cast<size_t>(got) != wanted) return false;
        CC_SHA256_Update(&sha, chunk.data(), static_cast<CC_LONG>(got));
        offset += got;
    }
    CC_SHA256_Final(digest, &sha);
    return true;
}

extern "C" q27_agent_status q27_agent_engine_load_session(
    q27_agent_engine *engine, const char *snapshot_path,
    const q27_agent_message *messages, size_t message_count,
    int enable_thinking, const unsigned char expected_sha256[32],
    uint32_t *snapshot_tokens,
    char *error, size_t error_cap) {
    if (snapshot_tokens) *snapshot_tokens = 0;
    if (!engine || !snapshot_path || !*snapshot_path || !expected_sha256 ||
        !snapshot_tokens) {
        reset_agent_state(engine);
        set_error(error, error_cap, "invalid session-load arguments");
        return Q27_AGENT_REJECTED;
    }
    try {
        std::vector<uint32_t> rendered;
        if (!render_messages(engine, messages, message_count, enable_thinking,
                             rendered, error, error_cap)) {
            reset_agent_state(engine);
            return Q27_AGENT_REJECTED;
        }
        const int snapshot_fd = open(snapshot_path,
                                     O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        if (snapshot_fd < 0)
            throw std::runtime_error("cannot pin session snapshot");
        struct SnapshotFd {
            int value;
            ~SnapshotFd() { if (value >= 0) close(value); }
        } pinned{snapshot_fd};
        unsigned char actual_sha256[32];
        if (!snapshot_sha256(pinned.value, actual_sha256) ||
            memcmp(actual_sha256, expected_sha256, sizeof(actual_sha256)))
            throw std::runtime_error(
                "snapshot payload digest does not match durable manifest");
        const q27::MetalEngine::SnapshotInfo info =
            q27::MetalEngine::peek_snapshot_fd(pinned.value, snapshot_path);
        if (!info.logits_resident || info.tokens.empty() ||
            info.tokens.size() > rendered.size() ||
            (info.tokens.size() != info.position &&
             info.tokens.size() != static_cast<size_t>(info.position) + 1) ||
            !std::equal(info.tokens.begin(), info.tokens.end(), rendered.begin())) {
            reset_agent_state(engine);
            set_error(error, error_cap,
                      "snapshot token ledger is not an exact transcript prefix");
            return Q27_AGENT_REJECTED;
        }
        const bool pending = info.tokens.size() ==
                             static_cast<size_t>(info.position) + 1;
        const uint32_t restored =
            engine->session->load_state_fd(pinned.value, snapshot_path);
        if (restored != info.position || restored > engine->context)
            throw std::runtime_error("snapshot restored an unexpected Metal position");
        if (pending && engine->session->pending_from_logits() != info.tokens.back())
            throw std::runtime_error("snapshot pending logits do not match token ledger");
        if (!engine->agent_session.restore(info.tokens, pending))
            throw std::runtime_error("snapshot token ledger could not be restored");
        engine->poisoned = false;
        engine->poison_error[0] = '\0';
        *snapshot_tokens = static_cast<uint32_t>(info.tokens.size());
        return Q27_AGENT_OK;
    } catch (const std::exception& e) {
        reset_agent_state(engine);
        set_error(error, error_cap, e.what());
    } catch (...) {
        reset_agent_state(engine);
        set_error(error, error_cap, "unknown session-load failure");
    }
    return Q27_AGENT_ERROR;
}

extern "C" q27_agent_status q27_agent_generate(
    q27_agent_engine *engine, const q27_agent_message *messages,
    size_t message_count, int enable_thinking, int enable_tools,
    uint32_t max_tokens, q27_agent_sampling sampling,
    q27_agent_text_sink sink,
    q27_agent_prefill_sink prefill_sink,
    q27_agent_alive_check alive, void *opaque,
    uint32_t *prompt_tokens, uint32_t *cached_tokens,
    uint32_t *prefill_tokens, uint32_t *output_tokens,
    int *tool_call_complete, int *eos_reached,
    char *error, size_t error_cap) {
    if (prompt_tokens) *prompt_tokens = 0;
    if (cached_tokens) *cached_tokens = 0;
    if (prefill_tokens) *prefill_tokens = 0;
    if (output_tokens) *output_tokens = 0;
    if (tool_call_complete) *tool_call_complete = 0;
    if (eos_reached) *eos_reached = 0;
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
        q27::SamplingParams params;
        params.temperature = sampling.temperature;
        params.top_p = sampling.top_p;
        params.top_k = sampling.top_k;
        params.seed = sampling.seed;
        q27::validate_sampling(params);
        // temperature 0 or top_k 1: keep the historical pure-argmax path so
        // tool grammar + session reuse stay bitwise-stable with prior agents.
        const bool use_sample =
            params.temperature > 0.0f && params.top_k != 1;
        std::mt19937_64 rng(params.seed);

        std::vector<uint32_t> ids;
        if (!render_messages(engine, messages, message_count, enable_thinking,
                             ids, error, error_cap))
            return Q27_AGENT_REJECTED;
        if (ids.size() > engine->context ||
            static_cast<uint64_t>(ids.size()) + max_tokens >
                static_cast<uint64_t>(engine->context) + 1) {
            set_error(error, error_cap, "prompt plus max_tokens exceeds context");
            return Q27_AGENT_REJECTED;
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
        uint32_t prefilled = 0;
        auto record_prefill = [&](uint32_t count) {
            prefilled += count;
            if (prefill_tokens) *prefill_tokens = prefilled;
        };
        auto publish_prefill = [&]() {
            return !prefill_sink || prefill_sink(
                static_cast<uint32_t>(ids.size()),
                static_cast<uint32_t>(plan.cached_tokens),
                prefilled, opaque);
        };
        if (!publish_prefill()) return cancelled();
        uint32_t last_reported_prefill = 0;

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
                record_prefill(1);
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
            record_prefill(count);
            if (!alive(opaque)) return cancelled();
            const uint32_t completed = prefilled;
            if (completed - last_reported_prefill >=
                q27::MetalEngine::prefill_chunk_max()) {
                if (!publish_prefill()) return cancelled();
                last_reported_prefill = completed;
            }
        }
        while (offset < ids.size()) {
            current = engine->session->step(ids[offset++]);
            record_prefill(1);
            if (!alive(opaque)) return cancelled();
            const uint32_t completed = prefilled;
            if (completed - last_reported_prefill >=
                q27::MetalEngine::prefill_chunk_max()) {
                if (!publish_prefill()) return cancelled();
                last_reported_prefill = completed;
            }
        }
        if (prefilled != last_reported_prefill &&
            !publish_prefill())
            return cancelled();
        // First generated token: sample from resident logits when requested.
        // Greedy keeps the old argmax path (pending_from_logits when prefill
        // left logits without a step return; otherwise last step's argmax).
        if (use_sample) {
            current = engine->session->sample_from_logits(params, rng);
        } else if (offset == plan.append_offset && !plan.finalize_pending) {
            current = engine->session->pending_from_logits();
        }

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
            const char *const *registered_names = nullptr;
            const size_t registered_count =
                q27_agent_tool_names(&registered_names);
            if (!registered_names || !registered_count)
                throw std::runtime_error("native tool registry is unavailable");
            std::vector<std::string> names;
            names.reserve(registered_count);
            for (size_t i = 0; i < registered_count; ++i)
                names.emplace_back(registered_names[i]);
            constrainer->begin(names);
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
        bool stopped_for_tool_call = false;
        bool awaiting_fenced_body = false;
        q27::agent::StallWatcher stall_watcher;
        // Adaptive live width for free-decode MTP (mirrors metal_server).
        uint32_t live_width =
            engine->mtp_width >= 2 ? std::min(engine->mtp_width, 4u) : 0;
        const bool sample_plain = getenv("Q27_SAMPLE_PLAIN") != nullptr;

        // Stream one non-EOS token: tool scan → stall → sink → produced++.
        // Returns: 0 ok continue, 1 stop (tool close / max), -1 cancelled,
        // -2 stalled. Sets call_closed / awaiting_fenced_body as needed.
        auto emit_one = [&](uint32_t token, bool *call_closed_out) -> int {
            if (call_closed_out) *call_closed_out = false;
            if (!alive(opaque)) return -1;
            bool call_closed = false;
            if (constrainer) {
                const long engaged_before = constrainer->engaged;
                const bool active_before = constrainer->active;
                const int t = static_cast<int>(token);
                (void)constrainer->scan_round(&t, 1);
                constrainer->on_id(t);
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
                if (call_closed &&
                    q27_agent_tool_name_expects_body(
                        constrainer->tg.tool_name().c_str())) {
                    // Body tools continue free-decoding for a same-turn
                    // markdown fence. Disable further engage so a literal
                    // <tool_call> inside file content cannot re-arm masks.
                    awaiting_fenced_body = true;
                    constrainer->enabled = false;
                    constrainer->active = false;
                    try {
                        engine->session->set_tool_constraint(-1);
                    } catch (...) {
                        // Mask clear is best-effort; free decode continues.
                    }
                }
            }
            if (call_closed_out) *call_closed_out = call_closed;

            const std::string bytes =
                engine->tokenizer->decode_one(static_cast<int>(token));
            if (stall_watcher.observe(token, bytes) !=
                q27::agent::StallReason::None) {
                // Prior streamed bytes cannot be withdrawn. The triggering
                // token is not streamed or recorded, and all resident reuse
                // authority is discarded before reporting the distinct
                // terminal reason.
                engine->agent_session.invalidate();
                if (output_tokens) *output_tokens = produced;
                set_error(error, error_cap, "generation stalled");
                return -2;
            }
            if (!bytes.empty() && !sink(bytes.data(), bytes.size(), opaque))
                return -1;
            ++produced;
            // The callback is irreversible: publish accounting before any
            // allocation, Metal step, or other bookkeeping can fail.
            if (output_tokens) *output_tokens = produced;
            if (call_closed && tool_call_complete) *tool_call_complete = 1;
            const bool stop_for_closed_call =
                call_closed && !awaiting_fenced_body;
            if (stop_for_closed_call) stopped_for_tool_call = true;
            if (produced == max_tokens || stop_for_closed_call) return 1;
            return 0;
        };

        auto finalize_last = [&](uint32_t token, bool already_encoded) -> void {
            try {
                if (!engine->agent_session.record_emitted(token))
                    throw std::runtime_error(
                        "agent session emitted-token mismatch");
            } catch (...) {
                engine->agent_session.invalidate();
                return;
            }
            if (already_encoded) {
                if (!engine->agent_session.mark_pending_encoded())
                    engine->agent_session.invalidate();
                return;
            }
            // Final-token Metal ingestion is bookkeeping: cancellation
            // invalidates reuse, while a runtime failure poisons the next
            // command, but neither reverses this result.
            if (engine->session->position() < engine->context) {
                if (!alive(opaque)) {
                    engine->agent_session.invalidate();
                    return;
                }
                try {
                    (void)engine->session->step(token);
                    if (!engine->agent_session.mark_pending_encoded())
                        throw std::runtime_error(
                            "agent session final-token mismatch");
                } catch (const std::exception& e) {
                    engine->agent_session.invalidate();
                    engine->poisoned = true;
                    set_error(engine->poison_error,
                              sizeof(engine->poison_error), e.what());
                } catch (...) {
                    engine->agent_session.invalidate();
                    engine->poisoned = true;
                    set_error(engine->poison_error,
                              sizeof(engine->poison_error),
                              "unknown final-token ingestion failure");
                }
                if (!alive(opaque)) engine->agent_session.invalidate();
            }
        };

        while (produced < max_tokens && current != eos) {
            if (!alive(opaque)) {
                if (output_tokens) *output_tokens = produced;
                return cancelled();
            }

            const uint32_t remaining = max_tokens - produced;
            // MTP only when tool grammar cannot open/close mid-burst:
            // active masks need serial sample (fail-closed), and an *enabled*
            // constrainer can still engage on free text — a multi-token MTP
            // draft would select unconstrained tokens after `<tool_call>`.
            // After body-tool close, awaiting_fenced_body disables engage so
            // MTP is safe for long fenced payloads. Matches metal_server's
            // "no MTP under tool constraint" boundary (codex P1).
            const bool tools_masking =
                constrainer.has_value() && constrainer->active;
            const bool tools_may_engage =
                constrainer.has_value() && constrainer->enabled &&
                !awaiting_fenced_body;
            const bool can_mtp =
                live_width >= 2 && engine->session->has_mtp() &&
                engine->session->chunked_prefill() && remaining >= 2 &&
                !tools_masking && !tools_may_engage && !sample_plain;

            if (can_mtp) {
                std::vector<uint32_t> committed;
                const uint32_t pos_before = engine->session->position();
                uint32_t next_pending;
                if (use_sample) {
                    next_pending = engine->session->mtp_sample_round(
                        current, remaining, eos, engine->mtp_width, live_width,
                        params, rng, committed);
                } else {
                    next_pending = engine->session->mtp_round(
                        current, remaining, eos, engine->mtp_width, live_width,
                        committed);
                }
                const uint32_t encoded =
                    engine->session->position() - pos_before;
                bool stop_after_burst = false;
                for (size_t i = 0; i < committed.size(); ++i) {
                    const uint32_t tok = committed[i];
                    if (tok == eos) {
                        // Do not stream EOS (serial loop also exits before).
                        current = eos;
                        stop_after_burst = true;
                        break;
                    }
                    bool call_closed = false;
                    const int er = emit_one(tok, &call_closed);
                    if (er == -2) return Q27_AGENT_STALLED;
                    if (er == -1) {
                        if (output_tokens) *output_tokens = produced;
                        return cancelled();
                    }
                    if (!engine->agent_session.record_emitted(tok))
                        throw std::runtime_error(
                            "agent session emitted-token mismatch");
                    if (i < encoded) {
                        if (!engine->agent_session.mark_pending_encoded())
                            throw std::runtime_error(
                                "agent session token-step mismatch");
                    } else {
                        // Remaining-edge final lane: not encoded by mtp_*.
                        (void)engine->session->step(tok);
                        if (!engine->agent_session.mark_pending_encoded())
                            throw std::runtime_error(
                                "agent session token-step mismatch");
                    }
                    // Non-body tool close: stop at the closing token like the
                    // serial path. Do not stream post-</tool_call> junk from
                    // the rest of the burst (parser rejects it). Any already-
                    // encoded trailing Metal rows desync the ledger — drop
                    // reuse fail-closed (codex P1).
                    if (er == 1 && call_closed && !awaiting_fenced_body) {
                        if (i + 1 < committed.size())
                            engine->agent_session.invalidate();
                        stop_after_burst = true;
                        break;
                    }
                    if (er == 1) {
                        // max_tokens (or body-tool path): drain remaining so
                        // encoded rows stay ledger-aligned when possible.
                        stop_after_burst = true;
                    }
                    if (!alive(opaque)) {
                        if (output_tokens) *output_tokens = produced;
                        return cancelled();
                    }
                }
                if (stop_after_burst || current == eos) break;
                current = next_pending;
                if (!alive(opaque)) {
                    if (output_tokens) *output_tokens = produced;
                    return cancelled();
                }
                continue;
            }

            // ---- serial quantum (tools active, no MTP, or remaining < 2) ----
            bool call_closed = false;
            const int er = emit_one(current, &call_closed);
            if (er == -2) return Q27_AGENT_STALLED;
            if (er == -1) {
                if (output_tokens) *output_tokens = produced;
                return cancelled();
            }
            if (er == 1) {
                finalize_last(current, /*already_encoded=*/false);
                break;
            }
            if (!engine->agent_session.record_emitted(current))
                throw std::runtime_error("agent session emitted-token mismatch");
            if (!alive(opaque)) {
                if (output_tokens) *output_tokens = produced;
                return cancelled();
            }
            // Encode the emitted token. Greedy uses step's returned argmax as
            // the next id; sampling advances with step then draws from the
            // (possibly tool-masked) logits — same mask surface as greedy.
            if (use_sample) {
                (void)engine->session->step(current);
                if (!engine->agent_session.mark_pending_encoded())
                    throw std::runtime_error(
                        "agent session token-step mismatch");
                if (!alive(opaque)) {
                    if (output_tokens) *output_tokens = produced;
                    return cancelled();
                }
                current = engine->session->sample_from_logits(params, rng);
            } else {
                current = engine->session->step(current);
                if (!engine->agent_session.mark_pending_encoded())
                    throw std::runtime_error(
                        "agent session token-step mismatch");
            }
            if (!alive(opaque)) {
                if (output_tokens) *output_tokens = produced;
                return cancelled();
            }
        }
        if (output_tokens) *output_tokens = produced;
        if (eos_reached)
            *eos_reached = current == eos && !stopped_for_tool_call;
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
