// Experimental C boundary for a DS4-style native q27 agent.
// The implementation catches every C++ exception; none may cross this ABI.
#ifndef Q27_AGENT_ENGINE_H
#define Q27_AGENT_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q27_agent_engine q27_agent_engine;

typedef struct {
    const char *role;
    const char *content;
    size_t content_len;
} q27_agent_message;

typedef int (*q27_agent_text_sink)(const char *bytes, size_t len, void *opaque);
typedef int (*q27_agent_prefill_sink)(uint32_t prompt_tokens,
                                      uint32_t cached_tokens,
                                      uint32_t prefill_tokens,
                                      void *opaque);
typedef int (*q27_agent_alive_check)(void *opaque);

typedef enum {
    Q27_AGENT_OK = 0,
    Q27_AGENT_CANCELLED = 1,
    Q27_AGENT_REJECTED = 2, // request validation failed; engine remains reusable
    Q27_AGENT_ERROR = 3,    // engine/runtime failure; worker may be poisoned
    Q27_AGENT_STALLED = 4   // bounded output watchdog; resident reuse invalidated
} q27_agent_status;

// Decode sampling. temperature == 0 (or top_k == 1) is pure greedy and must
// stay bitwise-identical to the historical agent path. top_p must be in (0,1];
// use 1.0f when temperature is 0. seed seeds mt19937_64 when sampling.
typedef struct {
    float temperature;
    float top_p;
    uint32_t top_k;
    uint64_t seed;
} q27_agent_sampling;

q27_agent_engine *q27_agent_engine_open(const char *model_path,
                                         const char *tokenizer_path,
                                         uint32_t context,
                                         char *error, size_t error_cap);
void q27_agent_engine_close(q27_agent_engine *engine);

// Optional think budget for streaming `<think>…</think>`. 0 = unlimited
// (default). When non-zero and still inside an open think span after N tokens,
// the engine *force-injects* `</think>\n\n` into the stream + KV (same close
// the `--no-think` prefill uses) and **continues free generation** for the
// answer within remaining max_tokens. Falls back to a length stop only if
// there is no room for the close sequence plus one answer token. Independent
// of the model; packs have no native thinking budget.
void q27_agent_set_max_think_tokens(uint32_t n);
uint32_t q27_agent_max_think_tokens(void);
q27_agent_status q27_agent_engine_tokenizer_sha1(
    q27_agent_engine *engine, unsigned char out_sha1[20],
    char *error, size_t error_cap);

// Renders the complete transcript on every call, but reuses resident state
// only when its exact generated-token ledger is a stable prefix of that render
// and the Metal position agrees. Any mismatch resets and re-prefills. Any
// cancellation/runtime error invalidates reuse for the next call. When
// enable_tools is set, decode grammar-locks registered <tool_call> (masks
// apply to both greedy and temperature sampling so tool JSON stays
// fail-closed). Body tools (write/overwrite/edit/edit_selection) continue
// free-decoding after </tool_call> so a same-turn markdown-fenced body can
// follow. eos_reached distinguishes a natural complete response (including
// EOS as the next token at the exact bound) from max-token/tool-call
// termination. Conservative empty-decode, whitespace-run, identical-token,
// and short-cycle bounds return Q27_AGENT_STALLED before streaming the
// triggering token and invalidate reuse. When prefill_sink is non-null, it
// receives an initial exact cached-prefix update, fixed 96-new-token updates
// at completed GPU boundaries, and a final update before the first generated
// text token. Returning zero cancels safely.
q27_agent_status q27_agent_generate(q27_agent_engine *engine,
                                     const q27_agent_message *messages,
                                     size_t message_count,
                                     int enable_thinking,
                                     int enable_tools,
                                     uint32_t max_tokens,
                                     q27_agent_sampling sampling,
                                     q27_agent_text_sink sink,
                                     q27_agent_prefill_sink prefill_sink,
                                     q27_agent_alive_check alive,
                                     void *opaque,
                                     uint32_t *prompt_tokens,
                                     uint32_t *cached_tokens,
                                     uint32_t *prefill_tokens,
                                     uint32_t *output_tokens,
                                     int *tool_call_complete,
                                     int *eos_reached,
                                     char *error, size_t error_cap);

// Owner-thread-only persistence primitives. Save writes Q27SNAP1 with the
// exact prompt-plus-generated token ledger. Load accepts it only when that
// ledger is a stable prefix of the supplied transcript render and Metal
// position/pending logits agree. A failed load invalidates and resets state.
q27_agent_status q27_agent_engine_save_session(q27_agent_engine *engine,
                                                const char *snapshot_path,
                                                const q27_agent_message *messages,
                                                size_t message_count,
                                                int enable_thinking,
                                                char *error, size_t error_cap);
q27_agent_status q27_agent_engine_load_session(q27_agent_engine *engine,
                                                const char *snapshot_path,
                                                const q27_agent_message *messages,
                                                size_t message_count,
                                                int enable_thinking,
                                                const unsigned char expected_sha256[32],
                                                uint32_t *snapshot_tokens,
                                                char *error, size_t error_cap);
q27_agent_status q27_agent_engine_count_prompt(q27_agent_engine *engine,
                                               const q27_agent_message *messages,
                                               size_t message_count,
                                               int enable_thinking,
                                               uint32_t *prompt_tokens,
                                               char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif
#endif
