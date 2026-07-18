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
typedef int (*q27_agent_alive_check)(void *opaque);

typedef enum {
    Q27_AGENT_OK = 0,
    Q27_AGENT_CANCELLED = 1,
    Q27_AGENT_ERROR = 2
} q27_agent_status;

q27_agent_engine *q27_agent_engine_open(const char *model_path,
                                         const char *tokenizer_path,
                                         uint32_t context,
                                         char *error, size_t error_cap);
void q27_agent_engine_close(q27_agent_engine *engine);

// Phase 0 deliberately re-renders and re-prefills the complete transcript on
// every call. This proves the direct engine boundary before the fork adopts
// append-only session ownership. No server or HTTP protocol is involved.
q27_agent_status q27_agent_generate(q27_agent_engine *engine,
                                     const q27_agent_message *messages,
                                     size_t message_count,
                                     int enable_thinking,
                                     uint32_t max_tokens,
                                     q27_agent_text_sink sink,
                                     q27_agent_alive_check alive,
                                     void *opaque,
                                     uint32_t *prompt_tokens,
                                     uint32_t *output_tokens,
                                     char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif
#endif
