// Dedicated engine-owner thread for the DS4-style q27 agent experiment.
#ifndef Q27_AGENT_WORKER_H
#define Q27_AGENT_WORKER_H

#include "q27_agent_engine.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q27_agent_worker q27_agent_worker;

typedef enum {
    Q27_WORKER_STARTING = 0,
    Q27_WORKER_IDLE,
    Q27_WORKER_GENERATING,
    Q27_WORKER_STOPPING,
    Q27_WORKER_ERROR,
    Q27_WORKER_STOPPED
} q27_agent_worker_state;

q27_agent_worker *q27_agent_worker_start(const char *model_path,
                                          const char *tokenizer_path,
                                          uint32_t context,
                                          char *error, size_t error_cap);

// Synchronous command submission for the first worker slice. The caller keeps
// messages, sink, alive, and opaque valid until this function returns. Only the
// worker thread touches the underlying engine.
q27_agent_status q27_agent_worker_generate(
    q27_agent_worker *worker,
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

q27_agent_worker_state q27_agent_worker_get_state(q27_agent_worker *worker);

// Non-destructive and callback-safe: closes admission and asks the worker to
// stop after any accepted command publishes its result.
void q27_agent_worker_request_stop(q27_agent_worker *worker);
// Owner/UI-thread helper: wait for the engine-owner thread to publish STOPPED
// after request_stop. Never call this from an engine callback.
void q27_agent_worker_wait_stopped(q27_agent_worker *worker);

// Lifetime contract: the owner must close admission before calling stop—no new
// generate/get_state call may begin concurrently. stop may overlap a generate
// call that has already been accepted; it waits for that submitter to consume
// its result before destroying the worker. It must be called by the owning/UI
// thread, never from an engine sink/alive callback (use request_stop there).
void q27_agent_worker_stop(q27_agent_worker *worker);

#ifdef Q27_AGENT_WORKER_TESTING
// Selftest-only synchronization point: invoked inside stop immediately before
// (phase 0) and after (phase 1) its accepted-submitter wait. call_active lets
// the test detect a removed/broken wait without scheduler timing.
void q27_agent_worker_set_stop_hook(q27_agent_worker *worker,
                                    void (*hook)(void *, int, int), void *opaque);
#endif

#ifdef __cplusplus
}
#endif
#endif
