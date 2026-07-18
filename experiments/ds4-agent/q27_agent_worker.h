// Dedicated engine-owner thread and event queue for the DS4-style experiment.
#ifndef Q27_AGENT_WORKER_H
#define Q27_AGENT_WORKER_H

#include "q27_agent_engine.h"
#include "q27_agent_tools.h"

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
    Q27_WORKER_TOOL_RUNNING,
    Q27_WORKER_STOPPING,
    Q27_WORKER_ERROR,
    Q27_WORKER_STOPPED
} q27_agent_worker_state;

typedef enum {
    Q27_EVENT_STATE = 0,
    Q27_EVENT_TEXT_DELTA,
    Q27_EVENT_TOOL_OUTPUT,
    Q27_EVENT_TURN_DONE,
    Q27_EVENT_TOOL_DONE,
    Q27_EVENT_REJECTED,
    Q27_EVENT_ERROR
} q27_agent_event_type;

typedef struct {
    q27_agent_event_type type;
    uint64_t sequence;
    uint64_t command_id;
    q27_agent_worker_state state;
    q27_agent_status status;
    unsigned char *data;
    size_t data_len;
    uint32_t prompt_tokens;
    uint32_t cached_tokens;
    uint32_t prefill_tokens;
    uint32_t output_tokens;
    q27_agent_tool_kind tool_kind;
    int32_t tool_exit_code;
    uint32_t tool_flags;
    uint32_t tool_output_bytes;
} q27_agent_event;

q27_agent_worker *q27_agent_worker_start(const char *model_path,
                                          const char *tokenizer_path,
                                          uint32_t context,
                                          char *error, size_t error_cap);
q27_agent_worker *q27_agent_worker_start_at(const char *model_path,
                                             const char *tokenizer_path,
                                             uint32_t context,
                                             const char *workspace_root,
                                             char *error, size_t error_cap);

// Deep-copies every message before returning. alive/opaque remain borrowed
// until the command's terminal event is consumed; the owner must keep them
// valid and close submission admission before destructive stop.
q27_agent_status q27_agent_worker_submit(
    q27_agent_worker *worker,
    const q27_agent_message *messages,
    size_t message_count,
    int enable_thinking,
    uint32_t max_tokens,
    q27_agent_alive_check alive,
    void *opaque,
    uint64_t *command_id,
    char *error, size_t error_cap);

// Deep-copies all request bytes. Exactly one generation or tool command may be
// active. Tool execution is serialized on the worker, bounded by the request,
// and publishes output/tool_done through the same owned event queue.
q27_agent_status q27_agent_worker_submit_tool(
    q27_agent_worker *worker,
    const q27_agent_tool_request *request,
    q27_agent_alive_check alive,
    void *opaque,
    uint64_t *command_id,
    char *error, size_t error_cap);

// Blocks until an event is available. Returns 1 with an owned event, 0 when a
// stopped worker has no events left, and -1 on invalid arguments. Consuming a
// terminal event acknowledges the command and reopens admission when safe.
int q27_agent_worker_next_event(q27_agent_worker *worker,
                                q27_agent_event *event,
                                char *error, size_t error_cap);
void q27_agent_event_free(q27_agent_event *event);

q27_agent_worker_state q27_agent_worker_get_state(q27_agent_worker *worker);

// Non-destructive: closes admission and asks the worker to stop after its
// current bounded engine quantum. Callback-safe.
void q27_agent_worker_request_stop(q27_agent_worker *worker);
// Owner/UI-thread helper; never call from an engine callback.
void q27_agent_worker_wait_stopped(q27_agent_worker *worker);

// Lifetime contract: destructive stop requires an owner-enforced API barrier:
// no submit/next_event/get_state call may be running or begin concurrently.
// Closing worker admission alone is not that barrier because submit clones
// before its admission check. Run stop on the sole owner/UI thread only after
// its API calls return, never inside an engine callback.
void q27_agent_worker_stop(q27_agent_worker *worker);

#ifdef Q27_AGENT_WORKER_TESTING
// Invoked inside destructive stop immediately before (phase 0) and after
// (phase 1) its accepted-command/terminal-consumption wait.
void q27_agent_worker_set_stop_hook(q27_agent_worker *worker,
                                    void (*hook)(void *, int, int), void *opaque);
#endif

#ifdef __cplusplus
}
#endif
#endif
