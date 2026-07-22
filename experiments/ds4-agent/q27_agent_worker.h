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
    Q27_WORKER_SESSION_IO,
    Q27_WORKER_STOPPING,
    Q27_WORKER_ERROR,
    Q27_WORKER_STOPPED
} q27_agent_worker_state;

typedef enum {
    Q27_EVENT_STATE = 0,
    Q27_EVENT_PREFILL_PROGRESS,
    Q27_EVENT_TEXT_DELTA,
    Q27_EVENT_TOOL_OUTPUT,
    // Control-plane metadata appended to the model-visible tool response after
    // the worker terminal. Carries a monotonic worker sequence and the parent
    // tool command id so JSONL remains a faithful transcript.
    Q27_EVENT_SELECTIONS,
    Q27_EVENT_TURN_DONE,
    Q27_EVENT_TOOL_DONE,
    Q27_EVENT_SESSION_DONE,
    Q27_EVENT_REJECTED,
    Q27_EVENT_STALLED,
    Q27_EVENT_ERROR
} q27_agent_event_type;

typedef enum {
    Q27_SESSION_SAVE = 1,
    Q27_SESSION_LOAD,
    Q27_SESSION_COUNT
} q27_agent_session_action;

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
    int tool_call_complete;
    int eos_reached;
    q27_agent_tool_kind tool_kind;
    int32_t tool_exit_code;
    uint32_t tool_flags;
    uint32_t tool_output_bytes;
    int tool_has_file_sha256;
    uint64_t tool_file_size;
    unsigned char tool_file_sha256[32];
    uint32_t tool_selection_count;
    q27_agent_tool_selection tool_selections[Q27_TOOL_MAX_SELECTIONS];
    // q27_agent_session_action value for SESSION_DONE events (0 = none).
    // FP1 requires the action on every session_done terminal (r11 codex P2).
    int session_action;
} q27_agent_event;

q27_agent_worker *q27_agent_worker_start(const char *model_path,
                                          const char *tokenizer_path,
                                          uint32_t context,
                                          char *error, size_t error_cap);
// mtp_width: 0 = serial decode only; 2..12 = free-decode MTP (sampled when
// temperature > 0). Active tool grammar still forces serial.
q27_agent_worker *q27_agent_worker_start_at(const char *model_path,
                                             const char *tokenizer_path,
                                             uint32_t context,
                                             const char *workspace_root,
                                             uint32_t mtp_width,
                                             char *error, size_t error_cap);

// Deep-copies every message before returning. enable_tools opts into the fixed
// native-tool grammar; a closed call ends generation and is reported on the
// terminal event. sampling is copied by value (temperature 0 keeps the greedy
// path). Prefill progress is reported before text at fixed 96-token intervals
// plus initial/final boundaries. Watchdog termination reports Q27_EVENT_STALLED
// with Q27_AGENT_STALLED and leaves worker admission reusable. alive/opaque
// remain borrowed until the command's terminal event is consumed; the owner
// must keep them valid and close submission admission before destructive stop.
q27_agent_status q27_agent_worker_submit(
    q27_agent_worker *worker,
    const q27_agent_message *messages,
    size_t message_count,
    int enable_thinking,
    int enable_tools,
    uint32_t max_tokens,
    q27_agent_sampling sampling,
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

// Serializes Q27SNAP1 save/load and tokenizer-only prompt counts on the engine
// owner. SAVE and LOAD require snapshot_path plus a transcript; COUNT requires
// a transcript and ignores snapshot_path. All three deep-copy every message;
// LOAD also requires the manifest's expected snapshot SHA-256. SESSION_DONE
// reports the loaded ledger or rendered
// prompt count in prompt_tokens.
q27_agent_status q27_agent_worker_submit_session(
    q27_agent_worker *worker,
    q27_agent_session_action action,
    const char *snapshot_path,
    const q27_agent_message *messages,
    size_t message_count,
    int enable_thinking,
    const unsigned char expected_snapshot_sha256[32],
    uint64_t *command_id,
    char *error, size_t error_cap);

// Blocks until an event is available. Returns 1 with an owned event, 0 when a
// stopped worker has no events left, and -1 on invalid arguments. Consuming a
// terminal event acknowledges the command and reopens admission when safe.
int q27_agent_worker_next_event(q27_agent_worker *worker,
                                q27_agent_event *event,
                                char *error, size_t error_cap);
// Like next_event, but timeout_ms >= 0 waits at most that many milliseconds.
// Returns 2 on timeout with no event (worker still live). timeout_ms < 0 blocks
// forever (same return codes as next_event).
int q27_agent_worker_next_event_timeout(q27_agent_worker *worker,
                                        q27_agent_event *event,
                                        char *error, size_t error_cap,
                                        int timeout_ms);
void q27_agent_event_free(q27_agent_event *event);

q27_agent_worker_state q27_agent_worker_get_state(q27_agent_worker *worker);
// Identity copied by the owner thread from the exact pinned tokenizer bytes
// used to construct the engine.
int q27_agent_worker_tokenizer_sha1(q27_agent_worker *worker,
                                    unsigned char out_sha1[20]);
// Creates one owned, monotonic post-publication terminal without requiring
// engine admission; usable even when snapshot I/O put the worker in ERROR.
int q27_agent_worker_selection_event(q27_agent_worker *worker,
                                     uint64_t parent_command_id,
                                     q27_agent_tool_kind tool_kind,
                                     const unsigned char *data, size_t data_len,
                                     q27_agent_event *event);

int q27_agent_worker_session_result_event(q27_agent_worker *worker,
                                          int success,
                                          const char *message,
                                          int action,
                                          q27_agent_event *event);

// Allocate a monotonic sequence for control-plane synthetic events
// (hello / idle / bye / notice / rejected / queue / tool_start). Same counter
// as worker events, which receive their seq at next_event (publish time) so
// stream order equals seq order even when reject_busy interleaves mid-turn.
// events so stream order equals seq order (FP1 §3.3). Returns 0 if worker is
// NULL; otherwise a strictly positive seq.
uint64_t q27_agent_worker_alloc_sequence(q27_agent_worker *worker);

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
