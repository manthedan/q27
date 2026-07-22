// Frontend Protocol v1 (FP1) emission helpers.
// See docs/metal/plans/2026-07-21-frontend-protocol-v1.md
#ifndef Q27_AGENT_FRONTEND_H
#define Q27_AGENT_FRONTEND_H

#include "q27_agent_worker.h"

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 0 = legacy jsonl / text; 1 = FP1. */
void q27_fp1_set_protocol(int protocol);
int q27_fp1_protocol(void);

/* True when stdout is the machine event stream (legacy jsonl or FP1). */
int q27_fp1_stream_events(int jsonl);

/* Wall-clock ms for envelope ts_ms. */
uint64_t q27_fp1_now_ms(void);

/* JSON-escape bytes into a malloc'd C string (quoted content only, no quotes).
 * Returns NULL on OOM. Empty input yields malloc'd "". */
char *q27_fp1_json_escape(const unsigned char *data, size_t len);

/* 1 if data is valid UTF-8 (empty is valid). */
int q27_fp1_utf8_valid(const unsigned char *data, size_t len);

/* Emit one worker event. mode 0 = legacy jsonl shape; mode 1 = FP1 envelope. */
int q27_fp1_print_event(FILE *out, const q27_agent_event *event, int mode,
                        const char *client_req_id);

/* Control-plane lifecycle (FP1 only). seq from worker counter; command_id 0. */
int q27_fp1_emit_hello(FILE *out, uint64_t seq, const char *model_path,
                       const char *tokenizer_path, uint32_t context,
                       const char *workspace, const char *session_path,
                       int auto_tools, int thinking, uint32_t max_tool_rounds,
                       int features_queue);

int q27_fp1_emit_idle(FILE *out, uint64_t seq, uint32_t ctx_used,
                      uint32_t ctx_size, uint32_t queue_len);

int q27_fp1_emit_bye(FILE *out, uint64_t seq, const char *reason);

/* Control-plane rejected (always has code). */
int q27_fp1_emit_rejected(FILE *out, uint64_t seq, const char *client_req_id,
                          const char *code, const char *text,
                          const char *state);

/* Optional notice. code may be NULL. */
int q27_fp1_emit_notice(FILE *out, uint64_t seq, const char *client_req_id,
                        const char *severity, const char *code,
                        const char *text, const char *state);

/* Control-plane session mutation terminal (action=save|load|count|compact|new).
 * command_id is 0 for compact/new; may be non-zero when a worker session cmd ran. */
int q27_fp1_emit_session_done(FILE *out, uint64_t seq, uint64_t command_id,
                              const char *client_req_id, const char *action,
                              const char *status, const char *text,
                              uint32_t prompt_tokens, const char *state);

/* Envelope-only state change (e.g. compacting). */
int q27_fp1_emit_state(FILE *out, uint64_t seq, const char *client_req_id,
                       const char *state);

/* Backend-owned prompt queue snapshot (P4). items[] from current queue. */
int q27_fp1_emit_queue(FILE *out, uint64_t seq, const char *state);

/* Control-plane tool card open (synthesized at tool submission). */
int q27_fp1_emit_tool_start(FILE *out, uint64_t seq, uint64_t command_id,
                            const char *client_req_id, const char *tool_kind,
                            const char *detail, int preflight);

/* Authoritative FP1 help text for the help op. */
const char *q27_fp1_help_text(void);

/* ── P2–P4 control channel (stdin NDJSON ClientMessage) ──────────────── */

typedef enum {
    Q27_FP1_OP_NONE = 0,
    Q27_FP1_OP_PROMPT,
    Q27_FP1_OP_CANCEL,
    Q27_FP1_OP_QUIT,
    Q27_FP1_OP_SAVE,
    Q27_FP1_OP_COMPACT,
    Q27_FP1_OP_NEW,
    Q27_FP1_OP_SESSION,
    Q27_FP1_OP_HELP,
    Q27_FP1_OP_TOOL,
    Q27_FP1_OP_QUEUE_CLEAR,
    Q27_FP1_OP_UNKNOWN,
    Q27_FP1_OP_BAD_VERSION,
    Q27_FP1_OP_MALFORMED
} q27_fp1_op_kind;

typedef struct {
    q27_fp1_op_kind kind;
    char *client_req_id; /* owned; may be NULL */
    char *text;          /* owned; prompt body or error detail */
    /* tool op fields (owned; may be NULL) */
    char *tool_kind; /* "read" | "search" | "shell" */
    char *path;
    char *needle;  /* search needle */
    char *command; /* shell command */
} q27_fp1_op;

void q27_fp1_op_free(q27_fp1_op *op);

/* Parse one NDJSON ClientMessage line into *out. Returns 1 always with a
 * kind (including MALFORMED / BAD_VERSION / UNKNOWN). */
int q27_fp1_parse_client_line(const char *line, size_t len, q27_fp1_op *out);

/* Start/stop the stdin reader thread. Only one control channel at a time. */
int q27_fp1_control_start(void);
void q27_fp1_control_stop(void);

/* 1 if cancel was requested (message or sticky). Does not clear. */
int q27_fp1_control_cancel_requested(void);
/* Clear cancel after the cancelled turn settles. */
void q27_fp1_control_clear_cancel(void);

/* 1 if quit/EOF was requested. reason is "quit" or "stdin_eof". */
int q27_fp1_control_quit_requested(void);

/* 1 when parsed-but-undispatched control ops remain (quit must not skip
 * them — r7 codex P1). */
int q27_fp1_control_pending(void);
const char *q27_fp1_control_quit_reason(void);

/* Block until a non-flag op is available, or quit. Returns 1 with *out filled
 * (caller frees), 0 on quit/stop. timeout_ms < 0 waits forever; 0 polls;
 * >0 waits up to that many ms (returns -1 on timeout with *out empty). */
int q27_fp1_control_wait_op(q27_fp1_op *out, int timeout_ms);

/* Snapshot for mid-turn session reports (point-in-time; may be stale). */
void q27_fp1_set_session_report(const char *session_path,
                                const char *snapshot_name, uint32_t context,
                                uint32_t ctx_used, int think, int auto_tools,
                                size_t turns_approx);

/* Pop pending ops while the worker is busy:
 *  - prompt → enqueue (feature queue) or rejected busy/empty/queue_full
 *  - queue_clear → clear prompt queue + emit queue
 *  - save/compact/new/tool → rejected code=busy
 *  - session/help → handled anytime (notice)
 *  - unknown/bad_version/malformed → rejected/notice as usual
 * Also flushes sticky drop markers. Returns 0 on emit failure. */
int q27_fp1_control_reject_busy(FILE *out, q27_agent_worker *worker,
                                const char *state);

/* While idle: emit sticky drop markers (queue overflow) if any. */
int q27_fp1_control_flush_drops(FILE *out, q27_agent_worker *worker,
                                const char *state);

/* Backend prompt queue (cap 8). Ownership of strings moves into the queue. */
uint32_t q27_fp1_prompt_queue_len(void);
/* Push a prompt. Returns 1 ok, 0 full, -1 OOM. Does not free on full. */
int q27_fp1_prompt_queue_push(const char *text, const char *client_req_id);
/* Pop into *out as Q27_FP1_OP_PROMPT (caller frees). Returns 1 if popped. */
int q27_fp1_prompt_queue_pop(q27_fp1_op *out);
/* Drop all queued prompts. */
void q27_fp1_prompt_queue_clear(void);
/* Emit queue event reflecting current contents. */
int q27_fp1_emit_queue_from_worker(FILE *out, q27_agent_worker *worker,
                                   const char *state);

/* Post an op as if from stdin (tests). */
int q27_fp1_control_post_for_test(const q27_fp1_op *op);

#ifdef __cplusplus
}
#endif
#endif
