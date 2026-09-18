// linenoise-backed interactive editor for q27-agent.
#ifndef Q27_AGENT_EDITOR_H
#define Q27_AGENT_EDITOR_H

#include "q27_agent_tui.h"

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q27_agent_editor q27_agent_editor;

/* signal_pipe_read is the self-pipe used for Ctrl-C wakeups; pass -1 to skip. */
q27_agent_editor *q27_agent_editor_create(int signal_pipe_read);
void q27_agent_editor_free(q27_agent_editor *ed);

void q27_agent_editor_set_status(q27_agent_editor *ed,
                                 const q27_tui_status *st);

/* Idle blocking-style read (multiplexed). Returns 1 with *out_line owned
 * (caller free), 0 on clean EOF, -1 on error, -2 on interrupt signal. */
int q27_agent_editor_read_line(q27_agent_editor *ed, char **out_line);

/* Blocking linenoise() path when multiplex is unavailable. */
int q27_agent_editor_read_line_simple(const char *prompt, char **out_line);

/* ---- Busy multiplex (queue-while-busy + sticky footer while streaming) ----
 * begin_busy starts a live linenoise session with a busy prompt. pump() polls
 * stdin for a short timeout and enqueues completed lines. write_above() hides
 * the prompt, writes model/tool chrome, then restores it. end_busy tears down
 * the live session so read_line can run again. */

int q27_agent_editor_begin_busy(q27_agent_editor *ed,
                                q27_tui_prompt_queue *queue);
/* Ends busy mode. Any unfinished (not Enter-submitted) buffer is retained and
 * restored into the next idle read_line. */
void q27_agent_editor_end_busy(q27_agent_editor *ed);
int q27_agent_editor_is_busy(const q27_agent_editor *ed);

/* Refresh sticky status while busy (no-op if not busy). */
void q27_agent_editor_refresh_status(q27_agent_editor *ed);

/* Poll stdin once (timeout_ms). Completed non-empty lines go into the queue
 * bound at begin_busy. Returns 1 if a line was queued, 0 if nothing, -1 error,
 * -2 interrupt (SIGINT via signal pipe; does not end busy). */
int q27_agent_editor_pump(q27_agent_editor *ed, int timeout_ms);

/* Write bytes above the live prompt (hide → write → show). FILE may be stdout
 * or stderr; hide/show only when fd is the linenoise output fd. */
int q27_agent_editor_write_above(q27_agent_editor *ed, FILE *out,
                                 const void *data, size_t len);

/* Ensure a trailing newline on the stream side before idle resume. */
void q27_agent_editor_note_output_newline(q27_agent_editor *ed, int has_nl);

/* Make the busy prompt visible. If model output is mid-line, terminates that
 * row first so linenoiseHide cannot erase streamed text on the next write. */
int q27_agent_editor_show_prompt(q27_agent_editor *ed);

#ifdef __cplusplus
}
#endif
#endif
