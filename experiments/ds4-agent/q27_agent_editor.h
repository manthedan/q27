// linenoise-backed interactive editor for q27-agent.
#ifndef Q27_AGENT_EDITOR_H
#define Q27_AGENT_EDITOR_H

#include "q27_agent_tui.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q27_agent_editor q27_agent_editor;

/* signal_pipe_read is the self-pipe used for Ctrl-C wakeups; pass -1 to skip. */
q27_agent_editor *q27_agent_editor_create(int signal_pipe_read);
void q27_agent_editor_free(q27_agent_editor *ed);

void q27_agent_editor_set_status(q27_agent_editor *ed,
                                 const q27_tui_status *st);

/* Read one line. Returns 1 with *out_line owned (caller free), 0 on clean EOF,
 * -1 on error, -2 on interrupt signal. */
int q27_agent_editor_read_line(q27_agent_editor *ed, char **out_line);

/* Blocking linenoise() path when multiplex is unavailable. */
int q27_agent_editor_read_line_simple(const char *prompt, char **out_line);

#ifdef __cplusplus
}
#endif
#endif
