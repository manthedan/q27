// Terminal status helpers for the native-agent TUI (ds4/pi inspired).
#ifndef Q27_AGENT_TUI_H
#define Q27_AGENT_TUI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Phase 1 wires IDLE (linenoise sticky footer between prompts) and PREFILL
 * (stderr progress during turns). GENERATING/TOOL/COMPACTING/SAVING/ERROR are
 * reserved for Phase 2 sticky multiphase chrome while the editor is stopped. */
typedef enum {
    Q27_TUI_IDLE = 0,
    Q27_TUI_PREFILL,
    Q27_TUI_GENERATING,
    Q27_TUI_TOOL,
    Q27_TUI_COMPACTING,
    Q27_TUI_SAVING,
    Q27_TUI_ERROR
} q27_tui_phase;

typedef struct {
    q27_tui_phase phase;
    uint32_t ctx_used;
    uint32_t ctx_size;
    uint32_t prefill_done;
    uint32_t prefill_total;
    double prefill_tps;
    uint32_t gen_tokens;
    double gen_tps;
    const char *tool_name; /* borrowed; may be NULL */
    const char *detail;    /* borrowed error/tool detail; may be NULL */
} q27_tui_status;

/* 1 when stdout and stdin are TTYs and TERM is usable for the TUI. */
int q27_tui_available(void);

/* Format a one-line status footer. Returns bytes written (excluding NUL), or
 * -1 if buf is too small / NULL. Never writes more than buf_len-1 chars. */
int q27_tui_format_status(const q27_tui_status *st, char *buf, size_t buf_len);

/* Compact token count for status (e.g. 32768 -> "32k"). */
void q27_tui_format_tokens(uint32_t n, char *buf, size_t buf_len);

/* Progress bar into buf; bar_width is the interior cell count. */
void q27_tui_format_progress_bar(int done, int total, int bar_width,
                                 char *buf, size_t buf_len);

/* ANSI escapes for linenoise status row (inverse-ish bar). Empty if no color. */
const char *q27_tui_status_start_escape(void);
const char *q27_tui_status_end_escape(void);

#ifdef __cplusplus
}
#endif
#endif
