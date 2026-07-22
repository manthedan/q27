// Terminal status helpers for the native-agent TUI (ds4/pi inspired).
#ifndef Q27_AGENT_TUI_H
#define Q27_AGENT_TUI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Multiphase chrome: IDLE between prompts; PREFILL/GENERATING/TOOL while a
 * turn is live (sticky footer via linenoise when the busy editor is open). */
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
    uint32_t queue_len;    /* pending user prompts while busy */
} q27_tui_status;

/* Bounded prompt queue for queue-while-busy. Owns each line. */
typedef struct {
    char **items;
    size_t len;
    size_t cap;
    size_t max_len; /* hard cap; 0 means default (8) */
} q27_tui_prompt_queue;

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

/* Tool card lines (ASCII, no emoji). Returns bytes written or -1. */
int q27_tui_format_tool_card_open(const char *kind, const char *detail,
                                  char *buf, size_t buf_len);
int q27_tui_format_tool_card_close(int exit_code, uint32_t output_bytes,
                                   const char *message, char *buf,
                                   size_t buf_len);

/* Collapse helper: copy up to max_lines of text; if truncated, append a
 * summary. Returns bytes written (excluding NUL) or -1. */
int q27_tui_collapse_text(const char *text, size_t text_len, int max_lines,
                          int max_chars, char *buf, size_t buf_len);

/* Sanitize text for terminal chrome (tool cards / status). Strips ESC and
 * other C0 controls so model- or filesystem-controlled strings cannot inject
 * terminal sequences. Always NUL-terminates when out_len > 0. */
void q27_tui_sanitize_display(const char *in, char *out, size_t out_len);

void q27_tui_prompt_queue_init(q27_tui_prompt_queue *q, size_t max_len);
void q27_tui_prompt_queue_free(q27_tui_prompt_queue *q);
/* Push a copy of text. Returns 1 on success, 0 if full or OOM. */
int q27_tui_prompt_queue_push(q27_tui_prompt_queue *q, const char *text,
                              size_t len);
/* Pop owned line (caller free). NULL if empty. */
char *q27_tui_prompt_queue_pop(q27_tui_prompt_queue *q);
size_t q27_tui_prompt_queue_len(const q27_tui_prompt_queue *q);

#ifdef __cplusplus
}
#endif
#endif
