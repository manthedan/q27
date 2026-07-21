#include "q27_agent_tui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

int q27_tui_available(void) {
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return 0;
    const char *term = getenv("TERM");
    /* Match linenoise's unsupported_term list (case-insensitive) so we never
     * enable the raw multiplex editor on terminals it refuses. */
    if (!term || !term[0]) return 0;
    if (!strcasecmp(term, "dumb") || !strcasecmp(term, "cons25") ||
        !strcasecmp(term, "emacs"))
        return 0;
    return 1;
}

void q27_tui_format_tokens(uint32_t n, char *buf, size_t buf_len) {
    if (!buf || buf_len == 0) return;
    if (n >= 1000000u)
        snprintf(buf, buf_len, "%.1fM", (double)n / 1000000.0);
    else if (n >= 10000u)
        snprintf(buf, buf_len, "%uk", n / 1000u);
    else if (n >= 1000u)
        snprintf(buf, buf_len, "%.1fk", (double)n / 1000.0);
    else
        snprintf(buf, buf_len, "%u", n);
}

void q27_tui_format_progress_bar(int done, int total, int bar_width,
                                 char *buf, size_t buf_len) {
    if (!buf || buf_len == 0) return;
    if (bar_width < 4) bar_width = 4;
    if (bar_width + 3 > (int)buf_len) bar_width = (int)buf_len - 3;
    if (bar_width < 1) {
        buf[0] = '\0';
        return;
    }
    if (total <= 0) total = 1;
    if (done < 0) done = 0;
    if (done > total) done = total;
    int filled = (int)(((long long)done * bar_width) / total);
    if (filled < 0) filled = 0;
    if (filled > bar_width) filled = bar_width;
    size_t pos = 0;
    if (pos + 1 < buf_len) buf[pos++] = '[';
    for (int i = 0; i < bar_width && pos + 1 < buf_len; i++)
        buf[pos++] = (i < filled) ? '#' : '-';
    if (pos + 1 < buf_len) buf[pos++] = ']';
    buf[pos] = '\0';
}

const char *q27_tui_status_start_escape(void) {
    return "\x1b[48;5;238;38;5;252m";
}

const char *q27_tui_status_end_escape(void) {
    return "\x1b[0m";
}

int q27_tui_format_status(const q27_tui_status *st, char *buf, size_t buf_len) {
    if (!st || !buf || buf_len == 0) return -1;
    char used[32], total[32];
    q27_tui_format_tokens(st->ctx_used, used, sizeof(used));
    q27_tui_format_tokens(st->ctx_size, total, sizeof(total));

    int n = 0;
    switch (st->phase) {
    case Q27_TUI_PREFILL: {
        char bar[48];
        int done = (int)st->prefill_done;
        int tot = st->prefill_total > 0 ? (int)st->prefill_total : 1;
        if (done > tot) done = tot;
        q27_tui_format_progress_bar(done, tot, 16, bar, sizeof(bar));
        double pct = 100.0 * (double)done / (double)tot;
        if (st->prefill_tps > 0.0)
            n = snprintf(buf, buf_len,
                         "ctx %s/%s | prefill %s %d/%d %.0f%% %.0f t/s",
                         used, total, bar, done, tot, pct, st->prefill_tps);
        else
            n = snprintf(buf, buf_len,
                         "ctx %s/%s | prefill %s %d/%d %.0f%%",
                         used, total, bar, done, tot, pct);
        break;
    }
    case Q27_TUI_GENERATING:
        if (st->gen_tps > 0.0)
            n = snprintf(buf, buf_len,
                         "ctx %s/%s | generating %u tok %.1f t/s",
                         used, total, st->gen_tokens, st->gen_tps);
        else
            n = snprintf(buf, buf_len,
                         "ctx %s/%s | generating %u tok",
                         used, total, st->gen_tokens);
        break;
    case Q27_TUI_TOOL:
        n = snprintf(buf, buf_len, "ctx %s/%s | tool %s%s%s",
                     used, total,
                     st->tool_name && st->tool_name[0] ? st->tool_name : "…",
                     st->detail && st->detail[0] ? " " : "",
                     st->detail && st->detail[0] ? st->detail : "");
        break;
    case Q27_TUI_COMPACTING:
        n = snprintf(buf, buf_len, "ctx %s/%s | compacting", used, total);
        break;
    case Q27_TUI_SAVING:
        n = snprintf(buf, buf_len, "ctx %s/%s | saving session", used, total);
        break;
    case Q27_TUI_ERROR:
        n = snprintf(buf, buf_len, "ctx %s/%s | error: %s",
                     used, total,
                     st->detail && st->detail[0] ? st->detail : "unknown");
        break;
    case Q27_TUI_IDLE:
    default:
        n = snprintf(buf, buf_len, "ctx %s/%s | idle", used, total);
        break;
    }
    if (n < 0) return -1;
    if ((size_t)n >= buf_len) return -1;
    return n;
}
