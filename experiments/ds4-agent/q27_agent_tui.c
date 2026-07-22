#include "q27_agent_tui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#define Q27_TUI_QUEUE_DEFAULT_MAX 8

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

static int append_queue_suffix(const q27_tui_status *st, char *buf, size_t buf_len,
                               int n) {
    if (!st || st->queue_len == 0 || n < 0) return n;
    if ((size_t)n >= buf_len) return -1;
    int extra = snprintf(buf + n, buf_len - (size_t)n, " | queue %u",
                         st->queue_len);
    if (extra < 0) return -1;
    if ((size_t)n + (size_t)extra >= buf_len) return -1;
    return n + extra;
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
    n = append_queue_suffix(st, buf, buf_len, n);
    return n;
}

int q27_tui_format_tool_card_open(const char *kind, const char *detail,
                                  char *buf, size_t buf_len) {
    if (!buf || buf_len == 0) return -1;
    const char *k = kind && kind[0] ? kind : "tool";
    int n;
    if (detail && detail[0])
        n = snprintf(buf, buf_len, "┌─ %s  %s\n", k, detail);
    else
        n = snprintf(buf, buf_len, "┌─ %s\n", k);
    if (n < 0 || (size_t)n >= buf_len) return -1;
    return n;
}

int q27_tui_format_tool_card_close(int exit_code, uint32_t output_bytes,
                                   const char *message, char *buf,
                                   size_t buf_len) {
    if (!buf || buf_len == 0) return -1;
    char size[32];
    q27_tui_format_tokens(output_bytes, size, sizeof(size));
    const char *mark = exit_code == 0 ? "ok" : "fail";
    int n;
    if (message && message[0])
        n = snprintf(buf, buf_len, "└─ %s  exit=%d  out=%sB  %s\n",
                     mark, exit_code, size, message);
    else
        n = snprintf(buf, buf_len, "└─ %s  exit=%d  out=%sB\n",
                     mark, exit_code, size);
    if (n < 0 || (size_t)n >= buf_len) return -1;
    return n;
}

size_t q27_tui_sanitize_bytes(const unsigned char *in, size_t in_len,
                              char *out, size_t out_cap) {
    if (!out || out_cap == 0) return 0;
    size_t o = 0;
    size_t i = 0;
    while (i < in_len && o + 4 < out_cap) {
        const unsigned char c = in[i];
        if (c == '\n') { out[o++] = '\n'; ++i; continue; }
        if (c == '\t') { out[o++] = ' '; ++i; continue; }
        if (c < 0x20 || c == 0x7f) { out[o++] = '.'; ++i; continue; }
        if (c < 0x80) { out[o++] = (char)c; ++i; continue; }
        /* Multibyte: decode-or-replace; never classify continuation bytes
         * as C1 controls (r13 codex P2). */
        size_t need = 0;
        if (c >= 0xC2 && c <= 0xDF) need = 2;
        else if (c >= 0xE0 && c <= 0xEF) need = 3;
        else if (c >= 0xF0 && c <= 0xF4) need = 4;
        int valid = need > 0 && i + need <= in_len;
        if (valid) {
            for (size_t k = 1; k < need; ++k)
                if ((in[i + k] & 0xC0) != 0x80) { valid = 0; break; }
        }
        if (valid) {
            const unsigned char c1 = in[i + 1];
            if (c == 0xE0 && c1 < 0xA0) valid = 0;
            if (c == 0xED && c1 > 0x9F) valid = 0;
            if (c == 0xF0 && c1 < 0x90) valid = 0;
            if (c == 0xF4 && c1 > 0x8F) valid = 0;
        }
        if (!valid) { out[o++] = '.'; ++i; continue; }
        /* The C1 control CODE POINT (U+0080..U+009F, 0xC2 0x80-0x9F) is a
         * control, not text. */
        if (c == 0xC2 && in[i + 1] <= 0x9F) { out[o++] = '.'; i += 2; continue; }
        for (size_t k = 0; k < need; ++k) out[o++] = (char)in[i + k];
        i += need;
    }
    out[o] = '\0';
    return o;
}

int q27_tui_collapse_text(const char *text, size_t text_len, int max_lines,
                          int max_chars, char *buf, size_t buf_len) {
    if (!buf || buf_len == 0) return -1;
    if (!text) text_len = 0;
    if (max_lines < 1) max_lines = 1;
    if (max_chars < 16) max_chars = 16;
    if ((size_t)max_chars + 64 > buf_len) max_chars = (int)buf_len - 64;
    if (max_chars < 1) {
        buf[0] = '\0';
        return -1;
    }

    /* Sanitize code-point-aware first (r13 codex P2); the rest of the
     * collapse only sees terminal-safe text. */
    char *clean = NULL;
    if (text_len) {
        clean = malloc(text_len + 1);
        if (!clean) return -1;
        text_len = q27_tui_sanitize_bytes((const unsigned char *)text,
                                          text_len, clean, text_len + 1);
        text = clean;
    }

    int result;
    size_t i = 0;
    int lines = 0;
    size_t out = 0;
    int truncated = 0;
    while (i < text_len && lines < max_lines) {
        /* Whole code point per step — byte-wise truncation could split a
         * multibyte character at max_chars (r14 codex P2). The input is
         * already sanitized to valid UTF-8. */
        const unsigned char c = (unsigned char)text[i];
        size_t cp_len = 1;
        if (c >= 0xC2 && c <= 0xDF) cp_len = 2;
        else if (c >= 0xE0 && c <= 0xEF) cp_len = 3;
        else if (c >= 0xF0 && c <= 0xF4) cp_len = 4;
        if (i + cp_len > text_len) cp_len = 1;
        if ((int)(out + cp_len) > max_chars || out + cp_len >= buf_len) {
            truncated = 1;
            break;
        }
        for (size_t k = 0; k < cp_len; ++k) buf[out++] = text[i + k];
        if (c == '\n') lines++;
        i += cp_len;
    }
    if (i < text_len) truncated = 1;
    if (truncated) {
        size_t rem = text_len - i;
        int extra = snprintf(buf + out, buf_len - out,
                             "%s… (%zu more bytes)\n",
                             (out && buf[out - 1] != '\n') ? "\n" : "",
                             rem);
        if (extra < 0 || (size_t)extra >= buf_len - out) {
            result = -1;
        } else {
            out += (size_t)extra;
            result = (int)out;
        }
    } else {
        buf[out] = '\0';
        result = (int)out;
    }
    free(clean);
    return result;
}

void q27_tui_sanitize_display(const char *in, char *out, size_t out_len) {
    if (!out || out_len == 0) return;
    out[0] = '\0';
    if (!in) return;
    size_t j = 0;
    for (size_t i = 0; in[i] && j + 1 < out_len; ) {
        unsigned char c = (unsigned char)in[i++];
        if (c == 0x1b || c == 0x9b) {
            /* Drop CSI / OSC-ish sequences (ESC or C1 CSI): skip until final. */
            while (in[i]) {
                unsigned char d = (unsigned char)in[i++];
                if (d >= 0x40 && d <= 0x7e) break;
            }
            continue;
        }
        /* C0, DEL, and C1 controls (0x80–0x9f) are not safe in chrome text. */
        if (c < 0x20 || c == 0x7f || (c >= 0x80 && c <= 0x9f)) {
            out[j++] = '?';
            continue;
        }
        out[j++] = (char)c;
    }
    out[j] = '\0';
}

void q27_tui_prompt_queue_init(q27_tui_prompt_queue *q, size_t max_len) {
    if (!q) return;
    memset(q, 0, sizeof(*q));
    q->max_len = max_len ? max_len : Q27_TUI_QUEUE_DEFAULT_MAX;
}

void q27_tui_prompt_queue_free(q27_tui_prompt_queue *q) {
    if (!q) return;
    for (size_t i = 0; i < q->len; i++) free(q->items[i]);
    free(q->items);
    memset(q, 0, sizeof(*q));
}

int q27_tui_prompt_queue_push(q27_tui_prompt_queue *q, const char *text,
                              size_t len) {
    if (!q || (!text && len)) return 0;
    if (q->len >= q->max_len) return 0;
    if (q->len == q->cap) {
        size_t ncap = q->cap ? q->cap * 2 : 4;
        if (ncap > q->max_len) ncap = q->max_len;
        if (ncap <= q->cap) return 0;
        char **ni = realloc(q->items, ncap * sizeof(*ni));
        if (!ni) return 0;
        q->items = ni;
        q->cap = ncap;
    }
    char *copy = malloc(len + 1);
    if (!copy) return 0;
    if (len && text) memcpy(copy, text, len);
    copy[len] = '\0';
    q->items[q->len++] = copy;
    return 1;
}

char *q27_tui_prompt_queue_pop(q27_tui_prompt_queue *q) {
    if (!q || !q->len) return NULL;
    char *line = q->items[0];
    memmove(q->items, q->items + 1, (q->len - 1) * sizeof(q->items[0]));
    q->items[q->len - 1] = NULL;   /* vacated duplicate alias (codex P3) */
    q->len--;
    return line;
}

size_t q27_tui_prompt_queue_len(const q27_tui_prompt_queue *q) {
    return q ? q->len : 0;
}
