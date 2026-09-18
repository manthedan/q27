#include "q27_agent_editor.h"

#include "third_party/linenoise/linenoise.h"

#include <errno.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define Q27_EDITOR_BUF_INITIAL 4096
#define Q27_EDITOR_BUF_MAX (1024 * 1024)

struct q27_agent_editor {
    int signal_pipe_read;
    char status[256];
    char prompt[32];
    char busy_prompt[32];
    int busy;
    int hidden;
    int output_line_open;
    char *edit_buf;
    struct linenoiseState ls;
    q27_tui_prompt_queue *queue; /* borrowed while busy */
    char *restore_line;          /* unfinished busy buffer for next idle edit */
};

q27_agent_editor *q27_agent_editor_create(int signal_pipe_read) {
    q27_agent_editor *ed = calloc(1, sizeof(*ed));
    if (!ed) return NULL;
    ed->signal_pipe_read = signal_pipe_read;
    snprintf(ed->prompt, sizeof(ed->prompt), "q27> ");
    snprintf(ed->busy_prompt, sizeof(ed->busy_prompt), "q27… ");
    snprintf(ed->status, sizeof(ed->status), "ctx ? | idle");
    linenoiseHistorySetMaxLen(256);
    linenoiseSetMultiLine(1);
    return ed;
}

void q27_agent_editor_free(q27_agent_editor *ed) {
    if (!ed) return;
    if (ed->busy) q27_agent_editor_end_busy(ed);
    free(ed->restore_line);
    free(ed);
}

void q27_agent_editor_set_status(q27_agent_editor *ed,
                                 const q27_tui_status *st) {
    if (!ed) return;
    if (!st) {
        snprintf(ed->status, sizeof(ed->status), "ctx ? | idle");
    } else if (q27_tui_format_status(st, ed->status, sizeof(ed->status)) < 0) {
        snprintf(ed->status, sizeof(ed->status), "ctx ? | idle");
    }
    if (ed->busy) q27_agent_editor_refresh_status(ed);
}

void q27_agent_editor_refresh_status(q27_agent_editor *ed) {
    if (!ed || !ed->busy) return;
    /* While the prompt is hidden for streaming, only update the cached status
     * string (set_status already did). SetStatus would repaint the prompt on
     * top of live model text. Applied on the next show_prompt. */
    if (ed->hidden) return;
    linenoiseEditSetStatus(&ed->ls, ed->status,
                           q27_tui_status_start_escape(),
                           q27_tui_status_end_escape());
}

int q27_agent_editor_is_busy(const q27_agent_editor *ed) {
    return ed && ed->busy;
}

void q27_agent_editor_note_output_newline(q27_agent_editor *ed, int has_nl) {
    if (!ed) return;
    ed->output_line_open = has_nl ? 0 : 1;
}

int q27_agent_editor_begin_busy(q27_agent_editor *ed,
                                q27_tui_prompt_queue *queue) {
    if (!ed || ed->busy) return 0;
    ed->edit_buf = malloc(Q27_EDITOR_BUF_INITIAL);
    if (!ed->edit_buf) return 0;
    memset(&ed->ls, 0, sizeof(ed->ls));
    if (linenoiseEditStart(&ed->ls, STDIN_FILENO, STDOUT_FILENO, ed->edit_buf,
                           Q27_EDITOR_BUF_INITIAL, ed->busy_prompt) != 0) {
        linenoiseEditStop(&ed->ls);
        free(ed->edit_buf);
        ed->edit_buf = NULL;
        return 0;
    }
    ed->ls.buflen_max = Q27_EDITOR_BUF_MAX;
    linenoiseEditSetStatus(&ed->ls, ed->status,
                           q27_tui_status_start_escape(),
                           q27_tui_status_end_escape());
    /* Carry unfinished text across consecutive busy sessions (queued turns
     * that never hit idle read_line would otherwise lose restore_line when
     * the next end_busy overwrote it). */
    if (ed->restore_line && ed->restore_line[0]) {
        size_t n = strlen(ed->restore_line);
        if (n > ed->ls.buflen) n = ed->ls.buflen;
        memcpy(ed->ls.buf, ed->restore_line, n);
        ed->ls.buf[n] = '\0';
        ed->ls.len = ed->ls.pos = n;
        linenoiseShow(&ed->ls);
        free(ed->restore_line);
        ed->restore_line = NULL;
    }
    ed->busy = 1;
    ed->hidden = 0;
    ed->output_line_open = 0;
    ed->queue = queue;
    return 1;
}

int q27_agent_editor_show_prompt(q27_agent_editor *ed) {
    if (!ed || !ed->busy) return 0;
    if (!ed->hidden) {
        linenoiseEditSetStatus(&ed->ls, ed->status,
                               q27_tui_status_start_escape(),
                               q27_tui_status_end_escape());
        return 1;
    }
    /* Prompt must not share a row with incomplete model output: the next
     * linenoiseHide would clear that entire row and erase streamed text. */
    if (ed->output_line_open) {
        if (fwrite("\n", 1, 1, stdout) != 1 || fflush(stdout) == EOF) return 0;
        ed->output_line_open = 0;
    }
    if (fputs("\x1b[0m", stdout) == EOF || fflush(stdout) == EOF) return 0;
    linenoiseEditSetStatus(&ed->ls, ed->status,
                           q27_tui_status_start_escape(),
                           q27_tui_status_end_escape());
    linenoiseShow(&ed->ls);
    ed->hidden = 0;
    return 1;
}

void q27_agent_editor_end_busy(q27_agent_editor *ed) {
    if (!ed || !ed->busy) return;
    /* Drain remaining raw stdin into the queue / edit buffer BEFORE
     * linenoiseEditStop's TCSAFLUSH, which would otherwise discard pasted
     * multi-prompt bytes that pump has not yet consumed. */
    for (int spins = 0; spins < 10000; spins++) {
        int pr = q27_agent_editor_pump(ed, 0);
        if (pr == 1)
            continue; /* line queued; more may follow */
        if (pr == -1 || pr == -2)
            break;
        struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
        int r;
        do {
            r = poll(&pfd, 1, 0);
        } while (r < 0 && errno == EINTR);
        if (r <= 0 || !(pfd.revents & (POLLIN | POLLHUP)))
            break;
        /* Still readable (partial UTF-8 / incomplete line). Keep pumping. */
    }
    /* Preserve unfinished follow-up text (typed but not Enter-submitted).
     * If restore_line still holds prior unfinished text (should be rare if
     * begin_busy re-seeded it), keep both rather than dropping the older. */
    if (ed->ls.buf && ed->ls.len > 0) {
        size_t cur = ed->ls.len;
        size_t prior = ed->restore_line ? strlen(ed->restore_line) : 0;
        size_t total = prior ? prior + 1 + cur : cur;
        char *hold = malloc(total + 1);
        if (hold) {
            size_t o = 0;
            if (prior) {
                memcpy(hold, ed->restore_line, prior);
                o = prior;
                hold[o++] = '\n';
            }
            memcpy(hold + o, ed->ls.buf, cur);
            hold[o + cur] = '\0';
            free(ed->restore_line);
            ed->restore_line = hold;
        }
    }
    /* Always clear the busy prompt row before stop. If we leave it visible,
     * the next idle q27> + restore_line redraws a duplicate line. */
    if (!ed->hidden) {
        linenoiseHide(&ed->ls);
        ed->hidden = 1;
    }
    if (ed->output_line_open) {
        (void)fwrite("\n", 1, 1, stdout);
        (void)fflush(stdout);
        ed->output_line_open = 0;
    }
    ed->hidden = 0;
    linenoiseEditStop(&ed->ls);
    free(ed->ls.buf);
    ed->ls.buf = NULL;
    ed->edit_buf = NULL;
    ed->busy = 0;
    ed->queue = NULL;
    ed->output_line_open = 0;
}

int q27_agent_editor_write_above(q27_agent_editor *ed, FILE *out,
                                 const void *data, size_t len) {
    if (!out || (!data && len)) return 0;
    if (!len) return 1;

    /* Hide the live prompt for any write while busy (stdout model stream or
     * stderr diagnostics). Tracking open-line state only for stdout so model
     * continuations stay correct; stderr diags always end a visual unit. */
    int use_hide = ed && ed->busy;
    int is_stdout = (out == stdout);
    if (use_hide && !ed->hidden) {
        linenoiseHide(&ed->ls);
        ed->hidden = 1;
        /* After hide, cursor sits on the cleared prompt row. If we previously
         * showed the prompt under a completed model line, that is the correct
         * place to append the next model/tool bytes. */
    }

    if (fwrite(data, 1, len, out) != len || fflush(out) == EOF) return 0;

    if (use_hide) {
        const unsigned char *b = (const unsigned char *)data;
        if (is_stdout)
            ed->output_line_open = (b[len - 1] != '\n');
        else {
            /* Diagnostics on stderr must not glue onto an unfinished model
             * line on stdout, and must not pretend stdout is complete. */
            if (ed->output_line_open) {
                if (fwrite("\n", 1, 1, stdout) != 1 || fflush(stdout) == EOF)
                    return 0;
                ed->output_line_open = 0;
            }
            if (b[len - 1] != '\n') {
                if (fwrite("\n", 1, 1, out) != 1 || fflush(out) == EOF)
                    return 0;
            }
        }
        /* Stay hidden while streaming on stdout mid-line. Restoring the prompt
         * after every token puts linenoise on the same row as incomplete text;
         * the next Hide then erases that row. Natural newlines leave a clean
         * boundary so the footer can reappear safely. */
        if (!ed->output_line_open) {
            if (fputs("\x1b[0m", stdout) == EOF || fflush(stdout) == EOF)
                return 0;
            linenoiseEditSetStatus(&ed->ls, ed->status,
                                   q27_tui_status_start_escape(),
                                   q27_tui_status_end_escape());
            linenoiseShow(&ed->ls);
            ed->hidden = 0;
        }
    }
    return 1;
}

int q27_agent_editor_pump(q27_agent_editor *ed, int timeout_ms) {
    if (!ed || !ed->busy) return 0;

    struct pollfd fds[2];
    int nfds = 1;
    fds[0].fd = STDIN_FILENO;
    fds[0].events = POLLIN;
    fds[0].revents = 0;
    if (ed->signal_pipe_read >= 0) {
        fds[1].fd = ed->signal_pipe_read;
        fds[1].events = POLLIN;
        fds[1].revents = 0;
        nfds = 2;
    }

    int ready;
    do {
        ready = poll(fds, (nfds_t)nfds, timeout_ms < 0 ? -1 : timeout_ms);
    } while (ready < 0 && errno == EINTR);

    if (ready < 0) return -1;

    if (nfds == 2 && (fds[1].revents & (POLLIN | POLLERR | POLLHUP))) {
        unsigned char sink[64];
        while (read(ed->signal_pipe_read, sink, sizeof(sink)) > 0) {
        }
        return -2;
    }

    if (ready == 0) {
        /* Soft refresh only when the prompt is already visible. */
        if (!ed->hidden)
            linenoiseEditSetStatus(&ed->ls, ed->status,
                                   q27_tui_status_start_escape(),
                                   q27_tui_status_end_escape());
        return 0;
    }

    if (!(fds[0].revents & (POLLIN | POLLHUP))) return 0;

    if (!q27_agent_editor_show_prompt(ed)) return -1;

    errno = 0;
    char *line = linenoiseEditFeed(&ed->ls);
    if (line == linenoiseEditMore) return 0;

    if (line == NULL) {
        const int err = errno;
        if (err == EAGAIN || err == EWOULDBLOCK) {
            /* Ctrl-C: leave ls.buf intact so end_busy can restore unfinished
             * follow-up text into restore_line. */
            return -2;
        }
        /* Ctrl-D/EOF (ENOENT) and other feed exits consume the scratch history
         * slot. Capture unfinished text before clear, then restore the sentinel
         * so a later Enter cannot underflow history_len. */
        if (ed->ls.buf && ed->ls.len > 0) {
            char *hold = malloc(ed->ls.len + 1);
            if (hold) {
                memcpy(hold, ed->ls.buf, ed->ls.len);
                hold[ed->ls.len] = '\0';
                free(ed->restore_line);
                ed->restore_line = hold;
            }
        }
        linenoiseEditClear(&ed->ls);
        linenoiseHistoryAdd("");
        ed->ls.history_index = 0;
        if (err == 0 || err == ENOENT) return 0; /* EOF while busy: ignore */
        return -1;
    }

    /* Reject embedded NULs. */
    if (memchr(ed->ls.buf, '\0', ed->ls.len) != NULL) {
        free(line);
        linenoiseEditClear(&ed->ls);
        linenoiseHistoryAdd("");
        ed->ls.history_index = 0;
        errno = EILSEQ;
        return -1;
    }

    size_t len = strlen(line);
    int queued = 0;
    if (len > 0 && ed->queue) {
        if (q27_tui_prompt_queue_push(ed->queue, line, len)) {
            if (line[0]) linenoiseHistoryAdd(line);
            queued = 1;
        } else {
            char msg[96];
            int n = snprintf(msg, sizeof(msg),
                             "q27-agent: prompt queue full (max %zu)\n",
                             ed->queue->max_len);
            if (n > 0)
                (void)q27_agent_editor_write_above(ed, stdout, msg,
                                                   (size_t)n);
        }
    }
    free(line);
    /* Restart edit buffer for next queued line without tearing down raw mode.
     * Enter removed linenoise's temporary current-history slot; restore an
     * empty scratch entry so the next Enter cannot underflow history_len. */
    linenoiseEditClear(&ed->ls);
    linenoiseHistoryAdd("");
    ed->ls.history_index = 0;
    return queued ? 1 : 0;
}

int q27_agent_editor_read_line_simple(const char *prompt, char **out_line) {
    if (!out_line) return -1;
    *out_line = NULL;
    char *line = linenoise(prompt ? prompt : "q27> ");
    if (!line) {
        if (errno == EAGAIN) return -2;
        return 0;
    }
    if (line[0]) linenoiseHistoryAdd(line);
    *out_line = line;
    return 1;
}

int q27_agent_editor_read_line(q27_agent_editor *ed, char **out_line) {
    if (!ed || !out_line) return -1;
    *out_line = NULL;
    if (ed->busy) return -1; /* must end_busy first */

    char *buf = malloc(Q27_EDITOR_BUF_INITIAL);
    if (!buf) return -1;

    struct linenoiseState ls;
    memset(&ls, 0, sizeof(ls));
    if (linenoiseEditStart(&ls, STDIN_FILENO, STDOUT_FILENO, buf,
                           Q27_EDITOR_BUF_INITIAL, ed->prompt) != 0) {
        /* EditStart enables raw mode before the first prompt write; stop
         * always so a failed write cannot leave the TTY in raw/no-echo. */
        linenoiseEditStop(&ls);
        free(buf);
        return -1;
    }
    ls.buflen_max = Q27_EDITOR_BUF_MAX;
    linenoiseEditSetStatus(&ls, ed->status,
                           q27_tui_status_start_escape(),
                           q27_tui_status_end_escape());
    /* Restore unfinished text captured when busy mode ended mid-edit. */
    if (ed->restore_line && ed->restore_line[0]) {
        size_t n = strlen(ed->restore_line);
        if (n > ls.buflen) n = ls.buflen;
        memcpy(ls.buf, ed->restore_line, n);
        ls.buf[n] = '\0';
        ls.len = ls.pos = n;
        linenoiseShow(&ls);
        free(ed->restore_line);
        ed->restore_line = NULL;
    } else {
        free(ed->restore_line);
        ed->restore_line = NULL;
    }

    for (;;) {
        struct pollfd fds[2];
        int nfds = 1;
        fds[0].fd = STDIN_FILENO;
        fds[0].events = POLLIN;
        fds[0].revents = 0;
        if (ed->signal_pipe_read >= 0) {
            fds[1].fd = ed->signal_pipe_read;
            fds[1].events = POLLIN;
            fds[1].revents = 0;
            nfds = 2;
        }

        int ready;
        do {
            ready = poll(fds, (nfds_t)nfds, 200);
        } while (ready < 0 && errno == EINTR);

        if (ready < 0) {
            linenoiseEditStop(&ls);
            free(ls.buf);
            return -1;
        }

        if (nfds == 2 && (fds[1].revents & (POLLIN | POLLERR | POLLHUP))) {
            unsigned char sink[64];
            while (read(ed->signal_pipe_read, sink, sizeof(sink)) > 0) {
            }
            linenoiseEditStop(&ls);
            free(ls.buf);
            return -2;
        }

        if (ready == 0) {
            linenoiseEditSetStatus(&ls, ed->status,
                                   q27_tui_status_start_escape(),
                                   q27_tui_status_end_escape());
            continue;
        }

        if (fds[0].revents & (POLLERR | POLLNVAL)) {
            linenoiseEditStop(&ls);
            free(ls.buf);
            return -1;
        }
        if (!(fds[0].revents & (POLLIN | POLLHUP))) continue;

        errno = 0;
        char *line = linenoiseEditFeed(&ls);
        if (line == linenoiseEditMore) continue;

        if (line == NULL) {
            const int err = errno;
            linenoiseEditStop(&ls);
            free(ls.buf);
            if (err == EAGAIN || err == EWOULDBLOCK) return -2; /* Ctrl-C */
            if (err == 0 || err == ENOENT) return 0; /* clean EOF / Ctrl-D */
            return -1; /* read/write/alloc or other I/O failure */
        }

        /* linenoise returns strdup(buf); embedded NULs would truncate. The
         * edit state still holds the full binary length — reject before stop
         * so we match the plain reader's NUL policy. */
        if (memchr(ls.buf, '\0', ls.len) != NULL) {
            free(line);
            linenoiseEditStop(&ls);
            free(ls.buf);
            errno = EILSEQ;
            return -1;
        }

        linenoiseEditStop(&ls);
        free(ls.buf);
        if (line[0]) linenoiseHistoryAdd(line);
        *out_line = line;
        return 1;
    }
}
