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
};

q27_agent_editor *q27_agent_editor_create(int signal_pipe_read) {
    q27_agent_editor *ed = calloc(1, sizeof(*ed));
    if (!ed) return NULL;
    ed->signal_pipe_read = signal_pipe_read;
    snprintf(ed->prompt, sizeof(ed->prompt), "q27> ");
    snprintf(ed->status, sizeof(ed->status), "ctx ? | idle");
    linenoiseHistorySetMaxLen(256);
    linenoiseSetMultiLine(1);
    return ed;
}

void q27_agent_editor_free(q27_agent_editor *ed) {
    free(ed);
}

void q27_agent_editor_set_status(q27_agent_editor *ed,
                                 const q27_tui_status *st) {
    if (!ed) return;
    if (!st) {
        snprintf(ed->status, sizeof(ed->status), "ctx ? | idle");
        return;
    }
    if (q27_tui_format_status(st, ed->status, sizeof(ed->status)) < 0)
        snprintf(ed->status, sizeof(ed->status), "ctx ? | idle");
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
