#define _POSIX_C_SOURCE 200809L

/*
 * Reduced DS4-style native-agent experiment for q27.
 *
 * The worker-owned direct-engine architecture is based on antirez/ds4's
 * ds4_agent.c. This Phase-0 file is intentionally small: it proves that a C
 * control loop can drive q27's C++/Metal engine without HTTP before we port
 * DS4's tools, jobs, persistence, and terminal UI. See THIRD_PARTY_NOTICES.md.
 */

#include "q27_agent_engine.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static volatile sig_atomic_t interrupted = 0;
static int signal_pipe[2] = {-1, -1};

static void on_signal(int sig) {
    (void)sig;
    int saved_errno = errno;
    interrupted = 1;
    if (signal_pipe[1] >= 0) {
        const unsigned char byte = 1;
        (void)write(signal_pipe[1], &byte, 1);
    }
    errno = saved_errno;
}

typedef struct {
    char *role;
    char *content;
    size_t content_len;
} owned_message;

typedef struct {
    owned_message *items;
    size_t len;
    size_t cap;
} transcript;

typedef struct {
    char *bytes;
    size_t len;
    size_t cap;
    int failed;
} output_buffer;

typedef struct {
    unsigned char pending[4096];
    size_t pos;
    size_t len;
    int eof;
} input_reader;

static void usage(FILE *out, const char *argv0) {
    fprintf(out,
        "usage: %s MODEL TOKENIZER [options]\n"
        "\n"
        "Direct, experimental q27 native chat loop (no HTTP server).\n"
        "\n"
        "options:\n"
        "  -p, --prompt TEXT       run one non-interactive turn\n"
        "  -s, --system TEXT       replace the default system prompt\n"
        "  -n, --max-tokens N      maximum generated tokens (default 512)\n"
        "  -c, --context N         engine context (default 8192)\n"
        "      --no-think          append the Qwen no-thinking prefix\n"
        "  -h, --help              show this help\n",
        argv0);
}

static int setup_signal_pipe(void) {
    if (pipe(signal_pipe) != 0) return 0;
    for (int i = 0; i < 2; ++i) {
        int flags = fcntl(signal_pipe[i], F_GETFL);
        int fdflags = fcntl(signal_pipe[i], F_GETFD);
        if (flags < 0 || fdflags < 0 ||
            fcntl(signal_pipe[i], F_SETFL, flags | O_NONBLOCK) < 0 ||
            fcntl(signal_pipe[i], F_SETFD, fdflags | FD_CLOEXEC) < 0) {
            int saved = errno;
            close(signal_pipe[0]);
            close(signal_pipe[1]);
            signal_pipe[0] = signal_pipe[1] = -1;
            errno = saved;
            return 0;
        }
    }
    return 1;
}

static void close_signal_pipe(void) {
    const int read_fd = signal_pipe[0], write_fd = signal_pipe[1];
    signal_pipe[0] = signal_pipe[1] = -1;
    if (read_fd >= 0) close(read_fd);
    if (write_fd >= 0) close(write_fd);
}

static int parse_u32(const char *text, uint32_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || !end || *end || value == 0 || value > UINT32_MAX) return 0;
    *out = (uint32_t)value;
    return 1;
}

static int transcript_append_len(transcript *t, const char *role,
                                 const char *content, size_t content_len) {
    if (!role || !content || content_len == SIZE_MAX) return 0;
    if (t->len == t->cap) {
        size_t next = t->cap ? t->cap * 2 : 8;
        if (next < t->cap || next > SIZE_MAX / sizeof(*t->items)) return 0;
        owned_message *items = realloc(t->items, next * sizeof(*items));
        if (!items) return 0;
        t->items = items;
        t->cap = next;
    }
    char *r = strdup(role);
    char *c = malloc(content_len + 1);
    if (!r || !c) {
        free(r);
        free(c);
        return 0;
    }
    memcpy(c, content, content_len);
    c[content_len] = '\0';
    t->items[t->len++] =
        (owned_message){.role = r, .content = c, .content_len = content_len};
    return 1;
}

static int transcript_append(transcript *t, const char *role, const char *content) {
    return transcript_append_len(t, role, content, strlen(content));
}

static void transcript_free(transcript *t) {
    for (size_t i = 0; i < t->len; ++i) {
        free(t->items[i].role);
        free(t->items[i].content);
    }
    free(t->items);
    *t = (transcript){0};
}

static int continue_running(void *opaque) {
    (void)opaque;
    return !interrupted;
}

static int output_sink(const char *bytes, size_t len, void *opaque) {
    output_buffer *out = opaque;
    if (interrupted) return 0;
    if (len && fwrite(bytes, 1, len, stdout) != len) {
        out->failed = 1;
        return 0;
    }
    if (fflush(stdout) == EOF) {
        out->failed = 1;
        return 0;
    }
    if (len > SIZE_MAX - out->len - 1) {
        out->failed = 1;
        return 0;
    }
    size_t need = out->len + len + 1;
    if (need > out->cap) {
        size_t next = out->cap ? out->cap : 4096;
        while (next < need) {
            if (next > SIZE_MAX / 2) {
                out->failed = 1;
                return 0;
            }
            next *= 2;
        }
        char *grown = realloc(out->bytes, next);
        if (!grown) {
            out->failed = 1;
            return 0;
        }
        out->bytes = grown;
        out->cap = next;
    }
    memcpy(out->bytes + out->len, bytes, len);
    out->len += len;
    out->bytes[out->len] = '\0';
    return 1;
}

// Returns 1 for a complete line (without newline), 0 for clean EOF,
// -1 for a read error, and -2 for a signal notification. Reading through
// poll + the self-pipe closes the check-before-getline signal race.
static int read_line_interruptible(input_reader *reader, char **line,
                                   size_t *line_len, size_t *line_cap,
                                   int *had_newline) {
    *line_len = 0;
    *had_newline = 0;
    for (;;) {
        while (reader->pos < reader->len) {
            const unsigned char byte = reader->pending[reader->pos++];
            if (byte == '\n') { *had_newline = 1; return 1; }
            if (*line_len == *line_cap) {
                size_t next = *line_cap ? *line_cap * 2 : 256;
                if (next < *line_cap) { errno = ENOMEM; return -1; }
                char *grown = realloc(*line, next);
                if (!grown) return -1;
                *line = grown;
                *line_cap = next;
            }
            (*line)[(*line_len)++] = (char)byte;
        }
        reader->pos = reader->len = 0;
        if (reader->eof) return *line_len ? 1 : 0;

        struct pollfd fds[2] = {
            {.fd = STDIN_FILENO, .events = POLLIN},
            {.fd = signal_pipe[0], .events = POLLIN}
        };
        int ready;
        do {
            ready = poll(fds, 2, -1);
        } while (ready < 0 && errno == EINTR && !interrupted);
        if (ready < 0) {
            if (interrupted) return -2;
            return -1;
        }
        if (interrupted || (fds[1].revents & (POLLIN | POLLERR | POLLHUP | POLLNVAL)))
            return -2;
        if (fds[0].revents & (POLLERR | POLLNVAL)) {
            errno = EIO;
            return -1;
        }
        if (fds[0].revents & (POLLIN | POLLHUP)) {
            ssize_t n;
            do {
                n = read(STDIN_FILENO, reader->pending, sizeof(reader->pending));
            } while (n < 0 && errno == EINTR && !interrupted);
            if (n < 0) {
                if (interrupted) return -2;
                return -1;
            }
            if (n == 0) reader->eof = 1;
            else reader->len = (size_t)n;
        }
    }
}

static int run_turn(q27_agent_engine *engine, transcript *chat, int think,
                    uint32_t max_tokens) {
    if (interrupted) {
        fprintf(stderr, "q27-agent: pending interrupt; generation not started\n");
        return 0;
    }
    q27_agent_message *view = calloc(chat->len, sizeof(*view));
    if (!view) {
        fprintf(stderr, "q27-agent: out of memory\n");
        return 0;
    }
    for (size_t i = 0; i < chat->len; ++i) {
        view[i].role = chat->items[i].role;
        view[i].content = chat->items[i].content;
        view[i].content_len = chat->items[i].content_len;
    }

    output_buffer output = {0};
    char error[512] = {0};
    uint32_t prompt_tokens = 0, output_tokens = 0;
    q27_agent_status status = q27_agent_generate(
        engine, view, chat->len, think, max_tokens, output_sink,
        continue_running, &output, &prompt_tokens, &output_tokens,
        error, sizeof(error));
    free(view);
    if (!output.failed && (status == Q27_AGENT_OK || output.len > 0) &&
        (fputc('\n', stdout) == EOF || fflush(stdout) == EOF))
        output.failed = 1;

    if (output.failed) {
        fprintf(stderr, "q27-agent: output failure\n");
        free(output.bytes);
        return 0;
    }
    if (status == Q27_AGENT_CANCELLED) {
        fprintf(stderr, "q27-agent: interrupted after %u tokens\n", output_tokens);
        free(output.bytes);
        return 0;
    }
    if (status != Q27_AGENT_OK) {
        fprintf(stderr, "q27-agent: generation failed: %s\n",
                error[0] ? error : "unknown error");
        free(output.bytes);
        return 0;
    }
    if (!transcript_append_len(chat, "assistant",
                               output.bytes ? output.bytes : "", output.len)) {
        fprintf(stderr, "q27-agent: could not retain assistant turn\n");
        free(output.bytes);
        return 0;
    }
    free(output.bytes);
    fprintf(stderr, "[q27-agent prompt=%u output=%u; phase0 full-prefill]\n",
            prompt_tokens, output_tokens);
    return 1;
}

int main(int argc, char **argv) {
    const char *model = NULL, *tokenizer = NULL, *prompt = NULL;
    const char *system =
        "You are q27-agent, an experimental local coding assistant. "
        "Answer concisely. Tool execution is not enabled in this Phase-0 build.";
    uint32_t context = 8192, max_tokens = 512;
    int think = 1;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout, argv[0]);
            return 0;
        } else if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            if (++i == argc) { usage(stderr, argv[0]); return 2; }
            prompt = argv[i];
        } else if (!strcmp(arg, "-s") || !strcmp(arg, "--system")) {
            if (++i == argc) { usage(stderr, argv[0]); return 2; }
            system = argv[i];
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--max-tokens")) {
            if (++i == argc || !parse_u32(argv[i], &max_tokens)) {
                fprintf(stderr, "q27-agent: invalid max token count\n");
                return 2;
            }
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--context")) {
            if (++i == argc || !parse_u32(argv[i], &context)) {
                fprintf(stderr, "q27-agent: invalid context\n");
                return 2;
            }
        } else if (!strcmp(arg, "--no-think")) {
            think = 0;
        } else if (arg[0] == '-') {
            fprintf(stderr, "q27-agent: unknown option: %s\n", arg);
            return 2;
        } else if (!model) {
            model = arg;
        } else if (!tokenizer) {
            tokenizer = arg;
        } else {
            fprintf(stderr, "q27-agent: unexpected argument: %s\n", arg);
            return 2;
        }
    }

    if (!model || !tokenizer) {
        usage(stderr, argv[0]);
        return 2;
    }

    if (!setup_signal_pipe()) {
        fprintf(stderr, "q27-agent: could not create signal pipe: %s\n",
                strerror(errno));
        return 1;
    }
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = on_signal;
    sigemptyset(&action.sa_mask);
    // Deliberately omit SA_RESTART so a signal wakes an interactive getline.
    if (sigaction(SIGINT, &action, NULL) || sigaction(SIGTERM, &action, NULL)) {
        fprintf(stderr, "q27-agent: could not install signal handlers: %s\n",
                strerror(errno));
        close_signal_pipe();
        return 1;
    }

    char error[512] = {0};
    q27_agent_engine *engine = q27_agent_engine_open(
        model, tokenizer, context, error, sizeof(error));
    if (!engine) {
        fprintf(stderr, "q27-agent: engine open failed: %s\n",
                error[0] ? error : "unknown error");
        close_signal_pipe();
        return 1;
    }

    transcript chat = {0};
    int ok = transcript_append(&chat, "system", system);
    if (!ok) fprintf(stderr, "q27-agent: out of memory\n");

    if (ok && prompt) {
        ok = transcript_append(&chat, "user", prompt) &&
             run_turn(engine, &chat, think, max_tokens);
    } else if (ok) {
        char *line = NULL;
        size_t len = 0, cap = 0;
        input_reader reader = {0};
        fprintf(stderr, "q27-agent Phase 0; enter :quit to exit\n");
        while (ok) {
            if (interrupted) { ok = 0; break; }
            fputs("q27> ", stderr);
            fflush(stderr);
            errno = 0;
            int had_newline = 0;
            int result = read_line_interruptible(&reader, &line, &len, &cap,
                                                 &had_newline);
            if (result <= 0) {
                if (result == -2) {
                    fputc('\n', stderr);
                    ok = 0;
                } else if (result == -1) {
                    fprintf(stderr, "q27-agent: stdin read failed: %s\n",
                            strerror(errno));
                    ok = 0;
                }
                break;
            }
            if (had_newline && len > 0 && line[len-1] == '\r') --len;
            if ((len == 5 && !memcmp(line, ":quit", 5)) ||
                (len == 2 && !memcmp(line, ":q", 2))) break;
            if (len == 0) continue;
            ok = transcript_append_len(&chat, "user", line, len) &&
                 run_turn(engine, &chat, think, max_tokens);
        }
        free(line);
    }

    transcript_free(&chat);
    q27_agent_engine_close(engine);
    close_signal_pipe();
    return ok && !interrupted ? 0 : 1;
}
