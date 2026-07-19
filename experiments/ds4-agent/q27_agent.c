#define _POSIX_C_SOURCE 200809L

/*
 * Reduced DS4-style native-agent experiment for q27.
 *
 * The worker-owned direct-engine architecture is based on antirez/ds4's
 * ds4_agent.c. This Phase-0 file is intentionally small: it proves that a C
 * control loop can drive q27's C++/Metal engine without HTTP. Narrow bounded
 * tools and opt-in model tool calls share its owned event boundary. Parsing
 * happens only after the generation terminal; the control thread then submits
 * a separate tool command. Persistence and terminal UI remain later slices.
 * See THIRD_PARTY_NOTICES.md.
 */

#include "q27_agent_protocol.h"
#include "q27_agent_worker.h"

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
    uint32_t prompt_tokens;
} turn_accounting;

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
        "      --output-format F   text (default) or jsonl events\n"
        "      --workspace DIR     root for relative tools (default .)\n"
        "      --auto-tools        opt into model-driven local side effects\n"
        "      --max-tool-rounds N automatic tool-call bound (default 8)\n"
        "  -h, --help              show this help\n"
        "\n"
        "interactive tools: :read PATH, :search PATH NEEDLE, :shell COMMAND\n",
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

static int transcript_append_assistant(transcript *t, const char *content,
                                       size_t content_len, int think) {
    static const char no_think_prefix[] = "<think>\n\n</think>\n\n";
    if (think)
        return transcript_append_len(t, "assistant", content, content_len);
    const size_t prefix_len = sizeof(no_think_prefix) - 1;
    if (content_len > SIZE_MAX - prefix_len) return 0;
    char *combined = malloc(prefix_len + content_len);
    if (!combined) return 0;
    memcpy(combined, no_think_prefix, prefix_len);
    if (content_len) memcpy(combined + prefix_len, content, content_len);
    int ok = transcript_append_len(t, "assistant", combined,
                                   prefix_len + content_len);
    free(combined);
    return ok;
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

static int output_append(output_buffer *out, const unsigned char *bytes, size_t len) {
    if (len > SIZE_MAX - out->len - 1) return 0;
    size_t need = out->len + len + 1;
    if (need > out->cap) {
        size_t next = out->cap ? out->cap : 4096;
        while (next < need) {
            if (next > SIZE_MAX / 2) return 0;
            next *= 2;
        }
        char *grown = realloc(out->bytes, next);
        if (!grown) return 0;
        out->bytes = grown;
        out->cap = next;
    }
    if (len) memcpy(out->bytes + out->len, bytes, len);
    out->len += len;
    out->bytes[out->len] = '\0';
    return 1;
}

static const char *event_type_name(q27_agent_event_type type) {
    switch (type) {
    case Q27_EVENT_STATE: return "state";
    case Q27_EVENT_TEXT_DELTA: return "text_delta";
    case Q27_EVENT_TOOL_OUTPUT: return "tool_output";
    case Q27_EVENT_TURN_DONE: return "turn_done";
    case Q27_EVENT_TOOL_DONE: return "tool_done";
    case Q27_EVENT_REJECTED: return "rejected";
    case Q27_EVENT_ERROR: return "error";
    }
    return "unknown";
}

static const char *status_name(q27_agent_status status) {
    switch (status) {
    case Q27_AGENT_OK: return "ok";
    case Q27_AGENT_CANCELLED: return "cancelled";
    case Q27_AGENT_REJECTED: return "rejected";
    case Q27_AGENT_ERROR: return "error";
    }
    return "unknown";
}

static const char *worker_state_name(q27_agent_worker_state state) {
    switch (state) {
    case Q27_WORKER_STARTING: return "starting";
    case Q27_WORKER_IDLE: return "idle";
    case Q27_WORKER_GENERATING: return "generating";
    case Q27_WORKER_TOOL_RUNNING: return "tool_running";
    case Q27_WORKER_STOPPING: return "stopping";
    case Q27_WORKER_ERROR: return "error";
    case Q27_WORKER_STOPPED: return "stopped";
    }
    return "unknown";
}

static const char *tool_kind_name(q27_agent_tool_kind kind) {
    switch (kind) {
    case Q27_TOOL_NONE: return "none";
    case Q27_TOOL_READ: return "read";
    case Q27_TOOL_SEARCH: return "search";
    case Q27_TOOL_EDIT: return "edit";
    case Q27_TOOL_SHELL: return "shell";
    }
    return "unknown";
}

static char *base64_encode(const unsigned char *data, size_t len) {
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (len > SIZE_MAX - 2) return NULL;
    size_t groups = (len + 2) / 3;
    if (groups > (SIZE_MAX - 1) / 4) return NULL;
    size_t out_len = 4 * groups;
    char *out = malloc(out_len + 1);
    if (!out) return NULL;
    size_t i = 0, o = 0;
    while (i < len) {
        uint32_t a = data[i++];
        uint32_t b = i < len ? data[i++] : 0;
        uint32_t c = i < len ? data[i++] : 0;
        uint32_t triple = (a << 16) | (b << 8) | c;
        out[o++] = table[(triple >> 18) & 63];
        out[o++] = table[(triple >> 12) & 63];
        out[o++] = table[(triple >> 6) & 63];
        out[o++] = table[triple & 63];
    }
    if (len % 3 == 1) out[out_len - 2] = out[out_len - 1] = '=';
    else if (len % 3 == 2) out[out_len - 1] = '=';
    out[out_len] = '\0';
    return out;
}

static int print_json_event(const q27_agent_event *event) {
    char *data = base64_encode(event->data, event->data_len);
    if (!data) return 0;
    int ok = fprintf(stdout,
        "{\"seq\":%llu,\"command_id\":%llu,\"type\":\"%s\","
        "\"state\":\"%s\",\"status\":\"%s\",\"data_b64\":\"%s\","
        "\"prompt_tokens\":%u,\"cached_tokens\":%u,"
        "\"prefill_tokens\":%u,\"output_tokens\":%u,"
        "\"tool_call_complete\":%s,"
        "\"tool_kind\":\"%s\",\"tool_exit_code\":%d,"
        "\"tool_flags\":%u,\"tool_output_bytes\":%u}\n",
        (unsigned long long)event->sequence,
        (unsigned long long)event->command_id,
        event_type_name(event->type), worker_state_name(event->state),
        status_name(event->status), data,
        event->prompt_tokens, event->cached_tokens, event->prefill_tokens,
        event->output_tokens,
        event->tool_call_complete ? "true" : "false",
        tool_kind_name(event->tool_kind),
        event->tool_exit_code, event->tool_flags, event->tool_output_bytes) >= 0 &&
        fflush(stdout) != EOF;
    free(data);
    return ok;
}

// Returns 1 for a complete line (without newline), 0 for clean EOF,
// -1 for a read error, and -2 for a signal notification. Reading through
// poll + the self-pipe closes the check-before-blocking-read signal race.
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

static int run_tool(q27_agent_worker *worker,
                    const q27_agent_tool_request *request, int jsonl,
                    int display_text, output_buffer *captured,
                    q27_agent_tool_result *completed) {
    if (captured) *captured = (output_buffer){0};
    if (completed) *completed = (q27_agent_tool_result){0};
    char error[512] = {0};
    uint64_t command_id = 0;
    q27_agent_status submitted = q27_agent_worker_submit_tool(
        worker, request, continue_running, NULL, &command_id,
        error, sizeof(error));
    if (submitted != Q27_AGENT_OK) {
        fprintf(stderr, "q27-agent: tool submission rejected: %s\n",
                error[0] ? error : "unknown error");
        return 0;
    }

    int terminal = 0;
    q27_agent_status status = Q27_AGENT_ERROR;
    int32_t exit_code = -1;
    uint32_t flags = 0, output_bytes = 0;
    while (!terminal) {
        q27_agent_event event;
        int got = q27_agent_worker_next_event(worker, &event,
                                               error, sizeof(error));
        if (got <= 0) {
            fprintf(stderr, "q27-agent: tool event stream ended: %s\n",
                    got < 0 && error[0] ? error : "worker stopped");
            if (captured) { free(captured->bytes); *captured = (output_buffer){0}; }
            return 0;
        }
        if (event.command_id != command_id) {
            fprintf(stderr, "q27-agent: tool event command mismatch\n");
            q27_agent_event_free(&event);
            q27_agent_worker_request_stop(worker);
            if (captured) { free(captured->bytes); *captured = (output_buffer){0}; }
            return 0;
        }
        int output_failed = 0;
        if (event.type == Q27_EVENT_TOOL_OUTPUT && event.data_len) {
            if (captured && !output_append(captured, event.data, event.data_len))
                output_failed = 1;
            if (!jsonl && display_text &&
                (fwrite(event.data, 1, event.data_len, stdout) != event.data_len ||
                 fflush(stdout) == EOF))
                output_failed = 1;
        }
        if (jsonl && !print_json_event(&event)) output_failed = 1;
        terminal = event.type == Q27_EVENT_TOOL_DONE ||
                   event.type == Q27_EVENT_REJECTED ||
                   event.type == Q27_EVENT_ERROR;
        if (terminal) {
            status = event.status;
            exit_code = event.tool_exit_code;
            flags = event.tool_flags;
            output_bytes = event.tool_output_bytes;
            if (completed) {
                completed->exit_code = event.tool_exit_code;
                completed->flags = event.tool_flags;
                completed->output_bytes = event.tool_output_bytes;
            }
            size_t n = event.data_len < sizeof(error) - 1 ?
                       event.data_len : sizeof(error) - 1;
            if (n) memcpy(error, event.data, n);
            error[n] = '\0';
            if (completed)
                snprintf(completed->message, sizeof(completed->message),
                         "%s", error);
        }
        q27_agent_event_free(&event);
        if (output_failed) {
            q27_agent_worker_request_stop(worker);
            fprintf(stderr, "q27-agent: tool output failure\n");
            if (captured) { free(captured->bytes); *captured = (output_buffer){0}; }
            return 0;
        }
    }
    if (status == Q27_AGENT_CANCELLED) {
        fprintf(stderr, "q27-agent: tool interrupted\n");
        if (captured) { free(captured->bytes); *captured = (output_buffer){0}; }
        return 0;
    }
    if (status != Q27_AGENT_OK) {
        fprintf(stderr, "q27-agent: tool infrastructure failure: %s\n",
                error[0] ? error : "unknown error");
        if (captured) { free(captured->bytes); *captured = (output_buffer){0}; }
        return 0;
    }
    if (captured && captured->len != output_bytes) {
        fprintf(stderr, "q27-agent: tool output accounting mismatch\n");
        free(captured->bytes);
        *captured = (output_buffer){0};
        return 0;
    }
    fprintf(stderr, "[q27-tool exit=%d flags=%u output=%u%s%s]\n",
            exit_code, flags, output_bytes, error[0] ? "; " : "",
            error[0] ? error : "");
    return 1;
}

static int run_turn(q27_agent_worker *worker, transcript *chat, int think,
                    int enable_tools, uint32_t max_tokens, int jsonl,
                    output_buffer *completed_output,
                    int *completed_tool_call,
                    turn_accounting *completed_accounting) {
    if (completed_output) *completed_output = (output_buffer){0};
    if (completed_tool_call) *completed_tool_call = 0;
    if (completed_accounting) *completed_accounting = (turn_accounting){0};
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

    char error[512] = {0};
    uint64_t command_id = 0;
    q27_agent_status submitted = q27_agent_worker_submit(
        worker, view, chat->len, think, enable_tools, max_tokens,
        continue_running, NULL, &command_id, error, sizeof(error));
    // submit deep-copies every message; the UI transcript is no longer pinned
    // for the duration of generation.
    free(view);
    if (submitted != Q27_AGENT_OK) {
        fprintf(stderr, "q27-agent: submission rejected: %s\n",
                error[0] ? error : "unknown error");
        return 0;
    }

    output_buffer output = {0};
    q27_agent_status status = Q27_AGENT_ERROR;
    uint32_t prompt_tokens = 0, cached_tokens = 0;
    uint32_t prefill_tokens = 0, output_tokens = 0;
    int tool_call_complete = 0, terminal = 0;
    while (!terminal) {
        q27_agent_event event;
        int got = q27_agent_worker_next_event(worker, &event,
                                               error, sizeof(error));
        if (got <= 0) {
            fprintf(stderr, "q27-agent: event stream ended before terminal: %s\n",
                    got < 0 && error[0] ? error : "worker stopped");
            free(output.bytes);
            return 0;
        }
        if (event.command_id != command_id) {
            fprintf(stderr, "q27-agent: event command mismatch\n");
            q27_agent_event_free(&event);
            free(output.bytes);
            q27_agent_worker_request_stop(worker);
            return 0;
        }
        if (event.type == Q27_EVENT_TEXT_DELTA) {
            if (!output_append(&output, event.data, event.data_len))
                output.failed = 1;
            if (!jsonl && !output.failed && event.data_len &&
                (fwrite(event.data, 1, event.data_len, stdout) != event.data_len ||
                 fflush(stdout) == EOF))
                output.failed = 1;
        }
        if (jsonl && !output.failed && !print_json_event(&event))
            output.failed = 1;

        terminal = event.type == Q27_EVENT_TURN_DONE ||
                   event.type == Q27_EVENT_REJECTED ||
                   event.type == Q27_EVENT_ERROR;
        if (terminal) {
            status = event.status;
            prompt_tokens = event.prompt_tokens;
            cached_tokens = event.cached_tokens;
            prefill_tokens = event.prefill_tokens;
            output_tokens = event.output_tokens;
            tool_call_complete = event.tool_call_complete;
            size_t n = event.data_len < sizeof(error) - 1 ?
                       event.data_len : sizeof(error) - 1;
            if (n) memcpy(error, event.data, n);
            error[n] = '\0';
        }
        q27_agent_event_free(&event);
        if (output.failed) {
            q27_agent_worker_request_stop(worker);
            fprintf(stderr, "q27-agent: output failure\n");
            free(output.bytes);
            return 0;
        }
    }

    if (!jsonl && (status == Q27_AGENT_OK || output.len > 0) &&
        (fputc('\n', stdout) == EOF || fflush(stdout) == EOF)) {
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
    if (!transcript_append_assistant(chat,
                                     output.bytes ? output.bytes : "",
                                     output.len, think)) {
        fprintf(stderr, "q27-agent: could not retain assistant turn\n");
        free(output.bytes);
        return 0;
    }
    if (completed_output) {
        *completed_output = output;
    } else {
        free(output.bytes);
    }
    if (completed_tool_call) *completed_tool_call = tool_call_complete;
    if (completed_accounting) {
        completed_accounting->prompt_tokens = prompt_tokens;
    }
    fprintf(stderr,
            "[q27-agent prompt=%u cached=%u prefill=%u output=%u; %s%s]\n",
            prompt_tokens, cached_tokens, prefill_tokens, output_tokens,
            cached_tokens ? "exact-prefix session reuse" : "reset/re-prefill",
            tool_call_complete ? "; closed tool call" : "");
    return 1;
}

static int append_tool_response(transcript *chat,
                                const output_buffer *output,
                                const q27_agent_tool_result *result) {
    output_buffer wrapped = {0};
    static const char open[] = "<tool_response>\n";
    static const char close[] = "\n</tool_response>";
    char status[512];
    int n = snprintf(status, sizeof(status),
                     "\n[q27-tool exit=%d flags=%u output=%u%s%s]",
                     result->exit_code, result->flags, result->output_bytes,
                     result->message[0] ? "; " : "",
                     result->message[0] ? result->message : "");
    int ok = n >= 0 && (size_t)n < sizeof(status) &&
             output_append(&wrapped, (const unsigned char *)open,
                           sizeof(open) - 1) &&
             output_append(&wrapped,
                           (const unsigned char *)(output->bytes ?
                                                   output->bytes : ""),
                           output->len) &&
             output_append(&wrapped, (const unsigned char *)status, (size_t)n) &&
             output_append(&wrapped, (const unsigned char *)close,
                           sizeof(close) - 1) &&
             transcript_append_len(chat, "user", wrapped.bytes, wrapped.len);
    free(wrapped.bytes);
    return ok;
}

static int run_agent_cycle(q27_agent_worker *worker, transcript *chat,
                           int think, uint32_t context_tokens,
                           uint32_t max_tokens, int jsonl,
                           uint32_t max_tool_rounds) {
    uint32_t tool_rounds = 0;
    for (;;) {
        output_buffer generated = {0};
        int engine_closed_call = 0;
        turn_accounting accounting = {0};
        if (!run_turn(worker, chat, think, 1, max_tokens, jsonl,
                      &generated, &engine_closed_call, &accounting)) {
            free(generated.bytes);
            return 0;
        }

        q27_agent_tool_call call = {0};
        char error[512] = {0};
        const size_t generated_bytes = generated.len;
        q27_agent_tool_call_status parsed = q27_agent_parse_tool_call(
            (const unsigned char *)(generated.bytes ? generated.bytes : ""),
            generated.len, &call, error, sizeof(error));
        if (parsed == Q27_TOOL_CALL_NONE) {
            free(generated.bytes);
            if (engine_closed_call) {
                fprintf(stderr,
                        "q27-agent: engine/parser tool-call mismatch\n");
                return 0;
            }
            return 1;
        }
        free(generated.bytes);
        if (parsed != Q27_TOOL_CALL_VALID || !engine_closed_call) {
            fprintf(stderr, "q27-agent: invalid model tool call: %s\n",
                    error[0] ? error : "constraint/parser mismatch");
            q27_agent_tool_call_free(&call);
            return 0;
        }
        if (tool_rounds >= max_tool_rounds) {
            fprintf(stderr, "q27-agent: automatic tool-round limit reached\n");
            q27_agent_tool_call_free(&call);
            return 0;
        }

        // Reserve enough worst-case one-byte tokens for the tool-response
        // tags, status metadata, ChatML role framing, and next assistant
        // prefix. The remaining byte cap is conservative because byte-BPE
        // cannot encode one input byte as more than one token.
        static const uint32_t response_reserve_tokens = 512;
        // Re-rendering assistant bytes can segment differently from the
        // emitted token IDs. Charge every generated byte as one token rather
        // than trusting output_tokens; byte-BPE cannot exceed that bound.
        const uint64_t committed = (uint64_t)accounting.prompt_tokens +
                                   generated_bytes;
        const uint64_t required = committed + max_tokens +
                                  response_reserve_tokens;
        if ((uint64_t)context_tokens + 1 <= required) {
            fprintf(stderr,
                    "q27-agent: insufficient context for bounded tool response\n");
            q27_agent_tool_call_free(&call);
            return 0;
        }
        const uint64_t room = (uint64_t)context_tokens + 1 - required;
        if (call.request.max_output_bytes > room)
            call.request.max_output_bytes = (uint32_t)room;

        fprintf(stderr,
                "[q27-agent automatic tool=%s round=%u/%u output-cap=%u]\n",
                tool_kind_name(call.request.kind), tool_rounds + 1,
                max_tool_rounds, call.request.max_output_bytes);
        output_buffer tool_output = {0};
        q27_agent_tool_result tool_result = {0};
        int ran = run_tool(worker, &call.request, jsonl, 0,
                           &tool_output, &tool_result);
        q27_agent_tool_call_free(&call);
        if (!ran) {
            free(tool_output.bytes);
            return 0;
        }
        if (!append_tool_response(chat, &tool_output, &tool_result)) {
            free(tool_output.bytes);
            fprintf(stderr, "q27-agent: could not retain tool response\n");
            return 0;
        }
        free(tool_output.bytes);
        ++tool_rounds;
    }
}

int main(int argc, char **argv) {
    const char *model = NULL, *tokenizer = NULL, *prompt = NULL;
    const char *workspace = ".";
    const char *system =
        "You are q27-agent, an experimental local coding assistant. "
        "Answer concisely.";
    uint32_t context = 8192, max_tokens = 512, max_tool_rounds = 8;
    int think = 1, jsonl = 0, auto_tools = 0;

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
        } else if (!strcmp(arg, "--auto-tools")) {
            auto_tools = 1;
        } else if (!strcmp(arg, "--max-tool-rounds")) {
            if (++i == argc || !parse_u32(argv[i], &max_tool_rounds) ||
                max_tool_rounds > 64) {
                fprintf(stderr, "q27-agent: max tool rounds must be 1..64\n");
                return 2;
            }
        } else if (!strcmp(arg, "--workspace")) {
            if (++i == argc || !argv[i][0]) {
                fprintf(stderr, "q27-agent: workspace path is required\n");
                return 2;
            }
            workspace = argv[i];
        } else if (!strcmp(arg, "--output-format")) {
            if (++i == argc ||
                (strcmp(argv[i], "text") && strcmp(argv[i], "jsonl"))) {
                fprintf(stderr, "q27-agent: output format must be text or jsonl\n");
                return 2;
            }
            jsonl = !strcmp(argv[i], "jsonl");
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
    q27_agent_worker *worker = q27_agent_worker_start_at(
        model, tokenizer, context, workspace, error, sizeof(error));
    if (!worker) {
        fprintf(stderr, "q27-agent: worker start failed: %s\n",
                error[0] ? error : "unknown error");
        close_signal_pipe();
        return 1;
    }

    transcript chat = {0};
    int ok;
    if (auto_tools) {
        const char *preamble = q27_agent_tool_preamble();
        output_buffer combined = {0};
        ok = preamble &&
             output_append(&combined, (const unsigned char *)preamble,
                           strlen(preamble)) &&
             output_append(&combined, (const unsigned char *)"\n\n", 2) &&
             output_append(&combined, (const unsigned char *)system,
                           strlen(system)) &&
             transcript_append_len(&chat, "system", combined.bytes,
                                   combined.len);
        free(combined.bytes);
    } else {
        ok = transcript_append(&chat, "system", system);
    }
    if (!ok) fprintf(stderr, "q27-agent: out of memory\n");

    if (ok && prompt) {
        ok = transcript_append(&chat, "user", prompt) &&
             (auto_tools ?
                  run_agent_cycle(worker, &chat, think, context, max_tokens,
                                  jsonl, max_tool_rounds) :
                  run_turn(worker, &chat, think, 0, max_tokens, jsonl,
                           NULL, NULL, NULL));
    } else if (ok) {
        char *line = NULL;
        size_t len = 0, cap = 0;
        input_reader reader = {0};
        fprintf(stderr, "q27-agent native session; enter :quit to exit\n");
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
            if (line[0] == ':') {
                if (memchr(line, '\0', len)) {
                    fprintf(stderr, "q27-agent: tool command contains NUL\n");
                    continue;
                }
                char *tool_line = strndup(line, len);
                if (!tool_line) { ok = 0; break; }
                q27_agent_tool_request request = {
                    .timeout_ms = 30000, .max_output_bytes = 256 * 1024};
                if (!strncmp(tool_line, ":read ", 6) && tool_line[6]) {
                    request.kind = Q27_TOOL_READ;
                    request.path = tool_line + 6;
                } else if (!strncmp(tool_line, ":search ", 8)) {
                    char *path = tool_line + 8;
                    char *space = strchr(path, ' ');
                    if (space && space[1]) {
                        *space = '\0';
                        request.kind = Q27_TOOL_SEARCH;
                        request.path = path;
                        request.input = (unsigned char *)(space + 1);
                        request.input_len = strlen(space + 1);
                    }
                } else if (!strncmp(tool_line, ":shell ", 7) && tool_line[7]) {
                    request.kind = Q27_TOOL_SHELL;
                    request.input = (unsigned char *)(tool_line + 7);
                    request.input_len = strlen(tool_line + 7);
                }
                if (request.kind == Q27_TOOL_NONE)
                    fprintf(stderr, "q27-agent: invalid tool command\n");
                else
                    ok = run_tool(worker, &request, jsonl, 1, NULL, NULL);
                free(tool_line);
                continue;
            }
            ok = transcript_append_len(&chat, "user", line, len) &&
                 (auto_tools ?
                      run_agent_cycle(worker, &chat, think, context, max_tokens,
                                      jsonl, max_tool_rounds) :
                      run_turn(worker, &chat, think, 0, max_tokens, jsonl,
                               NULL, NULL, NULL));
        }
        free(line);
    }

    transcript_free(&chat);
    q27_agent_worker_stop(worker);
    close_signal_pipe();
    return ok && !interrupted ? 0 : 1;
}
