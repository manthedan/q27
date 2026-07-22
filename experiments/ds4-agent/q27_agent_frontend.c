#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "q27_agent_frontend.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

static int g_fp1_protocol = 0;

void q27_fp1_set_protocol(int protocol) {
    g_fp1_protocol = protocol > 0 ? 1 : 0;
}

int q27_fp1_protocol(void) {
    return g_fp1_protocol;
}

int q27_fp1_stream_events(int jsonl) {
    return jsonl || g_fp1_protocol;
}

uint64_t q27_fp1_now_ms(void) {
    struct timeval tv;
    if (gettimeofday(&tv, NULL) != 0) return 0;
    return (uint64_t)tv.tv_sec * 1000ull + (uint64_t)tv.tv_usec / 1000ull;
}

int q27_fp1_utf8_valid(const unsigned char *data, size_t len) {
    if (!data && len) return 0;
    size_t i = 0;
    while (i < len) {
        unsigned char c = data[i];
        if (c <= 0x7f) {
            i++;
            continue;
        }
        size_t need;
        unsigned char lo, hi;
        if ((c & 0xe0) == 0xc0) {
            need = 2;
            lo = 0x80;
            hi = 0xbf;
            if (c < 0xc2) return 0; /* overlong */
        } else if ((c & 0xf0) == 0xe0) {
            need = 3;
            if (c == 0xe0) {
                lo = 0xa0;
                hi = 0xbf;
            } else if (c == 0xed) {
                lo = 0x80;
                hi = 0x9f;
            } else {
                lo = 0x80;
                hi = 0xbf;
            }
        } else if ((c & 0xf8) == 0xf0) {
            need = 4;
            if (c == 0xf0) {
                lo = 0x90;
                hi = 0xbf;
            } else if (c == 0xf4) {
                lo = 0x80;
                hi = 0x8f;
            } else if (c > 0xf4) {
                return 0;
            } else {
                lo = 0x80;
                hi = 0xbf;
            }
        } else {
            return 0;
        }
        if (i + need > len) return 0;
        unsigned char c1 = data[i + 1];
        if (c1 < lo || c1 > hi) return 0;
        for (size_t j = 2; j < need; ++j) {
            unsigned char cj = data[i + j];
            if (cj < 0x80 || cj > 0xbf) return 0;
        }
        i += need;
    }
    return 1;
}

char *q27_fp1_json_escape(const unsigned char *data, size_t len) {
    if (len && !data) return NULL;
    /* Worst case: every byte → \u00XX (6 chars). */
    if (len > (SIZE_MAX / 6) - 1) return NULL;
    size_t cap = len * 6 + 1;
    char *out = malloc(cap);
    if (!out) return NULL;
    size_t o = 0;
    for (size_t i = 0; i < len; ++i) {
        unsigned char c = data[i];
        if (c == '"' || c == '\\') {
            out[o++] = '\\';
            out[o++] = (char)c;
        } else if (c == '\b') {
            out[o++] = '\\';
            out[o++] = 'b';
        } else if (c == '\f') {
            out[o++] = '\\';
            out[o++] = 'f';
        } else if (c == '\n') {
            out[o++] = '\\';
            out[o++] = 'n';
        } else if (c == '\r') {
            out[o++] = '\\';
            out[o++] = 'r';
        } else if (c == '\t') {
            out[o++] = '\\';
            out[o++] = 't';
        } else if (c < 0x20) {
            static const char hex[] = "0123456789abcdef";
            out[o++] = '\\';
            out[o++] = 'u';
            out[o++] = '0';
            out[o++] = '0';
            out[o++] = hex[(c >> 4) & 0xf];
            out[o++] = hex[c & 0xf];
        } else {
            out[o++] = (char)c;
        }
    }
    out[o] = '\0';
    return out;
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

static const char *event_type_name(q27_agent_event_type type) {
    switch (type) {
    case Q27_EVENT_STATE: return "state";
    case Q27_EVENT_PREFILL_PROGRESS: return "prefill_progress";
    case Q27_EVENT_TEXT_DELTA: return "text_delta";
    case Q27_EVENT_TOOL_OUTPUT: return "tool_output";
    case Q27_EVENT_SELECTIONS: return "selection_handles";
    case Q27_EVENT_TURN_DONE: return "turn_done";
    case Q27_EVENT_TOOL_DONE: return "tool_done";
    case Q27_EVENT_SESSION_DONE: return "session_done";
    case Q27_EVENT_REJECTED: return "rejected";
    case Q27_EVENT_STALLED: return "generation_stalled";
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
    case Q27_AGENT_STALLED: return "stalled";
    }
    return "unknown";
}

static const char *worker_state_name(q27_agent_worker_state state) {
    switch (state) {
    case Q27_WORKER_STARTING: return "starting";
    case Q27_WORKER_IDLE: return "idle";
    case Q27_WORKER_GENERATING: return "generating";
    case Q27_WORKER_TOOL_RUNNING: return "tool_running";
    case Q27_WORKER_SESSION_IO: return "session_io";
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
    case Q27_TOOL_WRITE: return "write";
    case Q27_TOOL_OVERWRITE: return "overwrite";
    case Q27_TOOL_WRITE_PREFLIGHT: return "write_preflight";
    case Q27_TOOL_EDIT_PREFLIGHT: return "edit_preflight";
    case Q27_TOOL_OVERWRITE_PREFLIGHT: return "overwrite_preflight";
    }
    return "unknown";
}

static int json_put_escaped(FILE *out, const char *key,
                            const unsigned char *data, size_t len) {
    char *esc = q27_fp1_json_escape(data, len);
    if (!esc) return 0;
    int ok = fprintf(out, ",\"%s\":\"%s\"", key, esc) >= 0;
    free(esc);
    return ok;
}

static int json_put_cstr(FILE *out, const char *key, const char *value) {
    if (!value) {
        return fprintf(out, ",\"%s\":null", key) >= 0;
    }
    return json_put_escaped(out, key,
                            (const unsigned char *)value, strlen(value));
}

int q27_fp1_print_event(FILE *out, const q27_agent_event *event, int mode,
                        const char *client_req_id) {
    if (!out || !event) return 0;
    char *data_b64 = base64_encode(event->data, event->data_len);
    if (!data_b64) return 0;

    int ok = 0;
    if (mode <= 0) {
        /* Legacy jsonl (v0) — bit-for-bit shape of the pre-FP1 emitter. */
        ok = fprintf(out,
            "{\"seq\":%llu,\"command_id\":%llu,\"type\":\"%s\","
            "\"state\":\"%s\",\"status\":\"%s\",\"data_b64\":\"%s\","
            "\"prompt_tokens\":%u,\"cached_tokens\":%u,"
            "\"prefill_tokens\":%u,\"output_tokens\":%u,"
            "\"tool_call_complete\":%s,\"eos_reached\":%s,"
            "\"tool_kind\":\"%s\",\"tool_exit_code\":%d,"
            "\"tool_flags\":%u,\"tool_output_bytes\":%u}\n",
            (unsigned long long)event->sequence,
            (unsigned long long)event->command_id,
            event_type_name(event->type), worker_state_name(event->state),
            status_name(event->status), data_b64,
            event->prompt_tokens, event->cached_tokens, event->prefill_tokens,
            event->output_tokens,
            event->tool_call_complete ? "true" : "false",
            event->eos_reached ? "true" : "false",
            tool_kind_name(event->tool_kind),
            event->tool_exit_code, event->tool_flags,
            event->tool_output_bytes) >= 0;
    } else {
        const uint64_t ts = q27_fp1_now_ms();
        ok = fprintf(out,
            "{\"v\":1,\"seq\":%llu,\"type\":\"%s\",\"ts_ms\":%llu,"
            "\"command_id\":%llu",
            (unsigned long long)event->sequence,
            event_type_name(event->type),
            (unsigned long long)ts,
            (unsigned long long)event->command_id) >= 0;
        /* Correlation is per active command: the FP1 loop retains the
         * prompt's client_req_id for the whole turn and passes it here so
         * deltas/terminals/tool events can be matched to a request
         * (codex P1). NULL prints null. */
        if (ok) ok = json_put_cstr(out, "client_req_id", client_req_id);
        if (ok)
            ok = fprintf(out,
                ",\"state\":\"%s\",\"status\":\"%s\",\"code\":null",
                worker_state_name(event->state),
                status_name(event->status)) >= 0;

        /* Prefer UTF-8 text for chat-like payloads; always keep data_b64. */
        const int want_text =
            event->type == Q27_EVENT_TEXT_DELTA ||
            event->type == Q27_EVENT_TOOL_OUTPUT ||
            event->type == Q27_EVENT_SELECTIONS ||
            event->type == Q27_EVENT_REJECTED ||
            event->type == Q27_EVENT_STALLED ||
            event->type == Q27_EVENT_ERROR ||
            event->type == Q27_EVENT_SESSION_DONE;
        if (ok && want_text && event->data_len &&
            q27_fp1_utf8_valid(event->data, event->data_len)) {
            ok = json_put_escaped(out, "text", event->data, event->data_len);
        } else if (ok && want_text) {
            ok = fprintf(out, ",\"text\":null") >= 0;
        }

        if (ok && event->type == Q27_EVENT_TEXT_DELTA) {
            ok = fprintf(out, ",\"stream\":\"assistant\"") >= 0;
        }

        if (ok && event->type == Q27_EVENT_TOOL_OUTPUT) {
            if (event->data_len &&
                q27_fp1_utf8_valid(event->data, event->data_len)) {
                ok = fprintf(out, ",\"encoding\":\"utf8\"") >= 0;
            } else {
                ok = fprintf(out, ",\"encoding\":\"base64\"") >= 0;
            }
        }

        if (ok) {
            ok = fprintf(out,
                ",\"data_b64\":\"%s\","
                "\"prompt_tokens\":%u,\"cached_tokens\":%u,"
                "\"prefill_tokens\":%u,\"output_tokens\":%u,"
                "\"tool_call_complete\":%s,\"eos_reached\":%s,"
                "\"tool_kind\":\"%s\",\"tool_exit_code\":%d,"
                "\"tool_flags\":%u,\"tool_output_bytes\":%u}\n",
                data_b64,
                event->prompt_tokens, event->cached_tokens,
                event->prefill_tokens, event->output_tokens,
                event->tool_call_complete ? "true" : "false",
                event->eos_reached ? "true" : "false",
                tool_kind_name(event->tool_kind),
                event->tool_exit_code, event->tool_flags,
                event->tool_output_bytes) >= 0;
        }
    }

    free(data_b64);
    if (ok) ok = fflush(out) != EOF;
    return ok;
}

int q27_fp1_emit_hello(FILE *out, uint64_t seq, const char *model_path,
                       const char *tokenizer_path, uint32_t context,
                       const char *workspace, const char *session_path,
                       int auto_tools, int thinking, uint32_t max_tool_rounds,
                       int features_queue) {
    if (!out) return 0;
    const uint64_t ts = q27_fp1_now_ms();
    /* features without queue until P4 (features_queue gates it). */
    int ok = fprintf(out,
        "{\"v\":1,\"seq\":%llu,\"type\":\"hello\",\"ts_ms\":%llu,"
        "\"command_id\":0,\"client_req_id\":null,"
        "\"state\":\"idle\",\"status\":\"ok\",\"code\":null,"
        "\"protocol\":1",
        (unsigned long long)seq, (unsigned long long)ts) >= 0;
    if (ok) ok = json_put_cstr(out, "model_path", model_path);
    if (ok) ok = json_put_cstr(out, "tokenizer_path", tokenizer_path);
    if (ok) {
        ok = fprintf(out, ",\"context\":%u", context) >= 0;
    }
    if (ok) ok = json_put_cstr(out, "workspace", workspace);
    if (ok) ok = json_put_cstr(out, "session_path", session_path);
    if (ok) {
        ok = fprintf(out,
            ",\"auto_tools\":%s,\"thinking\":%s,\"max_tool_rounds\":%u,"
            "\"features\":[\"tools\",\"selections\",\"compact\",\"session\"%s]}\n",
            auto_tools ? "true" : "false",
            thinking ? "true" : "false",
            max_tool_rounds,
            features_queue ? ",\"queue\"" : "") >= 0;
    }
    if (ok) ok = fflush(out) != EOF;
    return ok;
}

int q27_fp1_emit_idle(FILE *out, uint64_t seq, uint32_t ctx_used,
                      uint32_t ctx_size, uint32_t queue_len) {
    if (!out) return 0;
    const uint64_t ts = q27_fp1_now_ms();
    int ok = fprintf(out,
        "{\"v\":1,\"seq\":%llu,\"type\":\"idle\",\"ts_ms\":%llu,"
        "\"command_id\":0,\"client_req_id\":null,"
        "\"state\":\"idle\",\"status\":\"ok\",\"code\":null,"
        "\"ctx_used\":%u,\"ctx_size\":%u,\"queue_len\":%u}\n",
        (unsigned long long)seq, (unsigned long long)ts,
        ctx_used, ctx_size, queue_len) >= 0;
    if (ok) ok = fflush(out) != EOF;
    return ok;
}

int q27_fp1_emit_bye(FILE *out, uint64_t seq, const char *reason) {
    if (!out) return 0;
    if (!reason) reason = "quit";
    const char *status = "ok";
    if (!strcmp(reason, "error")) status = "error";
    const uint64_t ts = q27_fp1_now_ms();
    int ok = fprintf(out,
        "{\"v\":1,\"seq\":%llu,\"type\":\"bye\",\"ts_ms\":%llu,"
        "\"command_id\":0,\"client_req_id\":null,"
        "\"state\":\"stopped\",\"status\":\"%s\",\"code\":null",
        (unsigned long long)seq, (unsigned long long)ts, status) >= 0;
    if (ok) ok = json_put_cstr(out, "reason", reason);
    if (ok) ok = fprintf(out, "}\n") >= 0;
    if (ok) ok = fflush(out) != EOF;
    return ok;
}

int q27_fp1_emit_rejected(FILE *out, uint64_t seq, const char *client_req_id,
                          const char *code, const char *text,
                          const char *state) {
    if (!out || !code) return 0;
    if (!state) state = "idle";
    if (!text) text = "";
    const uint64_t ts = q27_fp1_now_ms();
    int ok = fprintf(out,
        "{\"v\":1,\"seq\":%llu,\"type\":\"rejected\",\"ts_ms\":%llu,"
        "\"command_id\":0",
        (unsigned long long)seq, (unsigned long long)ts) >= 0;
    if (ok) ok = json_put_cstr(out, "client_req_id", client_req_id);
    if (ok) {
        ok = fprintf(out, ",\"state\":\"%s\",\"status\":\"rejected\"",
                     state) >= 0;
    }
    if (ok) ok = json_put_cstr(out, "code", code);
    if (ok) ok = json_put_cstr(out, "text", text);
    if (ok) ok = fprintf(out, "}\n") >= 0;
    if (ok) ok = fflush(out) != EOF;
    return ok;
}

int q27_fp1_emit_notice(FILE *out, uint64_t seq, const char *client_req_id,
                        const char *severity, const char *code,
                        const char *text, const char *state) {
    if (!out) return 0;
    if (!severity) severity = "info";
    if (!state) state = "idle";
    if (!text) text = "";
    const uint64_t ts = q27_fp1_now_ms();
    int ok = fprintf(out,
        "{\"v\":1,\"seq\":%llu,\"type\":\"notice\",\"ts_ms\":%llu,"
        "\"command_id\":0",
        (unsigned long long)seq, (unsigned long long)ts) >= 0;
    if (ok) ok = json_put_cstr(out, "client_req_id", client_req_id);
    if (ok) {
        ok = fprintf(out, ",\"state\":\"%s\",\"status\":\"ok\"", state) >= 0;
    }
    if (ok) ok = json_put_cstr(out, "severity", severity);
    if (ok && code) ok = json_put_cstr(out, "code", code);
    else if (ok) ok = fprintf(out, ",\"code\":null") >= 0;
    if (ok) ok = json_put_cstr(out, "text", text);
    if (ok) ok = fprintf(out, "}\n") >= 0;
    if (ok) ok = fflush(out) != EOF;
    return ok;
}

int q27_fp1_emit_session_done(FILE *out, uint64_t seq, uint64_t command_id,
                              const char *client_req_id, const char *action,
                              const char *status, const char *text,
                              uint32_t prompt_tokens, const char *state) {
    if (!out || !action) return 0;
    if (!status) status = "ok";
    if (!state) state = "idle";
    if (!text) text = "";
    const uint64_t ts = q27_fp1_now_ms();
    int ok = fprintf(out,
        "{\"v\":1,\"seq\":%llu,\"type\":\"session_done\",\"ts_ms\":%llu,"
        "\"command_id\":%llu",
        (unsigned long long)seq, (unsigned long long)ts,
        (unsigned long long)command_id) >= 0;
    if (ok) ok = json_put_cstr(out, "client_req_id", client_req_id);
    if (ok) {
        ok = fprintf(out, ",\"state\":\"%s\",\"status\":\"%s\",\"code\":null",
                     state, status) >= 0;
    }
    if (ok) ok = json_put_cstr(out, "action", action);
    if (ok) ok = json_put_cstr(out, "text", text);
    if (ok) {
        ok = fprintf(out, ",\"prompt_tokens\":%u}\n", prompt_tokens) >= 0;
    }
    if (ok) ok = fflush(out) != EOF;
    return ok;
}

int q27_fp1_emit_state(FILE *out, uint64_t seq, const char *client_req_id,
                       const char *state) {
    if (!out || !state) return 0;
    const uint64_t ts = q27_fp1_now_ms();
    int ok = fprintf(out,
        "{\"v\":1,\"seq\":%llu,\"type\":\"state\",\"ts_ms\":%llu,"
        "\"command_id\":0",
        (unsigned long long)seq, (unsigned long long)ts) >= 0;
    if (ok) ok = json_put_cstr(out, "client_req_id", client_req_id);
    if (ok) {
        ok = fprintf(out,
                     ",\"state\":\"%s\",\"status\":\"ok\",\"code\":null}\n",
                     state) >= 0;
    }
    if (ok) ok = fflush(out) != EOF;
    return ok;
}

int q27_fp1_emit_tool_start(FILE *out, uint64_t seq, uint64_t command_id,
                            const char *client_req_id, const char *tool_kind,
                            const char *detail, int preflight) {
    if (!out) return 0;
    if (!tool_kind) tool_kind = "unknown";
    if (!detail) detail = "";
    const uint64_t ts = q27_fp1_now_ms();
    int ok = fprintf(out,
        "{\"v\":1,\"seq\":%llu,\"type\":\"tool_start\",\"ts_ms\":%llu,"
        "\"command_id\":%llu",
        (unsigned long long)seq, (unsigned long long)ts,
        (unsigned long long)command_id) >= 0;
    if (ok) ok = json_put_cstr(out, "client_req_id", client_req_id);
    if (ok) {
        ok = fprintf(out,
                     ",\"state\":\"tool_running\",\"status\":\"ok\","
                     "\"code\":null") >= 0;
    }
    if (ok) ok = json_put_cstr(out, "tool_kind", tool_kind);
    if (ok) ok = json_put_cstr(out, "detail", detail);
    if (ok) {
        ok = fprintf(out, ",\"preflight\":%s}\n",
                     preflight ? "true" : "false") >= 0;
    }
    if (ok) ok = fflush(out) != EOF;
    return ok;
}

const char *q27_fp1_help_text(void) {
    return
        "Frontend Protocol v1 ops (NDJSON ClientMessage on stdin):\n"
        "  prompt   {\"v\":1,\"op\":\"prompt\",\"text\":\"…\"}\n"
        "  cancel   {\"v\":1,\"op\":\"cancel\"}\n"
        "  quit     {\"v\":1,\"op\":\"quit\"}\n"
        "  queue_clear  {\"v\":1,\"op\":\"queue_clear\"}  drop pending prompts\n"
        "  save     {\"v\":1,\"op\":\"save\"}           requires --session\n"
        "  compact  {\"v\":1,\"op\":\"compact\"}\n"
        "  session  {\"v\":1,\"op\":\"session\"}        point-in-time report\n"
        "  new      {\"v\":1,\"op\":\"new\"}            clear transcript\n"
        "  help     {\"v\":1,\"op\":\"help\"}\n"
        "  tool     {\"v\":1,\"op\":\"tool\",\"kind\":\"read|search|shell\",…}\n"
        "             read:   path\n"
        "             search: path, needle\n"
        "             shell:  command\n"
        "\n"
        "hello.features includes \"queue\": prompts while busy enqueue (cap 8).\n"
        "Outcomes arrive as ServerEvents on stdout (turn_done, session_done,\n"
        "queue, rejected, notice, …). Gate client behavior on hello.features.\n";
}

/* ── P2 control channel ──────────────────────────────────────────────── */

#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <ctype.h>
#include <poll.h>
#include <fcntl.h>

#define FP1_OP_QUEUE_CAP 16
#define FP1_DROP_CAP 8
/* Backend-owned prompt queue cap (§6.1) — mirrors linenoise default of 8. */
#define FP1_PROMPT_QUEUE_CAP 8
/* FP1 §3.3 soft cap: 1 MiB per NDJSON line. */
#define FP1_LINE_MAX (1024u * 1024u)

typedef struct {
    char *text;
    char *client_req_id;
} fp1_prompt_item;

static struct {
    fp1_prompt_item items[FP1_PROMPT_QUEUE_CAP];
    size_t head;
    size_t len;
    pthread_mutex_t mu;
} g_prompts = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
};

typedef struct {
    q27_fp1_op items[FP1_OP_QUEUE_CAP];
    size_t head;
    size_t len;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_t thread;
    int thread_started;
    int stop;
    int cancel_requested;
    int quit_requested;
    char quit_reason[32];
    int wake_pipe[2];
    /* Silent drops are forbidden (§2.6): bounded per-request rejection ring
     * for control-plane overflow (codex P2); beyond cap, coalesce into a
     * count emitted as one summary rejection. */
    size_t drop_head;
    size_t drop_len;
    char drop_reqs[FP1_DROP_CAP][65];
    char drop_codes[FP1_DROP_CAP][32];
    uint32_t drop_coalesced;
} fp1_control;

static fp1_control g_ctl = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .cv = PTHREAD_COND_INITIALIZER,
    .wake_pipe = {-1, -1},
};

void q27_fp1_op_free(q27_fp1_op *op) {
    if (!op) return;
    free(op->client_req_id);
    free(op->text);
    free(op->tool_kind);
    free(op->path);
    free(op->needle);
    free(op->command);
    *op = (q27_fp1_op){0};
}

static char *dup_n(const char *s, size_t n) {
    char *out = malloc(n + 1);
    if (!out) return NULL;
    if (n) memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static int json_unescape_to(const char *in, size_t in_len, char **out) {
    char *buf = malloc(in_len + 1);
    if (!buf) return 0;
    size_t o = 0;
    for (size_t i = 0; i < in_len; ++i) {
        char c = in[i];
        if (c != '\\') {
            buf[o++] = c;
            continue;
        }
        if (i + 1 >= in_len) {
            free(buf);
            return 0;
        }
        char e = in[++i];
        switch (e) {
        case '"':
        case '\\':
        case '/':
            buf[o++] = e;
            break;
        case 'b':
            buf[o++] = '\b';
            break;
        case 'f':
            buf[o++] = '\f';
            break;
        case 'n':
            buf[o++] = '\n';
            break;
        case 'r':
            buf[o++] = '\r';
            break;
        case 't':
            buf[o++] = '\t';
            break;
        case 'u': {
            if (i + 4 >= in_len) {
                free(buf);
                return 0;
            }
            unsigned code = 0;
            for (int k = 0; k < 4; ++k) {
                char h = in[++i];
                code <<= 4;
                if (h >= '0' && h <= '9') code |= (unsigned)(h - '0');
                else if (h >= 'a' && h <= 'f') code |= (unsigned)(h - 'a' + 10);
                else if (h >= 'A' && h <= 'F') code |= (unsigned)(h - 'A' + 10);
                else {
                    free(buf);
                    return 0;
                }
            }
            if (code < 0x80) {
                buf[o++] = (char)code;
            } else if (code < 0x800) {
                buf[o++] = (char)(0xc0 | (code >> 6));
                buf[o++] = (char)(0x80 | (code & 0x3f));
            } else {
                buf[o++] = (char)(0xe0 | (code >> 12));
                buf[o++] = (char)(0x80 | ((code >> 6) & 0x3f));
                buf[o++] = (char)(0x80 | (code & 0x3f));
            }
            break;
        }
        default:
            free(buf);
            return 0;
        }
    }
    buf[o] = '\0';
    *out = buf;
    return 1;
}

/* Find a top-level "key": in the outermost object and return a pointer to
 * the first value byte (past the colon and surrounding whitespace), or
 * NULL. Control-plane fields live at depth 1 only: nested metadata and
 * string VALUES that merely equal a key name must never be read as fields
 * (codex P1 — `{"v":1,"meta":{"op":"quit"}}` is not a quit). */
static const char *json_top_value(const char *line, size_t len,
                                  const char *key) {
    char pattern[80];
    int pn = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (pn <= 0 || (size_t)pn >= sizeof(pattern)) return NULL;
    const char *p = line;
    const char *end = line + len;
    int depth = 0;
    while (p < end) {
        const char c = *p;
        if (c == '"') {
            const char *str_start = p;
            ++p;
            while (p < end) {
                if (*p == '\\') { p += 2; continue; }
                if (*p == '"') break;
                ++p;
            }
            if (p >= end) return NULL;
            /* [str_start, p] spans one string token, quotes included. */
            if (depth == 1 && (size_t)(p - str_start + 1) == (size_t)pn &&
                !memcmp(str_start, pattern, (size_t)pn)) {
                const char *q = p + 1;
                while (q < end && (*q == ' ' || *q == '\t')) ++q;
                if (q < end && *q == ':') {
                    ++q;
                    while (q < end && (*q == ' ' || *q == '\t')) ++q;
                    if (q < end) return q;
                }
            }
            ++p;
            continue;
        }
        if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']') --depth;
        ++p;
    }
    return NULL;
}

/* Find top-level "key": value where value is a JSON string; owned unescape. */
static int json_get_string(const char *line, size_t len, const char *key,
                           char **out) {
    *out = NULL;
    const char *q = json_top_value(line, len, key);
    if (!q) return 0;
    const char *end = line + len;
    if (*q == 'n' && q + 4 <= end && !memcmp(q, "null", 4)) {
        *out = NULL;
        return 1;
    }
    if (*q != '"') return 0;
    ++q;
    const char *s = q;
    while (q < end) {
        if (*q == '\\') { q += 2; continue; }
        if (*q == '"') break;
        ++q;
    }
    if (q >= end || *q != '"') return 0;
    return json_unescape_to(s, (size_t)(q - s), out);
}

static int json_get_number_int(const char *line, size_t len, const char *key,
                               long *out) {
    const char *q = json_top_value(line, len, key);
    if (!q) return 0;
    char *endptr = NULL;
    long v = strtol(q, &endptr, 10);
    if (endptr == q) return 0;
    *out = v;
    return 1;
}

int q27_fp1_parse_client_line(const char *line, size_t len, q27_fp1_op *out) {
    if (!out) return 0;
    *out = (q27_fp1_op){0};
    if (!line) {
        out->kind = Q27_FP1_OP_MALFORMED;
        return 1;
    }
    /* Trim trailing CR. */
    while (len && (line[len - 1] == '\r' || line[len - 1] == '\n' ||
                   line[len - 1] == ' ' || line[len - 1] == '\t'))
        --len;
    size_t start = 0;
    while (start < len &&
           (line[start] == ' ' || line[start] == '\t'))
        start++;
    if (start >= len || line[start] != '{') {
        out->kind = Q27_FP1_OP_MALFORMED;
        return 1;
    }
    line += start;
    len -= start;

    long version = 0;
    if (!json_get_number_int(line, len, "v", &version)) {
        out->kind = Q27_FP1_OP_MALFORMED;
        return 1;
    }
    if (version != 1) {
        out->kind = Q27_FP1_OP_BAD_VERSION;
        (void)json_get_string(line, len, "client_req_id", &out->client_req_id);
        return 1;
    }

    char *op = NULL;
    if (!json_get_string(line, len, "op", &op) || !op) {
        free(op);
        out->kind = Q27_FP1_OP_MALFORMED;
        return 1;
    }
    (void)json_get_string(line, len, "client_req_id", &out->client_req_id);
    if (out->client_req_id && strlen(out->client_req_id) > 64) {
        /* Soft-trim for safety; keep the prefix. */
        out->client_req_id[64] = '\0';
    }

    if (!strcmp(op, "prompt")) {
        out->kind = Q27_FP1_OP_PROMPT;
        if (!json_get_string(line, len, "text", &out->text))
            out->text = dup_n("", 0);
    } else if (!strcmp(op, "cancel")) {
        out->kind = Q27_FP1_OP_CANCEL;
    } else if (!strcmp(op, "quit")) {
        out->kind = Q27_FP1_OP_QUIT;
    } else if (!strcmp(op, "save")) {
        out->kind = Q27_FP1_OP_SAVE;
    } else if (!strcmp(op, "compact")) {
        out->kind = Q27_FP1_OP_COMPACT;
    } else if (!strcmp(op, "new")) {
        out->kind = Q27_FP1_OP_NEW;
    } else if (!strcmp(op, "session")) {
        out->kind = Q27_FP1_OP_SESSION;
    } else if (!strcmp(op, "help")) {
        out->kind = Q27_FP1_OP_HELP;
    } else if (!strcmp(op, "tool")) {
        out->kind = Q27_FP1_OP_TOOL;
        (void)json_get_string(line, len, "kind", &out->tool_kind);
        (void)json_get_string(line, len, "path", &out->path);
        (void)json_get_string(line, len, "needle", &out->needle);
        (void)json_get_string(line, len, "command", &out->command);
    } else if (!strcmp(op, "queue_clear")) {
        out->kind = Q27_FP1_OP_QUEUE_CLEAR;
    } else {
        out->kind = Q27_FP1_OP_UNKNOWN;
        out->text = op;
        op = NULL;
    }
    free(op);
    return 1;
}

/* ── Backend prompt queue (P4) ───────────────────────────────────────── */

uint32_t q27_fp1_prompt_queue_len(void) {
    pthread_mutex_lock(&g_prompts.mu);
    uint32_t n = (uint32_t)g_prompts.len;
    pthread_mutex_unlock(&g_prompts.mu);
    return n;
}

int q27_fp1_prompt_queue_push(const char *text, const char *client_req_id) {
    if (!text) text = "";
    char *t = dup_n(text, strlen(text));
    if (!t) return -1;
    char *r = NULL;
    if (client_req_id && client_req_id[0]) {
        r = dup_n(client_req_id, strlen(client_req_id));
        if (!r) {
            free(t);
            return -1;
        }
    }
    pthread_mutex_lock(&g_prompts.mu);
    if (g_prompts.len >= FP1_PROMPT_QUEUE_CAP) {
        pthread_mutex_unlock(&g_prompts.mu);
        free(t);
        free(r);
        return 0;
    }
    size_t idx = (g_prompts.head + g_prompts.len) % FP1_PROMPT_QUEUE_CAP;
    g_prompts.items[idx].text = t;
    g_prompts.items[idx].client_req_id = r;
    g_prompts.len++;
    pthread_mutex_unlock(&g_prompts.mu);
    return 1;
}

int q27_fp1_prompt_queue_pop(q27_fp1_op *out) {
    if (!out) return 0;
    *out = (q27_fp1_op){0};
    pthread_mutex_lock(&g_prompts.mu);
    if (g_prompts.len == 0) {
        pthread_mutex_unlock(&g_prompts.mu);
        return 0;
    }
    fp1_prompt_item item = g_prompts.items[g_prompts.head];
    g_prompts.items[g_prompts.head] = (fp1_prompt_item){0};
    g_prompts.head = (g_prompts.head + 1) % FP1_PROMPT_QUEUE_CAP;
    g_prompts.len--;
    pthread_mutex_unlock(&g_prompts.mu);
    out->kind = Q27_FP1_OP_PROMPT;
    out->text = item.text;
    out->client_req_id = item.client_req_id;
    return 1;
}

void q27_fp1_prompt_queue_clear(void) {
    pthread_mutex_lock(&g_prompts.mu);
    while (g_prompts.len) {
        fp1_prompt_item *item = &g_prompts.items[g_prompts.head];
        free(item->text);
        free(item->client_req_id);
        *item = (fp1_prompt_item){0};
        g_prompts.head = (g_prompts.head + 1) % FP1_PROMPT_QUEUE_CAP;
        g_prompts.len--;
    }
    g_prompts.head = 0;
    pthread_mutex_unlock(&g_prompts.mu);
}

int q27_fp1_emit_queue(FILE *out, uint64_t seq, const char *state) {
    if (!out) return 0;
    if (!state) state = "idle";
    const uint64_t ts = q27_fp1_now_ms();

    /* Snapshot string pointers under lock (valid until next queue mutation). */
    const char *texts[FP1_PROMPT_QUEUE_CAP];
    const char *reqs[FP1_PROMPT_QUEUE_CAP];
    size_t n = 0;
    pthread_mutex_lock(&g_prompts.mu);
    n = g_prompts.len;
    for (size_t i = 0; i < n; ++i) {
        size_t idx = (g_prompts.head + i) % FP1_PROMPT_QUEUE_CAP;
        texts[i] = g_prompts.items[idx].text ? g_prompts.items[idx].text : "";
        reqs[i] = g_prompts.items[idx].client_req_id;
    }
    pthread_mutex_unlock(&g_prompts.mu);

    int ok = fprintf(out,
        "{\"v\":1,\"seq\":%llu,\"type\":\"queue\",\"ts_ms\":%llu,"
        "\"command_id\":0,\"client_req_id\":null,"
        "\"state\":\"%s\",\"status\":\"ok\",\"code\":null,"
        "\"queue_len\":%u,\"items\":[",
        (unsigned long long)seq, (unsigned long long)ts, state,
        (unsigned)n) >= 0;

    for (size_t i = 0; ok && i < n; ++i) {
        if (i) ok = fputc(',', out) != EOF;
        if (!ok) break;
        /* preview: first ≤80 bytes, whitespace flattened; never cut inside
         * a UTF-8 codepoint — an invalid preview frame kills the Rust line
         * reader and looks like a child exit (codex P1). */
        char preview[81];
        size_t j = 0;
        const char *p = texts[i];
        for (; *p && j < 80; ++p) {
            char c = *p;
            if (c == '\n' || c == '\r' || c == '\t') c = ' ';
            preview[j++] = c;
        }
        if (*p && j) {
            /* Truncated: back off an incomplete trailing multibyte sequence. */
            size_t k = j;
            while (k > 0 && ((unsigned char)preview[k - 1] & 0xC0) == 0x80)
                --k;   /* k-1 = lead byte of the trailing sequence */
            if (k > 0) {
                const unsigned char lead = (unsigned char)preview[k - 1];
                size_t need = 1;
                if (lead >= 0xF0) need = 4;
                else if (lead >= 0xE0) need = 3;
                else if (lead >= 0xC0) need = 2;
                if (k - 1 + need > j) j = k - 1;
            }
        }
        preview[j] = '\0';
        char *esc_prev =
            q27_fp1_json_escape((const unsigned char *)preview, j);
        if (!esc_prev) {
            ok = 0;
            break;
        }
        if (reqs[i]) {
            char *esc_req = q27_fp1_json_escape(
                (const unsigned char *)reqs[i], strlen(reqs[i]));
            if (!esc_req) {
                free(esc_prev);
                ok = 0;
                break;
            }
            ok = fprintf(out, "{\"client_req_id\":\"%s\",\"preview\":\"%s\"}",
                         esc_req, esc_prev) >= 0;
            free(esc_req);
        } else {
            ok = fprintf(out, "{\"client_req_id\":null,\"preview\":\"%s\"}",
                         esc_prev) >= 0;
        }
        free(esc_prev);
    }
    if (ok) ok = fprintf(out, "]}\n") >= 0;
    if (ok) ok = fflush(out) != EOF;
    return ok;
}

int q27_fp1_emit_queue_from_worker(FILE *out, q27_agent_worker *worker,
                                   const char *state) {
    if (!out || !worker) return 0;
    const uint64_t seq = q27_agent_worker_alloc_sequence(worker);
    return q27_fp1_emit_queue(out, seq, state);
}

static int ctl_push_locked(const q27_fp1_op *op) {
    if (g_ctl.len >= FP1_OP_QUEUE_CAP) return 0;
    size_t idx = (g_ctl.head + g_ctl.len) % FP1_OP_QUEUE_CAP;
    g_ctl.items[idx] = *op;
    g_ctl.len++;
    pthread_cond_broadcast(&g_ctl.cv);
    return 1;
}

static void ctl_note_drop_locked(const q27_fp1_op *op, const char *code) {
    /* Per-request record so every dropped caller gets its own rejection
     * (codex P2); the ring is bounded, overflow coalesces to a count. */
    if (g_ctl.drop_len >= FP1_DROP_CAP) {
        g_ctl.drop_coalesced++;
        pthread_cond_broadcast(&g_ctl.cv);
        return;
    }
    const size_t idx = (g_ctl.drop_head + g_ctl.drop_len) % FP1_DROP_CAP;
    snprintf(g_ctl.drop_codes[idx], sizeof(g_ctl.drop_codes[idx]), "%s",
             code ? code : "busy");
    if (op && op->client_req_id) {
        snprintf(g_ctl.drop_reqs[idx], sizeof(g_ctl.drop_reqs[idx]),
                 "%s", op->client_req_id);
    } else {
        g_ctl.drop_reqs[idx][0] = '\0';
    }
    g_ctl.drop_len++;
    pthread_cond_broadcast(&g_ctl.cv);
}

/* Pop one drop record (caller holds no lock after return). Returns 1 with
 * the record filled, 0 when the ring is empty. *coalesced receives (and
 * clears) the overflow count only when the ring has just drained. */
static int ctl_pop_drop_locked(char *code, size_t code_cap, char *req,
                               size_t req_cap, uint32_t *coalesced) {
    if (g_ctl.drop_len == 0) {
        if (g_ctl.drop_coalesced && coalesced) {
            *coalesced = g_ctl.drop_coalesced;
            g_ctl.drop_coalesced = 0;
            return 2;
        }
        return 0;
    }
    snprintf(code, code_cap, "%s", g_ctl.drop_codes[g_ctl.drop_head]);
    snprintf(req, req_cap, "%s", g_ctl.drop_reqs[g_ctl.drop_head]);
    g_ctl.drop_head = (g_ctl.drop_head + 1) % FP1_DROP_CAP;
    g_ctl.drop_len--;
    return 1;
}

static int ctl_apply_op(q27_fp1_op *op) {
    /* cancel / quit are flags; other ops go on the queue. */
    pthread_mutex_lock(&g_ctl.mu);
    if (op->kind == Q27_FP1_OP_CANCEL) {
        g_ctl.cancel_requested = 1;
        q27_fp1_op_free(op);
        pthread_cond_broadcast(&g_ctl.cv);
        pthread_mutex_unlock(&g_ctl.mu);
        return 1;
    }
    if (op->kind == Q27_FP1_OP_QUIT) {
        g_ctl.quit_requested = 1;
        snprintf(g_ctl.quit_reason, sizeof(g_ctl.quit_reason), "quit");
        q27_fp1_op_free(op);
        pthread_cond_broadcast(&g_ctl.cv);
        pthread_mutex_unlock(&g_ctl.mu);
        return 1;
    }
    int ok = ctl_push_locked(op);
    if (!ok) {
        /* Control queue full: surface on-stream (pre-P4 has no prompt queue). */
        const char *code = "busy";
        if (op->kind == Q27_FP1_OP_UNKNOWN) code = "unknown_op";
        else if (op->kind == Q27_FP1_OP_BAD_VERSION) code = "bad_version";
        ctl_note_drop_locked(op, code);
        q27_fp1_op_free(op);
    } else {
        *op = (q27_fp1_op){0}; /* ownership moved */
    }
    pthread_mutex_unlock(&g_ctl.mu);
    return ok;
}

static void *fp1_reader_main(void *arg) {
    (void)arg;
    char *line = NULL;
    size_t cap = 0;
    size_t len = 0;
    int line_overflow = 0;
    unsigned char buf[4096];
    for (;;) {
        pthread_mutex_lock(&g_ctl.mu);
        int stop = g_ctl.stop || g_ctl.quit_requested;
        int wake_fd = g_ctl.wake_pipe[0];
        pthread_mutex_unlock(&g_ctl.mu);
        if (stop) break;

        struct pollfd fds[2];
        int nfds = 1;
        fds[0].fd = STDIN_FILENO;
        fds[0].events = POLLIN;
        if (wake_fd >= 0) {
            fds[1].fd = wake_fd;
            fds[1].events = POLLIN;
            nfds = 2;
        }
        int ready = poll(fds, nfds, 200);
        if (ready < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (nfds == 2 && (fds[1].revents & (POLLIN | POLLHUP | POLLERR))) {
            unsigned char sink[64];
            while (read(wake_fd, sink, sizeof(sink)) > 0) {
            }
            pthread_mutex_lock(&g_ctl.mu);
            stop = g_ctl.stop;
            pthread_mutex_unlock(&g_ctl.mu);
            if (stop) break;
        }
        if (!(fds[0].revents & (POLLIN | POLLHUP | POLLERR))) continue;

        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (n == 0) {
            pthread_mutex_lock(&g_ctl.mu);
            g_ctl.quit_requested = 1;
            snprintf(g_ctl.quit_reason, sizeof(g_ctl.quit_reason), "stdin_eof");
            pthread_cond_broadcast(&g_ctl.cv);
            pthread_mutex_unlock(&g_ctl.mu);
            break;
        }
        for (ssize_t i = 0; i < n; ++i) {
            unsigned char c = buf[i];
            if (c == '\n') {
                q27_fp1_op op = {0};
                if (line_overflow) {
                    op.kind = Q27_FP1_OP_MALFORMED;
                    op.text = dup_n("ClientMessage exceeds 1 MiB", 28);
                    (void)ctl_apply_op(&op);
                } else if (len == 0) {
                    /* Empty lines: ignore. */
                } else {
                    if (!q27_fp1_parse_client_line(line ? line : "", len,
                                                   &op)) {
                        op.kind = Q27_FP1_OP_MALFORMED;
                    }
                    (void)ctl_apply_op(&op);
                }
                len = 0;
                line_overflow = 0;
                continue;
            }
            if (line_overflow) continue; /* discard until newline */
            if (len >= FP1_LINE_MAX) {
                line_overflow = 1;
                len = 0;
                continue;
            }
            if (len + 1 >= cap) {
                size_t next = cap ? cap * 2 : 256;
                if (next > FP1_LINE_MAX + 1) next = FP1_LINE_MAX + 1;
                if (next < len + 1) next = len + 1;
                char *grown = realloc(line, next);
                if (!grown) {
                    line_overflow = 1;
                    len = 0;
                    continue;
                }
                line = grown;
                cap = next;
            }
            line[len++] = (char)c;
        }
    }
    free(line);
    return NULL;
}

int q27_fp1_control_start(void) {
    pthread_mutex_lock(&g_ctl.mu);
    if (g_ctl.thread_started) {
        pthread_mutex_unlock(&g_ctl.mu);
        return 1;
    }
    g_ctl.stop = 0;
    g_ctl.cancel_requested = 0;
    g_ctl.quit_requested = 0;
    g_ctl.quit_reason[0] = '\0';
    g_ctl.head = 0;
    g_ctl.len = 0;
    if (pipe(g_ctl.wake_pipe) != 0) {
        pthread_mutex_unlock(&g_ctl.mu);
        return 0;
    }
    int fl0 = fcntl(g_ctl.wake_pipe[0], F_GETFL);
    int fl1 = fcntl(g_ctl.wake_pipe[1], F_GETFL);
    if (fl0 >= 0) (void)fcntl(g_ctl.wake_pipe[0], F_SETFL, fl0 | O_NONBLOCK);
    if (fl1 >= 0) (void)fcntl(g_ctl.wake_pipe[1], F_SETFL, fl1 | O_NONBLOCK);
    if (pthread_create(&g_ctl.thread, NULL, fp1_reader_main, NULL) != 0) {
        close(g_ctl.wake_pipe[0]);
        close(g_ctl.wake_pipe[1]);
        g_ctl.wake_pipe[0] = g_ctl.wake_pipe[1] = -1;
        pthread_mutex_unlock(&g_ctl.mu);
        return 0;
    }
    g_ctl.thread_started = 1;
    pthread_mutex_unlock(&g_ctl.mu);
    return 1;
}

void q27_fp1_control_stop(void) {
    pthread_mutex_lock(&g_ctl.mu);
    if (!g_ctl.thread_started) {
        pthread_mutex_unlock(&g_ctl.mu);
        return;
    }
    g_ctl.stop = 1;
    if (g_ctl.wake_pipe[1] >= 0) {
        unsigned char b = 0;
        (void)write(g_ctl.wake_pipe[1], &b, 1);
    }
    pthread_t th = g_ctl.thread;
    pthread_mutex_unlock(&g_ctl.mu);
    pthread_join(th, NULL);
    pthread_mutex_lock(&g_ctl.mu);
    g_ctl.thread_started = 0;
    if (g_ctl.wake_pipe[0] >= 0) close(g_ctl.wake_pipe[0]);
    if (g_ctl.wake_pipe[1] >= 0) close(g_ctl.wake_pipe[1]);
    g_ctl.wake_pipe[0] = g_ctl.wake_pipe[1] = -1;
    while (g_ctl.len) {
        q27_fp1_op *op = &g_ctl.items[g_ctl.head];
        q27_fp1_op_free(op);
        g_ctl.head = (g_ctl.head + 1) % FP1_OP_QUEUE_CAP;
        g_ctl.len--;
    }
    pthread_mutex_unlock(&g_ctl.mu);
    q27_fp1_prompt_queue_clear();
}

int q27_fp1_control_cancel_requested(void) {
    pthread_mutex_lock(&g_ctl.mu);
    int v = g_ctl.cancel_requested;
    pthread_mutex_unlock(&g_ctl.mu);
    return v;
}

void q27_fp1_control_clear_cancel(void) {
    pthread_mutex_lock(&g_ctl.mu);
    g_ctl.cancel_requested = 0;
    pthread_mutex_unlock(&g_ctl.mu);
}

int q27_fp1_control_quit_requested(void) {
    pthread_mutex_lock(&g_ctl.mu);
    int v = g_ctl.quit_requested;
    pthread_mutex_unlock(&g_ctl.mu);
    return v;
}

const char *q27_fp1_control_quit_reason(void) {
    pthread_mutex_lock(&g_ctl.mu);
    const char *r =
        g_ctl.quit_reason[0] ? g_ctl.quit_reason : "quit";
    pthread_mutex_unlock(&g_ctl.mu);
    return r;
}

int q27_fp1_control_wait_op(q27_fp1_op *out, int timeout_ms) {
    if (!out) return 0;
    *out = (q27_fp1_op){0};
    pthread_mutex_lock(&g_ctl.mu);
    for (;;) {
        /* Drain pending ops before quit so malformed/prompt/notice work that
         * arrived on the same burst as quit is still delivered to the control
         * plane. stop always wins immediately. */
        if (g_ctl.len > 0) {
            *out = g_ctl.items[g_ctl.head];
            g_ctl.items[g_ctl.head] = (q27_fp1_op){0};
            g_ctl.head = (g_ctl.head + 1) % FP1_OP_QUEUE_CAP;
            g_ctl.len--;
            pthread_mutex_unlock(&g_ctl.mu);
            return 1;
        }
        if (g_ctl.quit_requested || g_ctl.stop) {
            pthread_mutex_unlock(&g_ctl.mu);
            return 0;
        }
        if (timeout_ms == 0) {
            pthread_mutex_unlock(&g_ctl.mu);
            return -1;
        }
        if (timeout_ms < 0) {
            pthread_cond_wait(&g_ctl.cv, &g_ctl.mu);
            continue;
        }
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += timeout_ms / 1000;
        ts.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000L;
        }
        int rc = pthread_cond_timedwait(&g_ctl.cv, &g_ctl.mu, &ts);
        if (rc == ETIMEDOUT) {
            pthread_mutex_unlock(&g_ctl.mu);
            return -1;
        }
    }
}

static int text_is_whitespace_only(const char *text) {
    if (!text) return 1;
    for (const char *p = text; *p; ++p) {
        if (!isspace((unsigned char)*p)) return 0;
    }
    return 1;
}

/* Point-in-time session report for mid-turn session/help (may be stale). */
static struct {
    char session_path[512];
    char snapshot_name[256];
    uint32_t context;
    uint32_t ctx_used;
    int think;
    int auto_tools;
    size_t turns;
    int set;
} g_fp1_report;

void q27_fp1_set_session_report(const char *session_path,
                                const char *snapshot_name, uint32_t context,
                                uint32_t ctx_used, int think, int auto_tools,
                                size_t turns_approx) {
    g_fp1_report.set = 1;
    g_fp1_report.context = context;
    g_fp1_report.ctx_used = ctx_used;
    g_fp1_report.think = think;
    g_fp1_report.auto_tools = auto_tools;
    g_fp1_report.turns = turns_approx;
    if (session_path)
        snprintf(g_fp1_report.session_path, sizeof(g_fp1_report.session_path),
                 "%s", session_path);
    else
        g_fp1_report.session_path[0] = '\0';
    if (snapshot_name)
        snprintf(g_fp1_report.snapshot_name, sizeof(g_fp1_report.snapshot_name),
                 "%s", snapshot_name);
    else
        g_fp1_report.snapshot_name[0] = '\0';
}

static int emit_session_report(FILE *out, uint64_t seq,
                               const char *client_req_id, const char *state) {
    char buf[768];
    if (g_fp1_report.set) {
        snprintf(buf, sizeof(buf),
                 "session: path=%s snapshot=%s ctx=%u/%u think=%s "
                 "auto_tools=%s turns≈%zu",
                 g_fp1_report.session_path[0] ? g_fp1_report.session_path
                                              : "(none)",
                 g_fp1_report.snapshot_name[0] ? g_fp1_report.snapshot_name
                                               : "(none)",
                 g_fp1_report.ctx_used, g_fp1_report.context,
                 g_fp1_report.think ? "on" : "off",
                 g_fp1_report.auto_tools ? "on" : "off", g_fp1_report.turns);
    } else {
        snprintf(buf, sizeof(buf), "session: (report unavailable)");
    }
    return q27_fp1_emit_notice(out, seq, client_req_id, "info", NULL, buf,
                               state);
}

static int emit_help_notice(FILE *out, uint64_t seq, const char *client_req_id,
                            const char *state) {
    return q27_fp1_emit_notice(out, seq, client_req_id, "info", NULL,
                               q27_fp1_help_text(), state);
}

static int op_needs_worker_exclusive(q27_fp1_op_kind kind) {
    return kind == Q27_FP1_OP_SAVE || kind == Q27_FP1_OP_COMPACT ||
           kind == Q27_FP1_OP_NEW || kind == Q27_FP1_OP_TOOL;
}

int q27_fp1_control_reject_busy(FILE *out, q27_agent_worker *worker,
                                const char *state) {
    if (!out || !worker) return 0;
    if (!state) state = "generating";
    for (;;) {
        q27_fp1_op op = {0};
        int drop = 0;
        uint32_t coalesced = 0;
        char drop_code[32] = {0};
        char drop_req[65] = {0};
        pthread_mutex_lock(&g_ctl.mu);
        const int dr = ctl_pop_drop_locked(drop_code, sizeof(drop_code),
                                           drop_req, sizeof(drop_req),
                                           &coalesced);
        if (dr != 0) {
            drop = dr;
            pthread_mutex_unlock(&g_ctl.mu);
        } else if (g_ctl.len == 0) {
            pthread_mutex_unlock(&g_ctl.mu);
            return 1;
        } else {
            op = g_ctl.items[g_ctl.head];
            g_ctl.items[g_ctl.head] = (q27_fp1_op){0};
            g_ctl.head = (g_ctl.head + 1) % FP1_OP_QUEUE_CAP;
            g_ctl.len--;
            pthread_mutex_unlock(&g_ctl.mu);
        }

        const uint64_t seq = q27_agent_worker_alloc_sequence(worker);
        int ok = 1;
        if (drop == 2) {
            char text[96];
            snprintf(text, sizeof(text),
                     "%u further control ops dropped (rejection cap)",
                     (unsigned)coalesced);
            ok = q27_fp1_emit_rejected(out, seq, NULL, "busy", text, state);
            if (!ok) return 0;
            continue;
        }
        if (drop) {
            const char *req = drop_req[0] ? drop_req : NULL;
            const char *text =
                !strcmp(drop_code, "busy")
                    ? "control queue full or worker busy; op dropped"
                    : "control queue full; op dropped";
            ok = q27_fp1_emit_rejected(out, seq, req,
                                       drop_code[0] ? drop_code : "busy",
                                       text, state);
            if (!ok) return 0;
            continue;
        }
        if (op.kind == Q27_FP1_OP_PROMPT) {
            if (text_is_whitespace_only(op.text)) {
                ok = q27_fp1_emit_rejected(out, seq, op.client_req_id, "empty",
                                           "prompt text is empty", state);
            } else {
                /* P4: backend-owned queue (feature always advertised). */
                int pr = q27_fp1_prompt_queue_push(op.text, op.client_req_id);
                if (pr < 0) {
                    ok = q27_fp1_emit_rejected(out, seq, op.client_req_id,
                                               "busy", "out of memory", state);
                } else if (pr == 0) {
                    ok = q27_fp1_emit_rejected(
                        out, seq, op.client_req_id, "queue_full",
                        "prompt queue full (cap 8)", state);
                } else {
                    ok = q27_fp1_emit_queue(out, seq, state);
                }
            }
        } else if (op.kind == Q27_FP1_OP_QUEUE_CLEAR) {
            q27_fp1_prompt_queue_clear();
            ok = q27_fp1_emit_queue(out, seq, state);
        } else if (op.kind == Q27_FP1_OP_HELP) {
            ok = emit_help_notice(out, seq, op.client_req_id, state);
        } else if (op.kind == Q27_FP1_OP_SESSION) {
            ok = emit_session_report(out, seq, op.client_req_id, state);
        } else if (op_needs_worker_exclusive(op.kind)) {
            ok = q27_fp1_emit_rejected(out, seq, op.client_req_id, "busy",
                                       "worker is busy", state);
        } else if (op.kind == Q27_FP1_OP_UNKNOWN) {
            ok = q27_fp1_emit_rejected(out, seq, op.client_req_id, "unknown_op",
                                       "unrecognized op", state);
        } else if (op.kind == Q27_FP1_OP_BAD_VERSION) {
            ok = q27_fp1_emit_rejected(out, seq, op.client_req_id, "bad_version",
                                       "protocol version must be 1", state);
        } else if (op.kind == Q27_FP1_OP_MALFORMED) {
            const char *msg = op.text && op.text[0] ? op.text
                                                    : "malformed ClientMessage";
            ok = q27_fp1_emit_notice(out, seq, op.client_req_id, "error", NULL,
                                     msg, state);
        } else {
            /* cancel/quit are flags and never queued. */
            ok = q27_fp1_emit_rejected(out, seq, op.client_req_id, "busy",
                                       "worker is busy", state);
        }
        q27_fp1_op_free(&op);
        if (!ok) return 0;
    }
}

int q27_fp1_control_flush_drops(FILE *out, q27_agent_worker *worker,
                                const char *state) {
    if (!out || !worker) return 0;
    if (!state) state = "idle";
    for (;;) {
        char drop_code[32] = {0};
        char drop_req[65] = {0};
        uint32_t coalesced = 0;
        pthread_mutex_lock(&g_ctl.mu);
        const int dr = ctl_pop_drop_locked(drop_code, sizeof(drop_code),
                                           drop_req, sizeof(drop_req),
                                           &coalesced);
        pthread_mutex_unlock(&g_ctl.mu);
        if (dr == 0) return 1;
        const uint64_t seq = q27_agent_worker_alloc_sequence(worker);
        if (dr == 2) {
            char text[96];
            snprintf(text, sizeof(text),
                     "%u further control ops dropped (rejection cap)",
                     (unsigned)coalesced);
            if (!q27_fp1_emit_rejected(out, seq, NULL, "busy", text, state))
                return 0;
            continue;
        }
        const char *req = drop_req[0] ? drop_req : NULL;
        if (!q27_fp1_emit_rejected(
                out, seq, req, drop_code[0] ? drop_code : "busy",
                "control queue full; op dropped", state))
            return 0;
    }
}

int q27_fp1_control_post_for_test(const q27_fp1_op *op) {
    if (!op) return 0;
    q27_fp1_op copy = {0};
    copy.kind = op->kind;
    if (op->client_req_id) {
        copy.client_req_id = dup_n(op->client_req_id, strlen(op->client_req_id));
        if (!copy.client_req_id) return 0;
    }
    if (op->text) {
        copy.text = dup_n(op->text, strlen(op->text));
        if (!copy.text) {
            q27_fp1_op_free(&copy);
            return 0;
        }
    }
    if (op->tool_kind) {
        copy.tool_kind = dup_n(op->tool_kind, strlen(op->tool_kind));
        if (!copy.tool_kind) {
            q27_fp1_op_free(&copy);
            return 0;
        }
    }
    if (op->path) {
        copy.path = dup_n(op->path, strlen(op->path));
        if (!copy.path) {
            q27_fp1_op_free(&copy);
            return 0;
        }
    }
    if (op->needle) {
        copy.needle = dup_n(op->needle, strlen(op->needle));
        if (!copy.needle) {
            q27_fp1_op_free(&copy);
            return 0;
        }
    }
    if (op->command) {
        copy.command = dup_n(op->command, strlen(op->command));
        if (!copy.command) {
            q27_fp1_op_free(&copy);
            return 0;
        }
    }
    return ctl_apply_op(&copy);
}
