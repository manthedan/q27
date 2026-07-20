#define _GNU_SOURCE
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "q27_agent_worker.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define EVENT_MAX_COUNT 4096u
#define EVENT_MAX_BYTES (1024u * 1024u)

typedef struct event_node {
    q27_agent_event event;
    size_t charge;
    struct event_node *next;
} event_node;

typedef struct {
    q27_agent_message *items;
    size_t len;
} owned_messages;

typedef struct {
    q27_agent_tool_request request;
    char *path;
    unsigned char *input;
    unsigned char *replacement;
} owned_tool_request;

typedef enum {
    REQUEST_NONE = 0,
    REQUEST_GENERATE,
    REQUEST_TOOL,
    REQUEST_SESSION
} request_kind;

struct q27_agent_worker {
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    char *model_path;
    char *tokenizer_path;
    int workspace_fd;
    uint32_t context;

    q27_agent_worker_state state;
    int init_done;
    int init_ok;
    int stop;
    int request_ready;
    int generation_active;
    int command_active;
    int terminal_pending;

    request_kind request_type;
    owned_messages request;
    owned_tool_request tool_request;
    q27_agent_session_action session_action;
    char *session_path;
    unsigned char expected_snapshot_sha256[32];
    int enable_thinking;
    int enable_tools;
    uint32_t max_tokens;
    q27_agent_alive_check alive;
    void *alive_opaque;
    uint64_t command_id;
    uint64_t next_command_id;
    uint64_t next_sequence;

    event_node *event_head;
    event_node *event_tail;
    size_t event_count;
    size_t event_bytes;
    unsigned char tokenizer_sha1[20];
    char error[512];
#ifdef Q27_AGENT_WORKER_TESTING
    void (*stop_hook)(void *, int, int);
    void *stop_hook_opaque;
#endif
};

typedef struct {
    q27_agent_worker *worker;
    q27_agent_alive_check external_alive;
    void *external_opaque;
    uint64_t command_id;
    q27_agent_tool_kind tool_kind;
    uint64_t deadline_ms;
    int deadline_expired;
    int queue_error;
} generation_context;

static void copy_error(char *out, size_t cap, const char *message) {
    if (!out || cap == 0) return;
    snprintf(out, cap, "%s", message ? message : "unknown worker error");
}

static void messages_free(owned_messages *messages) {
    if (!messages) return;
    for (size_t i = 0; i < messages->len; ++i) {
        free((char *)messages->items[i].role);
        free((char *)messages->items[i].content);
    }
    free(messages->items);
    *messages = (owned_messages){0};
}

static int messages_clone(const q27_agent_message *source, size_t count,
                          owned_messages *out, char *error, size_t error_cap) {
    *out = (owned_messages){0};
    if (!source || count == 0 || count > SIZE_MAX / sizeof(*out->items)) {
        copy_error(error, error_cap, "invalid message array");
        return 0;
    }
    out->items = calloc(count, sizeof(*out->items));
    if (!out->items) {
        copy_error(error, error_cap, "out of memory copying messages");
        return 0;
    }
    out->len = count;
    for (size_t i = 0; i < count; ++i) {
        if (!source[i].role || !source[i].content ||
            source[i].content_len == SIZE_MAX) {
            copy_error(error, error_cap, "invalid message role/content");
            messages_free(out);
            return 0;
        }
        char *role = strdup(source[i].role);
        char *content = malloc(source[i].content_len + 1);
        if (!role || !content) {
            free(role);
            free(content);
            copy_error(error, error_cap, "out of memory copying messages");
            messages_free(out);
            return 0;
        }
        memcpy(content, source[i].content, source[i].content_len);
        content[source[i].content_len] = '\0';
        out->items[i] = (q27_agent_message){
            .role = role, .content = content,
            .content_len = source[i].content_len};
    }
    return 1;
}

static void tool_request_free(owned_tool_request *owned) {
    if (!owned) return;
    free(owned->path);
    free(owned->input);
    free(owned->replacement);
    *owned = (owned_tool_request){0};
}

static int tool_request_clone(const q27_agent_tool_request *source,
                              owned_tool_request *out,
                              char *error, size_t error_cap) {
    *out = (owned_tool_request){0};
    if (!source || source->kind < Q27_TOOL_READ ||
        source->kind > Q27_TOOL_EDIT_PREFLIGHT || !source->max_output_bytes ||
        source->max_output_bytes > 256u * 1024u ||
        source->input_len > 8u * 1024u * 1024u ||
        source->replacement_len > 8u * 1024u * 1024u ||
        (source->input_len && !source->input) ||
        (source->replacement_len && !source->replacement)) {
        copy_error(error, error_cap, "invalid tool request");
        return 0;
    }
    if (source->kind != Q27_TOOL_SHELL && (!source->path || !*source->path)) {
        copy_error(error, error_cap, "tool path is required");
        return 0;
    }
    if (source->kind == Q27_TOOL_SHELL &&
        (!source->input_len || !source->timeout_ms || source->timeout_ms > 60000 ||
         memchr(source->input, '\0', source->input_len))) {
        copy_error(error, error_cap, "invalid shell command or timeout");
        return 0;
    }
    if (source->kind == Q27_TOOL_SEARCH && !source->input_len) {
        copy_error(error, error_cap, "search needle is required");
        return 0;
    }
    if ((source->kind == Q27_TOOL_EDIT ||
         source->kind == Q27_TOOL_EDIT_PREFLIGHT) && !source->input_len) {
        copy_error(error, error_cap, "edit old bytes are required");
        return 0;
    }
    if (source->path) {
        out->path = strdup(source->path);
        if (!out->path) goto oom;
    }
    if (source->input_len) {
        out->input = malloc(source->input_len);
        if (!out->input) goto oom;
        memcpy(out->input, source->input, source->input_len);
    }
    if (source->replacement_len) {
        out->replacement = malloc(source->replacement_len);
        if (!out->replacement) goto oom;
        memcpy(out->replacement, source->replacement, source->replacement_len);
    }
    out->request = *source;
    out->request.path = out->path;
    out->request.input = out->input;
    out->request.replacement = out->replacement;
    return 1;

oom:
    tool_request_free(out);
    copy_error(error, error_cap, "out of memory copying tool request");
    return 0;
}

static void event_node_free(event_node *node) {
    if (!node) return;
    free(node->event.data);
    free(node);
}

void q27_agent_event_free(q27_agent_event *event) {
    if (!event) return;
    free(event->data);
    *event = (q27_agent_event){0};
}

static int external_alive_unlocked(q27_agent_alive_check alive, void *opaque) {
    return !alive || alive(opaque);
}

static int event_enqueue(q27_agent_worker *worker, q27_agent_event event,
                         q27_agent_alive_check alive, void *alive_opaque,
                         int allow_stopping) {
    event_node *node = calloc(1, sizeof(*node));
    if (!node) return 0;
    node->event = event;
    node->event.data = NULL;
    node->charge = sizeof(*node) + event.data_len;
    if (event.data_len) {
        node->event.data = malloc(event.data_len);
        if (!node->event.data) { free(node); return 0; }
        memcpy(node->event.data, event.data, event.data_len);
    }
    if (node->charge > EVENT_MAX_BYTES) {
        event_node_free(node);
        return 0;
    }

    if (!external_alive_unlocked(alive, alive_opaque)) {
        event_node_free(node);
        return 0;
    }
    const size_t count_limit = allow_stopping ? EVENT_MAX_COUNT : EVENT_MAX_COUNT - 2;
    const size_t byte_limit = allow_stopping ? EVENT_MAX_BYTES : EVENT_MAX_BYTES - 4096;
    if (node->charge > byte_limit) {
        event_node_free(node);
        return 0;
    }
    int alive_ok = 1;
    pthread_mutex_lock(&worker->mu);
    while ((!worker->stop || allow_stopping) &&
           (worker->event_count >= count_limit ||
            worker->event_bytes > byte_limit - node->charge)) {
        if (alive) {
            pthread_mutex_unlock(&worker->mu);
            int ok = external_alive_unlocked(alive, alive_opaque);
            pthread_mutex_lock(&worker->mu);
            if (!ok) { alive_ok = 0; break; }
        }
        struct timespec until;
        clock_gettime(CLOCK_REALTIME, &until);
        until.tv_nsec += 50 * 1000 * 1000;
        if (until.tv_nsec >= 1000 * 1000 * 1000) {
            until.tv_sec++;
            until.tv_nsec -= 1000 * 1000 * 1000;
        }
        pthread_cond_timedwait(&worker->cv, &worker->mu, &until);
    }
    if (!alive_ok || (worker->stop && !allow_stopping)) {
        pthread_mutex_unlock(&worker->mu);
        event_node_free(node);
        return 0;
    }
    if (allow_stopping && worker->stop)
        node->event.state = Q27_WORKER_STOPPING;
    node->event.sequence = ++worker->next_sequence;
    if (worker->event_tail) worker->event_tail->next = node;
    else worker->event_head = node;
    worker->event_tail = node;
    worker->event_count++;
    worker->event_bytes += node->charge;
    pthread_cond_broadcast(&worker->cv);
    pthread_mutex_unlock(&worker->mu);
    return 1;
}

static uint64_t worker_monotonic_ms(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000 + (uint64_t)now.tv_nsec / 1000000;
}

static int combined_alive(void *opaque) {
    generation_context *context = opaque;
    if (context->deadline_ms && worker_monotonic_ms() >= context->deadline_ms) {
        context->deadline_expired = 1;
        return 0;
    }
    pthread_mutex_lock(&context->worker->mu);
    int stopped = context->worker->stop;
    pthread_mutex_unlock(&context->worker->mu);
    return !stopped && external_alive_unlocked(
        context->external_alive, context->external_opaque);
}

static int event_text_sink(const char *bytes, size_t len, void *opaque) {
    generation_context *context = opaque;
    q27_agent_event event = {
        .type = Q27_EVENT_TEXT_DELTA,
        .command_id = context->command_id,
        .state = Q27_WORKER_GENERATING,
        .status = Q27_AGENT_OK,
        .data = (unsigned char *)bytes,
        .data_len = len
    };
    int queued = event_enqueue(context->worker, event,
                               context->external_alive,
                               context->external_opaque, 0);
    if (!queued && combined_alive(context)) context->queue_error = 1;
    return queued;
}

static int event_tool_sink(const unsigned char *bytes, size_t len, void *opaque) {
    generation_context *context = opaque;
    q27_agent_event event = {
        .type = Q27_EVENT_TOOL_OUTPUT,
        .command_id = context->command_id,
        .state = Q27_WORKER_TOOL_RUNNING,
        .status = Q27_AGENT_OK,
        .tool_kind = context->tool_kind,
        .data = (unsigned char *)bytes,
        .data_len = len
    };
    int queued = event_enqueue(context->worker, event,
                               combined_alive, context, 0);
    if (!queued && combined_alive(context)) context->queue_error = 1;
    return queued;
}

static void publish_terminal(q27_agent_worker *worker, uint64_t command_id,
                             q27_agent_event_type terminal_type,
                             q27_agent_status status, uint32_t prompt_tokens,
                             uint32_t cached_tokens, uint32_t prefill_tokens,
                             uint32_t output_tokens,
                             int tool_call_complete, int eos_reached,
                             const q27_agent_tool_result *tool_result,
                             q27_agent_tool_kind tool_kind,
                             const char *error) {
    q27_agent_event event = {
        .type = terminal_type,
        .command_id = command_id,
        .state = status == Q27_AGENT_ERROR ? Q27_WORKER_ERROR : Q27_WORKER_IDLE,
        .status = status,
        .prompt_tokens = prompt_tokens,
        .cached_tokens = cached_tokens,
        .prefill_tokens = prefill_tokens,
        .output_tokens = output_tokens,
        .tool_call_complete = tool_call_complete,
        .eos_reached = eos_reached,
        .tool_kind = tool_kind,
        .tool_exit_code = tool_result ? tool_result->exit_code : 0,
        .tool_flags = tool_result ? tool_result->flags : 0,
        .tool_output_bytes = tool_result ? tool_result->output_bytes : 0,
        .data = (unsigned char *)(error ? error : ""),
        .data_len = error ? strlen(error) : 0
    };
    // Mark completion before publication so a fast consumer cannot acknowledge
    // the terminal and submit the next command before this generation retires.
    pthread_mutex_lock(&worker->mu);
    worker->generation_active = 0;
    worker->terminal_pending = 1;
    pthread_cond_broadcast(&worker->cv);
    pthread_mutex_unlock(&worker->mu);

    int queued = event_enqueue(worker, event, NULL, NULL, 1);
    if (!queued) {
        pthread_mutex_lock(&worker->mu);
        worker->terminal_pending = 0;
        worker->command_active = 0;
        worker->state = worker->stop ? Q27_WORKER_STOPPING : Q27_WORKER_ERROR;
        copy_error(worker->error, sizeof(worker->error),
                   "terminal event publication failed");
        pthread_cond_broadcast(&worker->cv);
        pthread_mutex_unlock(&worker->mu);
    }
}

static void *worker_main(void *opaque) {
    q27_agent_worker *worker = opaque;
    char error[512] = {0};
    q27_agent_engine *engine = q27_agent_engine_open(
        worker->model_path, worker->tokenizer_path, worker->context,
        error, sizeof(error));
    if (engine && q27_agent_engine_tokenizer_sha1(
                      engine, worker->tokenizer_sha1,
                      error, sizeof(error)) != Q27_AGENT_OK) {
        q27_agent_engine_close(engine);
        engine = NULL;
    }

    pthread_mutex_lock(&worker->mu);
    worker->init_done = 1;
    worker->init_ok = engine != NULL;
    worker->state = engine ? Q27_WORKER_IDLE : Q27_WORKER_ERROR;
    if (!engine) copy_error(worker->error, sizeof(worker->error), error);
    pthread_cond_broadcast(&worker->cv);
    pthread_mutex_unlock(&worker->mu);
    if (!engine) return NULL;

    for (;;) {
        pthread_mutex_lock(&worker->mu);
        while (!worker->stop && !worker->request_ready)
            pthread_cond_wait(&worker->cv, &worker->mu);
        if (worker->stop && !worker->request_ready) {
            worker->state = Q27_WORKER_STOPPED;
            pthread_cond_broadcast(&worker->cv);
            pthread_mutex_unlock(&worker->mu);
            break;
        }

        const request_kind kind = worker->request_type;
        worker->request_type = REQUEST_NONE;
        owned_messages messages = worker->request;
        worker->request = (owned_messages){0};
        owned_tool_request tool = worker->tool_request;
        worker->tool_request = (owned_tool_request){0};
        const q27_agent_session_action session_action = worker->session_action;
        char *session_path = worker->session_path;
        worker->session_path = NULL;
        const int enable_thinking = worker->enable_thinking;
        const int enable_tools = worker->enable_tools;
        const uint32_t max_tokens = worker->max_tokens;
        q27_agent_alive_check alive = worker->alive;
        void *alive_opaque = worker->alive_opaque;
        const uint64_t command_id = worker->command_id;
        worker->request_ready = 0;
        worker->generation_active = 1;
        const q27_agent_worker_state running_state =
            kind == REQUEST_TOOL ? Q27_WORKER_TOOL_RUNNING :
            kind == REQUEST_SESSION ? Q27_WORKER_SESSION_IO :
                                      Q27_WORKER_GENERATING;
        worker->state = worker->stop ? Q27_WORKER_STOPPING : running_state;
        pthread_mutex_unlock(&worker->mu);

        q27_agent_event state_event = {
            .type = Q27_EVENT_STATE, .command_id = command_id,
            .state = running_state, .status = Q27_AGENT_OK,
            .tool_kind = tool.request.kind};
        int state_queued = event_enqueue(worker, state_event,
                                         alive, alive_opaque, 0);

        generation_context context = {
            .worker = worker, .external_alive = alive,
            .external_opaque = alive_opaque, .command_id = command_id,
            .tool_kind = tool.request.kind,
            .deadline_ms = kind == REQUEST_TOOL &&
                           tool.request.kind == Q27_TOOL_SHELL ?
                           worker_monotonic_ms() + tool.request.timeout_ms : 0};
        uint32_t prompt_tokens = 0, cached_tokens = 0;
        uint32_t prefill_tokens = 0, output_tokens = 0;
        int tool_call_complete = 0, eos_reached = 0;
        q27_agent_tool_result tool_result = {0};
        error[0] = '\0';
        q27_agent_status status;
        if (!state_queued) {
            status = combined_alive(&context) ? Q27_AGENT_ERROR :
                                               Q27_AGENT_CANCELLED;
            if (status == Q27_AGENT_ERROR)
                copy_error(error, sizeof(error), "state event publication failed");
        } else if (kind == REQUEST_TOOL) {
            status = q27_agent_tool_execute(
                worker->workspace_fd, &tool.request,
                event_tool_sink, combined_alive, &context, &tool_result);
            if (context.queue_error && status == Q27_AGENT_CANCELLED) {
                status = Q27_AGENT_ERROR;
                copy_error(error, sizeof(error), "tool event publication failed");
            } else if (context.deadline_expired &&
                       status == Q27_AGENT_CANCELLED) {
                status = Q27_AGENT_OK;
                tool_result.exit_code = -1;
                tool_result.flags |= Q27_TOOL_FLAG_TIMED_OUT;
                copy_error(tool_result.message, sizeof(tool_result.message),
                           "shell job timed out");
                copy_error(error, sizeof(error), tool_result.message);
            } else if (tool_result.message[0]) {
                copy_error(error, sizeof(error), tool_result.message);
            }
        } else if (kind == REQUEST_SESSION) {
            if (session_action == Q27_SESSION_SAVE) {
                status = q27_agent_engine_save_session(
                    engine, session_path, messages.items, messages.len,
                    enable_thinking, error, sizeof(error));
            } else if (session_action == Q27_SESSION_LOAD) {
                status = q27_agent_engine_load_session(
                    engine, session_path, messages.items, messages.len,
                    enable_thinking, worker->expected_snapshot_sha256,
                    &prompt_tokens, error, sizeof(error));
            } else {
                status = q27_agent_engine_count_prompt(
                    engine, messages.items, messages.len, enable_thinking,
                    &prompt_tokens, error, sizeof(error));
            }
        } else {
            status = q27_agent_generate(
                engine, messages.items, messages.len, enable_thinking,
                enable_tools, max_tokens, event_text_sink, combined_alive,
                &context, &prompt_tokens, &cached_tokens, &prefill_tokens,
                &output_tokens, &tool_call_complete, &eos_reached,
                error, sizeof(error));
            if (context.queue_error && status == Q27_AGENT_CANCELLED) {
                status = Q27_AGENT_ERROR;
                copy_error(error, sizeof(error), "text event publication failed");
            }
        }
        const q27_agent_tool_kind completed_tool_kind = tool.request.kind;
        messages_free(&messages);
        tool_request_free(&tool);
        free(session_path);
        q27_agent_event_type terminal_type = kind == REQUEST_TOOL ?
            Q27_EVENT_TOOL_DONE : kind == REQUEST_SESSION ?
            Q27_EVENT_SESSION_DONE :
            status == Q27_AGENT_REJECTED ? Q27_EVENT_REJECTED :
            status == Q27_AGENT_ERROR ? Q27_EVENT_ERROR : Q27_EVENT_TURN_DONE;
        publish_terminal(worker, command_id, terminal_type, status, prompt_tokens,
                         cached_tokens, prefill_tokens, output_tokens,
                         tool_call_complete, eos_reached,
                         kind == REQUEST_TOOL ? &tool_result : NULL,
                         completed_tool_kind, error);
    }

    q27_agent_engine_close(engine);
    return NULL;
}

q27_agent_worker *q27_agent_worker_start_at(const char *model_path,
                                             const char *tokenizer_path,
                                             uint32_t context,
                                             const char *workspace_root,
                                             char *error, size_t error_cap) {
    if (!model_path || !tokenizer_path || !workspace_root) {
        copy_error(error, error_cap,
                   "model, tokenizer, and workspace paths are required");
        return NULL;
    }
    q27_agent_worker *worker = calloc(1, sizeof(*worker));
    if (!worker) {
        copy_error(error, error_cap, "out of memory");
        return NULL;
    }
    worker->workspace_fd = -1;
    worker->model_path = strdup(model_path);
    worker->tokenizer_path = strdup(tokenizer_path);
    worker->workspace_fd = open(workspace_root,
                                O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    worker->context = context;
    worker->state = Q27_WORKER_STARTING;
    struct stat workspace_stat;
    if (!worker->model_path || !worker->tokenizer_path ||
        worker->workspace_fd < 0 || fstat(worker->workspace_fd, &workspace_stat) != 0 ||
        !S_ISDIR(workspace_stat.st_mode)) {
        copy_error(error, error_cap, "could not pin workspace directory");
        free(worker->model_path);
        free(worker->tokenizer_path);
        if (worker->workspace_fd >= 0) close(worker->workspace_fd);
        free(worker);
        return NULL;
    }

    int mu_ok = pthread_mutex_init(&worker->mu, NULL) == 0;
    int cv_ok = mu_ok && pthread_cond_init(&worker->cv, NULL) == 0;
    if (!cv_ok) {
        copy_error(error, error_cap, "could not initialize worker synchronization");
        if (mu_ok) pthread_mutex_destroy(&worker->mu);
        free(worker->model_path);
        free(worker->tokenizer_path);
        if (worker->workspace_fd >= 0) close(worker->workspace_fd);
        free(worker);
        return NULL;
    }
    int rc = pthread_create(&worker->thread, NULL, worker_main, worker);
    if (rc != 0) {
        copy_error(error, error_cap, strerror(rc));
        pthread_cond_destroy(&worker->cv);
        pthread_mutex_destroy(&worker->mu);
        free(worker->model_path);
        free(worker->tokenizer_path);
        if (worker->workspace_fd >= 0) close(worker->workspace_fd);
        free(worker);
        return NULL;
    }

    pthread_mutex_lock(&worker->mu);
    while (!worker->init_done)
        pthread_cond_wait(&worker->cv, &worker->mu);
    int init_ok = worker->init_ok;
    copy_error(error, error_cap, worker->error);
    pthread_mutex_unlock(&worker->mu);
    if (!init_ok) {
        pthread_join(worker->thread, NULL);
        pthread_cond_destroy(&worker->cv);
        pthread_mutex_destroy(&worker->mu);
        free(worker->model_path);
        free(worker->tokenizer_path);
        if (worker->workspace_fd >= 0) close(worker->workspace_fd);
        free(worker);
        return NULL;
    }
    return worker;
}

q27_agent_worker *q27_agent_worker_start(const char *model_path,
                                          const char *tokenizer_path,
                                          uint32_t context,
                                          char *error, size_t error_cap) {
    return q27_agent_worker_start_at(model_path, tokenizer_path, context, ".",
                                     error, error_cap);
}

q27_agent_status q27_agent_worker_submit(
    q27_agent_worker *worker, const q27_agent_message *messages,
    size_t message_count, int enable_thinking, int enable_tools,
    uint32_t max_tokens, q27_agent_alive_check alive, void *opaque,
    uint64_t *command_id,
    char *error, size_t error_cap) {
    if (command_id) *command_id = 0;
    if (!worker || !messages || !message_count || !max_tokens || !alive) {
        copy_error(error, error_cap, "invalid worker submission");
        return Q27_AGENT_REJECTED;
    }
    owned_messages copied;
    if (!messages_clone(messages, message_count, &copied, error, error_cap))
        return Q27_AGENT_REJECTED;

    pthread_mutex_lock(&worker->mu);
    if (worker->state != Q27_WORKER_IDLE || worker->stop ||
        worker->command_active || worker->request_ready) {
        pthread_mutex_unlock(&worker->mu);
        messages_free(&copied);
        copy_error(error, error_cap, "worker is not idle");
        return Q27_AGENT_REJECTED;
    }
    worker->request_type = REQUEST_GENERATE;
    worker->request = copied;
    worker->enable_thinking = enable_thinking;
    worker->enable_tools = enable_tools;
    worker->max_tokens = max_tokens;
    worker->alive = alive;
    worker->alive_opaque = opaque;
    worker->command_id = ++worker->next_command_id;
    worker->command_active = 1;
    worker->request_ready = 1;
    worker->state = Q27_WORKER_GENERATING;
    if (command_id) *command_id = worker->command_id;
    pthread_cond_broadcast(&worker->cv);
    pthread_mutex_unlock(&worker->mu);
    return Q27_AGENT_OK;
}

q27_agent_status q27_agent_worker_submit_tool(
    q27_agent_worker *worker, const q27_agent_tool_request *request,
    q27_agent_alive_check alive, void *opaque, uint64_t *command_id,
    char *error, size_t error_cap) {
    if (command_id) *command_id = 0;
    if (!worker || !request || !alive) {
        copy_error(error, error_cap, "invalid worker tool submission");
        return Q27_AGENT_REJECTED;
    }
    owned_tool_request copied;
    if (!tool_request_clone(request, &copied, error, error_cap))
        return Q27_AGENT_REJECTED;

    pthread_mutex_lock(&worker->mu);
    if (worker->state != Q27_WORKER_IDLE || worker->stop ||
        worker->command_active || worker->request_ready) {
        pthread_mutex_unlock(&worker->mu);
        tool_request_free(&copied);
        copy_error(error, error_cap, "worker is not idle");
        return Q27_AGENT_REJECTED;
    }
    worker->request_type = REQUEST_TOOL;
    worker->tool_request = copied;
    worker->alive = alive;
    worker->alive_opaque = opaque;
    worker->command_id = ++worker->next_command_id;
    worker->command_active = 1;
    worker->request_ready = 1;
    worker->state = Q27_WORKER_TOOL_RUNNING;
    if (command_id) *command_id = worker->command_id;
    pthread_cond_broadcast(&worker->cv);
    pthread_mutex_unlock(&worker->mu);
    return Q27_AGENT_OK;
}

q27_agent_status q27_agent_worker_submit_session(
    q27_agent_worker *worker, q27_agent_session_action action,
    const char *snapshot_path, const q27_agent_message *messages,
    size_t message_count, int enable_thinking,
    const unsigned char expected_snapshot_sha256[32],
    uint64_t *command_id, char *error, size_t error_cap) {
    if (command_id) *command_id = 0;
    const int needs_path = action == Q27_SESSION_SAVE || action == Q27_SESSION_LOAD;
    const int needs_messages = action == Q27_SESSION_SAVE ||
                               action == Q27_SESSION_LOAD ||
                               action == Q27_SESSION_COUNT;
    if (!worker || action < Q27_SESSION_SAVE || action > Q27_SESSION_COUNT ||
        (needs_path && (!snapshot_path || !*snapshot_path)) ||
        (needs_messages && (!messages || !message_count)) ||
        (action == Q27_SESSION_LOAD && !expected_snapshot_sha256)) {
        copy_error(error, error_cap, "invalid worker session submission");
        return Q27_AGENT_REJECTED;
    }
    owned_messages copied = {0};
    if (needs_messages &&
        !messages_clone(messages, message_count, &copied, error, error_cap))
        return Q27_AGENT_REJECTED;
    char *path = needs_path ? strdup(snapshot_path) : NULL;
    if (needs_path && !path) {
        messages_free(&copied);
        copy_error(error, error_cap, "out of memory copying snapshot path");
        return Q27_AGENT_REJECTED;
    }

    pthread_mutex_lock(&worker->mu);
    if (worker->state != Q27_WORKER_IDLE || worker->stop ||
        worker->command_active || worker->request_ready) {
        pthread_mutex_unlock(&worker->mu);
        messages_free(&copied); free(path);
        copy_error(error, error_cap, "worker is not idle");
        return Q27_AGENT_REJECTED;
    }
    worker->request_type = REQUEST_SESSION;
    worker->request = copied;
    worker->session_action = action;
    worker->session_path = path;
    memset(worker->expected_snapshot_sha256, 0,
           sizeof(worker->expected_snapshot_sha256));
    if (expected_snapshot_sha256)
        memcpy(worker->expected_snapshot_sha256,
               expected_snapshot_sha256,
               sizeof(worker->expected_snapshot_sha256));
    worker->enable_thinking = enable_thinking;
    worker->alive = NULL;
    worker->alive_opaque = NULL;
    worker->command_id = ++worker->next_command_id;
    worker->command_active = 1;
    worker->request_ready = 1;
    worker->state = Q27_WORKER_SESSION_IO;
    if (command_id) *command_id = worker->command_id;
    pthread_cond_broadcast(&worker->cv);
    pthread_mutex_unlock(&worker->mu);
    return Q27_AGENT_OK;
}

int q27_agent_worker_next_event(q27_agent_worker *worker,
                                q27_agent_event *event,
                                char *error, size_t error_cap) {
    if (!worker || !event) {
        copy_error(error, error_cap, "invalid event request");
        return -1;
    }
    *event = (q27_agent_event){0};
    pthread_mutex_lock(&worker->mu);
    while (!worker->event_head && worker->state != Q27_WORKER_STOPPED &&
           worker->state != Q27_WORKER_ERROR)
        pthread_cond_wait(&worker->cv, &worker->mu);
    if (!worker->event_head) {
        int failed = worker->state == Q27_WORKER_ERROR;
        if (failed) copy_error(error, error_cap, worker->error);
        pthread_mutex_unlock(&worker->mu);
        return failed ? -1 : 0;
    }
    event_node *node = worker->event_head;
    worker->event_head = node->next;
    if (!worker->event_head) worker->event_tail = NULL;
    worker->event_count--;
    worker->event_bytes -= node->charge;
    *event = node->event;
    node->event.data = NULL;
    const int terminal = event->type == Q27_EVENT_TURN_DONE ||
                         event->type == Q27_EVENT_TOOL_DONE ||
                         event->type == Q27_EVENT_SESSION_DONE ||
                         event->type == Q27_EVENT_REJECTED ||
                         event->type == Q27_EVENT_ERROR;
    if (terminal && event->command_id == worker->command_id) {
        worker->terminal_pending = 0;
        worker->command_active = 0;
        if (worker->state != Q27_WORKER_STOPPED && !worker->stop) {
            worker->state = event->status == Q27_AGENT_ERROR ?
                Q27_WORKER_ERROR : Q27_WORKER_IDLE;
        }
    }
    pthread_cond_broadcast(&worker->cv);
    pthread_mutex_unlock(&worker->mu);
    free(node);
    return 1;
}

int q27_agent_worker_tokenizer_sha1(q27_agent_worker *worker,
                                    unsigned char out_sha1[20]) {
    if (!worker || !out_sha1) return 0;
    pthread_mutex_lock(&worker->mu);
    const int ok = worker->init_ok;
    if (ok) memcpy(out_sha1, worker->tokenizer_sha1, 20);
    pthread_mutex_unlock(&worker->mu);
    return ok;
}

int q27_agent_worker_session_result_event(q27_agent_worker *worker,
                                          int success,
                                          const char *message,
                                          q27_agent_event *event) {
    if (!worker || !event) return 0;
    *event = (q27_agent_event){0};
    const char *text = message ? message : "";
    const size_t len = strlen(text);
    unsigned char *copy = NULL;
    if (len) {
        copy = malloc(len);
        if (!copy) return 0;
        memcpy(copy, text, len);
    }
    pthread_mutex_lock(&worker->mu);
    event->sequence = ++worker->next_sequence;
    event->command_id = ++worker->next_command_id;
    event->type = Q27_EVENT_SESSION_DONE;
    event->state = success ? Q27_WORKER_IDLE : Q27_WORKER_ERROR;
    event->status = success ? Q27_AGENT_OK : Q27_AGENT_ERROR;
    event->data = copy;
    event->data_len = len;
    pthread_mutex_unlock(&worker->mu);
    return 1;
}

q27_agent_worker_state q27_agent_worker_get_state(q27_agent_worker *worker) {
    if (!worker) return Q27_WORKER_ERROR;
    pthread_mutex_lock(&worker->mu);
    q27_agent_worker_state state = worker->state;
    pthread_mutex_unlock(&worker->mu);
    return state;
}

void q27_agent_worker_request_stop(q27_agent_worker *worker) {
    if (!worker) return;
    pthread_mutex_lock(&worker->mu);
    worker->stop = 1;
    if (worker->state != Q27_WORKER_STOPPED)
        worker->state = Q27_WORKER_STOPPING;
    pthread_cond_broadcast(&worker->cv);
    pthread_mutex_unlock(&worker->mu);
}

void q27_agent_worker_wait_stopped(q27_agent_worker *worker) {
    if (!worker) return;
    pthread_mutex_lock(&worker->mu);
    while (worker->state != Q27_WORKER_STOPPED)
        pthread_cond_wait(&worker->cv, &worker->mu);
    pthread_mutex_unlock(&worker->mu);
}

void q27_agent_worker_stop(q27_agent_worker *worker) {
    if (!worker) return;
    q27_agent_worker_request_stop(worker);
    pthread_mutex_lock(&worker->mu);
#ifdef Q27_AGENT_WORKER_TESTING
    if (worker->stop_hook)
        worker->stop_hook(worker->stop_hook_opaque, 0,
                          worker->generation_active || worker->command_active);
#endif
    pthread_mutex_unlock(&worker->mu);
    pthread_join(worker->thread, NULL);
    pthread_mutex_lock(&worker->mu);
#ifdef Q27_AGENT_WORKER_TESTING
    if (worker->stop_hook)
        worker->stop_hook(worker->stop_hook_opaque, 1, worker->generation_active);
#endif
    messages_free(&worker->request);
    tool_request_free(&worker->tool_request);
    free(worker->session_path);
    event_node *node = worker->event_head;
    while (node) {
        event_node *next = node->next;
        event_node_free(node);
        node = next;
    }
    pthread_mutex_unlock(&worker->mu);
    pthread_cond_destroy(&worker->cv);
    pthread_mutex_destroy(&worker->mu);
    free(worker->model_path);
    free(worker->tokenizer_path);
    if (worker->workspace_fd >= 0) close(worker->workspace_fd);
    free(worker);
}

#ifdef Q27_AGENT_WORKER_TESTING
void q27_agent_worker_set_stop_hook(q27_agent_worker *worker,
                                    void (*hook)(void *, int, int), void *opaque) {
    if (!worker) return;
    pthread_mutex_lock(&worker->mu);
    worker->stop_hook = hook;
    worker->stop_hook_opaque = opaque;
    pthread_mutex_unlock(&worker->mu);
}
#endif
