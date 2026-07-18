#define _POSIX_C_SOURCE 200809L

#include "q27_agent_worker.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

struct q27_agent_worker {
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    char *model_path;
    char *tokenizer_path;
    uint32_t context;

    q27_agent_worker_state state;
    int init_done;
    int init_ok;
    int stop;
    int request_ready;
    int generation_active;
    int command_active;
    int terminal_pending;

    owned_messages request;
    int enable_thinking;
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

static int combined_alive(void *opaque) {
    generation_context *context = opaque;
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

static void publish_terminal(q27_agent_worker *worker, uint64_t command_id,
                             q27_agent_status status, uint32_t prompt_tokens,
                             uint32_t output_tokens, const char *error) {
    q27_agent_event event = {
        .type = status == Q27_AGENT_REJECTED ? Q27_EVENT_REJECTED :
                status == Q27_AGENT_ERROR ? Q27_EVENT_ERROR : Q27_EVENT_TURN_DONE,
        .command_id = command_id,
        .state = status == Q27_AGENT_ERROR ? Q27_WORKER_ERROR : Q27_WORKER_IDLE,
        .status = status,
        .prompt_tokens = prompt_tokens,
        .output_tokens = output_tokens,
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

        owned_messages messages = worker->request;
        worker->request = (owned_messages){0};
        const int enable_thinking = worker->enable_thinking;
        const uint32_t max_tokens = worker->max_tokens;
        q27_agent_alive_check alive = worker->alive;
        void *alive_opaque = worker->alive_opaque;
        const uint64_t command_id = worker->command_id;
        worker->request_ready = 0;
        worker->generation_active = 1;
        worker->state = worker->stop ? Q27_WORKER_STOPPING :
                                       Q27_WORKER_GENERATING;
        pthread_mutex_unlock(&worker->mu);

        q27_agent_event state_event = {
            .type = Q27_EVENT_STATE, .command_id = command_id,
            .state = Q27_WORKER_GENERATING, .status = Q27_AGENT_OK};
        int state_queued = event_enqueue(worker, state_event,
                                         alive, alive_opaque, 0);

        generation_context context = {
            .worker = worker, .external_alive = alive,
            .external_opaque = alive_opaque, .command_id = command_id};
        uint32_t prompt_tokens = 0, output_tokens = 0;
        error[0] = '\0';
        q27_agent_status status;
        if (!state_queued) {
            status = combined_alive(&context) ? Q27_AGENT_ERROR :
                                               Q27_AGENT_CANCELLED;
            if (status == Q27_AGENT_ERROR)
                copy_error(error, sizeof(error), "state event publication failed");
        } else {
            status = q27_agent_generate(
                engine, messages.items, messages.len, enable_thinking, max_tokens,
                event_text_sink, combined_alive, &context,
                &prompt_tokens, &output_tokens, error, sizeof(error));
            if (context.queue_error && status == Q27_AGENT_CANCELLED) {
                status = Q27_AGENT_ERROR;
                copy_error(error, sizeof(error), "text event publication failed");
            }
        }
        messages_free(&messages);
        publish_terminal(worker, command_id, status, prompt_tokens,
                         output_tokens, error);
    }

    q27_agent_engine_close(engine);
    return NULL;
}

q27_agent_worker *q27_agent_worker_start(const char *model_path,
                                          const char *tokenizer_path,
                                          uint32_t context,
                                          char *error, size_t error_cap) {
    if (!model_path || !tokenizer_path) {
        copy_error(error, error_cap, "model and tokenizer paths are required");
        return NULL;
    }
    q27_agent_worker *worker = calloc(1, sizeof(*worker));
    if (!worker) {
        copy_error(error, error_cap, "out of memory");
        return NULL;
    }
    worker->model_path = strdup(model_path);
    worker->tokenizer_path = strdup(tokenizer_path);
    worker->context = context;
    worker->state = Q27_WORKER_STARTING;
    if (!worker->model_path || !worker->tokenizer_path) {
        copy_error(error, error_cap, "out of memory");
        free(worker->model_path);
        free(worker->tokenizer_path);
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
        free(worker);
        return NULL;
    }
    return worker;
}

q27_agent_status q27_agent_worker_submit(
    q27_agent_worker *worker, const q27_agent_message *messages,
    size_t message_count, int enable_thinking, uint32_t max_tokens,
    q27_agent_alive_check alive, void *opaque, uint64_t *command_id,
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
    worker->request = copied;
    worker->enable_thinking = enable_thinking;
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
