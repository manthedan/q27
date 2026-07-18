#define _POSIX_C_SOURCE 200809L

#include "q27_agent_worker.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    int call_active;
    int request_ready;
    int result_ready;

    const q27_agent_message *messages;
    size_t message_count;
    int enable_thinking;
    uint32_t max_tokens;
    q27_agent_text_sink sink;
    q27_agent_alive_check alive;
    void *opaque;

    q27_agent_status result;
    uint32_t prompt_tokens;
    uint32_t output_tokens;
    char error[512];
#ifdef Q27_AGENT_WORKER_TESTING
    void (*stop_hook)(void *, int, int);
    void *stop_hook_opaque;
#endif
};

static void copy_error(char *out, size_t cap, const char *message) {
    if (!out || cap == 0) return;
    snprintf(out, cap, "%s", message ? message : "unknown worker error");
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

        const q27_agent_message *messages = worker->messages;
        const size_t message_count = worker->message_count;
        const int enable_thinking = worker->enable_thinking;
        const uint32_t max_tokens = worker->max_tokens;
        q27_agent_text_sink sink = worker->sink;
        q27_agent_alive_check alive = worker->alive;
        void *request_opaque = worker->opaque;
        worker->request_ready = 0;
        worker->state = Q27_WORKER_GENERATING;
        pthread_mutex_unlock(&worker->mu);

        uint32_t prompt_tokens = 0, output_tokens = 0;
        error[0] = '\0';
        q27_agent_status result = q27_agent_generate(
            engine, messages, message_count, enable_thinking, max_tokens,
            sink, alive, request_opaque, &prompt_tokens, &output_tokens,
            error, sizeof(error));

        pthread_mutex_lock(&worker->mu);
        worker->result = result;
        worker->prompt_tokens = prompt_tokens;
        worker->output_tokens = output_tokens;
        copy_error(worker->error, sizeof(worker->error), error);
        worker->result_ready = 1;
        // Remain GENERATING until the submitting caller consumes these shared
        // result fields. call_active prevents a second submitter overwriting
        // them in the meantime.
        pthread_cond_broadcast(&worker->cv);
        pthread_mutex_unlock(&worker->mu);
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

q27_agent_status q27_agent_worker_generate(
    q27_agent_worker *worker, const q27_agent_message *messages,
    size_t message_count, int enable_thinking, uint32_t max_tokens,
    q27_agent_text_sink sink, q27_agent_alive_check alive, void *opaque,
    uint32_t *prompt_tokens, uint32_t *output_tokens,
    char *error, size_t error_cap) {
    if (prompt_tokens) *prompt_tokens = 0;
    if (output_tokens) *output_tokens = 0;
    if (!worker || !messages || message_count == 0 || !sink || !alive) {
        copy_error(error, error_cap, "invalid worker generation arguments");
        return Q27_AGENT_REJECTED;
    }

    pthread_mutex_lock(&worker->mu);
    if (worker->state != Q27_WORKER_IDLE || worker->stop || worker->call_active) {
        pthread_mutex_unlock(&worker->mu);
        copy_error(error, error_cap, "worker is not idle");
        return Q27_AGENT_REJECTED;
    }
    worker->messages = messages;
    worker->message_count = message_count;
    worker->enable_thinking = enable_thinking;
    worker->max_tokens = max_tokens;
    worker->sink = sink;
    worker->alive = alive;
    worker->opaque = opaque;
    worker->call_active = 1;
    worker->result_ready = 0;
    worker->request_ready = 1;
    pthread_cond_broadcast(&worker->cv);
    while (!worker->result_ready)
        pthread_cond_wait(&worker->cv, &worker->mu);

    const q27_agent_status result = worker->result;
    if (prompt_tokens) *prompt_tokens = worker->prompt_tokens;
    if (output_tokens) *output_tokens = worker->output_tokens;
    copy_error(error, error_cap, worker->error);
    worker->result_ready = 0;
    worker->call_active = 0;
    if (worker->stop) {
        if (worker->state != Q27_WORKER_STOPPED)
            worker->state = Q27_WORKER_STOPPING;
    } else {
        worker->state = result == Q27_AGENT_ERROR ?
            Q27_WORKER_ERROR : Q27_WORKER_IDLE;
    }
    pthread_cond_broadcast(&worker->cv);
    pthread_mutex_unlock(&worker->mu);
    return result;
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
        worker->stop_hook(worker->stop_hook_opaque, 0, worker->call_active);
#endif
    // q27_agent_worker_generate owns borrowed request/result storage until it
    // returns. Never destroy the worker underneath that synchronous caller.
    while (worker->call_active)
        pthread_cond_wait(&worker->cv, &worker->mu);
#ifdef Q27_AGENT_WORKER_TESTING
    if (worker->stop_hook)
        worker->stop_hook(worker->stop_hook_opaque, 1, worker->call_active);
#endif
    pthread_cond_broadcast(&worker->cv);
    pthread_mutex_unlock(&worker->mu);
    pthread_join(worker->thread, NULL);
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
