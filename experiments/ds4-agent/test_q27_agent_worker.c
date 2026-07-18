#include "q27_agent_worker.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct q27_agent_engine { int marker; };

q27_agent_engine *q27_agent_engine_open(const char *model, const char *tokenizer,
                                         uint32_t context,
                                         char *error, size_t error_cap) {
    (void)tokenizer;
    (void)context;
    if (!strcmp(model, "fail")) {
        snprintf(error, error_cap, "injected open failure");
        return NULL;
    }
    q27_agent_engine *engine = malloc(sizeof(*engine));
    if (engine) engine->marker = 27;
    return engine;
}

void q27_agent_engine_close(q27_agent_engine *engine) {
    free(engine);
}

q27_agent_status q27_agent_generate(
    q27_agent_engine *engine, const q27_agent_message *messages,
    size_t message_count, int enable_thinking, uint32_t max_tokens,
    q27_agent_text_sink sink, q27_agent_alive_check alive, void *opaque,
    uint32_t *prompt_tokens, uint32_t *output_tokens,
    char *error, size_t error_cap) {
    (void)enable_thinking;
    (void)max_tokens;
    if (message_count == 1 && messages[0].content_len == 3 &&
        memcmp(messages[0].content, "bad", 3) == 0) {
        snprintf(error, error_cap, "injected request rejection");
        return Q27_AGENT_REJECTED;
    }
    if (!engine || engine->marker != 27 || message_count != 1 ||
        messages[0].content_len != 3 ||
        memcmp(messages[0].content, "x\0y", 3) != 0) {
        snprintf(error, error_cap, "bad forwarded request");
        return Q27_AGENT_ERROR;
    }
    if (!alive(opaque)) return Q27_AGENT_CANCELLED;
    const char reply[] = {'a', '\0', 'b'};
    if (!sink(reply, sizeof(reply), opaque)) return Q27_AGENT_CANCELLED;
    *prompt_tokens = 11;
    *output_tokens = 1;
    return Q27_AGENT_OK;
}

typedef struct {
    char bytes[8];
    size_t len;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int block;
    int entered;
    int release;
} capture;

static int alive(void *opaque) {
    (void)opaque;
    return 1;
}

static int gated_alive(void *opaque) {
    capture *gate = opaque;
    pthread_mutex_lock(&gate->mu);
    gate->entered = 1;
    pthread_cond_broadcast(&gate->cv);
    while (gate->block && !gate->release)
        pthread_cond_wait(&gate->cv, &gate->mu);
    pthread_mutex_unlock(&gate->mu);
    return 1;
}

static int sink(const char *bytes, size_t len, void *opaque) {
    capture *out = opaque;
    if (len > sizeof(out->bytes) - out->len) return 0;
    memcpy(out->bytes + out->len, bytes, len);
    out->len += len;
    return 1;
}

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

typedef struct {
    q27_agent_worker *worker;
    q27_agent_message message;
    capture *out;
    q27_agent_status status;
} generate_args;

static void *generate_main(void *opaque) {
    generate_args *args = opaque;
    char error[128];
    uint32_t prompt_tokens, output_tokens;
    args->status = q27_agent_worker_generate(
        args->worker, &args->message, 1, 0, 8, sink, gated_alive, args->out,
        &prompt_tokens, &output_tokens, error, sizeof(error));
    return NULL;
}

static void capture_init(capture *out, int block) {
    memset(out, 0, sizeof(*out));
    pthread_mutex_init(&out->mu, NULL);
    pthread_cond_init(&out->cv, NULL);
    out->block = block;
}

static void capture_release(capture *out) {
    pthread_mutex_lock(&out->mu);
    out->release = 1;
    pthread_cond_broadcast(&out->cv);
    pthread_mutex_unlock(&out->mu);
}

static void capture_destroy(capture *out) {
    pthread_cond_destroy(&out->cv);
    pthread_mutex_destroy(&out->mu);
}

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int reached_before;
    int reached_after;
    int after_call_active;
    int returned;
} stop_latch;

typedef struct {
    q27_agent_worker *worker;
    stop_latch *latch;
} stop_args;

static void stop_hook(void *opaque, int phase, int call_active) {
    stop_latch *latch = opaque;
    pthread_mutex_lock(&latch->mu);
    if (phase == 0) latch->reached_before = 1;
    else {
        latch->reached_after = 1;
        latch->after_call_active = call_active;
    }
    pthread_cond_broadcast(&latch->cv);
    pthread_mutex_unlock(&latch->mu);
}

static void *stop_main(void *opaque) {
    stop_args *args = opaque;
    q27_agent_worker_stop(args->worker);
    pthread_mutex_lock(&args->latch->mu);
    args->latch->returned = 1;
    pthread_cond_broadcast(&args->latch->cv);
    pthread_mutex_unlock(&args->latch->mu);
    return NULL;
}

int main(void) {
    char error[128] = {0};
    q27_agent_worker *failed =
        q27_agent_worker_start("fail", "tok", 128, error, sizeof(error));
    CHECK(!failed && strstr(error, "injected open failure"),
          "worker must propagate engine-open failure");

    q27_agent_worker *worker =
        q27_agent_worker_start("model", "tok", 128, error, sizeof(error));
    CHECK(worker, "worker starts");
    CHECK(q27_agent_worker_get_state(worker) == Q27_WORKER_IDLE,
          "opened worker is idle");

    const char binary[] = {'x', '\0', 'y'};
    q27_agent_message message = {
        .role = "user", .content = binary, .content_len = sizeof(binary)};
    capture out;
    capture_init(&out, 0);
    uint32_t prompt_tokens = 0, output_tokens = 0;
    q27_agent_status status = q27_agent_worker_generate(
        worker, &message, 1, 0, 8, sink, alive, &out,
        &prompt_tokens, &output_tokens, error, sizeof(error));
    CHECK(status == Q27_AGENT_OK, "generation succeeds");
    CHECK(prompt_tokens == 11 && output_tokens == 1,
          "generation accounting crosses worker boundary");
    CHECK(out.len == 3 && memcmp(out.bytes, "a\0b", 3) == 0,
          "binary stream crosses worker boundary");
    CHECK(q27_agent_worker_get_state(worker) == Q27_WORKER_IDLE,
          "worker returns to idle");
    capture_destroy(&out);

    q27_agent_message bad = {.role = "user", .content = "bad", .content_len = 3};
    capture_init(&out, 0);
    status = q27_agent_worker_generate(
        worker, &bad, 1, 0, 8, sink, alive, &out,
        &prompt_tokens, &output_tokens, error, sizeof(error));
    CHECK(status == Q27_AGENT_REJECTED &&
          q27_agent_worker_get_state(worker) == Q27_WORKER_IDLE,
          "request rejection leaves worker reusable");
    capture_destroy(&out);

    // Hold one accepted command inside its liveness callback. A concurrent
    // submitter must be rejected without overwriting the first result.
    capture gate;
    capture_init(&gate, 1);
    generate_args args = {.worker = worker, .message = message, .out = &gate};
    pthread_t generating;
    CHECK(pthread_create(&generating, NULL, generate_main, &args) == 0,
          "start blocked generation");
    pthread_mutex_lock(&gate.mu);
    while (!gate.entered) pthread_cond_wait(&gate.cv, &gate.mu);
    pthread_mutex_unlock(&gate.mu);
    capture second;
    capture_init(&second, 0);
    status = q27_agent_worker_generate(
        worker, &message, 1, 0, 8, sink, alive, &second,
        &prompt_tokens, &output_tokens, error, sizeof(error));
    CHECK(status == Q27_AGENT_REJECTED && strstr(error, "not idle"),
          "concurrent submitter is rejected");
    capture_release(&gate);
    pthread_join(generating, NULL);
    CHECK(args.status == Q27_AGENT_OK, "first submitter keeps its result");
    capture_destroy(&second);
    capture_destroy(&gate);

    // Repeat with shutdown racing an accepted command. stop must wait until
    // the synchronous submitter consumes its result before freeing the worker.
    capture_init(&gate, 1);
    args = (generate_args){.worker = worker, .message = message, .out = &gate};
    CHECK(pthread_create(&generating, NULL, generate_main, &args) == 0,
          "start generation before stop");
    pthread_mutex_lock(&gate.mu);
    while (!gate.entered) pthread_cond_wait(&gate.cv, &gate.mu);
    pthread_mutex_unlock(&gate.mu);
    q27_agent_worker_request_stop(worker);
    capture_release(&gate);
    pthread_join(generating, NULL);
    CHECK(args.status == Q27_AGENT_OK, "accepted command completes during stop");
    q27_agent_worker_wait_stopped(worker);
    CHECK(q27_agent_worker_get_state(worker) == Q27_WORKER_STOPPED,
          "non-destructive stop reaches terminal state");
    q27_agent_worker_request_stop(worker); // idempotent: must remain STOPPED
    CHECK(q27_agent_worker_get_state(worker) == Q27_WORKER_STOPPED,
          "repeated stop request preserves terminal state");
    q27_agent_worker_stop(worker);
    capture_destroy(&gate);

    // Destructive stop itself may overlap an already accepted call. The test
    // hook proves stop reached its call_active wait before generation resumes.
    worker = q27_agent_worker_start("model", "tok", 128, error, sizeof(error));
    CHECK(worker, "worker for destructive overlap starts");
    capture_init(&gate, 1);
    args = (generate_args){.worker = worker, .message = message, .out = &gate};
    CHECK(pthread_create(&generating, NULL, generate_main, &args) == 0,
          "start generation before destructive stop");
    pthread_mutex_lock(&gate.mu);
    while (!gate.entered) pthread_cond_wait(&gate.cv, &gate.mu);
    pthread_mutex_unlock(&gate.mu);

    stop_latch latch;
    memset(&latch, 0, sizeof(latch));
    pthread_mutex_init(&latch.mu, NULL);
    pthread_cond_init(&latch.cv, NULL);
    q27_agent_worker_set_stop_hook(worker, stop_hook, &latch);
    stop_args stopping_args = {.worker = worker, .latch = &latch};
    pthread_t stopping;
    CHECK(pthread_create(&stopping, NULL, stop_main, &stopping_args) == 0,
          "start destructive overlapping stop");
    pthread_mutex_lock(&latch.mu);
    while (!latch.reached_before) pthread_cond_wait(&latch.cv, &latch.mu);
    pthread_mutex_unlock(&latch.mu);

    capture_release(&gate);
    pthread_join(generating, NULL);
    pthread_join(stopping, NULL);
    CHECK(args.status == Q27_AGENT_OK,
          "submitter consumes result before destructive stop returns");
    pthread_mutex_lock(&latch.mu);
    CHECK(latch.reached_after && latch.after_call_active == 0,
          "destructive stop passes wait only after result consumption");
    CHECK(latch.returned, "destructive stop eventually returns");
    pthread_mutex_unlock(&latch.mu);
    pthread_cond_destroy(&latch.cv);
    pthread_mutex_destroy(&latch.mu);
    capture_destroy(&gate);

    puts("q27 agent worker selftest: PASS");
    return 0;
}
