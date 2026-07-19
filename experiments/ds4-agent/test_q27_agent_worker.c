#include "q27_agent_worker.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct q27_agent_engine { int marker; };

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int block;
    int entered;
    int release;
} alive_gate;

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
    size_t message_count, int enable_thinking, int enable_tools,
    uint32_t max_tokens, q27_agent_text_sink sink,
    q27_agent_alive_check alive, void *opaque,
    uint32_t *prompt_tokens, uint32_t *cached_tokens,
    uint32_t *prefill_tokens, uint32_t *output_tokens,
    int *tool_call_complete, char *error, size_t error_cap) {
    (void)enable_thinking;
    (void)max_tokens;
    if (!engine || engine->marker != 27 || message_count != 1) {
        snprintf(error, error_cap, "bad forwarded request");
        return Q27_AGENT_ERROR;
    }
    if (messages[0].content_len == 3 &&
        memcmp(messages[0].content, "bad", 3) == 0) {
        snprintf(error, error_cap, "injected request rejection");
        return Q27_AGENT_REJECTED;
    }
    if (!alive(opaque)) return Q27_AGENT_CANCELLED;
    if (messages[0].content_len == 4 &&
        memcmp(messages[0].content, "call", 4) == 0) {
        static const char body[] =
            "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}"
            "</tool_call>";
        if (!enable_tools || !sink(body, sizeof(body) - 1, opaque))
            return Q27_AGENT_CANCELLED;
        *prompt_tokens = 9;
        *prefill_tokens = 9;
        *output_tokens = 12;
        *tool_call_complete = 1;
        return Q27_AGENT_OK;
    }
    if (messages[0].content_len == 5 &&
        memcmp(messages[0].content, "burst", 5) == 0) {
        for (uint32_t i = 0; i < 5000; ++i) {
            if (!sink("z", 1, opaque)) return Q27_AGENT_CANCELLED;
        }
        *prompt_tokens = 7;
        *cached_tokens = 3;
        *prefill_tokens = 4;
        *output_tokens = 5000;
        return Q27_AGENT_OK;
    }
    if (messages[0].content_len != 3 ||
        memcmp(messages[0].content, "x\0y", 3) != 0) {
        snprintf(error, error_cap, "message copy lost binary bytes");
        return Q27_AGENT_ERROR;
    }
    const char reply[] = {'a', '\0', 'b'};
    if (!sink(reply, sizeof(reply), opaque)) return Q27_AGENT_CANCELLED;
    *prompt_tokens = 11;
    *cached_tokens = 5;
    *prefill_tokens = 6;
    *output_tokens = 1;
    return Q27_AGENT_OK;
}

static int alive(void *opaque) {
    (void)opaque;
    return 1;
}

static int gated_alive(void *opaque) {
    alive_gate *gate = opaque;
    pthread_mutex_lock(&gate->mu);
    gate->entered = 1;
    pthread_cond_broadcast(&gate->cv);
    while (gate->block && !gate->release)
        pthread_cond_wait(&gate->cv, &gate->mu);
    pthread_mutex_unlock(&gate->mu);
    return 1;
}

static void gate_init(alive_gate *gate, int block) {
    memset(gate, 0, sizeof(*gate));
    pthread_mutex_init(&gate->mu, NULL);
    pthread_cond_init(&gate->cv, NULL);
    gate->block = block;
}

static void gate_wait_entered(alive_gate *gate) {
    pthread_mutex_lock(&gate->mu);
    while (!gate->entered) pthread_cond_wait(&gate->cv, &gate->mu);
    pthread_mutex_unlock(&gate->mu);
}

static void gate_release(alive_gate *gate) {
    pthread_mutex_lock(&gate->mu);
    gate->release = 1;
    pthread_cond_broadcast(&gate->cv);
    pthread_mutex_unlock(&gate->mu);
}

static void gate_destroy(alive_gate *gate) {
    pthread_cond_destroy(&gate->cv);
    pthread_mutex_destroy(&gate->mu);
}

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

typedef struct {
    unsigned char bytes[16];
    size_t len;
    size_t deltas;
    uint64_t last_sequence;
    q27_agent_status terminal_status;
    q27_agent_worker_state terminal_state;
    uint32_t prompt_tokens;
    uint32_t cached_tokens;
    uint32_t prefill_tokens;
    uint32_t output_tokens;
    int tool_call_complete;
    int states;
    int tool_outputs;
    int terminals;
    q27_agent_tool_kind tool_kind;
    int32_t tool_exit_code;
    uint32_t tool_flags;
    uint32_t tool_output_bytes;
} drained;

static int drain_command(q27_agent_worker *worker, uint64_t command_id,
                         drained *out) {
    char error[128] = {0};
    memset(out, 0, sizeof(*out));
    for (;;) {
        q27_agent_event event;
        int got = q27_agent_worker_next_event(worker, &event, error, sizeof(error));
        if (got != 1) return 0;
        if (event.command_id != command_id || event.sequence <= out->last_sequence) {
            q27_agent_event_free(&event);
            return 0;
        }
        out->last_sequence = event.sequence;
        if (event.type == Q27_EVENT_STATE) out->states++;
        if (event.type == Q27_EVENT_TEXT_DELTA ||
            event.type == Q27_EVENT_TOOL_OUTPUT) {
            if (event.type == Q27_EVENT_TEXT_DELTA) out->deltas++;
            else out->tool_outputs++;
            if (event.data_len <= sizeof(out->bytes) - out->len) {
                memcpy(out->bytes + out->len, event.data, event.data_len);
                out->len += event.data_len;
            }
        }
        int terminal = event.type == Q27_EVENT_TURN_DONE ||
                       event.type == Q27_EVENT_TOOL_DONE ||
                       event.type == Q27_EVENT_REJECTED ||
                       event.type == Q27_EVENT_ERROR;
        if (terminal) {
            out->terminals++;
            out->terminal_status = event.status;
            out->terminal_state = event.state;
            out->prompt_tokens = event.prompt_tokens;
            out->cached_tokens = event.cached_tokens;
            out->prefill_tokens = event.prefill_tokens;
            out->output_tokens = event.output_tokens;
            out->tool_call_complete = event.tool_call_complete;
            out->tool_kind = event.tool_kind;
            out->tool_exit_code = event.tool_exit_code;
            out->tool_flags = event.tool_flags;
            out->tool_output_bytes = event.tool_output_bytes;
        }
        q27_agent_event_free(&event);
        if (terminal) return 1;
    }
}

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int reached_before;
    int reached_after;
    int after_active;
    int returned;
} stop_latch;

typedef struct {
    q27_agent_worker *worker;
    stop_latch *latch;
} stop_args;

static void stop_hook(void *opaque, int phase, int active) {
    stop_latch *latch = opaque;
    pthread_mutex_lock(&latch->mu);
    if (phase == 0) latch->reached_before = 1;
    else {
        latch->reached_after = 1;
        latch->after_active = active;
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
          "worker propagates engine-open failure");

    q27_agent_worker *worker =
        q27_agent_worker_start("model", "tok", 128, error, sizeof(error));
    CHECK(worker && q27_agent_worker_get_state(worker) == Q27_WORKER_IDLE,
          "worker starts idle");

    // submit owns a deep binary copy: mutate the source immediately afterward.
    char binary[] = {'x', '\0', 'y'};
    q27_agent_message message = {
        .role = "user", .content = binary, .content_len = sizeof(binary)};
    uint64_t command = 0;
    CHECK(q27_agent_worker_submit(worker, &message, 1, 0, 0, 8, alive, NULL,
                                  &command, error, sizeof(error)) == Q27_AGENT_OK,
          "binary command submits");
    memset(binary, '!', sizeof(binary));
    drained result;
    CHECK(drain_command(worker, command, &result), "binary command drains");
    CHECK(result.states == 1 && result.deltas == 1 && result.terminals == 1,
          "state/delta/terminal lifecycle is exact");
    CHECK(result.len == 3 && memcmp(result.bytes, "a\0b", 3) == 0,
          "binary delta survives event queue");
    CHECK(result.terminal_status == Q27_AGENT_OK &&
          result.prompt_tokens == 11 && result.cached_tokens == 5 &&
          result.prefill_tokens == 6 && result.output_tokens == 1,
          "terminal session accounting survives event queue");
    CHECK(q27_agent_worker_get_state(worker) == Q27_WORKER_IDLE,
          "terminal consumption reopens admission");

    q27_agent_message call = {
        .role = "user", .content = "call", .content_len = 4};
    CHECK(q27_agent_worker_submit(worker, &call, 1, 0, 1, 64, alive, NULL,
                                  &command, error, sizeof(error)) == Q27_AGENT_OK,
          "constrained generation command submits");
    CHECK(drain_command(worker, command, &result) &&
          result.terminal_status == Q27_AGENT_OK &&
          result.tool_call_complete == 1 && result.output_tokens == 12,
          "closed model tool call reaches terminal metadata");

    char shell_command[] = "printf 't\\000l'";
    q27_agent_tool_request tool_request = {
        .kind = Q27_TOOL_SHELL,
        .input = (const unsigned char *)shell_command,
        .input_len = strlen(shell_command),
        .timeout_ms = 1000,
        .max_output_bytes = 1024};
    CHECK(q27_agent_worker_submit_tool(worker, &tool_request, alive, NULL,
                                       &command, error, sizeof(error)) ==
              Q27_AGENT_OK,
          "asynchronous shell tool submits");
    memset(shell_command, '!', strlen(shell_command));
    CHECK(drain_command(worker, command, &result), "shell tool drains");
    CHECK(result.states == 1 && result.tool_outputs >= 1 &&
          result.terminals == 1 && result.len == 3 &&
          !memcmp(result.bytes, "t\0l", 3),
          "tool output uses the owned binary event stream");
    CHECK(result.terminal_status == Q27_AGENT_OK &&
          result.tool_kind == Q27_TOOL_SHELL && result.tool_exit_code == 0 &&
          result.tool_flags == 0 && result.tool_output_bytes == 3 &&
          q27_agent_worker_get_state(worker) == Q27_WORKER_IDLE,
          "tool terminal accounting reopens shared admission");

    q27_agent_message bad = {.role = "user", .content = "bad", .content_len = 3};
    CHECK(q27_agent_worker_submit(worker, &bad, 1, 0, 0, 8, alive, NULL,
                                  &command, error, sizeof(error)) == Q27_AGENT_OK,
          "rejected request is accepted as a command");
    CHECK(drain_command(worker, command, &result) &&
          result.terminal_status == Q27_AGENT_REJECTED &&
          q27_agent_worker_get_state(worker) == Q27_WORKER_IDLE,
          "request rejection is terminal but reusable");

    // 5000 one-byte deltas exceed the bounded 4094-delta producer allowance;
    // draining proves backpressure resumes without loss or duplicate sequence.
    q27_agent_message burst = {
        .role = "user", .content = "burst", .content_len = 5};
    CHECK(q27_agent_worker_submit(worker, &burst, 1, 0, 0, 6000, alive, NULL,
                                  &command, error, sizeof(error)) == Q27_AGENT_OK,
          "burst command submits");
    CHECK(drain_command(worker, command, &result) &&
          result.deltas == 5000 && result.output_tokens == 5000,
          "bounded queue drains all burst events exactly");

    // A command stays active until its terminal event is consumed.
    alive_gate gate;
    gate_init(&gate, 1);
    message = (q27_agent_message){
        .role = "user", .content = "x\0y", .content_len = 3};
    CHECK(q27_agent_worker_submit(worker, &message, 1, 0, 0, 8, gated_alive, &gate,
                                  &command, error, sizeof(error)) == Q27_AGENT_OK,
          "blocked command submits");
    gate_wait_entered(&gate);
    uint64_t rejected_id = 99;
    CHECK(q27_agent_worker_submit(worker, &message, 1, 0, 0, 8, alive, NULL,
                                  &rejected_id, error, sizeof(error)) ==
                                      Q27_AGENT_REJECTED && rejected_id == 0,
          "concurrent submission is rejected");
    gate_release(&gate);
    CHECK(drain_command(worker, command, &result) &&
          result.terminal_status == Q27_AGENT_OK,
          "first command retains its events");
    gate_destroy(&gate);
    q27_agent_worker_stop(worker);

    // Non-destructive stop cancels an accepted command and still publishes its
    // sole terminal before reaching STOPPED.
    worker = q27_agent_worker_start("model", "tok", 128, error, sizeof(error));
    CHECK(worker, "worker for cancellation starts");
    gate_init(&gate, 1);
    CHECK(q27_agent_worker_submit(worker, &message, 1, 0, 0, 8, gated_alive, &gate,
                                  &command, error, sizeof(error)) == Q27_AGENT_OK,
          "command before request_stop submits");
    gate_wait_entered(&gate);
    q27_agent_worker_request_stop(worker);
    gate_release(&gate);
    q27_agent_worker_wait_stopped(worker);
    CHECK(drain_command(worker, command, &result) && result.terminals == 1 &&
          result.terminal_status == Q27_AGENT_CANCELLED &&
          result.terminal_state == Q27_WORKER_STOPPING,
          "request_stop preserves one truthful cancellation terminal");
    q27_agent_worker_stop(worker);
    gate_destroy(&gate);

    // Destructive stop waits for the engine thread; test-only phases make the
    // ordering deterministic without scheduler sleeps.
    worker = q27_agent_worker_start("model", "tok", 128, error, sizeof(error));
    CHECK(worker, "worker for destructive stop starts");
    gate_init(&gate, 1);
    CHECK(q27_agent_worker_submit(worker, &message, 1, 0, 0, 8, gated_alive, &gate,
                                  &command, error, sizeof(error)) == Q27_AGENT_OK,
          "command before destructive stop submits");
    gate_wait_entered(&gate);
    stop_latch latch;
    memset(&latch, 0, sizeof(latch));
    pthread_mutex_init(&latch.mu, NULL);
    pthread_cond_init(&latch.cv, NULL);
    q27_agent_worker_set_stop_hook(worker, stop_hook, &latch);
    stop_args args = {.worker = worker, .latch = &latch};
    pthread_t stopping;
    CHECK(pthread_create(&stopping, NULL, stop_main, &args) == 0,
          "destructive stop thread starts");
    pthread_mutex_lock(&latch.mu);
    while (!latch.reached_before) pthread_cond_wait(&latch.cv, &latch.mu);
    pthread_mutex_unlock(&latch.mu);
    gate_release(&gate);
    pthread_join(stopping, NULL);
    pthread_mutex_lock(&latch.mu);
    CHECK(latch.reached_after && latch.after_active == 0 && latch.returned,
          "destructive stop joins only after generation ends");
    pthread_mutex_unlock(&latch.mu);
    pthread_cond_destroy(&latch.cv);
    pthread_mutex_destroy(&latch.mu);
    gate_destroy(&gate);

    puts("q27 agent worker selftest: PASS");
    return 0;
}
