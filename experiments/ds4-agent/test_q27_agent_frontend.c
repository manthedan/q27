/* Unit tests for Frontend Protocol v1 emit helpers. */
#include "q27_agent_frontend.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Stub worker sequence allocator so the frontend unit binary does not link
 * the full worker / engine stack. */
uint64_t q27_agent_worker_alloc_sequence(q27_agent_worker *worker) {
    static uint64_t seq = 100;
    (void)worker;
    return ++seq;
}

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static char *slurp_tmp(FILE *f) {
    if (fseek(f, 0, SEEK_END) != 0) return NULL;
    long n = ftell(f);
    if (n < 0) return NULL;
    if (fseek(f, 0, SEEK_SET) != 0) return NULL;
    char *buf = malloc((size_t)n + 1);
    if (!buf) return NULL;
    size_t got = fread(buf, 1, (size_t)n, f);
    buf[got] = '\0';
    return buf;
}

static void test_utf8(void) {
    CHECK(q27_fp1_utf8_valid(NULL, 0));
    CHECK(q27_fp1_utf8_valid((const unsigned char *)"hi", 2));
    CHECK(q27_fp1_utf8_valid((const unsigned char *)"café", 5));
    /* overlong 2-byte encoding of ASCII */
    unsigned char overlong[] = {0xc0, 0xaf};
    CHECK(!q27_fp1_utf8_valid(overlong, 2));
    unsigned char bad[] = {0xff};
    CHECK(!q27_fp1_utf8_valid(bad, 1));
    /* truncated multi-byte */
    unsigned char trunc[] = {0xe2, 0x82};
    CHECK(!q27_fp1_utf8_valid(trunc, 2));
}

static void test_json_escape(void) {
    char *e = q27_fp1_json_escape((const unsigned char *)"a\"b\\c\n", 6);
    CHECK(e != NULL);
    CHECK(strcmp(e, "a\\\"b\\\\c\\n") == 0);
    free(e);
    e = q27_fp1_json_escape((const unsigned char *)"", 0);
    CHECK(e != NULL && e[0] == '\0');
    free(e);
}

static void test_v0_shape(void) {
    q27_agent_event ev = {0};
    ev.sequence = 7;
    ev.command_id = 3;
    ev.type = Q27_EVENT_TEXT_DELTA;
    ev.state = Q27_WORKER_GENERATING;
    ev.status = Q27_AGENT_OK;
    unsigned char msg[] = "hi";
    ev.data = msg;
    ev.data_len = 2;

    FILE *f = tmpfile();
    CHECK(f != NULL);
    CHECK(q27_fp1_print_event(f, &ev, /*mode=*/0, /*client_req_id=*/NULL));
    char *out = slurp_tmp(f);
    CHECK(out != NULL);
    /* v0 has no "v" field and no "text" field. */
    CHECK(strstr(out, "\"seq\":7") != NULL);
    CHECK(strstr(out, "\"type\":\"text_delta\"") != NULL);
    CHECK(strstr(out, "\"data_b64\"") != NULL);
    CHECK(strstr(out, "\"v\":1") == NULL);
    CHECK(strstr(out, "\"text\"") == NULL);
    CHECK(strstr(out, "\"ts_ms\"") == NULL);
    free(out);
    fclose(f);
}

static void test_v1_shape(void) {
    q27_agent_event ev = {0};
    ev.sequence = 12;
    ev.command_id = 3;
    ev.type = Q27_EVENT_TEXT_DELTA;
    ev.state = Q27_WORKER_GENERATING;
    ev.status = Q27_AGENT_OK;
    unsigned char msg[] = "Here is the fix";
    ev.data = msg;
    ev.data_len = sizeof(msg) - 1;

    FILE *f = tmpfile();
    CHECK(f != NULL);
    CHECK(q27_fp1_print_event(f, &ev, /*mode=*/1, "req-42"));
    char *out = slurp_tmp(f);
    CHECK(out != NULL);
    CHECK(strstr(out, "\"v\":1") != NULL);
    CHECK(strstr(out, "\"seq\":12") != NULL);
    CHECK(strstr(out, "\"ts_ms\":") != NULL);
    CHECK(strstr(out, "\"text\":\"Here is the fix\"") != NULL);
    CHECK(strstr(out, "\"stream\":\"assistant\"") != NULL);
    CHECK(strstr(out, "\"data_b64\"") != NULL);
    /* Correlation id threads through to worker events (codex P1). */
    CHECK(strstr(out, "\"client_req_id\":\"req-42\"") != NULL);
    free(out);
    fclose(f);

    /* NULL correlation still prints an explicit null field. */
    f = tmpfile();
    CHECK(f != NULL);
    CHECK(q27_fp1_print_event(f, &ev, /*mode=*/1, NULL));
    out = slurp_tmp(f);
    CHECK(out != NULL);
    CHECK(strstr(out, "\"client_req_id\":null") != NULL);
    free(out);
    fclose(f);
}

static void test_lifecycle(void) {
    FILE *f = tmpfile();
    CHECK(f != NULL);
    CHECK(q27_fp1_emit_hello(f, 1, "/m.q27", "/m.tok", 8192, "/ws", NULL,
                             1, 1, 8, /*features_queue=*/0));
    CHECK(q27_fp1_emit_idle(f, 2, 100, 8192, 0));
    CHECK(q27_fp1_emit_bye(f, 3, "quit"));
    char *out = slurp_tmp(f);
    CHECK(out != NULL);
    CHECK(strstr(out, "\"type\":\"hello\"") != NULL);
    CHECK(strstr(out, "\"protocol\":1") != NULL);
    CHECK(strstr(out, "\"features\":[\"tools\",\"selections\",\"compact\",\"session\"]") !=
          NULL);
    CHECK(strstr(out, "\"queue\"") == NULL); /* not until P4 */
    CHECK(strstr(out, "\"session_path\":null") != NULL);
    CHECK(strstr(out, "\"type\":\"idle\"") != NULL);
    CHECK(strstr(out, "\"ctx_used\":100") != NULL);
    CHECK(strstr(out, "\"type\":\"bye\"") != NULL);
    CHECK(strstr(out, "\"reason\":\"quit\"") != NULL);
    CHECK(strstr(out, "\"status\":\"ok\"") != NULL);
    free(out);
    fclose(f);

    f = tmpfile();
    CHECK(f != NULL);
    CHECK(q27_fp1_emit_bye(f, 4, "error"));
    out = slurp_tmp(f);
    CHECK(out != NULL);
    CHECK(strstr(out, "\"reason\":\"error\"") != NULL);
    CHECK(strstr(out, "\"status\":\"error\"") != NULL);
    free(out);
    fclose(f);

    f = tmpfile();
    CHECK(f != NULL);
    CHECK(q27_fp1_emit_hello(f, 1, "m", "t", 1, ".", "s.q27agent", 0, 0, 4,
                             /*features_queue=*/1));
    out = slurp_tmp(f);
    CHECK(out != NULL);
    CHECK(strstr(out, "\"queue\"") != NULL);
    free(out);
    fclose(f);
}

static void test_rejected_notice(void) {
    FILE *f = tmpfile();
    CHECK(f != NULL);
    CHECK(q27_fp1_emit_rejected(f, 9, "ui-1", "busy", "worker busy", "generating"));
    CHECK(q27_fp1_emit_notice(f, 10, NULL, "error", "no_session",
                              "save requires --session FILE", "idle"));
    char *out = slurp_tmp(f);
    CHECK(out != NULL);
    CHECK(strstr(out, "\"type\":\"rejected\"") != NULL);
    CHECK(strstr(out, "\"code\":\"busy\"") != NULL);
    CHECK(strstr(out, "\"client_req_id\":\"ui-1\"") != NULL);
    CHECK(strstr(out, "\"type\":\"notice\"") != NULL);
    CHECK(strstr(out, "\"code\":\"no_session\"") != NULL);
    free(out);
    fclose(f);
}

static void test_parse_client(void) {
    q27_fp1_op op = {0};
    const char *line =
        "{\"v\":1,\"op\":\"prompt\",\"client_req_id\":\"ui-1\","
        "\"text\":\"fix the\\ntest\"}";
    CHECK(q27_fp1_parse_client_line(line, strlen(line), &op));
    CHECK(op.kind == Q27_FP1_OP_PROMPT);
    CHECK(op.client_req_id && !strcmp(op.client_req_id, "ui-1"));
    CHECK(op.text && !strcmp(op.text, "fix the\ntest"));
    q27_fp1_op_free(&op);

    CHECK(q27_fp1_parse_client_line("{\"v\":2,\"op\":\"prompt\"}", 22, &op));
    CHECK(op.kind == Q27_FP1_OP_BAD_VERSION);
    q27_fp1_op_free(&op);

    CHECK(q27_fp1_parse_client_line("{\"v\":1,\"op\":\"cancel\"}", 22, &op));
    CHECK(op.kind == Q27_FP1_OP_CANCEL);
    q27_fp1_op_free(&op);

    CHECK(q27_fp1_parse_client_line("{\"v\":1,\"op\":\"quit\"}", 20, &op));
    CHECK(op.kind == Q27_FP1_OP_QUIT);
    q27_fp1_op_free(&op);

    /* Top-level only (codex P1): nested op strings and string values that
     * equal a key name are NOT control-plane fields. */
    {
        const char *nested = "{\"v\":1,\"meta\":{\"op\":\"quit\"}}";
        CHECK(q27_fp1_parse_client_line(nested, strlen(nested), &op));
        CHECK(op.kind != Q27_FP1_OP_QUIT);
        q27_fp1_op_free(&op);
        const char *strval = "{\"v\":1,\"note\":\"op\",\"op\":\"quit\"}";
        CHECK(q27_fp1_parse_client_line(strval, strlen(strval), &op));
        CHECK(op.kind == Q27_FP1_OP_QUIT);
        q27_fp1_op_free(&op);
        const char *nested_tool =
            "{\"v\":1,\"wrap\":{\"op\":\"tool\",\"kind\":\"shell\","
            "\"command\":\"rm -rf x\"}}";
        CHECK(q27_fp1_parse_client_line(nested_tool, strlen(nested_tool),
                                        &op));
        CHECK(op.kind != Q27_FP1_OP_TOOL);
        q27_fp1_op_free(&op);
    }

    /* Strict version token (codex P2): fractional/trailing-junk versions
     * are malformed, not v1. */
    {
        const char *frac = "{\"v\":1.5,\"op\":\"quit\"}";
        CHECK(q27_fp1_parse_client_line(frac, strlen(frac), &op));
        CHECK(op.kind == Q27_FP1_OP_MALFORMED);
        q27_fp1_op_free(&op);
        const char *junk = "{\"v\":1junk,\"op\":\"quit\"}";
        CHECK(q27_fp1_parse_client_line(junk, strlen(junk), &op));
        CHECK(op.kind == Q27_FP1_OP_MALFORMED);
        q27_fp1_op_free(&op);
    }

    /* Surrogate pairs combine; unpaired surrogates reject (codex P2). */
    {
        const char *emoji =
            "{\"v\":1,\"op\":\"prompt\",\"text\":\"\\uD83D\\uDE00\"}";
        CHECK(q27_fp1_parse_client_line(emoji, strlen(emoji), &op));
        CHECK(op.kind == Q27_FP1_OP_PROMPT);
        CHECK(op.text && strlen(op.text) == 4 &&
              (unsigned char)op.text[0] == 0xF0 &&
              (unsigned char)op.text[1] == 0x9F &&
              (unsigned char)op.text[2] == 0x98 &&
              (unsigned char)op.text[3] == 0x80);
        q27_fp1_op_free(&op);
        const char *lone =
            "{\"v\":1,\"op\":\"prompt\",\"text\":\"\\uD83D x\"}";
        CHECK(q27_fp1_parse_client_line(lone, strlen(lone), &op));
        CHECK(op.kind == Q27_FP1_OP_PROMPT);
        CHECK(op.text && op.text[0] == '\0');   /* unpaired -> dropped to empty */
        q27_fp1_op_free(&op);
    }

    /* Structural validation (r6 codex P2): unbalanced or trailing-garbage
     * frames are malformed, not dispatched. */
    {
        const char *open = "{\"v\":1,\"op\":\"quit\"";
        CHECK(q27_fp1_parse_client_line(open, strlen(open), &op));
        CHECK(op.kind == Q27_FP1_OP_MALFORMED);
        q27_fp1_op_free(&op);
        const char *trailing = "{\"v\":1,\"op\":\"quit\"} garbage";
        CHECK(q27_fp1_parse_client_line(trailing, strlen(trailing), &op));
        CHECK(op.kind == Q27_FP1_OP_MALFORMED);
        q27_fp1_op_free(&op);
        const char *unter = "{\"v\":1,\"op\":\"quit}";
        CHECK(q27_fp1_parse_client_line(unter, strlen(unter), &op));
        CHECK(op.kind == Q27_FP1_OP_MALFORMED);
        q27_fp1_op_free(&op);
        const char *mismatch = "{\"v\":1,\"op\":\"quit\"]";
        CHECK(q27_fp1_parse_client_line(mismatch, strlen(mismatch), &op));
        CHECK(op.kind == Q27_FP1_OP_MALFORMED);
        q27_fp1_op_free(&op);
        const char *nocomma =
            "{\"v\":1,\"op\":\"quit\" \"ignored\":true}";
        CHECK(q27_fp1_parse_client_line(nocomma, strlen(nocomma), &op));
        CHECK(op.kind == Q27_FP1_OP_MALFORMED);
        q27_fp1_op_free(&op);
    }

    /* Overlong client_req_id is rejected, not truncated (r18 codex P2). */
    {
        const char *longid =
            "{\"v\":1,\"op\":\"quit\",\"client_req_id\":\""
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}";
        CHECK(q27_fp1_parse_client_line(longid, strlen(longid), &op));
        CHECK(op.kind == Q27_FP1_OP_MALFORMED);
        q27_fp1_op_free(&op);
    }

    /* Invalid UTF-8 in a client string is dropped, not emitted (r6 codex P2). */
    {
        char badline[] = "{\"v\":1,\"op\":\"prompt\",\"text\":\"ok \xff\xfe\"}";
        CHECK(q27_fp1_parse_client_line(badline, strlen(badline), &op));
        CHECK(op.kind == Q27_FP1_OP_PROMPT);
        CHECK(op.text && op.text[0] == '\0');
        q27_fp1_op_free(&op);
    }

    /* Embedded NUL (raw or escaped) is rejected, not truncated (r9 codex P2). */
    {
        const char *nul =
            "{\"v\":1,\"op\":\"prompt\",\"text\":\"abc\\u0000def\"}";
        CHECK(q27_fp1_parse_client_line(nul, strlen(nul), &op));
        CHECK(op.kind == Q27_FP1_OP_PROMPT);
        CHECK(op.text && op.text[0] == '\0');
        q27_fp1_op_free(&op);
    }

    CHECK(q27_fp1_parse_client_line("{\"v\":1,\"op\":\"save\"}", 20, &op));
    CHECK(op.kind == Q27_FP1_OP_SAVE);
    q27_fp1_op_free(&op);

    CHECK(q27_fp1_parse_client_line("{\"v\":1,\"op\":\"compact\"}", 23, &op));
    CHECK(op.kind == Q27_FP1_OP_COMPACT);
    q27_fp1_op_free(&op);

    CHECK(q27_fp1_parse_client_line("{\"v\":1,\"op\":\"new\"}", 19, &op));
    CHECK(op.kind == Q27_FP1_OP_NEW);
    q27_fp1_op_free(&op);

    CHECK(q27_fp1_parse_client_line("{\"v\":1,\"op\":\"session\"}", 23, &op));
    CHECK(op.kind == Q27_FP1_OP_SESSION);
    q27_fp1_op_free(&op);

    CHECK(q27_fp1_parse_client_line("{\"v\":1,\"op\":\"help\"}", 20, &op));
    CHECK(op.kind == Q27_FP1_OP_HELP);
    q27_fp1_op_free(&op);

    const char *tool =
        "{\"v\":1,\"op\":\"tool\",\"kind\":\"search\",\"path\":\"src\","
        "\"needle\":\"foo\"}";
    CHECK(q27_fp1_parse_client_line(tool, strlen(tool), &op));
    CHECK(op.kind == Q27_FP1_OP_TOOL);
    CHECK(op.tool_kind && !strcmp(op.tool_kind, "search"));
    CHECK(op.path && !strcmp(op.path, "src"));
    CHECK(op.needle && !strcmp(op.needle, "foo"));
    q27_fp1_op_free(&op);

    CHECK(q27_fp1_parse_client_line("{\"v\":1,\"op\":\"queue_clear\"}", 27, &op));
    CHECK(op.kind == Q27_FP1_OP_QUEUE_CLEAR);
    q27_fp1_op_free(&op);

    CHECK(q27_fp1_parse_client_line("not json", 8, &op));
    CHECK(op.kind == Q27_FP1_OP_MALFORMED);
    q27_fp1_op_free(&op);
}

static void test_session_done_state(void) {
    FILE *f = tmpfile();
    CHECK(f != NULL);
    CHECK(q27_fp1_emit_session_done(f, 70, 0, "ui-5", "new", "ok",
                                    "new transcript; system retained", 0,
                                    "idle"));
    CHECK(q27_fp1_emit_state(f, 71, "ui-4", "compacting"));
    CHECK(q27_fp1_emit_session_done(f, 72, 0, "ui-4", "compact", "ok",
                                    "compacted older turns", 1200, "idle"));
    char *out = slurp_tmp(f);
    CHECK(out != NULL);
    CHECK(strstr(out, "\"type\":\"session_done\"") != NULL);
    CHECK(strstr(out, "\"action\":\"new\"") != NULL);
    CHECK(strstr(out, "\"client_req_id\":\"ui-5\"") != NULL);
    CHECK(strstr(out, "\"type\":\"state\"") != NULL);
    CHECK(strstr(out, "\"state\":\"compacting\"") != NULL);
    CHECK(strstr(out, "\"action\":\"compact\"") != NULL);
    CHECK(strstr(out, "\"prompt_tokens\":1200") != NULL);
    free(out);
    fclose(f);

    CHECK(strstr(q27_fp1_help_text(), "op\":\"prompt\"") != NULL);
}

static void test_p4_queue_and_tool_start(void) {
    q27_fp1_prompt_queue_clear();
    CHECK(q27_fp1_prompt_queue_len() == 0);
    CHECK(q27_fp1_prompt_queue_push("first", "ui-1") == 1);
    CHECK(q27_fp1_prompt_queue_push("second\nline", "ui-2") == 1);
    CHECK(q27_fp1_prompt_queue_len() == 2);

    FILE *f = tmpfile();
    CHECK(f != NULL);
    CHECK(q27_fp1_emit_queue(f, 50, "generating"));
    char *out = slurp_tmp(f);
    CHECK(out != NULL);
    CHECK(strstr(out, "\"type\":\"queue\"") != NULL);
    CHECK(strstr(out, "\"queue_len\":2") != NULL);
    CHECK(strstr(out, "\"client_req_id\":\"ui-1\"") != NULL);
    CHECK(strstr(out, "\"preview\":\"first\"") != NULL);
    CHECK(strstr(out, "\"preview\":\"second line\"") != NULL); /* flattened */
    free(out);
    fclose(f);

    q27_fp1_op op = {0};
    CHECK(q27_fp1_prompt_queue_pop(&op));
    CHECK(op.kind == Q27_FP1_OP_PROMPT);
    CHECK(op.text && !strcmp(op.text, "first"));
    CHECK(op.client_req_id && !strcmp(op.client_req_id, "ui-1"));
    q27_fp1_op_free(&op);
    CHECK(q27_fp1_prompt_queue_len() == 1);
    q27_fp1_prompt_queue_clear();
    CHECK(q27_fp1_prompt_queue_len() == 0);

    /* Preview truncation never splits a UTF-8 codepoint (codex P1): 79 ASCII
     * bytes + a 2-byte é straddling byte 80 must preview as the 79 ASCII
     * bytes only, keeping the frame valid UTF-8. */
    {
        char longprompt[128];
        memset(longprompt, 'a', 79);
        longprompt[79] = (char)0xC3;
        longprompt[80] = (char)0xA9;   /* é */
        longprompt[81] = 'z';
        longprompt[82] = '\0';
        CHECK(q27_fp1_prompt_queue_push(longprompt, "ui-utf8") == 1);
        f = tmpfile();
        CHECK(f != NULL);
        CHECK(q27_fp1_emit_queue(f, 51, "idle"));
        out = slurp_tmp(f);
        CHECK(out != NULL);
        char expect[128];
        memset(expect, 'a', 79);
        expect[79] = '\0';
        char needle[96];
        snprintf(needle, sizeof(needle), "\"preview\":\"%s\"", expect);
        CHECK(strstr(out, needle) != NULL);
        CHECK(q27_fp1_utf8_valid((const unsigned char *)out, strlen(out)));
        free(out);
        fclose(f);
        q27_fp1_prompt_queue_clear();
    }

    /* Cap 8 */
    for (int i = 0; i < 8; ++i) {
        char buf[16];
        snprintf(buf, sizeof(buf), "p%d", i);
        CHECK(q27_fp1_prompt_queue_push(buf, NULL) == 1);
    }
    CHECK(q27_fp1_prompt_queue_push("overflow", NULL) == 0);
    q27_fp1_prompt_queue_clear();

    f = tmpfile();
    CHECK(f != NULL);
    CHECK(q27_fp1_emit_tool_start(f, 51, 4, "ui-6", "read", "path=README.md",
                                  0));
    CHECK(q27_fp1_emit_tool_start(f, 52, 5, NULL, "write_preflight",
                                  "path=x", 1));
    out = slurp_tmp(f);
    CHECK(out != NULL);
    CHECK(strstr(out, "\"type\":\"tool_start\"") != NULL);
    CHECK(strstr(out, "\"tool_kind\":\"read\"") != NULL);
    CHECK(strstr(out, "\"detail\":\"path=README.md\"") != NULL);
    CHECK(strstr(out, "\"preflight\":false") != NULL);
    CHECK(strstr(out, "\"preflight\":true") != NULL);
    free(out);
    fclose(f);
}

int main(void) {
    test_utf8();
    test_json_escape();
    test_v0_shape();
    test_v1_shape();
    test_lifecycle();
    test_rejected_notice();
    test_parse_client();
    test_session_done_state();
    test_p4_queue_and_tool_start();
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("test_q27_agent_frontend: ok\n");
    return 0;
}
