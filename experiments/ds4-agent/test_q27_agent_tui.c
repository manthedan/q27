#include "q27_agent_commands.h"
#include "q27_agent_tui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

int main(void) {
    char buf[256];
    q27_tui_status st = {
        .phase = Q27_TUI_IDLE,
        .ctx_used = 1200,
        .ctx_size = 32768,
    };
    CHECK(q27_tui_format_status(&st, buf, sizeof(buf)) > 0);
    CHECK(strstr(buf, "idle") != NULL);
    CHECK(strstr(buf, "32k") != NULL || strstr(buf, "32768") != NULL);

    st.phase = Q27_TUI_PREFILL;
    st.prefill_done = 48;
    st.prefill_total = 96;
    st.prefill_tps = 40.0;
    CHECK(q27_tui_format_status(&st, buf, sizeof(buf)) > 0);
    CHECK(strstr(buf, "prefill") != NULL);
    CHECK(strstr(buf, "48/96") != NULL);

    st.phase = Q27_TUI_GENERATING;
    st.gen_tokens = 12;
    st.gen_tps = 25.5;
    CHECK(q27_tui_format_status(&st, buf, sizeof(buf)) > 0);
    CHECK(strstr(buf, "generating") != NULL);

    st.phase = Q27_TUI_TOOL;
    st.tool_name = "read";
    st.detail = "path=README.md";
    CHECK(q27_tui_format_status(&st, buf, sizeof(buf)) > 0);
    CHECK(strstr(buf, "read") != NULL);

    st.phase = Q27_TUI_IDLE;
    st.queue_len = 2;
    CHECK(q27_tui_format_status(&st, buf, sizeof(buf)) > 0);
    CHECK(strstr(buf, "queue 2") != NULL);

    q27_tui_format_tokens(500, buf, sizeof(buf));
    CHECK(!strcmp(buf, "500"));
    q27_tui_format_tokens(1500, buf, sizeof(buf));
    CHECK(strstr(buf, "k") != NULL);

    char bar[64];
    q27_tui_format_progress_bar(8, 16, 8, bar, sizeof(bar));
    CHECK(bar[0] == '[');
    CHECK(strchr(bar, ']') != NULL);

    CHECK(q27_tui_format_tool_card_open("read", "path=foo.c", buf,
                                        sizeof(buf)) > 0);
    CHECK(strstr(buf, "read") != NULL);
    CHECK(strstr(buf, "path=foo.c") != NULL);
    CHECK(q27_tui_format_tool_card_close(0, 1200, NULL, buf, sizeof(buf)) > 0);
    CHECK(strstr(buf, "ok") != NULL);
    CHECK(q27_tui_format_tool_card_close(1, 0, "denied", buf, sizeof(buf)) > 0);
    CHECK(strstr(buf, "fail") != NULL);
    CHECK(strstr(buf, "denied") != NULL);

    const char long_text[] = "line1\nline2\nline3\nline4\nline5\n";
    CHECK(q27_tui_collapse_text(long_text, strlen(long_text), 2, 200, buf,
                                sizeof(buf)) > 0);
    CHECK(strstr(buf, "line1") != NULL);
    CHECK(strstr(buf, "more bytes") != NULL);

    /* Binary / C1 controls must not survive collapse previews. */
    char binary[] = {'a', '\0', (char)0x9b, '3', '1', 'm', 'x', '\n'};
    int bn = q27_tui_collapse_text(binary, sizeof(binary), 4, 64, buf,
                                   sizeof(buf));
    CHECK(bn > 0);
    CHECK(memchr(buf, '\0', (size_t)bn) == NULL ||
          (size_t)bn == strlen(buf)); /* no interior NULs in printable span */
    CHECK(memchr(buf, 0x9b, (size_t)bn) == NULL);
    CHECK(strchr(buf, 'a') != NULL);

    char dirty[] = "path=foo\x1b[31mbar\n";
    char clean[64];
    q27_tui_sanitize_display(dirty, clean, sizeof(clean));
    CHECK(strstr(clean, "\x1b") == NULL);
    CHECK(strchr(clean, '\n') == NULL);
    CHECK(strstr(clean, "path=foo") != NULL);
    char c1[] = "p\x9b" "31mx";
    q27_tui_sanitize_display(c1, clean, sizeof(clean));
    CHECK((unsigned char)clean[0] != 0x9b);
    CHECK(strchr(clean, 'p') != NULL);

    /* Code-point sanitization (r13 codex P2): valid multibyte survives —
     * U+0100 (C4 80) must NOT be corrupted by the old byte-wise C1 rule —
     * while ESC, raw C1 bytes, and malformed sequences become '.'. */
    {
        const unsigned char mixed[] = {
            'A', 0xC4, 0x80,        /* U+0100 — valid, keep */
            0x1B,                   /* ESC — control, '.' */
            0xC2, 0x85,             /* U+0085 (NEL) — C1 code point, '.' */
            (unsigned char)0xFF,    /* invalid lead, '.' */
            'z', 0
        };
        char sout[32];
        const size_t sn = q27_tui_sanitize_bytes(mixed, sizeof(mixed) - 1,
                                                 sout, sizeof(sout));
        CHECK(sn > 0);
        CHECK((unsigned char)sout[0] == 'A');
        CHECK((unsigned char)sout[1] == 0xC4 &&
              (unsigned char)sout[2] == 0x80);
        CHECK(sout[3] == '.' && sout[4] == '.' && sout[5] == '.');
        CHECK(sout[6] == 'z');
        /* Collapse inherits the code-point rule. */
        int cn = q27_tui_collapse_text((const char *)mixed,
                                       sizeof(mixed) - 1, 4, 64, buf,
                                       sizeof(buf));
        CHECK(cn > 0);
        CHECK(memchr(buf, 0xC4, (size_t)cn) != NULL);
        CHECK(memchr(buf, 0x1B, (size_t)cn) == NULL);
    }

    q27_tui_prompt_queue q;
    q27_tui_prompt_queue_init(&q, 2);
    CHECK(q27_tui_prompt_queue_push(&q, "one", 3));
    CHECK(q27_tui_prompt_queue_push(&q, "two", 3));
    CHECK(!q27_tui_prompt_queue_push(&q, "three", 5)); /* full */
    CHECK(q27_tui_prompt_queue_len(&q) == 2);
    char *a = q27_tui_prompt_queue_pop(&q);
    char *b = q27_tui_prompt_queue_pop(&q);
    CHECK(a && !strcmp(a, "one"));
    CHECK(b && !strcmp(b, "two"));
    CHECK(q27_tui_prompt_queue_pop(&q) == NULL);
    free(a);
    free(b);
    q27_tui_prompt_queue_free(&q);

    char line1[] = "/help";
    q27_agent_cmd cmd;
    CHECK(q27_agent_cmd_parse(line1, strlen(line1), &cmd) == 1);
    CHECK(cmd.kind == Q27_CMD_HELP);

    char line2[] = ":quit";
    CHECK(q27_agent_cmd_parse(line2, strlen(line2), &cmd) == 1);
    CHECK(cmd.kind == Q27_CMD_QUIT);

    char line3[] = "/read src/foo.c";
    CHECK(q27_agent_cmd_parse(line3, strlen(line3), &cmd) == 1);
    CHECK(cmd.kind == Q27_CMD_READ);
    CHECK(cmd.args && !strcmp(cmd.args, "src/foo.c"));

    char line4[] = "hello world";
    CHECK(q27_agent_cmd_parse(line4, strlen(line4), &cmd) == 0);

    char line5[] = "/nope";
    CHECK(q27_agent_cmd_parse(line5, strlen(line5), &cmd) == 1);
    CHECK(cmd.kind == Q27_CMD_UNKNOWN);

    CHECK(q27_agent_cmd_help_text() != NULL);
    CHECK(strstr(q27_agent_cmd_help_text(), "/help") != NULL);

    printf("test_q27_agent_tui: ok\n");
    return 0;
}
