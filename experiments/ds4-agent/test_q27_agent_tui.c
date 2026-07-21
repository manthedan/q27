#include "q27_agent_commands.h"
#include "q27_agent_tui.h"

#include <stdio.h>
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

    q27_tui_format_tokens(500, buf, sizeof(buf));
    CHECK(!strcmp(buf, "500"));
    q27_tui_format_tokens(1500, buf, sizeof(buf));
    CHECK(strstr(buf, "k") != NULL);

    char bar[64];
    q27_tui_format_progress_bar(8, 16, 8, bar, sizeof(bar));
    CHECK(bar[0] == '[');
    CHECK(strchr(bar, ']') != NULL);

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
