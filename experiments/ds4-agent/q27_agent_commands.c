#include "q27_agent_commands.h"

#include <ctype.h>
#include <string.h>

static int starts_with_cmd(const char *line, size_t len, const char *name,
                           size_t *name_len_out) {
    size_t n = strlen(name);
    if (len < n) return 0;
    if (memcmp(line, name, n) != 0) return 0;
    if (len == n || isspace((unsigned char)line[n]) || line[n] == '\0') {
        *name_len_out = n;
        return 1;
    }
    return 0;
}

int q27_agent_cmd_parse(char *line, size_t len, q27_agent_cmd *out) {
    if (!out) return 0;
    out->kind = Q27_CMD_NONE;
    out->args = NULL;
    if (!line || len == 0) return 0;
    if (line[0] != '/' && line[0] != ':') return 0;

    /* Normalize leading colon to slash for matching. */
    char mark = line[0];
    (void)mark;

    size_t name_len = 0;
    q27_agent_cmd_kind kind = Q27_CMD_UNKNOWN;

    if (starts_with_cmd(line, len, "/help", &name_len) ||
        starts_with_cmd(line, len, ":help", &name_len) ||
        starts_with_cmd(line, len, "/hotkeys", &name_len) ||
        starts_with_cmd(line, len, ":hotkeys", &name_len))
        kind = Q27_CMD_HELP;
    else if (starts_with_cmd(line, len, "/quit", &name_len) ||
             starts_with_cmd(line, len, ":quit", &name_len) ||
             starts_with_cmd(line, len, "/exit", &name_len) ||
             starts_with_cmd(line, len, ":exit", &name_len) ||
             starts_with_cmd(line, len, "/q", &name_len) ||
             starts_with_cmd(line, len, ":q", &name_len))
        kind = Q27_CMD_QUIT;
    else if (starts_with_cmd(line, len, "/save", &name_len) ||
             starts_with_cmd(line, len, ":save", &name_len))
        kind = Q27_CMD_SAVE;
    else if (starts_with_cmd(line, len, "/compact", &name_len) ||
             starts_with_cmd(line, len, ":compact", &name_len))
        kind = Q27_CMD_COMPACT;
    else if (starts_with_cmd(line, len, "/session", &name_len) ||
             starts_with_cmd(line, len, ":session", &name_len))
        kind = Q27_CMD_SESSION;
    else if (starts_with_cmd(line, len, "/new", &name_len) ||
             starts_with_cmd(line, len, ":new", &name_len))
        kind = Q27_CMD_NEW;
    else if (starts_with_cmd(line, len, "/read", &name_len) ||
             starts_with_cmd(line, len, ":read", &name_len))
        kind = Q27_CMD_READ;
    else if (starts_with_cmd(line, len, "/search", &name_len) ||
             starts_with_cmd(line, len, ":search", &name_len))
        kind = Q27_CMD_SEARCH;
    else if (starts_with_cmd(line, len, "/shell", &name_len) ||
             starts_with_cmd(line, len, ":shell", &name_len))
        kind = Q27_CMD_SHELL;
    else {
        out->kind = Q27_CMD_UNKNOWN;
        return 1;
    }

    out->kind = kind;
    size_t i = name_len;
    while (i < len && isspace((unsigned char)line[i])) i++;
    if (i < len) {
        line[name_len] = '\0'; /* isolate name for debug; args follow spaces */
        out->args = line + i;
    } else {
        out->args = NULL;
    }
    return 1;
}

const char *q27_agent_cmd_help_text(void) {
    return
        "q27-agent commands:\n"
        "  /help              this text\n"
        "  /quit  /exit  /q   leave the session\n"
        "  /save              persist session (requires --session)\n"
        "  /compact           compact older turns now\n"
        "  /session           show session path and flags\n"
        "  /new               clear transcript (keeps system + tools preamble)\n"
        "  /read PATH         run the workspace read tool\n"
        "  /search PATH NEEDLE\n"
        "  /shell COMMAND     run the sandboxed shell tool\n"
        "\n"
        "Colon forms (:help, :quit, …) still work.\n"
        "Editing: arrows, Ctrl-A/E, history Up/Down (linenoise).\n"
        "Ctrl-C cancels the active turn or exits if idle.\n";
}
