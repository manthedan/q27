// Slash and colon command parsing for the native agent TUI.
#ifndef Q27_AGENT_COMMANDS_H
#define Q27_AGENT_COMMANDS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    Q27_CMD_NONE = 0,
    Q27_CMD_HELP,
    Q27_CMD_QUIT,
    Q27_CMD_SAVE,
    Q27_CMD_COMPACT,
    Q27_CMD_SESSION,
    Q27_CMD_NEW,
    Q27_CMD_READ,
    Q27_CMD_SEARCH,
    Q27_CMD_SHELL,
    Q27_CMD_UNKNOWN
} q27_agent_cmd_kind;

typedef struct {
    q27_agent_cmd_kind kind;
    /* Points into the mutable line buffer after the command name, or NULL. */
    char *args;
} q27_agent_cmd;

/* Returns 1 if line is a command (starts with / or :), 0 if normal user text.
 * Mutates line in place for argument splitting (NUL after command name). */
int q27_agent_cmd_parse(char *line, size_t len, q27_agent_cmd *out);

/* Multi-line help text for /help. */
const char *q27_agent_cmd_help_text(void);

#ifdef __cplusplus
}
#endif
#endif
