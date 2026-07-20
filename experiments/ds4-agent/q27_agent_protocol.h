// Strict q27 <tool_call> protocol and fixed native-tool registry.
#ifndef Q27_AGENT_PROTOCOL_H
#define Q27_AGENT_PROTOCOL_H

#include "q27_agent_tools.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    Q27_TOOL_CALL_NONE = 0,
    Q27_TOOL_CALL_VALID = 1,
    Q27_TOOL_CALL_INVALID = 2
} q27_agent_tool_call_status;

typedef struct {
    q27_agent_tool_request request;
    char *path;
    unsigned char *input;
    unsigned char *replacement;
} q27_agent_tool_call;

// Returns the fixed read/search/write/edit/shell registry and instructions inserted
// into the system message when automatic tools are explicitly enabled.
const char *q27_agent_tool_preamble(void);

// Returns the same ordered fixed registry used by the prompt and constrained
// decoder. The returned array and strings have static lifetime.
size_t q27_agent_tool_names(const char *const **names_out);

// Parses exactly one closed wrapped call. Prefix prose is allowed; bytes after
// the closer must be whitespace. JSON and each fixed tool schema are strict.
// VALID owns all request bytes in `call`; release with tool_call_free.
q27_agent_tool_call_status q27_agent_parse_tool_call(
    const unsigned char *bytes, size_t len,
    q27_agent_tool_call *call,
    char *error, size_t error_cap);
void q27_agent_tool_call_free(q27_agent_tool_call *call);

#ifdef __cplusplus
}
#endif
#endif
