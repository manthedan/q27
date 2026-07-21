// Strict q27 <tool_call> protocol and fixed native-tool registry.
//
// Body tools (write / overwrite / edit / edit_selection) use thin JSON headers
// and carry file bytes as a same-turn markdown-fenced body after </tool_call>.
// Content is never JSON-escaped, and the harness no longer free-decodes a
// second raw-payload chat turn for those tools.
#ifndef Q27_AGENT_PROTOCOL_H
#define Q27_AGENT_PROTOCOL_H

#include "q27_agent_tools.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Hard cap on literal edit `old` so models use surgical patches or overwrite.
enum { Q27_AGENT_EDIT_OLD_MAX_BYTES = 512 };

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
    char *selection;
    // 1 when a body-tool call is schema-valid but missing its required fence.
    int missing_body;
} q27_agent_tool_call;

// Returns the fixed registry and instructions inserted into the system message
// when automatic tools are explicitly enabled.
const char *q27_agent_tool_preamble(void);

// Returns the same ordered fixed registry used by the prompt and constrained
// decoder. The returned array and strings have static lifetime.
size_t q27_agent_tool_names(const char *const **names_out);

// True when the constrained decoder should keep generating after </tool_call>
// so the model can emit a fenced body in the same turn.
int q27_agent_tool_name_expects_body(const char *name);

// True when this request kind attaches file/replacement bytes via a same-turn
// fenced body rather than a second free generation.
int q27_agent_tool_kind_expects_body(q27_agent_tool_kind kind);

// Rejects payloads that look like another tool call or empty/control junk.
// Returns 0 when safe to publish, non-zero with a message when not.
int q27_agent_payload_rejected(const unsigned char *bytes, size_t len,
                               char *error, size_t error_cap);

// Parses exactly one closed wrapped call. Prefix prose is allowed.
// Non-body tools forbid non-whitespace after the closer.
// Body tools require a markdown-fenced body after the closer (optional for
// empty overwrite/write only when the fence body is empty). VALID owns all
// request bytes in `call`; release with tool_call_free.
q27_agent_tool_call_status q27_agent_parse_tool_call(
    const unsigned char *bytes, size_t len,
    q27_agent_tool_call *call,
    char *error, size_t error_cap);
void q27_agent_tool_call_free(q27_agent_tool_call *call);

// Removes one unambiguous outer Markdown fence from a known whole-file
// source-code payload. Callers must not apply this to edit fragments. This is
// a weak-model transport repair, not general Markdown parsing: a three-backtick
// opening language must match the file extension (or be empty), the closing
// fence must be the final line, and prose/Markdown files remain untouched.
// Returns 1 when bytes were normalized, 0 when unchanged, and -1 on invalid
// arguments. The buffer is modified in place and remains NUL-terminated.
int q27_agent_unwrap_whole_file_source_fence(
    const char *path, unsigned char *bytes, size_t *len);

// Extracts a required outer markdown fence from `bytes` (already a body
// region). On success, allocates a NUL-terminated body into *out and sets
// *out_len. Returns 1 on success, 0 when no complete fence is present.
int q27_agent_extract_fenced_body(const unsigned char *bytes, size_t len,
                                  unsigned char **out, size_t *out_len,
                                  char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif
#endif
