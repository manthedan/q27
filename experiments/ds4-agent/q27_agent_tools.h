#ifndef Q27_AGENT_TOOLS_H
#define Q27_AGENT_TOOLS_H

#include "q27_agent_engine.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    Q27_TOOL_NONE = 0,
    Q27_TOOL_READ,
    Q27_TOOL_SEARCH,
    Q27_TOOL_EDIT,
    Q27_TOOL_SHELL,
    Q27_TOOL_WRITE,
    // Create-or-replace whole file (same payload channel as write).
    Q27_TOOL_OVERWRITE,
    // Internal validation-only commands. They are never model-visible and
    // perform no filesystem mutation.
    Q27_TOOL_WRITE_PREFLIGHT,
    Q27_TOOL_EDIT_PREFLIGHT,
    Q27_TOOL_OVERWRITE_PREFLIGHT
} q27_agent_tool_kind;

enum { Q27_TOOL_MAX_SELECTIONS = 32 };

typedef struct {
    uint64_t file_offset;
    uint64_t length;
    uint64_t output_offset;
    uint64_t output_length;
    uint64_t first_line;
    uint64_t last_line;
} q27_agent_tool_selection;

typedef struct {
    q27_agent_tool_kind kind;
    const char *path;             // relative to the configured workspace
    const unsigned char *input;   // search needle, edit old bytes, shell command, write content
    size_t input_len;
    const unsigned char *replacement; // edit replacement bytes
    size_t replacement_len;
    // Resolved edit-selection authority. The model sees only a short opaque
    // handle; the control plane supplies the exact selected bytes, range, and
    // full-file digest copied from a prior successful read/search.
    int has_selection;
    uint64_t selection_offset;
    uint64_t selection_length;
    unsigned char selection_file_sha256[32];
    uint32_t timeout_ms;          // shell only; 1..60000
    uint32_t max_output_bytes;    // 1..262144
} q27_agent_tool_request;

enum {
    Q27_TOOL_FLAG_TIMED_OUT = 1u << 0,
    Q27_TOOL_FLAG_SIGNALED = 1u << 1,
    Q27_TOOL_FLAG_OUTPUT_LIMIT = 1u << 2,
    Q27_TOOL_FLAG_DURABILITY_UNCERTAIN = 1u << 3
};

typedef struct {
    int32_t exit_code; // 0 file-tool success; shell exit/128+signal; -1 tool failure
    uint32_t flags;
    uint32_t output_bytes;
    int has_file_sha256;
    uint64_t file_size;
    unsigned char file_sha256[32];
    uint32_t selection_count;
    q27_agent_tool_selection selections[Q27_TOOL_MAX_SELECTIONS];
    char message[256];
} q27_agent_tool_result;

typedef int (*q27_agent_tool_sink)(const unsigned char *bytes, size_t len,
                                   void *opaque);

// Executes one bounded request beneath a caller-owned, pinned workspace fd.
// File paths must be
// relative and every component is opened with O_NOFOLLOW. File tools accept
// regular files up to 8 MiB. Write atomically creates only a nonexistent file;
// edit requires exactly one old-byte match and publishes atomically. Shell runs
// in its own process group, captures combined
// stdout/stderr, denies process creation so the complete job stays in one
// bounded process group, and is killed/reaped on timeout, cancellation, or
// output cap.
// Operational failures are returned in result (function returns OK); only
// caller cancellation returns CANCELLED.
q27_agent_status q27_agent_tool_execute(
    int workspace_fd,
    const q27_agent_tool_request *request,
    q27_agent_tool_sink sink,
    q27_agent_alive_check alive,
    void *opaque,
    q27_agent_tool_result *result);

#ifdef __cplusplus
}
#endif
#endif
