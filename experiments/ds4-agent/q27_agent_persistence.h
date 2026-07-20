// Durable transcript manifest paired with one immutable Q27SNAP1 engine file.
#ifndef Q27_AGENT_PERSISTENCE_H
#define Q27_AGENT_PERSISTENCE_H

#include "q27_agent_engine.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    q27_agent_message *messages;
    size_t message_count;
    int enable_thinking;
    int enable_tools;
    uint32_t context;
    unsigned char tokenizer_sha1[20];
    unsigned char snapshot_sha256[32];
    char *snapshot_path;
    char *snapshot_name;
    int lock_fd; // held shared until engine pins/validates the snapshot
    int lock_held;
} q27_agent_saved_session;

// Returns 1 and a fresh, non-existent snapshot path in the manifest's parent
// directory. The basename is random and is the only path stored in the
// manifest. Returns 0 with errno/error on failure.
int q27_agent_session_new_snapshot_path(const char *manifest_path,
                                        char **snapshot_path,
                                        char **snapshot_name,
                                        char *error, size_t error_cap);

// Atomically publishes a mode-0600 manifest after verifying that snapshot_path
// names a private regular file in the same directory. old_snapshot_name, when
// supplied, is removed only after the new manifest and directory are durable.
// Returns 1 when durable, 2 when rename committed but directory durability is
// uncertain (the new snapshot must not be deleted), and 0 before commit.
int q27_agent_session_publish(const char *manifest_path,
                              const char *snapshot_path,
                              const char *snapshot_name,
                              const char *old_snapshot_name,
                              const q27_agent_message *messages,
                              size_t message_count,
                              int enable_thinking,
                              int enable_tools,
                              uint32_t context,
                              const unsigned char tokenizer_sha1[20],
                              const unsigned char snapshot_sha256[32],
                              char *error, size_t error_cap);

// Loads a complete, CRC-checked private manifest. The referenced snapshot is
// not interpreted here; the engine owner validates Q27SNAP1 identity, layout,
// exact token-prefix metadata, and resident position before accepting it.
int q27_agent_session_load(const char *manifest_path,
                           q27_agent_saved_session *session,
                           char *error, size_t error_cap);
void q27_agent_saved_session_free(q27_agent_saved_session *session);

// Chooses the first retained root user message while treating an assistant
// tool call plus following <tool_response> user message as one indivisible
// root turn. Returns 1 with cut_index, or 0 when there are not enough complete
// earlier root turns to compact.
int q27_agent_compaction_cut(const q27_agent_message *messages,
                             size_t message_count,
                             uint32_t keep_root_turns,
                             size_t *cut_index);

#ifdef __cplusplus
}
#endif
#endif
