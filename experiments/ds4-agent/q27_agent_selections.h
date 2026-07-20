#ifndef Q27_AGENT_SELECTIONS_H
#define Q27_AGENT_SELECTIONS_H

#include "q27_agent_tools.h"

#include <stddef.h>
#include <stdint.h>

typedef struct q27_agent_selection_ledger q27_agent_selection_ledger;

q27_agent_selection_ledger *q27_agent_selection_ledger_create(uint32_t nonce);
void q27_agent_selection_ledger_free(q27_agent_selection_ledger *ledger);

// Converts trusted read/search metadata plus exact captured output into a
// bounded model-visible annotation and process-local ledger entries. Returns 1
// on success, including a budget-driven skip (*annotation_out == NULL), and 0
// on invalid metadata or allocation failure. The caller owns annotation_out.
int q27_agent_selection_annotate(
    q27_agent_selection_ledger *ledger,
    const q27_agent_tool_request *request,
    const q27_agent_tool_result *result,
    const unsigned char *raw_output, size_t raw_output_len,
    size_t total_output_cap,
    unsigned char **annotation_out, size_t *annotation_len,
    char *error, size_t error_cap);

// Resolves one short handle into exact selected bytes plus digest/range
// authority on request. The caller owns *input_out and must keep it alive until
// the request has been deep-copied by the worker.
int q27_agent_selection_resolve(
    q27_agent_selection_ledger *ledger,
    const char *selection, const char *path,
    q27_agent_tool_request *request,
    unsigned char **input_out,
    char *error, size_t error_cap);

#endif
