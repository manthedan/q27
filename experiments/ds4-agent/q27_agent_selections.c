#define _POSIX_C_SOURCE 200809L

#include "q27_agent_selections.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SELECTION_LEDGER_CAP 256u
#define FILE_MAX_BYTES (8u * 1024u * 1024u)

typedef struct {
    char id[32];
    char *path;
    unsigned char *bytes;
    size_t len;
    uint64_t offset;
    unsigned char file_sha256[32];
} edit_selection;

struct q27_agent_selection_ledger {
    edit_selection entries[SELECTION_LEDGER_CAP];
    uint32_t nonce;
    uint64_t next_id;
};

typedef struct {
    unsigned char *bytes;
    size_t len;
    size_t cap;
} byte_buffer;

static void set_error(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message ? message : "selection failure");
}

static int append_bytes(byte_buffer *out, const void *data, size_t len) {
    if (len > SIZE_MAX - out->len) return 0;
    const size_t need = out->len + len;
    if (need > out->cap) {
        size_t cap = out->cap ? out->cap : 256;
        while (cap < need) {
            if (cap > SIZE_MAX / 2) return 0;
            cap *= 2;
        }
        unsigned char *grown = realloc(out->bytes, cap);
        if (!grown) return 0;
        out->bytes = grown;
        out->cap = cap;
    }
    if (len) memcpy(out->bytes + out->len, data, len);
    out->len += len;
    return 1;
}

q27_agent_selection_ledger *q27_agent_selection_ledger_create(uint32_t nonce) {
    q27_agent_selection_ledger *ledger = calloc(1, sizeof(*ledger));
    if (ledger) ledger->nonce = nonce;
    return ledger;
}

void q27_agent_selection_ledger_free(q27_agent_selection_ledger *ledger) {
    if (!ledger) return;
    for (size_t i = 0; i < SELECTION_LEDGER_CAP; ++i) {
        free(ledger->entries[i].path);
        free(ledger->entries[i].bytes);
    }
    free(ledger);
}

static edit_selection *find_selection(q27_agent_selection_ledger *ledger,
                                      const char *id) {
    if (!ledger || !id) return NULL;
    for (size_t i = 0; i < SELECTION_LEDGER_CAP; ++i)
        if (ledger->entries[i].id[0] && !strcmp(ledger->entries[i].id, id))
            return &ledger->entries[i];
    return NULL;
}

static edit_selection *add_selection(
    q27_agent_selection_ledger *ledger, const char *path,
    const unsigned char file_sha256[32], uint64_t offset,
    const unsigned char *bytes, size_t len) {
    if (!ledger || !path || !*path || !file_sha256 || !bytes || !len ||
        ledger->next_id == UINT64_MAX)
        return NULL;
    const uint64_t number = ++ledger->next_id;
    edit_selection *entry =
        &ledger->entries[(size_t)((number - 1) % SELECTION_LEDGER_CAP)];
    char *path_copy = strdup(path);
    unsigned char *byte_copy = malloc(len);
    if (!path_copy || !byte_copy) {
        free(path_copy); free(byte_copy);
        --ledger->next_id;
        return NULL;
    }
    memcpy(byte_copy, bytes, len);
    free(entry->path);
    free(entry->bytes);
    memset(entry, 0, sizeof(*entry));
    snprintf(entry->id, sizeof(entry->id), "s%08x-%llu", ledger->nonce,
             (unsigned long long)number);
    entry->path = path_copy;
    entry->bytes = byte_copy;
    entry->len = len;
    entry->offset = offset;
    memcpy(entry->file_sha256, file_sha256, 32);
    return entry;
}

int q27_agent_selection_annotate(
    q27_agent_selection_ledger *ledger,
    const q27_agent_tool_request *request,
    const q27_agent_tool_result *result,
    const unsigned char *raw_output, size_t raw_output_len,
    size_t total_output_cap,
    unsigned char **annotation_out, size_t *annotation_len,
    char *error, size_t error_cap) {
    if (!annotation_out || !annotation_len) return 0;
    *annotation_out = NULL;
    *annotation_len = 0;
    if (!ledger || !request || !result ||
        (raw_output_len && !raw_output) || result->exit_code != 0 ||
        !result->has_file_sha256 || !result->selection_count) {
        if (result && result->exit_code == 0 && result->selection_count)
            set_error(error, error_cap, "invalid selection metadata inputs");
        return result && (result->exit_code != 0 || !result->selection_count);
    }
    if ((request->kind != Q27_TOOL_READ &&
         request->kind != Q27_TOOL_SEARCH) ||
        !request->path || !*request->path ||
        result->selection_count > Q27_TOOL_MAX_SELECTIONS ||
        result->file_size > FILE_MAX_BYTES ||
        ledger->next_id > UINT64_MAX - result->selection_count) {
        set_error(error, error_cap, "invalid selection metadata bounds");
        return 0;
    }

    static const char open[] =
        "\n<q27_selections>\n"
        "Use edit_selection with the same path and one handle below.\n";
    static const char close[] = "</q27_selections>\n";
    size_t needed = sizeof(open) - 1 + sizeof(close) - 1;
    for (uint32_t i = 0; i < result->selection_count; ++i) {
        const q27_agent_tool_selection *s = &result->selections[i];
        if (!s->length || !s->first_line || s->last_line < s->first_line ||
            s->file_offset > result->file_size ||
            s->length > result->file_size - s->file_offset ||
            s->output_offset > raw_output_len ||
            s->output_length != s->length ||
            s->output_length > raw_output_len - s->output_offset) {
            set_error(error, error_cap, "invalid selection range metadata");
            return 0;
        }
        char id[32], line[160];
        int id_n = snprintf(id, sizeof(id), "s%08x-%llu", ledger->nonce,
                            (unsigned long long)(ledger->next_id + i + 1));
        int line_n = id_n < 0 || (size_t)id_n >= sizeof(id) ? -1 :
            snprintf(line, sizeof(line),
                     "[%s] lines %llu-%llu bytes %llu-%llu\n", id,
                     (unsigned long long)s->first_line,
                     (unsigned long long)s->last_line,
                     (unsigned long long)s->file_offset,
                     (unsigned long long)(s->file_offset + s->length));
        if (line_n < 0 || (size_t)line_n >= sizeof(line) ||
            (size_t)line_n > SIZE_MAX - needed) {
            set_error(error, error_cap, "selection annotation is too large");
            return 0;
        }
        needed += (size_t)line_n;
    }
    if (raw_output_len > total_output_cap ||
        needed > total_output_cap - raw_output_len)
        return 1;

    byte_buffer annotation = {0};
    if (!append_bytes(&annotation, open, sizeof(open) - 1)) goto oom;
    for (uint32_t i = 0; i < result->selection_count; ++i) {
        const q27_agent_tool_selection *s = &result->selections[i];
        edit_selection *entry = add_selection(
            ledger, request->path, result->file_sha256, s->file_offset,
            raw_output + s->output_offset, (size_t)s->output_length);
        if (!entry) goto oom;
        char line[160];
        int n = snprintf(line, sizeof(line),
                         "[%s] lines %llu-%llu bytes %llu-%llu\n", entry->id,
                         (unsigned long long)s->first_line,
                         (unsigned long long)s->last_line,
                         (unsigned long long)s->file_offset,
                         (unsigned long long)(s->file_offset + s->length));
        if (n < 0 || (size_t)n >= sizeof(line) ||
            !append_bytes(&annotation, line, (size_t)n))
            goto oom;
    }
    if (!append_bytes(&annotation, close, sizeof(close) - 1)) goto oom;
    *annotation_out = annotation.bytes;
    *annotation_len = annotation.len;
    return 1;

oom:
    free(annotation.bytes);
    set_error(error, error_cap, "out of memory building selection annotation");
    return 0;
}

int q27_agent_selection_resolve(
    q27_agent_selection_ledger *ledger,
    const char *selection, const char *path,
    q27_agent_tool_request *request,
    unsigned char **input_out,
    char *error, size_t error_cap) {
    if (!input_out || !request) return 0;
    *input_out = NULL;
    if (request->kind != Q27_TOOL_EDIT || request->input ||
        request->input_len || request->has_selection) {
        set_error(error, error_cap, "edit selection request is not empty");
        return 0;
    }
    edit_selection *entry = find_selection(ledger, selection);
    if (!entry) {
        set_error(error, error_cap,
                  "unknown or expired edit selection; read or search the file again");
        return 0;
    }
    if (!path || strcmp(path, entry->path)) {
        set_error(error, error_cap, "edit selection path does not match");
        return 0;
    }
    unsigned char *copy = malloc(entry->len);
    if (!copy) {
        set_error(error, error_cap, "out of memory resolving edit selection");
        return 0;
    }
    memcpy(copy, entry->bytes, entry->len);
    request->input = copy;
    request->input_len = entry->len;
    request->has_selection = 1;
    request->selection_offset = entry->offset;
    request->selection_length = entry->len;
    memcpy(request->selection_file_sha256, entry->file_sha256, 32);
    *input_out = copy;
    return 1;
}
