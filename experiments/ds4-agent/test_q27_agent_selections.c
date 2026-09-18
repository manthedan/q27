#include "q27_agent_selections.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

static int contains(const unsigned char *haystack, size_t hlen,
                    const char *needle) {
    const size_t nlen = strlen(needle);
    if (!nlen || nlen > hlen) return 0;
    for (size_t i = 0; i <= hlen - nlen; ++i)
        if (!memcmp(haystack + i, needle, nlen)) return 1;
    return 0;
}

static q27_agent_tool_result fixture_result(void) {
    q27_agent_tool_result result = {0};
    result.exit_code = 0;
    result.has_file_sha256 = 1;
    result.file_size = 8;
    memset(result.file_sha256, 0x27, sizeof(result.file_sha256));
    result.selection_count = 2;
    result.selections[0] = (q27_agent_tool_selection){
        .file_offset = 0, .length = 3,
        .output_offset = 0, .output_length = 3,
        .first_line = 1, .last_line = 1};
    result.selections[1] = (q27_agent_tool_selection){
        .file_offset = 4, .length = 3,
        .output_offset = 4, .output_length = 3,
        .first_line = 2, .last_line = 2};
    return result;
}

int main(void) {
    q27_agent_selection_ledger *ledger =
        q27_agent_selection_ledger_create(0x12345678u);
    CHECK(ledger, "selection ledger allocates");
    const unsigned char raw[] = "one\ntwo\n";
    q27_agent_tool_request read_request = {
        .kind = Q27_TOOL_READ, .path = "src/a.txt", .max_output_bytes = 1024};
    q27_agent_tool_result result = fixture_result();
    unsigned char *annotation = NULL;
    size_t annotation_len = 0;
    char error[256] = {0};
    CHECK(q27_agent_selection_annotate(
              ledger, &read_request, &result, raw, sizeof(raw) - 1, 1024,
              &annotation, &annotation_len, error, sizeof(error)) &&
          annotation && annotation_len &&
          contains(annotation, annotation_len, "[s12345678-1] lines 1-1") &&
          contains(annotation, annotation_len, "[s12345678-2] lines 2-2"),
          "annotation renders predictable handles and line ranges");
    free(annotation);

    q27_agent_tool_request edit = {
        .kind = Q27_TOOL_EDIT, .path = "src/a.txt", .max_output_bytes = 1024};
    unsigned char *selected = NULL;
    CHECK(q27_agent_selection_resolve(
              ledger, "s12345678-1", "src/a.txt", &edit, &selected,
              error, sizeof(error)) && selected && edit.has_selection &&
          edit.selection_offset == 0 && edit.selection_length == 3 &&
          edit.input_len == 3 && !memcmp(edit.input, "one", 3) &&
          edit.selection_file_sha256[0] == 0x27,
          "handle resolves to owned bytes, range, and full-file digest");
    free(selected);
    edit.input = NULL;
    edit.input_len = 0;
    edit.has_selection = 0;
    CHECK(!q27_agent_selection_resolve(
              ledger, "s12345678-2", "src/other.txt", &edit, &selected,
              error, sizeof(error)) && strstr(error, "path"),
          "path-mismatched handle fails closed");
    CHECK(!q27_agent_selection_resolve(
              ledger, "s12345678-999", "src/a.txt", &edit, &selected,
              error, sizeof(error)) && strstr(error, "unknown"),
          "unknown handle fails closed");

    annotation = NULL;
    annotation_len = 99;
    CHECK(q27_agent_selection_annotate(
              ledger, &read_request, &result, raw, sizeof(raw) - 1,
              sizeof(raw) - 1, &annotation, &annotation_len,
              error, sizeof(error)) && !annotation && annotation_len == 0,
          "insufficient annotation budget preserves exact raw output");

    q27_agent_tool_result single = fixture_result();
    single.file_size = 1;
    single.selection_count = 1;
    single.selections[0] = (q27_agent_tool_selection){
        .file_offset = 0, .length = 1,
        .output_offset = 0, .output_length = 1,
        .first_line = 1, .last_line = 1};
    const unsigned char x = 'x';
    for (unsigned i = 0; i < 255; ++i) {
        annotation = NULL;
        annotation_len = 0;
        CHECK(q27_agent_selection_annotate(
                  ledger, &read_request, &single, &x, 1, 1024,
                  &annotation, &annotation_len, error, sizeof(error)) &&
              annotation && annotation_len,
              "bounded ledger accepts replacement selection");
        free(annotation);
    }
    CHECK(!q27_agent_selection_resolve(
              ledger, "s12345678-1", "src/a.txt", &edit, &selected,
              error, sizeof(error)) && strstr(error, "expired"),
          "256-slot reuse expires the oldest handle without aliasing");

    q27_agent_selection_ledger_free(ledger);
    puts("q27 agent selections selftest: PASS");
    return 0;
}
