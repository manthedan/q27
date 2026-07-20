#include "q27_agent_protocol.h"

#include <cstdio>
#include <cstring>
#include <string>

#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

static q27_agent_tool_call_status parse(const std::string& text,
                                        q27_agent_tool_call& call,
                                        char *error, size_t cap) {
    return q27_agent_parse_tool_call(
        reinterpret_cast<const unsigned char *>(text.data()), text.size(),
        &call, error, cap);
}

int main() {
    char error[256] = {0};
    q27_agent_tool_call call{};
    CHECK(parse("plain answer", call, error, sizeof(error)) == Q27_TOOL_CALL_NONE,
          "plain text is not a call");

    std::string read = "I will inspect it.\n<tool_call>\n"
        "{\"name\":\"read\",\"arguments\":{\"path\":\"src/a.cc\"}}\n"
        "</tool_call>\n";
    CHECK(parse(read, call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.request.kind == Q27_TOOL_READ &&
          std::strcmp(call.request.path, "src/a.cc") == 0,
          "strict read call parses with prefix prose");
    q27_agent_tool_call_free(&call);

    std::string write = "<tool_call>{\"name\":\"write\",\"arguments\":{"
        "\"path\":\"new.bin\"}}</tool_call>";
    CHECK(parse(write, call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.request.kind == Q27_TOOL_WRITE && !call.request.input &&
          call.request.input_len == 0,
          "write control call carries only a path");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"bad\",\"content\":\"embedded\"}}</tool_call>",
                call, error, sizeof(error)) == Q27_TOOL_CALL_INVALID,
          "write rejects JSON-embedded content");

    std::string edit = "<tool_call>{\"name\":\"edit\",\"arguments\":{"
        "\"path\":\"a.bin\",\"old\":\"a\\u0000b\"}}</tool_call>";
    CHECK(parse(edit, call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.request.kind == Q27_TOOL_EDIT && call.request.input_len == 3 &&
          !std::memcmp(call.request.input, "a\0b", 3) &&
          !call.request.replacement && call.request.replacement_len == 0,
          "edit control call carries old bytes but no replacement");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"edit\",\"arguments\":{"
                "\"path\":\"a\",\"old\":\"b\","
                "\"replacement\":\"embedded\"}}</tool_call>",
                call, error, sizeof(error)) == Q27_TOOL_CALL_INVALID,
          "edit rejects JSON-embedded replacement");

    std::string shell = "<tool_call>{\"name\":\"shell\",\"arguments\":{"
        "\"command\":\"pwd\",\"timeout_ms\":123,"
        "\"max_output_bytes\":99}}</tool_call>";
    CHECK(parse(shell, call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.request.kind == Q27_TOOL_SHELL &&
          call.request.timeout_ms == 123 && call.request.max_output_bytes == 99,
          "bounded shell options parse");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\",\"x\":1}}</tool_call>",
                call, error, sizeof(error)) == Q27_TOOL_CALL_INVALID,
          "unknown arguments fail closed");
    CHECK(parse("<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"x\"}}</tool_call>junk",
                call, error, sizeof(error)) == Q27_TOOL_CALL_INVALID,
          "post-call prose fails closed");
    CHECK(parse("<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}",
                call, error, sizeof(error)) == Q27_TOOL_CALL_INVALID,
          "truncated wrapper fails closed");
    CHECK(parse("<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\\u0000b\"}}</tool_call>",
                call, error, sizeof(error)) == Q27_TOOL_CALL_INVALID,
          "NUL pathname fails closed");
    CHECK(parse("<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}</tool_call>"
                "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"b\"}}</tool_call>",
                call, error, sizeof(error)) == Q27_TOOL_CALL_INVALID,
          "multiple calls fail closed");

    unsigned char fenced_python[] =
        "```python\nprint(f\"hello {name}\")\n```\n";
    size_t fenced_python_len = sizeof(fenced_python) - 1;
    static const char expected_python[] = "print(f\"hello {name}\")\n";
    CHECK(q27_agent_unwrap_whole_file_source_fence(
              "src/primes.py", fenced_python, &fenced_python_len) == 1 &&
          fenced_python_len == sizeof(expected_python) - 1 &&
          !std::memcmp(fenced_python, expected_python, fenced_python_len) &&
          fenced_python[fenced_python_len] == 0,
          "matching outer source fence is removed in place");

    unsigned char fenced_crlf[] =
        "```python\r\nprint(27)\r\n```\r\n";
    size_t fenced_crlf_len = sizeof(fenced_crlf) - 1;
    static const char expected_crlf[] = "print(27)\r\n";
    CHECK(q27_agent_unwrap_whole_file_source_fence(
              "windows.py", fenced_crlf, &fenced_crlf_len) == 1 &&
          fenced_crlf_len == sizeof(expected_crlf) - 1 &&
          !std::memcmp(fenced_crlf, expected_crlf, fenced_crlf_len),
          "matching CRLF source fence preserves CRLF body bytes");

    unsigned char long_closer[] =
        "```python\nprint(27)\n   ````  \n";
    size_t long_closer_len = sizeof(long_closer) - 1;
    static const char expected_long_closer[] = "print(27)\n";
    CHECK(q27_agent_unwrap_whole_file_source_fence(
              "long.py", long_closer, &long_closer_len) == 1 &&
          long_closer_len == sizeof(expected_long_closer) - 1 &&
          !std::memcmp(long_closer, expected_long_closer, long_closer_len),
          "longer whitespace-suffixed final closer is accepted");

    unsigned char empty_fence[] = "```py\n```";
    size_t empty_fence_len = sizeof(empty_fence) - 1;
    CHECK(q27_agent_unwrap_whole_file_source_fence(
              "empty.py", empty_fence, &empty_fence_len) == 1 &&
          empty_fence_len == 0 && empty_fence[0] == 0,
          "empty matching source fence becomes an empty payload");

    unsigned char markdown_fence[] = "```python\nprint(1)\n```\n";
    size_t markdown_fence_len = sizeof(markdown_fence) - 1;
    CHECK(q27_agent_unwrap_whole_file_source_fence(
              "README.md", markdown_fence, &markdown_fence_len) == 0 &&
          markdown_fence_len == sizeof(markdown_fence) - 1,
          "Markdown files retain intentional fences");

    unsigned char wrong_label[] = "```javascript\nprint(1)\n```\n";
    size_t wrong_label_len = sizeof(wrong_label) - 1;
    CHECK(q27_agent_unwrap_whole_file_source_fence(
              "script.py", wrong_label, &wrong_label_len) == 0,
          "mismatched source language remains exact");

    unsigned char trailing_prose[] = "```python\nprint(1)\n```\nDone.";
    size_t trailing_prose_len = sizeof(trailing_prose) - 1;
    CHECK(q27_agent_unwrap_whole_file_source_fence(
              "script.py", trailing_prose, &trailing_prose_len) == 0,
          "fence with trailing prose is ambiguous and remains exact");

    unsigned char multiple_fences[] =
        "```python\nprint(1)\n  ````  \nexplanation\n```\n";
    size_t multiple_fences_len = sizeof(multiple_fences) - 1;
    CHECK(q27_agent_unwrap_whole_file_source_fence(
              "script.py", multiple_fences, &multiple_fences_len) == 0 &&
          multiple_fences_len == sizeof(multiple_fences) - 1,
          "earlier closing fence makes the wrapper ambiguous");

    const char *const *names = nullptr;
    CHECK(q27_agent_tool_names(&names) == 5 && names &&
          !std::strcmp(names[0], "read") && !std::strcmp(names[1], "search") &&
          !std::strcmp(names[2], "write") && !std::strcmp(names[3], "edit") &&
          !std::strcmp(names[4], "shell"),
          "prompt and constrained decoder share one ordered registry");

    const char *preamble = q27_agent_tool_preamble();
    CHECK(preamble && std::strstr(preamble, "<tools>") &&
          std::strstr(preamble, "\"name\":\"read\"") &&
          std::strstr(preamble, "\"name\":\"write\"") &&
          std::strstr(preamble, "\"name\":\"shell\"") &&
          std::strstr(preamble, "additionalProperties"),
          "fixed strict registry is present in preamble");

    std::puts("q27 agent protocol selftest: PASS");
    return 0;
}
