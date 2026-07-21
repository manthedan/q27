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
        "\"path\":\"new.py\"}}</tool_call>\n"
        "```python\nprint(f\"hi {name}\")\n```\n";
    CHECK(parse(write, call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.request.kind == Q27_TOOL_WRITE && !call.missing_body &&
          call.request.input_len == std::strlen("print(f\"hi {name}\")\n") &&
          !std::memcmp(call.request.input, "print(f\"hi {name}\")\n",
                       call.request.input_len),
          "write carries same-turn fenced body with f-string intact");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"bad\",\"content\":\"embedded\"}}</tool_call>",
                call, error, sizeof(error)) == Q27_TOOL_CALL_INVALID,
          "write rejects JSON-embedded content");

    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.py\"}}</tool_call>",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.missing_body == 1 && call.request.kind == Q27_TOOL_WRITE,
          "write without fence is valid-but-missing-body");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.py\"}}</tool_call>\n"
                "print(\"no fence\")\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.missing_body == 1 && call.body_error,
          "unfenced body soft-fails with missing_body");
    q27_agent_tool_call_free(&call);

    // Observed T2 pattern: well-formed fence, then a spurious second
    // </tool_call> after free-decode. Body must still publish.
    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"amazing.py\"}}</tool_call>\n"
                "```python\nprint(f\"wow {n}\")\n```\n"
                "</tool_call>\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          !call.missing_body && call.request.kind == Q27_TOOL_WRITE &&
          call.request.input_len == std::strlen("print(f\"wow {n}\")\n") &&
          !std::memcmp(call.request.input, "print(f\"wow {n}\")\n",
                       call.request.input_len),
          "trailing protocol-echo </tool_call> after fence is ignored");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.py\"}}</tool_call>\n"
                "```\nbody\n```\n"
                "</tool_call>\n</tool_call>\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          !call.missing_body &&
          call.request.input_len == std::strlen("body\n"),
          "repeated trailing </tool_call> echo is ignored");
    q27_agent_tool_call_free(&call);

    // Premature fence closer must still soft-fail: remaining source is not
    // protocol echo, so we refuse to publish a truncated body.
    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.py\"}}</tool_call>\n"
                "```python\n"
                "s = \"\"\"\n"
                "```\n"
                "\"\"\"\n"
                "print(s)\n"
                "```\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.missing_body == 1,
          "premature fence closer with trailing source still soft-fails");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.py\"}}</tool_call>\n"
                "```\nok\n```\n"
                "Done writing.\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.missing_body == 1,
          "trailing prose after fence still soft-fails");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.py\"}}</tool_call>\n"
                "```\nok\n```\n"
                "<tool_call>{\"name\":\"shell\",\"arguments\":{"
                "\"command\":\"pwd\"}}</tool_call>\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.missing_body == 1,
          "second tool call after fence still soft-fails (no silent multi-tool)");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"read\",\"arguments\":{"
                "\"path\":\"a\"}}</tool_call>"
                "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"b\"}}"
                "</tool_call>",
                call, error, sizeof(error)) == Q27_TOOL_CALL_INVALID,
          "second tool_call for non-body tools still fails");

    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"pkg.json\"}}</tool_call>\n"
                "```json\n{\"name\":\"demo\",\"version\":\"1\"}\n```\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          !call.missing_body &&
          call.request.input_len == std::strlen("{\"name\":\"demo\",\"version\":\"1\"}\n"),
          "package.json-style {\"name\" body is allowed");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.py\"}}</tool_call>\n"
                "```\n<tool_call>{\"name\":\"write\"}</tool_call>\n```\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.missing_body == 1 && call.body_error,
          "tool-shaped fenced body soft-fails");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"doc.md\"}}</tool_call>\n"
                "````md\n"
                "example:\n"
                "```\n"
                "code\n"
                "```\n"
                "````\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          !call.missing_body &&
          std::string(reinterpret_cast<const char *>(call.request.input),
                      call.request.input_len) ==
              "example:\n```\ncode\n```\n",
          "longer outer fence allows nested triple-backtick content");
    q27_agent_tool_call_free(&call);

    std::string overwrite = "<tool_call>{\"name\":\"overwrite\",\"arguments\":{"
        "\"path\":\"x.py\"}}</tool_call>\n```\nfull file\n```\n";
    CHECK(parse(overwrite, call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.request.kind == Q27_TOOL_OVERWRITE &&
          call.request.input_len == std::strlen("full file\n") &&
          !std::memcmp(call.request.input, "full file\n", call.request.input_len),
          "overwrite accepts fenced whole-file body");
    q27_agent_tool_call_free(&call);

    std::string edit = "<tool_call>{\"name\":\"edit\",\"arguments\":{"
        "\"path\":\"a.bin\",\"old\":\"a\\u0000b\"}}</tool_call>\n"
        "```\nrepl\n```\n";
    CHECK(parse(edit, call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.request.kind == Q27_TOOL_EDIT && call.request.input_len == 3 &&
          !std::memcmp(call.request.input, "a\0b", 3) &&
          call.request.replacement_len == 5 &&
          !std::memcmp(call.request.replacement, "repl\n", 5),
          "edit control keeps old in JSON; replacement is fenced body");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"edit\",\"arguments\":{"
                "\"path\":\"a\",\"old\":\"b\","
                "\"replacement\":\"embedded\"}}</tool_call>",
                call, error, sizeof(error)) == Q27_TOOL_CALL_INVALID,
          "edit rejects JSON-embedded replacement");

    std::string long_old(600, 'x');
    std::string huge_edit =
        "<tool_call>{\"name\":\"edit\",\"arguments\":{\"path\":\"a.py\","
        "\"old\":\"" + long_old + "\"}}</tool_call>\n```\ny\n```\n";
    CHECK(parse(huge_edit, call, error, sizeof(error)) == Q27_TOOL_CALL_INVALID,
          "edit rejects oversized old; force overwrite");

    std::string selected = "<tool_call>{\"name\":\"edit_selection\",\"arguments\":{"
        "\"path\":\"a.bin\",\"selection\":\"s12345678-9\"}}</tool_call>\n"
        "```\nnew\n```\n";
    CHECK(parse(selected, call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.request.kind == Q27_TOOL_EDIT && !call.request.input &&
          call.request.input_len == 0 && call.selection &&
          !std::strcmp(call.selection, "s12345678-9") &&
          call.request.replacement_len == 4,
          "edit_selection carries handle plus fenced replacement");
    q27_agent_tool_call_free(&call);
    CHECK(parse("<tool_call>{\"name\":\"edit_selection\",\"arguments\":{"
                "\"path\":\"a\",\"selection\":\"s1\",\"old\":\"x\"}}"
                "</tool_call>", call, error, sizeof(error)) ==
              Q27_TOOL_CALL_INVALID,
          "edit_selection rejects mixed literal authority");

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
          "post-call prose fails closed for non-body tools");
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

    CHECK(q27_agent_tool_name_expects_body("write") == 1 &&
          q27_agent_tool_name_expects_body("overwrite") == 1 &&
          q27_agent_tool_name_expects_body("edit") == 1 &&
          q27_agent_tool_name_expects_body("read") == 0,
          "body-tool routing helper");
    char reject_err[128] = {0};
    static const unsigned char toolish[] =
        "<tool_call>\n{\"name\":\"write\"}\n</tool_call>";
    CHECK(q27_agent_payload_rejected(toolish, sizeof(toolish) - 1, reject_err,
                                     sizeof(reject_err)) != 0,
          "payload rejector catches tool-shaped bytes");

    const char *const *names = nullptr;
    CHECK(q27_agent_tool_names(&names) == 7 && names &&
          !std::strcmp(names[0], "read") && !std::strcmp(names[1], "search") &&
          !std::strcmp(names[2], "write") && !std::strcmp(names[3], "overwrite") &&
          !std::strcmp(names[4], "edit") &&
          !std::strcmp(names[5], "edit_selection") &&
          !std::strcmp(names[6], "shell"),
          "prompt and constrained decoder share one ordered registry");

    const char *preamble = q27_agent_tool_preamble();
    CHECK(preamble && std::strstr(preamble, "<tools>") &&
          std::strstr(preamble, "\"name\":\"read\"") &&
          std::strstr(preamble, "\"name\":\"write\"") &&
          std::strstr(preamble, "\"name\":\"overwrite\"") &&
          std::strstr(preamble, "\"name\":\"edit_selection\"") &&
          std::strstr(preamble, "markdown fenced") &&
          std::strstr(preamble, "additionalProperties"),
          "fixed strict registry is present in preamble");

    std::puts("q27 agent protocol selftest: PASS");
    return 0;
}
