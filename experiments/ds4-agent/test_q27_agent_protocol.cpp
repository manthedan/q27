#include "q27_agent_protocol.h"

#include <cstdio>
#include <cstring>
#include <string>

#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

static q27_agent_tool_call_status parse(const std::string& text,
                                        q27_agent_tool_call& call,
                                        char *error, size_t cap, int eos = 1,
                                        int xml_dialect = 0) {
    return q27_agent_parse_tool_call(
        reinterpret_cast<const unsigned char *>(text.data()), text.size(),
        &call, error, cap, eos, xml_dialect);
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

    std::string native_read = "<tool_call>\n<function=read>\n<parameter=path>\n"
        "src/native.cc\n</parameter>\n</function>\n</tool_call>\n";
    CHECK(parse(native_read, call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.request.kind == Q27_TOOL_READ &&
          std::strcmp(call.request.path, "src/native.cc") == 0,
          "native XML read call parses");
    q27_agent_tool_call_free(&call);

    std::string bare_native_read = "<function=read>\n<parameter=path>\n"
        "src/bare-native.cc\n</parameter>\n</function>\n";
    CHECK(parse(bare_native_read, call, error, sizeof(error), 1, 1) ==
              Q27_TOOL_CALL_VALID && call.request.kind == Q27_TOOL_READ &&
          std::strcmp(call.request.path, "src/bare-native.cc") == 0,
          "selected XML dialect normalizes a wrapperless native read call");
    q27_agent_tool_call_free(&call);
    std::string thinking_then_native =
        "reasoning about the file\n</think>\n"
        "<function=read><parameter=path>src/after-think.cc"
        "</parameter></function>";
    CHECK(parse(thinking_then_native, call, error, sizeof(error), 1, 1) ==
              Q27_TOOL_CALL_VALID &&
          std::strcmp(call.request.path, "src/after-think.cc") == 0,
          "wrapperless native call parses after the thinking span");
    q27_agent_tool_call_free(&call);
    CHECK(parse(bare_native_read, call, error, sizeof(error)) ==
              Q27_TOOL_CALL_NONE,
          "JSON dialect leaves wrapperless native XML as text");
    CHECK(parse("Example:\n```xml\n<function=read><parameter=path>x"
                "</parameter></function>\n```", call, error, sizeof(error), 1, 1) ==
              Q27_TOOL_CALL_NONE,
          "wrapperless native XML inside a markdown fence is not executed");
    CHECK(parse("Example:\n``````xml\n<function=shell><parameter=command>"
                "echo unsafe</parameter></function>\n``````", call, error,
                sizeof(error), 1, 1) == Q27_TOOL_CALL_NONE,
          "six-backtick native XML examples are not executed");
    CHECK(parse("Example:\n~~~~xml\n<function=shell><parameter=command>"
                "echo unsafe</parameter></function>\n~~~~", call, error,
                sizeof(error), 1, 1) == Q27_TOOL_CALL_NONE,
          "tilde-fenced native XML examples are not executed");
    CHECK(parse("    <function=shell><parameter=command>echo unsafe"
                "</parameter></function>", call, error, sizeof(error), 1, 1) ==
              Q27_TOOL_CALL_NONE,
          "indented-code native XML examples are not executed");
    CHECK(parse("<function=fixture><parameter=value>\n<function=shell>"
                "<parameter=command>echo unsafe</parameter></function>"
                "</parameter></function>", call, error, sizeof(error), 1, 1) !=
              Q27_TOOL_CALL_VALID,
          "native XML nested in another parameter is not executed");
    CHECK(parse("{\"example\":\"<function=read><parameter=path>/tmp/x"
                "</parameter></function>\"}", call, error, sizeof(error), 1, 1) ==
              Q27_TOOL_CALL_NONE,
          "quoted native XML examples are not executed");
    std::string example_then_call =
        "Example:\n```xml\n<function=read><parameter=path>ignored"
        "</parameter></function>\n```\nNow inspect it:\n"
        "<function=read><parameter=path>src/later.cc</parameter></function>";
    CHECK(parse(example_then_call, call, error, sizeof(error), 1, 1) ==
              Q27_TOOL_CALL_NONE,
          "native parser never executes a later call after displayed examples");
    CHECK(parse("Quoted output:\n\"\n<function=shell><parameter=command>"
                "echo unsafe</parameter></function>\n\"", call, error,
                sizeof(error), 1, 1) == Q27_TOOL_CALL_NONE,
          "multiline quoted native XML examples are not executed");

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

    std::string native_write =
        "<tool_call>\n<function=write>\n<parameter=path>\nnative.py\n</parameter>\n"
        "</function>\n</tool_call>\n```python\nprint('native')\n```\n";
    CHECK(parse(native_write, call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.request.kind == Q27_TOOL_WRITE && !call.missing_body &&
          std::strcmp(call.request.path, "native.py") == 0 &&
          call.request.input_len == std::strlen("print('native')\n"),
          "native XML write carries same-turn fenced body");
    q27_agent_tool_call_free(&call);

    std::string bare_native_write =
        "<function=write>\n<parameter=path>\nbare-native.py\n</parameter>\n"
        "</function>\n```python\nprint('bare native')\n```\n";
    CHECK(parse(bare_native_write, call, error, sizeof(error), 1, 1) ==
              Q27_TOOL_CALL_VALID && call.request.kind == Q27_TOOL_WRITE &&
          !call.missing_body &&
          std::strcmp(call.request.path, "bare-native.py") == 0 &&
          call.request.input_len == std::strlen("print('bare native')\n"),
          "wrapperless native XML write preserves its same-turn fenced body");
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

    // Live T2/sampled failure: opener + full file + trailing protocol echo,
    // no closing ```. Recover the body rather than soft-fail forever.
    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"python_file.py\"}}</tool_call>\n"
                "```python\n"
                "def main():\n"
                "    print(f\"hi {name}\")\n"
                "\n"
                "if __name__ == \"__main__\":\n"
                "    main()\n"
                "</tool_call>\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          !call.missing_body &&
          std::string(reinterpret_cast<const char *>(call.request.input),
                      call.request.input_len) ==
              "def main():\n"
              "    print(f\"hi {name}\")\n"
              "\n"
              "if __name__ == \"__main__\":\n"
              "    main()\n",
          "unclosed fence recovers body and strips trailing </tool_call>");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.py\"}}</tool_call>\n"
                "```python\nprint(1)\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          !call.missing_body &&
          std::string(reinterpret_cast<const char *>(call.request.input),
                      call.request.input_len) == "print(1)\n",
          "unclosed fence at EOS recovers body");
    q27_agent_tool_call_free(&call);

    // Output-limit stop: the same unclosed fence must NOT recover — the body
    // is truncated and publishing it would silently write a partial file
    // (codex branch-review P1). Soft-fail with an actionable error instead.
    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.py\"}}</tool_call>\n"
                "```python\nprint(1)\n",
                call, error, sizeof(error), /*eos=*/0) == Q27_TOOL_CALL_VALID &&
          call.missing_body == 1 &&
          call.body_error &&
          std::string(call.body_error).find("output limit") != std::string::npos,
          "unclosed fence at max_tokens fails closed (no truncated publish)");
    q27_agent_tool_call_free(&call);

    // CommonMark-length outer fence: content containing ``` lines stays
    // byte-exact when the outer fence is longer (the documented escape for
    // the inner-closer ambiguity, branch-review P2).
    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.md\"}}</tool_call>\n"
                "````md\n"
                "# Title\n"
                "```python\n"
                "code\n"
                "```\n"
                "tail\n"
                "````\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          !call.missing_body &&
          std::string(reinterpret_cast<const char *>(call.request.input),
                      call.request.input_len) ==
              "# Title\n```python\ncode\n```\ntail\n",
          "longer outer fence preserves inner fence lines byte-exactly");
    q27_agent_tool_call_free(&call);

    // Tick plumbing: the cycle's legacy double-wrap unwrap may only run for a
    // minimal 3-tick transport; a longer fence marks inner fences as content
    // (codex branch-review P2). Extract always preserves the body either way.
    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.py\"}}</tool_call>\n"
                "```python\nprint(1)\n```\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          !call.missing_body && call.body_fence_ticks == 3,
          "minimal transport reports 3 ticks (unwrap eligible)");
    q27_agent_tool_call_free(&call);
    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.py\"}}</tool_call>\n"
                "````python\n```python\nprint(1)\n```\n````\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          !call.missing_body && call.body_fence_ticks == 4 &&
          std::string(reinterpret_cast<const char *>(call.request.input),
                      call.request.input_len) == "```python\nprint(1)\n```\n",
          "longer transport reports 4 ticks and keeps the inner wrapper");
    q27_agent_tool_call_free(&call);

    // Bare tool-call JSON as a body fails closed (re-emitted call as
    // content); ordinary JSON like package.json stays legal (branch-review P2).
    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.json\"}}</tool_call>\n"
                "```json\n"
                "{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}\n"
                "```\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.missing_body == 1 && call.body_error &&
          std::string(call.body_error).find("tool call") != std::string::npos,
          "bare tool-call JSON body fails closed");
    q27_agent_tool_call_free(&call);
    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"package.json\"}}</tool_call>\n"
                "```json\n"
                "{\"name\":\"my-package\",\"version\":\"1.0.0\"}\n"
                "```\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          !call.missing_body &&
          std::string(reinterpret_cast<const char *>(call.request.input),
                      call.request.input_len) ==
              "{\"name\":\"my-package\",\"version\":\"1.0.0\"}\n",
          "package.json body stays legal");
    q27_agent_tool_call_free(&call);

    // Second tool call after an unclosed fence must not land as file content.
    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.py\"}}</tool_call>\n"
                "```python\nprint(1)\n"
                "<tool_call>{\"name\":\"shell\",\"arguments\":{"
                "\"command\":\"pwd\"}}</tool_call>\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.missing_body == 1,
          "unclosed fence with a second tool call soft-fails");
    q27_agent_tool_call_free(&call);

    // Content that ends with the literal tag mid-line is kept (not protocol
    // echo, which is always a whole trailing line).
    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.txt\"}}</tool_call>\n"
                "```\n"
                "note: ends with </tool_call>\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          !call.missing_body &&
          std::string(reinterpret_cast<const char *>(call.request.input),
                      call.request.input_len) ==
              "note: ends with </tool_call>\n",
          "unclosed recovery keeps mid-line </tool_call> content");
    q27_agent_tool_call_free(&call);

    CHECK(parse("<tool_call>{\"name\":\"write\",\"arguments\":{"
                "\"path\":\"x.py\"}}</tool_call>\n"
                "```python\nprint(1)\n"
                "  </tool_call>\n",
                call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          !call.missing_body &&
          std::string(reinterpret_cast<const char *>(call.request.input),
                      call.request.input_len) == "print(1)\n",
          "unclosed recovery strips indented trailing protocol echo");
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

    std::string native_shell =
        "<tool_call>\n<function=shell>\n<parameter=command>\nprintf 'x &amp; y'\n"
        "</parameter>\n<parameter=timeout_ms>\n321\n</parameter>\n"
        "<parameter=max_output_bytes>\n654\n</parameter>\n</function>\n</tool_call>";
    CHECK(parse(native_shell, call, error, sizeof(error)) == Q27_TOOL_CALL_VALID &&
          call.request.kind == Q27_TOOL_SHELL && call.request.timeout_ms == 321 &&
          call.request.max_output_bytes == 654 &&
          std::string(reinterpret_cast<const char *>(call.request.input),
                      call.request.input_len) == "printf 'x & y'",
          "native XML shell uses schema types and decodes escaped string bytes");
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
    static const unsigned char literal_function_file[] =
        "<function=fixture><parameter=value>x</parameter></function>";
    CHECK(q27_agent_payload_rejected(
              literal_function_file, sizeof(literal_function_file) - 1,
              reject_err, sizeof(reject_err)) == 0,
          "legitimate files beginning with a function-like XML tag remain writable");

    const char *const *names = nullptr;
    CHECK(q27_agent_tool_names(&names) == 7 && names &&
          !std::strcmp(names[0], "read") && !std::strcmp(names[1], "search") &&
          !std::strcmp(names[2], "write") && !std::strcmp(names[3], "overwrite") &&
          !std::strcmp(names[4], "edit") &&
          !std::strcmp(names[5], "edit_selection") &&
          !std::strcmp(names[6], "shell"),
          "prompt and constrained decoder share one ordered registry");

    const char *preamble = q27_agent_tool_preamble(1);
    CHECK(preamble && std::strstr(preamble, "<tools>") &&
          std::strstr(preamble, "\"name\":\"read\"") &&
          std::strstr(preamble, "\"name\":\"write\"") &&
          std::strstr(preamble, "\"name\":\"overwrite\"") &&
          std::strstr(preamble, "\"name\":\"edit_selection\"") &&
          std::strstr(preamble, "markdown fenced") &&
          std::strstr(preamble, "<function=write>") &&
          std::strstr(preamble, "additionalProperties"),
          "fixed strict registry and native XML example are present in preamble");
    const char *json_preamble = q27_agent_tool_preamble(0);
    CHECK(json_preamble && !std::strstr(json_preamble, "<function=write>") &&
          std::strstr(json_preamble, "{\"name\":\"write\"") &&
          std::strcmp(json_preamble, preamble) != 0,
          "native preamble cache remains dialect-specific");

    std::puts("q27 agent protocol selftest: PASS");
    return 0;
}
