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
