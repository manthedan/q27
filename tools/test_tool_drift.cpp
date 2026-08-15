// CPU-only regression test for the bare-tool-call drift recoveries in
// api_common.h (parse_bare_tool_calls). Covers the modes that have bitten
// real Claude Code sessions:
//   mode 10 -- dropped `{"name": "` opener (issue: flask-5014 early quit)
//   mode 11 -- raw code-body string value, unescaped inner quotes (issue #4)
// plus the negatives (prose must not false-recover, well-formed calls take
// the normal path).
//
// Build + run (no CUDA needed):
//   g++ -std=c++17 -I src tools/test_tool_drift.cpp -o build/test_tool_drift && ./build/test_tool_drift
#include "api_common.h"
#include "toolgram.h"
#include <cstdio>
#include <string>

using json = nlohmann::json;

static int failures = 0;
static void ok(bool cond, const char* name) {
    printf("  %s %s\n", cond ? "PASS" : "FAIL", name);
    if (!cond) failures++;
}

static json tool(const char* name, std::vector<std::pair<std::string, bool>> params) {
    // params: (key, is_required); all typed string for these tests
    json props = json::object(), req = json::array();
    for (auto& p : params) {
        props[p.first] = {{"type", "string"}};
        if (p.second) req.push_back(p.first);
    }
    return {{"type", "function"},
            {"function",
             {{"name", name},
              {"parameters", {{"type", "object"}, {"properties", props}, {"required", req}}}}}};
}

// Drift mode 13 (2026-07-24, found live by the grammar-engage probe): a
// wrapper-less call truncated INSIDE an escape sequence. The repair used to
// append the closing quote straight after a dangling backslash, which escaped
// it -- string still open, object never parsed, call UN-RESCUED. The real
// payload was a Write whose markdown content was cut while writing an escaped
// JSON example (`\\"role\\": \\"assistant\\",\\`).
// Drift mode 14 (2026-08-14, captured live from a thunderdome run on the
// Qwen3.8-27B q5f repack, bench-time-tracker trial-1). The model was told to
// emit `<tool_call>\n{"name":..., "arguments":{...}}\n</tool_call>` by
// tools_preamble and DID so correctly on turn 1, then drifted on turn 2 into
// its chat-template's XML dialect with no <tool_call> wrapper at all:
//
//   <tool_name>Read</tool_name>
//   <parameter=file_path>/workspace/tests/index.test.ts</parameter>
//   <tool_name>Bash</tool_name>
//   <parameter=arguments>
//   {"command":"...","description":"..."}}
//
// Three things make this its own mode rather than a variant of 10/11:
//   1. TWO calls are concatenated in one assistant block.
//   2. The conventions DISAGREE between them -- Read uses a scalar
//      <parameter=KEY>VALUE</parameter>, Bash uses <parameter=arguments>
//      wrapping a whole JSON object and never closes the tag.
//   3. The Bash payload carries one unbalanced trailing brace.
//
// Claude Code saw plain text, made no tool call, and the trial ended at
// num_turns=2 scoring 0.000 -- the same shape as the pre-fix 3.6 agentic
// ceiling, which was parser-bound rather than quality-bound.
//
// NOT YET RESCUED. This fixture pins the observed bytes and the CURRENT
// behaviour so a future parser change has a real target and a regression
// witness. Flip `expect_rescued` to true in the same commit that teaches
// parse_bare_tool_calls this dialect.
// Drift mode 15 (2026-08-14, Qwen3.8-27B): surfaced by re-running the same
// thunderdome task once mode 14 stopped the earlier stall. An unclosed <name>
// pseudo-tag, then a bare identifier and JSON args with no opening brace or
// quote, plus the same trailing unbalanced brace seen in mode 14.
static void test_mode15_name_tag_bare_args() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command"]}}}])");
    // Verbatim bytes from the captured transcript.
    std::string t =
        "<name>Bash, \"arguments\": {\"command\":\"ls /workspace/tests /workspace/src && cat "
        "/workspace/.eslintrc.cjs /workspace/vitest.config.ts\",\"description\":\"List tests "
        "and src, show eslint and vitest config\"}}";
    std::string pre;
    auto v = q27::parse_bare_tool_calls(t, &pre, &tools);
    ok(v.size() == 1 && v[0].ok && v[0].name == "Bash",
       "mode15: <name> pseudo-tag with bare args recovered");
    if (v.size() == 1)
        ok(v[0].arguments.value("description", std::string()) ==
               "List tests and src, show eslint and vitest config",
           "mode15: trailing unbalanced brace tolerated, args intact");
    // An undeclared name must NOT be rescued.
    std::string bad = "<name>NotATool, \"arguments\": {\"x\":1}}";
    auto v2 = q27::parse_bare_tool_calls(bad, &pre, &tools);
    ok(v2.empty(), "mode15: undeclared name is rejected");
}

// Drift mode 16 and the mode-15 attribute variant (2026-08-14), both harvested
// from the same batch of Qwen3.8 thunderdome runs. Also pins the one form that
// must NEVER be rescued.
static void test_mode16_and_15_variants() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command"]}}},{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}}])");
    std::string pre;

    // mode 15 variant: attribute spelling <name="Read", ...
    std::string attr = "<name=\"Read\", \"arguments\": {\"file_path\": \"/workspace/tests/tracker.test.ts\"}}";
    auto v1 = q27::parse_bare_tool_calls(attr, &pre, &tools);
    ok(v1.size() == 1 && v1[0].ok && v1[0].name == "Read" &&
           v1[0].arguments.value("file_path", std::string()) ==
               "/workspace/tests/tracker.test.ts",
       "mode15-attr: <name=\"X\" spelling recovered");

    // mode 16: correct JSON, wrong wrapper.
    std::string fn = "I'll start by exploring the workspace.\n\n<function>\n"
        "{\"name\": \"Bash\", \"arguments\": {\"command\": \"ls -la /workspace\", "
        "\"description\": \"List workspace\"}}";
    auto v2 = q27::parse_bare_tool_calls(fn, &pre, &tools);
    ok(v2.size() == 1 && v2[0].ok && v2[0].name == "Bash" &&
           v2[0].arguments.value("command", std::string()) == "ls -la /workspace",
       "mode16: <function>-wrapped JSON recovered");
    ok(pre.find("exploring the workspace") != std::string::npos,
       "mode16: prose before the wrapper is preserved as prefix");

    // MUST NOT RESCUE: a hallucinated tool RESULT, not a call. Rescuing this
    // would feed invented command output back as though a tool had run.
    std::string halluc =
        "I'll start by exploring the codebase.\n\n<tool_calls>\n<result>\n"
        "<name>Bash</name>\n<output>total 40\ndrwxr-xr-x 1 node node 4096 .\n</output>\n"
        "</result>\n</tool_calls>";
    auto v3 = q27::parse_bare_tool_calls(halluc, &pre, &tools);
    ok(v3.empty(), "hallucinated <result>/<output> block is NOT rescued as a call");
}

static void test_think_mode_drift() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Write","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}},{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}}])");
    std::string prefix;
    auto bare = q27::parse_bare_tool_calls(
        "\n<function=Read>\n<parameter=file_path>\n/workspace/tests/phase-06.test.ts\n</parameter>\n</function>\n",
        &prefix, &tools, true, true, true);
    ok(bare.size() == 1 && bare[0].ok && bare[0].name == "Read" &&
           bare[0].arguments.value("file_path", std::string()) ==
               "/workspace/tests/phase-06.test.ts",
       "bare native dialect without wrapper recovered");
    auto bare_multi = q27::parse_bare_tool_calls(
        "<function=Read>\n<parameter=file_path>\n/a\n</parameter>\n</function>\n"
        "<function=Read>\n<parameter=file_path>\n/b\n</parameter>\n</function>",
        &prefix, &tools, true, true, true);
    ok(bare_multi.size() == 2 && bare_multi[0].name == "Read" &&
           bare_multi[1].name == "Read" &&
           bare_multi[0].arguments.value("file_path", std::string()) == "/a" &&
           bare_multi[1].arguments.value("file_path", std::string()) == "/b",
       "bare native XML preserves consecutive multi-call turns");
    auto json_dialect_xml = q27::parse_bare_tool_calls(
        "<function=Read>\n<parameter=file_path>\n/workspace/tests/phase-06.test.ts\n"
        "</parameter>\n</function>", &prefix, &tools);
    ok(json_dialect_xml.empty(),
       "bare native XML is not executable outside the selected XML dialect");
    auto quoted_xml = q27::parse_bare_tool_calls(
        "{\"example\":\"<function=Read><parameter=file_path>/tmp/x"
        "</parameter></function>\"}", &prefix, &tools, true, true, true);
    ok(quoted_xml.empty(),
       "bare native XML embedded in a quoted example is not executed");
    auto long_fenced_xml = q27::parse_bare_tool_calls(
        "Example:\n``````xml\n<function=Read><parameter=file_path>/unsafe"
        "</parameter></function>\n``````", &prefix, &tools, true, true, true);
    ok(long_fenced_xml.empty(),
       "six-backtick native XML examples are not executed");
    auto tilde_fenced_xml = q27::parse_bare_tool_calls(
        "Example:\n~~~~xml\n<function=Read><parameter=file_path>/unsafe"
        "</parameter></function>\n~~~~", &prefix, &tools, true, true, true);
    ok(tilde_fenced_xml.empty(),
       "tilde-fenced native XML examples are not executed");
    auto indented_xml = q27::parse_bare_tool_calls(
        "    <function=Read><parameter=file_path>/unsafe</parameter></function>",
        &prefix, &tools, true, true, true);
    ok(indented_xml.empty(),
       "four-space indented native XML examples are not executed");
    auto nested_xml = q27::parse_bare_tool_calls(
        "<function=Fixture><parameter=value>\n"
        "<function=Read><parameter=file_path>/unsafe</parameter></function>\n"
        "</parameter></function>",
        &prefix, &tools, true, true, true);
    ok(nested_xml.empty(),
       "native XML nested in another parameter is not executed");
    auto example_then_call = q27::parse_bare_tool_calls(
        "Example:\n```xml\n<function=Read><parameter=file_path>/ignored"
        "</parameter></function>\n```\nNow inspect:\n"
        "<function=Read><parameter=file_path>/later</parameter></function>",
        &prefix, &tools, true, true, true);
    ok(example_then_call.empty(),
       "bare recovery never executes a later call after displayed prose/examples");
    auto multiline_quoted_xml = q27::parse_bare_tool_calls(
        "Quoted output:\n\"\n<function=Read><parameter=file_path>/unsafe"
        "</parameter></function>\n\"",
        &prefix, &tools, true, true, true);
    ok(multiline_quoted_xml.empty(),
       "multiline quoted native XML examples are not executed");
    auto literal_result = q27::parse_bare_tool_calls(
        "<function=Write>\n<parameter=file_path>\n/out\n</parameter>\n"
        "<parameter=content>\n<output>literal</output>\n</parameter>\n</function>",
        &prefix, &tools, true, true, true);
    ok(literal_result.size() == 1 &&
           literal_result[0].arguments.value("content", std::string()) ==
               "<output>literal</output>",
       "native XML arguments may contain literal result/output tags");

    auto chimera = q27::parse_bare_tool_calls(
        "The test suite is clear. Writing the implementation:\n\n"
        "{\"name\": \"Write\",\n<parameter=file_path>\n/workspace/src/index.ts\n</parameter>\n"
        "<parameter=content>\nimport { existsSync, mkdirSync } from 'fs';\nconst x = 1;\n</parameter>",
        &prefix, &tools, true, true, true);
    ok(chimera.size() == 1 && chimera[0].ok && chimera[0].name == "Write" &&
           chimera[0].arguments.value("file_path", std::string()) ==
               "/workspace/src/index.ts" &&
           chimera[0].arguments.value("content", std::string()).rfind(
               "import { existsSync", 0) == 0,
       "mode17: json-head/xml-params chimera recovered");
    ok(prefix.find("Writing the implementation") != std::string::npos,
       "mode17: prose before the chimera preserved as prefix");

    auto truncated_chimera = q27::parse_bare_tool_calls(
        "{\"name\": \"Write\",\n<parameter=file_path>\n/workspace/src/index.ts\n</parameter>\n"
        "<parameter=content>\npartial",
        &prefix, &tools, false, true, true);
    ok(truncated_chimera.empty(),
       "mode17: truncation repair disabled rejects incomplete XML parameters");
    auto complete_chimera = q27::parse_bare_tool_calls(
        "{\"name\": \"Write\",\n<parameter=file_path>\n/workspace/src/index.ts\n</parameter>\n"
        "<parameter=content>\ncomplete\n</parameter>\n</function>",
        &prefix, &tools, false, true, true);
    ok(complete_chimera.size() == 1 && complete_chimera[0].ok &&
           complete_chimera[0].arguments.value("content", std::string()) == "complete",
       "mode17: complete XML chimera survives fail-closed mode");

    auto undeclared = q27::parse_bare_tool_calls(
        "{\"name\": \"NotATool\",\n<parameter=x>\n1\n</parameter>",
        &prefix, &tools, true, true, true);
    ok(undeclared.empty(), "mode17: undeclared chimera name is not rescued");
    auto fenced = q27::parse_bare_tool_calls(
        "Example only:\n```xml\n<function=Read>\n<parameter=file_path>\n/etc/passwd\n"
        "</parameter>\n</function>\n```",
        &prefix, &tools, true, true, true);
    ok(fenced.empty(), "mode17: fenced native XML example is not executed");
}

static void test_dialect_default_keying() {
    unsetenv("Q27_TOOL_DIALECT");
    auto metadata = [](const char* name) {
        return std::string("{\"general.name\": \"") + name + "\"}";
    };
    ok(q27::select_tool_dialect_for_model(metadata("Qwen38 27b Hf")),
       "dialect: mangled 3.8 name selects xml");
    const bool clean_xml = q27::select_tool_dialect_for_model(metadata("Qwen3.8-27B"));
    ok(clean_xml, "dialect: clean 3.8 name selects xml");
    json tools = json::array({tool("Read", {{"file_path", true}})});
    const std::string xml_preamble = q27::tools_preamble(tools, clean_xml);
    ok(xml_preamble.find("<function=example_function_name>") != std::string::npos &&
           xml_preamble.find("{\"name\": <function-name>") == std::string::npos,
       "dialect: qwen3.8 preamble uses trained XML format");
    ok(!q27::select_tool_dialect_for_model(metadata("Qwen3.6-27B")),
       "dialect: 3.6 stays json");
    ok(!q27::select_tool_dialect_for_model(metadata("Qwopus3.6 27B v2")),
       "dialect: qwopus fine-tune stays json");
    setenv("Q27_TOOL_DIALECT", "json", 1);
    ok(!q27::select_tool_dialect_for_model(metadata("Qwen3.8-27B")),
       "dialect: env json overrides a 3.8 model");
    setenv("Q27_TOOL_DIALECT", "xml", 1);
    ok(q27::select_tool_dialect_for_model(metadata("Qwen3.6-27B")),
       "dialect: env xml overrides a 3.6 model");
    unsetenv("Q27_TOOL_DIALECT");
}

static void test_native_xml_dialect() {
    const json xml_tools = json::parse(R"([
        {"type":"function","function":{"name":"get_weather","parameters":{
            "type":"object","properties":{"city":{"type":"string"},
            "units":{"type":"string"},"count":{"type":"integer"},
            "notes":{"type":"string"}}}}},
        {"type":"function","function":{"name":"Write","parameters":{
            "type":"object","properties":{"content":{"type":"string"},
            "count":{"type":"integer"},"force":{"type":"boolean"}}}}}
    ])");
    q27::ToolCall first;
    ok(q27::parse_native_xml_call(
           "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
           "<parameter=units>\nmetric\n</parameter>\n</function>\n", first, &xml_tools) &&
           first.ok && first.name == "get_weather" &&
           first.arguments.value("city", std::string()) == "Paris" &&
           first.arguments.value("units", std::string()) == "metric",
       "native-xml: two scalar parameters round-trip");
    q27::ToolCall typed;
    ok(q27::parse_native_xml_call(
           "<function=Write>\n<parameter=content>\n123\n</parameter>\n"
           "<parameter=count>\n3\n</parameter>\n<parameter=force>\ntrue\n</parameter>\n"
           "</function>", typed, &xml_tools) &&
           typed.arguments.value("content", std::string()) == "123" &&
           typed.arguments.value("count", 0) == 3 &&
           typed.arguments.value("force", false),
       "native-xml: schema preserves JSON-looking strings and types declared scalars");
    const json union_tools = json::parse(R"([
        {"type":"function","function":{"name":"typed","parameters":{
            "type":"object","properties":{
                "nullable_count":{"type":["integer","null"]},
                "choice":{"anyOf":[{"type":"integer"},{"type":"boolean"}]},
                "mode":{"enum":["safe","fast"]}}}}}
    ])");
    q27::ToolCall union_typed;
    ok(q27::parse_native_xml_call(
           "<function=typed>\n<parameter=nullable_count>\n3\n</parameter>\n"
           "<parameter=choice>\nfalse\n</parameter>\n"
           "<parameter=mode>\nsafe\n</parameter>\n</function>",
           union_typed, &union_tools) &&
           union_typed.arguments.value("nullable_count", 0) == 3 &&
           !union_typed.arguments.value("choice", true) &&
           union_typed.arguments.value("mode", std::string()) == "safe",
       "native-xml: union, anyOf, and enum schemas preserve declared types");
    q27::ToolCall invalid_typed;
    ok(!q27::parse_native_xml_call(
           "<function=typed>\n<parameter=nullable_count>\nnot-a-number\n"
           "</parameter>\n</function>", invalid_typed, &union_tools),
       "native-xml: schema-invalid typed values are rejected");
    std::string bare_prefix;
    const auto bare = q27::parse_bare_tool_calls(
        "<function=Write>\n<parameter=content>\n123\n</parameter>\n"
        "<parameter=count>\n8\n</parameter>\n<parameter=force>\nfalse\n</parameter>\n"
        "</function>", &bare_prefix, &xml_tools, true, true, true);
    ok(bare.size() == 1 && bare[0].arguments.value("content", std::string()) == "123" &&
           bare[0].arguments.value("count", 0) == 8 &&
           !bare[0].arguments.value("force", true) && bare_prefix.empty(),
       "native-xml: bare recovery preserves schema-declared argument types");
    q27::ToolCall empty;
    ok(q27::parse_native_xml_call("<function=list_files>\n</function>", empty) &&
           empty.ok && empty.arguments.empty(),
       "native-xml: zero-parameter call is legal");
    q27::ToolCall truncated;
    ok(!q27::parse_native_xml_call(
           "<function=Write>\n<parameter=content>\ntruncated with no closer", truncated),
       "native-xml: truncated parameter is refused, not guessed");
    q27::ToolCall missing_function_close;
    ok(!q27::parse_native_xml_call(
           "<function=Write><parameter=content>x</parameter>",
           missing_function_close),
       "native-xml: missing function closer is refused");
    q27::ToolCall trailing;
    ok(!q27::parse_native_xml_call(
           "<function=Write></function> ignored suffix", trailing),
       "native-xml: trailing body text is refused");
    q27::ToolCall prose;
    ok(!q27::parse_native_xml_call("plain text, no dialect", prose),
       "native-xml: non-dialect text is not consumed");
    const json history_args = {
        {"city", "Paris"}, {"count", 3},
        {"notes", "line one\nliteral </parameter> & <x>"}};
    const std::string history = q27::tool_call_text("get_weather", history_args, true);
    const size_t body_begin = history.find('\n') + 1;
    const size_t body_end = history.rfind("\n</tool_call>");
    q27::ToolCall replayed;
    ok(history.find("<function=get_weather>") != std::string::npos &&
           history.find("{\"name\"") == std::string::npos &&
           history.find("&lt;/parameter&gt; &amp; &lt;x&gt;") != std::string::npos &&
           body_begin > 0 && body_end != std::string::npos &&
           q27::parse_native_xml_call(
               history.substr(body_begin, body_end - body_begin), replayed, &xml_tools) &&
           replayed.arguments == history_args,
       "native-xml: escaped assistant tool history round-trips losslessly");
    ok(q27::tool_call_text("get_weather", history_args, false).find(
           "<tool_call>\n{\"name\": \"get_weather\"") == 0,
       "native-json: assistant tool history remains legacy JSON");
    const json unusual_tools = json::parse(R"([{"type":"function","function":{
        "name":"f>x","parameters":{"type":"object","properties":{"a>b":{"type":"string"}}}}}])");
    const json unusual_args = {{"a>b", "literal <tag> & bytes"}};
    const std::string unusual_history = q27::tool_call_text("f>x", unusual_args, true);
    q27::ToolCall unusual_call;
    const size_t unusual_begin = unusual_history.find('\n') + 1;
    const size_t unusual_end = unusual_history.rfind("\n</tool_call>");
    q27::ToolGrammar grammar;
    grammar.reset({"f>x"}, true);
    ok(unusual_history.find("<function=f&gt;x>") != std::string::npos &&
           unusual_history.find("<parameter=a&gt;b>") != std::string::npos &&
           q27::parse_native_xml_call(
               unusual_history.substr(unusual_begin, unusual_end - unusual_begin),
               unusual_call, &unusual_tools) && unusual_call.name == "f>x" &&
           unusual_call.arguments == unusual_args &&
           grammar.advance_str("<function=f&gt;x>\n</function>\n</tool_call>") &&
           grammar.closed(),
       "native-xml: function and parameter tag names escape losslessly");
}

static void test_strict_native_xml_dialect() {
    const json tools = json::parse(R"([{"type":"function","function":{"name":"count",
        "parameters":{"type":"object","properties":{"value":{"type":"integer"}}}}}])");
    const q27::ToolCall call = q27::parse_tool_call(
        "<function=count>\n<parameter=value>\n7\n</parameter>\n</function>",
        &tools, true);
    ok(call.ok && call.name == "count" && call.arguments.value("value", 0) == 7,
       "native-xml: strict mode accepts the selected trained dialect");
}

static void test_mode14_tool_name_xml_dialect() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}},{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command"]}}}])");
    // Verbatim bytes from the captured transcript.
    std::string t =
        "<tool_name>Read</tool_name>\n"
        "<parameter=file_path>/workspace/tests/index.test.ts</parameter>\n"
        "<tool_name>Bash</tool_name>\n"
        "<parameter=arguments>\n"
        "{\"command\":\"ls /workspace/src /workspace/tests && cat /workspace/.eslintrc.cjs "
        "/workspace/vitest.config.ts\",\"description\":\"List src/tests and show eslint and "
        "vitest configs\"}}";
    const bool expect_rescued = true;    // rescued as of the mode-14 parser path
    std::string pre;
    auto v = q27::parse_bare_tool_calls(t, &pre, &tools);
    ok((!v.empty()) == expect_rescued,
       expect_rescued ? "mode14: <tool_name>/<parameter=> dialect rescued"
                      : "mode14: <tool_name>/<parameter=> dialect UN-RESCUED (documented)");
    // Both calls must come back, with the scalar parameter AND the JSON-object
    // parameter each mapped correctly, and the trailing brace tolerated.
    ok(v.size() == 2, "mode14: both concatenated calls recovered");
    if (v.size() == 2) {
        ok(v[0].ok && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) ==
                   "/workspace/tests/index.test.ts",
           "mode14: scalar <parameter=file_path> mapped");
        ok(v[1].ok && v[1].name == "Bash" &&
               v[1].arguments.value("command", std::string()).rfind("ls /workspace/src", 0) == 0 &&
               v[1].arguments.value("description", std::string()) ==
                   "List src/tests and show eslint and vitest configs",
           "mode14: <parameter=arguments> JSON object merged, trailing brace tolerated");
    }
}

static void test_mode13_truncated_mid_escape() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Write","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}}])");
    // trailing dangling backslash
    std::string t1 = "Let me write that.\n{\"name\": \"Write\", \"arguments\": {\"file_path\": \"g.md\", "
                     "\"content\": \"# Guide\\n\\n```json\\n{\\n  \\\"role\\\": \\\"assistant\\\",\\";
    std::string pre;
    auto v1 = q27::parse_bare_tool_calls(t1, &pre, &tools);
    ok(v1.size() == 1 && v1[0].ok && v1[0].name == "Write",
       "mode13: truncated at a dangling backslash");
    // partial \uXXXX at the cut
    std::string t2 = "{\"name\": \"Write\", \"arguments\": {\"file_path\": \"g.md\", \"content\": \"caf\\u00";
    auto v2 = q27::parse_bare_tool_calls(t2, &pre, &tools);
    ok(v2.size() == 1 && v2[0].ok && v2[0].name == "Write",
       "mode13: truncated inside a partial \\uXXXX");
    // a COMPLETE escape at the cut must still round-trip (no over-trim)
    std::string t3 = "{\"name\": \"Write\", \"arguments\": {\"file_path\": \"g.md\", \"content\": \"caf\\u00e9";
    auto v3 = q27::parse_bare_tool_calls(t3, &pre, &tools);
    ok(v3.size() == 1 && v3[0].ok &&
           v3[0].arguments.value("content", std::string()) == "caf\u00e9",
       "mode13: a COMPLETE escape at the cut is not over-trimmed");
}

int main() {
    if (std::getenv("Q27_TEST_STRICT_XML")) {
        test_strict_native_xml_dialect();
        return failures ? 1 : 0;
    }
    test_mode13_truncated_mid_escape();
    test_mode14_tool_name_xml_dialect();
    test_mode15_name_tag_bare_args();
    test_mode16_and_15_variants();
    test_think_mode_drift();
    test_dialect_default_keying();
    test_native_xml_dialect();
    json tools = json::array();
    tools.push_back(tool("Write", {{"content", true}, {"file_path", true}}));
    tools.push_back(tool("Read", {{"file_path", true}}));

    auto call = [&](const std::string& txt) {
        std::string pre;
        return q27::parse_bare_tool_calls(txt, &pre, &tools);
    };

    // mode 10: dropped `{"name": "` opener
    {
        auto v = call("prose.\n\nRead\", \"file_path\": \"/x/y.py\"}");
        ok(v.size() == 1 && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) == "/x/y.py",
           "mode10 dropped-opener");
    }
    // mode 11: raw code, unescaped inner quotes, content last
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"content\": \"package main\n"
                      "import \"fmt\"\nfunc main(){ fmt.Println(\"hi\") }\n\"}}");
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("content", std::string()).find("fmt.Println") !=
                   std::string::npos,
           "mode11 raw-content-last");
    }
    // mode 11: inner []string{"a","b"} + a scalar AFTER content
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"content\": \"a := []string{\"x\", "
                      "\"y\"}\nfmt.Println(a)\n\", \"file_path\": \"m.go\"}}");
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("file_path", std::string()) == "m.go" &&
               v[0].arguments.value("content", std::string()).find("[]string") !=
                   std::string::npos,
           "mode11 inner-braces + scalar-after");
    }
    // mode 11: scalar BEFORE content
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"file_path\": \"m.go\", \"content\": "
                      "\"func f(){ s := \"hi\" }\n\"}}");
        ok(v.size() == 1 && v[0].arguments.value("file_path", std::string()) == "m.go" &&
               !v[0].arguments.value("content", std::string()).empty(),
           "mode11 scalar-before-content");
    }
    // mode 11 refinement (issue #4, 2026-07-20, @chaudhryfaisal): content is
    // MOSTLY-escaped JSON (\n \" all escaped) with ONE sparse escape error
    // (\"x" -- bare closing quote) AND a trailing </tool_call>. json().dump()
    // would double-escape the already-escaped body, and the trailing tag broke
    // the reconstruction's parse -> UN-RESCUED. minimal-escape + first-balanced-
    // object recover it with the CORRECT content ("fmt" quotes preserved).
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"content\":\"package main\\n"
                      "import \\\"fmt\\\"\\nvar s = \\\"x\"\\n\",\"file_path\":\"m.go\"}}\n"
                      "</tool_call>");
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("file_path", std::string()) == "m.go" &&
               v[0].arguments.value("content", std::string()).find("\"fmt\"") != std::string::npos,
           "mode11 mostly-escaped content + trailing tag");
    }
    // multi-call: a normal call followed by a mode-11 raw-value Write must
    // keep BOTH calls. The mode-11 preference replaces only the trailing
    // m10-suspect candidate; a wholesale vector swap would silently drop the
    // earlier normal call (codex P2, 2026-07-20).
    {
        auto v = call("{\"name\": \"Read\", \"arguments\": {\"file_path\": \"/a.py\"}}\n"
                      "{\"name\": \"Write\", \"arguments\": {\"content\": \"a := []string{\"x\", "
                      "\"y\"}\n\", \"file_path\": \"m.go\"}}");
        bool rd = false, wr = false;
        for (auto& c : v) {
            if (c.name == "Read" && c.arguments.value("file_path", std::string()) == "/a.py") rd = true;
            if (c.name == "Write" && c.arguments.value("file_path", std::string()) == "m.go" &&
                c.arguments.value("content", std::string()).find("[]string") != std::string::npos) wr = true;
        }
        ok(rd && wr, "multi-call: normal + mode-11 keeps both");
    }
    // mode 12: unquoted tool-name value {"name": Read, "arguments": {...}}
    // (club-3090 cli-40: the model emitted {"name": bash, ...} and the whole
    // call went UN-RESCUED -> agent turn stopped at turnsUsed=0).
    {
        auto v = call("{\"name\": Read, \"arguments\": {\"file_path\": \"/x/y.py\"}}");
        ok(v.size() == 1 && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) == "/x/y.py",
           "mode12 unquoted-name");
    }
    // mode 12b: dropped OPENING quote of the name value -> {"name": Read", ...}
    // (thunderdome 2026-07-20: model emitted {"name": read", ...} -- bareword +
    // stray closing quote; naive quoting would make "Read"" (invalid)).
    {
        auto v = call("{\"name\": Read\", \"arguments\": {\"file_path\": \"/x/y.py\"}}");
        ok(v.size() == 1 && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) == "/x/y.py",
           "mode12b dropped-opening-quote");
    }
    // mode 12 negative: an unquoted name that is NOT a registered tool must be
    // left untouched (never quote arbitrary barewords).
    {
        auto v = call("{\"name\": notatool, \"arguments\": {\"file_path\": \"/x\"}}");
        ok(v.empty(), "mode12 unknown unquoted-name rejected");
    }
    // negative: well-formed call recovers via the normal path (not a drift mode)
    {
        auto v =
            call("{\"name\": \"Write\", \"arguments\": {\"file_path\": \"a.txt\", \"content\": "
                 "\"hello\"}}");
        ok(v.size() == 1 && v[0].arguments.value("content", std::string()) == "hello",
           "wellformed via normal path");
    }
    // negative: prose JSON with an unregistered "name" must NOT recover
    {
        auto v = call("config: {\"name\": \"my-app\", \"version\": \"1.0\"} shipped.");
        ok(v.empty(), "prose-unknown-name rejected");
    }
    // fence-skip (thunderdome 2026-07-20): a COMPLETE, well-formed call inside a
    // ```fenced``` block is a displayed example / echoed injection, NOT a call
    // the model is making -> must not recover (prose-to-execution guard).
    {
        auto v = call("Here is how it works:\n```json\n{\"name\": \"Read\", \"arguments\": "
                      "{\"file_path\": \"/etc/passwd\"}}\n```\nThat is the format.");
        ok(v.empty(), "fence-skip: fenced example not recovered");
    }
    // but a write whose CONTENT contains fences still recovers (its ``` are
    // after the call's opener, so the guard -- which looks only before -- ignores them)
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"content\": \"# doc\\n```go\\nx := 1\\n"
                      "```\\n\", \"file_path\": \"/x.md\"}}");
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("file_path", std::string()) == "/x.md",
           "fence-skip: write w/ fenced content still recovers");
    }

    printf(failures ? "\nDRIFT TESTS: %d FAIL\n" : "\nDRIFT TESTS: all pass\n", failures);
    return failures ? 1 : 0;
}
