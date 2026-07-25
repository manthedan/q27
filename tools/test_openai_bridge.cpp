// CPU unit tests for the new /v1/chat/completions bridges in api_common.h:
// openai_tools_json, openai_msgs, parse_tool_choice. Pure header logic, no
// CUDA/engine dependency -- same rationale as test_toolconstrain.cpp.
//
// Build+run: g++ -std=c++17 -I src tools/test_openai_bridge.cpp -o build/test_openai_bridge && ./build/test_openai_bridge
#include "api_common.h"

#include <cassert>
#include <cstdio>

using json = nlohmann::json;
using q27::Msg;

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static void test_tools_passthrough() {
    json body = {{"tools", json::array({
        {{"type", "function"}, {"function", {{"name", "get_weather"},
            {"description", "get weather"},
            {"parameters", {{"type","object"},{"properties", {{"location", {{"type","string"}}}}}}}}}},
        // malformed entries must be dropped, not throw / take down the request
        {{"type", "web_search"}},                      // hosted type, not "function"
        {{"type", "function"}},                         // missing "function" key
        {{"type", "function"}, {"function", json::object()}}, // missing name
    })}};
    json tools = q27::openai_tools_json(body);
    CHECK(tools.is_array());
    CHECK(tools.size() == 1);
    CHECK(tools[0]["function"]["name"] == "get_weather");
    CHECK(tools[0]["type"] == "function");
}

static void test_tools_absent() {
    json body = json::object();
    json tools = q27::openai_tools_json(body);
    CHECK(tools.is_array());
    CHECK(tools.empty());
}

static void test_msgs_plain_roundtrip() {
    json body = {{"messages", json::array({
        {{"role","system"},{"content","be terse"}},
        {{"role","user"},{"content","hi"}},
    })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 2);
    CHECK(msgs[0].role == "system" && msgs[0].content == "be terse");
    CHECK(msgs[1].role == "user" && msgs[1].content == "hi");
}

static void test_msgs_content_parts_array() {
    json body = {{"messages", json::array({
        {{"role","user"},{"content", json::array({
            {{"type","text"},{"text","part one "}},
            {{"type","image_url"},{"image_url", {{"url","http://x"}}}}, // non-text part ignored, not crash
            {{"type","text"},{"text","part two"}},
        })}},
    })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 1);
    CHECK(msgs[0].content == "part one part two");
}

static void test_msgs_developer_role_maps_to_system() {
    json body = {{"messages", json::array({
        {{"role","developer"},{"content","dev system prompt"}},
    })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 1);
    CHECK(msgs[0].role == "system");
}

static void test_msgs_assistant_tool_calls_reconstructed() {
    // OpenAI wire shape: function.arguments is a JSON-ENCODED STRING.
    json body = {{"messages", json::array({
        {{"role","user"},{"content","what's the weather in Tokyo?"}},
        {{"role","assistant"},
         {"content", nullptr},
         {"tool_calls", json::array({
             {{"id","call_1"},{"type","function"},
              {"function", {{"name","get_weather"},{"arguments","{\"location\":\"Tokyo\"}"}}}}
         })}},
        {{"role","tool"},{"tool_call_id","call_1"},{"content","72F and sunny"}},
        {{"role","user"},{"content","thanks, and tomorrow?"}},
    })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 4);
    CHECK(msgs[0].role == "user");
    CHECK(msgs[1].role == "assistant");
    CHECK(msgs[1].content.find("<tool_call>") != std::string::npos);
    CHECK(msgs[1].content.find("\"name\": \"get_weather\"") != std::string::npos);
    CHECK(msgs[1].content.find("\"location\":\"Tokyo\"") != std::string::npos);
    // role:"tool" folds into a USER turn wrapped in <tool_response>
    CHECK(msgs[2].role == "user");
    CHECK(msgs[2].content.find("<tool_response>") != std::string::npos);
    CHECK(msgs[2].content.find("72F and sunny") != std::string::npos);
    CHECK(msgs[3].role == "user");
    CHECK(msgs[3].content == "thanks, and tomorrow?");
}

static void test_msgs_assistant_content_plus_tool_calls() {
    // real-world shape: assistant text AND a tool call in the same turn
    json body = {{"messages", json::array({
        {{"role","assistant"}, {"content","Let me check that."},
         {"tool_calls", json::array({
             {{"id","call_9"},{"type","function"},
              {"function", {{"name","search"},{"arguments","{\"q\":\"x\"}"}}}}
         })}},
    })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 1);
    CHECK(msgs[0].content.rfind("Let me check that.", 0) == 0);
    CHECK(msgs[0].content.find("<tool_call>") != std::string::npos);
}

static void test_msgs_malformed_arguments_string_kept_not_dropped() {
    json body = {{"messages", json::array({
        {{"role","assistant"}, {"content", nullptr},
         {"tool_calls", json::array({
             {{"id","call_x"},{"type","function"},
              {"function", {{"name","broken"},{"arguments","not-json{{"}}}}
         })}},
    })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 1);
    // must not throw, and the call must still be present in some form
    CHECK(msgs[0].content.find("<tool_call>") != std::string::npos);
    CHECK(msgs[0].content.find("broken") != std::string::npos);
}

static void test_msgs_no_messages_key() {
    json body = json::object();
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.empty());
}

static void test_msgs_content_less_message_no_crash() {
    // a message with no "content" key at all must not abort (json.hpp assertion)
    json body = {{"messages", json::array({ {{"role","user"}} })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 1);
    CHECK(msgs[0].content.empty());
}

// ---- tolerant request-field readers (jnum/jint/jbool/jstr) ----------------
// json::value() throws type_error.302 on a PRESENT-but-null key, and httplib
// turns a handler throw into a 500 -- but {"max_tokens": null} is how a large
// share of OpenAI-compatible clients spell "unset". null / wrong-typed must
// read exactly like absent.
static void test_jread_null_reads_as_absent() {
    json body = json::parse(R"({"max_tokens":null,"stream":null,"temperature":null,
                                "top_p":null,"prompt":null})");
    CHECK(q27::jint(body, "max_tokens", 8192) == 8192);
    CHECK(!q27::jbool(body, "stream", false));
    CHECK(q27::jbool(body, "stream", true));
    CHECK(q27::jnum(body, "temperature", 0.0) == 0.0);
    CHECK(q27::jnum(body, "top_p", 1.0) == 1.0);
    CHECK(q27::jstr(body, "prompt", "dflt") == "dflt");
}

static void test_jread_wrong_type_reads_as_absent() {
    json body = json::parse(R"({"max_tokens":"512","stream":"yes","temperature":{},"prompt":7})");
    CHECK(q27::jint(body, "max_tokens", 8192) == 8192);
    CHECK(!q27::jbool(body, "stream", false));
    CHECK(q27::jnum(body, "temperature", 0.0) == 0.0);
    CHECK(q27::jstr(body, "prompt", "dflt") == "dflt");
}

static void test_jread_present_values_win() {
    json body = json::parse(R"({"max_tokens":512,"stream":true,"temperature":0.7,
                                "top_p":0.95,"prompt":"hi"})");
    CHECK(q27::jint(body, "max_tokens", 8192) == 512);
    CHECK(q27::jbool(body, "stream", false));
    CHECK(q27::jnum(body, "temperature", 0.0) == 0.7);
    CHECK(q27::jnum(body, "top_p", 1.0) == 0.95);
    CHECK(q27::jstr(body, "prompt", "dflt") == "hi");
}

static void test_jint_float_and_absurd_magnitude() {
    // an integer field sent as a float still works; a nonsense magnitude
    // clamps instead of overflowing the int the callers assign into
    json body = json::parse(R"({"a":8192.0,"big":1e30,"neg":-1e30})");
    CHECK(q27::jint(body, "a", 0) == 8192);
    CHECK(q27::jint(body, "big", 0) == 2147483647L);
    CHECK(q27::jint(body, "neg", 0) == -2147483648L);
    CHECK(q27::jint(json::array(), "a", 5) == 5); // non-object body
}

// ---- malformed-shape robustness (must not throw -> no spurious 500) -------
static void test_anthropic_msgs_non_object_message_skipped() {
    json body = json::parse(R"({"messages":["hi",3,{"role":"user","content":"real"}]})");
    auto msgs = q27::anthropic_msgs(body);
    CHECK(msgs.size() == 1);
    if (msgs.size() == 1) CHECK(msgs[0].content == "real");
}

static void test_anthropic_msgs_bare_string_content_part_skipped() {
    json body = json::parse(
        R"({"messages":[{"role":"user","content":["bare",{"type":"text","text":"kept"}]}]})");
    auto msgs = q27::anthropic_msgs(body);
    CHECK(msgs.size() == 1);
    if (msgs.size() == 1) CHECK(msgs[0].content == "kept");
}

static void test_anthropic_msgs_system_array_of_strings() {
    json body = json::parse(R"({"system":["bare",{"type":"text","text":"sys"}],"messages":[]})");
    auto msgs = q27::anthropic_msgs(body);
    CHECK(msgs.size() == 1);
    if (msgs.size() == 1) {
        CHECK(msgs[0].role == "system");
        CHECK(msgs[0].content == "sys");
    }
}

static void test_anthropic_msgs_tool_result_bare_string_content() {
    json body = json::parse(R"({"messages":[{"role":"user","content":[
        {"type":"tool_result","content":["bare",{"type":"text","text":"out"}]}]}]})");
    auto msgs = q27::anthropic_msgs(body);
    CHECK(msgs.size() == 1);
    if (msgs.size() == 1) CHECK(msgs[0].content.find("out") != std::string::npos);
}

static void test_anthropic_tools_non_array_and_non_object_entries() {
    json body = json::parse(R"({"tools":["str",7,{"name":"Read","input_schema":{}}]})");
    json out = q27::anthropic_tools_json(body);
    CHECK(out.size() == 1);
    json body2 = json::parse(R"({"tools":{"not":"an array"}})");
    CHECK(q27::anthropic_tools_json(body2).empty());
}

// Claude Code's billing header carries a stamp that changes between
// conversations; if it is not pinned, the first ~15 tokens of every system
// prompt differ and NO prefix tier (P8 snapshot, P9 ring, P16 disk) can ever
// share state across sessions. CC 2.1.220 moved that stamp: the `cch=` field is
// gone and the volatile part rides a 4th component on cc_version. Captured live
// 2026-07-24 -- these two strings are verbatim from two real sessions.
static void test_billing_header_2_1_220_shape() {
    std::string a = "x-anthropic-billing-header: cc_version=2.1.220.473; cc_entrypoint=sdk-cli;You are";
    std::string b = "x-anthropic-billing-header: cc_version=2.1.220.c50; cc_entrypoint=sdk-cli;You are";
    q27::normalize_cc_billing_header(a);
    q27::normalize_cc_billing_header(b);
    CHECK(a == b);
    CHECK(a.find("cc_version=2.1.220.fff") != std::string::npos);
}
static void test_billing_header_legacy_cch_still_pinned() {
    std::string a = "x-anthropic-billing-header: cc_version=2.1.1; cc_entrypoint=cli; cch=a5145;X";
    std::string b = "x-anthropic-billing-header: cc_version=2.1.1; cc_entrypoint=cli; cch=b9e02;X";
    q27::normalize_cc_billing_header(a);
    q27::normalize_cc_billing_header(b);
    CHECK(a == b);
}
static void test_billing_header_leaves_other_prompts_alone() {
    std::string a = "You are a helpful assistant.", a0 = a;
    q27::normalize_cc_billing_header(a);
    CHECK(a == a0);
    std::string b = "x-anthropic-billing-header: cc_version=2.1.220; cc_entrypoint=sdk-cli;X", b0 = b;
    q27::normalize_cc_billing_header(b);
    CHECK(b == b0);  // no 4th component -> nothing to pin
}

static void test_tool_choice_absent_is_auto() {
    json body = json::object();
    auto tc = q27::parse_tool_choice(body);
    CHECK(tc.mode == q27::ToolChoice::AUTO);
}

static void test_tool_choice_none() {
    json body = {{"tool_choice", "none"}};
    auto tc = q27::parse_tool_choice(body);
    CHECK(tc.mode == q27::ToolChoice::NONE);
}

static void test_tool_choice_required() {
    json body = {{"tool_choice", "required"}};
    auto tc = q27::parse_tool_choice(body);
    CHECK(tc.mode == q27::ToolChoice::FORCED);
    CHECK(tc.forced_name.empty());
}

static void test_tool_choice_named_function() {
    json body = {{"tool_choice", {{"type","function"},{"function",{{"name","get_weather"}}}}}};
    auto tc = q27::parse_tool_choice(body);
    CHECK(tc.mode == q27::ToolChoice::FORCED);
    CHECK(tc.forced_name == "get_weather");
}

static void test_tool_choice_unknown_string_is_auto() {
    json body = {{"tool_choice", "auto"}};
    auto tc = q27::parse_tool_choice(body);
    CHECK(tc.mode == q27::ToolChoice::AUTO);
}

// End-to-end sanity: openai_msgs + openai_tools_json feed correctly into the
// SAME chatml_prompt() the /v1/messages path already uses in production.
static void test_end_to_end_chatml_prompt() {
    json body = {
        {"tools", json::array({
            {{"type","function"},{"function",{{"name","get_weather"},{"description","w"},
                {"parameters", {{"type","object"},{"properties",{{"location",{{"type","string"}}}}}}}}}}
        })},
        {"messages", json::array({
            {{"role","system"},{"content","be terse"}},
            {{"role","user"},{"content","weather in tokyo?"}},
        })},
    };
    json tools = q27::openai_tools_json(body);
    auto msgs = q27::openai_msgs(body);
    size_t stable_off = 0;
    std::string rendered = q27::chatml_prompt(msgs, tools, /*think=*/false, &stable_off);
    CHECK(rendered.find("# Tools") != std::string::npos);
    CHECK(rendered.find("get_weather") != std::string::npos);
    CHECK(rendered.find("be terse") != std::string::npos);
    CHECK(rendered.find("weather in tokyo?") != std::string::npos);
    CHECK(rendered.rfind("<|im_start|>assistant\n<think>\n\n</think>\n\n") ==
          rendered.size() - std::string("<|im_start|>assistant\n<think>\n\n</think>\n\n").size());
    CHECK(stable_off > 0 && stable_off < rendered.size());
    // FORCED-mode prompt injection (server.cu appends this after chatml_prompt
    // returns): must land in the volatile tail, past stable_off.
    std::string forced = rendered + "<tool_call>\n";
    CHECK(forced.substr(0, stable_off) == rendered.substr(0, stable_off));
}

static void test_chat_message_plain_text_no_calls() {
    json msg = q27::openai_chat_message_json("hello there", {}, 42);
    CHECK(msg["role"] == "assistant");
    CHECK(msg["content"] == "hello there");
    CHECK(!msg.contains("tool_calls"));
}

static void test_chat_message_empty_text_no_calls_is_empty_string_not_null() {
    json msg = q27::openai_chat_message_json("", {}, 42);
    CHECK(msg["content"].is_string());
    CHECK(msg["content"] == "");
    CHECK(!msg.contains("tool_calls"));
}

static void test_chat_message_call_no_leftover_text_content_null() {
    q27::ToolCall c; c.ok = true; c.name = "get_weather"; c.arguments = {{"location","Tokyo"}};
    json msg = q27::openai_chat_message_json("", {c}, 7);
    CHECK(msg["content"].is_null());
    CHECK(msg["tool_calls"].size() == 1);
    CHECK(msg["tool_calls"][0]["id"] == "call_q27_7_0");
    CHECK(msg["tool_calls"][0]["type"] == "function");
    CHECK(msg["tool_calls"][0]["function"]["name"] == "get_weather");
    json args = json::parse(msg["tool_calls"][0]["function"]["arguments"].get<std::string>());
    CHECK(args["location"] == "Tokyo");
}

static void test_chat_message_call_plus_leftover_text() {
    q27::ToolCall c; c.ok = true; c.name = "search"; c.arguments = json::object();
    json msg = q27::openai_chat_message_json("Let me check that.", {c}, 1);
    CHECK(msg["content"] == "Let me check that.");
    CHECK(msg["tool_calls"].size() == 1);
}

static void test_chat_message_parallel_calls_indexed_and_ordered() {
    q27::ToolCall a; a.ok = true; a.name = "one"; a.arguments = json::object();
    q27::ToolCall bad; bad.ok = false; bad.raw = "garbage"; // must be skipped, not crash
    q27::ToolCall b; b.ok = true; b.name = "two"; b.arguments = json::object();
    json msg = q27::openai_chat_message_json("", {a, bad, b}, 3);
    CHECK(msg["tool_calls"].size() == 2);
    CHECK(msg["tool_calls"][0]["id"] == "call_q27_3_0");
    CHECK(msg["tool_calls"][0]["function"]["name"] == "one");
    CHECK(msg["tool_calls"][1]["id"] == "call_q27_3_1");
    CHECK(msg["tool_calls"][1]["function"]["name"] == "two");
}

static void test_chat_message_reasoning_content_included_when_present() {
    json msg = q27::openai_chat_message_json("the answer", {}, 1, "thinking it through");
    CHECK(msg["content"] == "the answer");
    CHECK(msg["reasoning_content"] == "thinking it through");
}

static void test_chat_message_reasoning_content_absent_when_empty() {
    json msg = q27::openai_chat_message_json("the answer", {}, 1, "");
    CHECK(!msg.contains("reasoning_content"));
}

static void test_chat_message_reasoning_content_with_tool_call() {
    q27::ToolCall c; c.ok = true; c.name = "get_weather"; c.arguments = json::object();
    json msg = q27::openai_chat_message_json("", {c}, 5, "deciding to check weather");
    CHECK(msg["content"].is_null());
    CHECK(msg["reasoning_content"] == "deciding to check weather");
    CHECK(msg["tool_calls"].size() == 1);
}

static void test_reasoning_delta_shape() {
    json d = q27::openai_reasoning_delta("partial thought");
    CHECK(d["reasoning_content"] == "partial thought");
}

static void test_stream_chunk_shape() {
    json j = q27::openai_stream_chunk("chatcmpl-1", "chat.completion.chunk", 123, "q27model",
                                      json{{"content", "hi"}});
    CHECK(j["id"] == "chatcmpl-1");
    CHECK(j["choices"][0]["delta"]["content"] == "hi");
    CHECK(j["choices"][0]["finish_reason"].is_null());
}

static void test_stream_chunk_finish_reason() {
    json j = q27::openai_stream_chunk("id", "obj", 0, "m", json::object(), "tool_calls");
    CHECK(j["choices"][0]["finish_reason"] == "tool_calls");
    CHECK(j["choices"][0]["delta"].empty());
}

static void test_tool_call_delta_shape() {
    q27::ToolCall c; c.ok = true; c.name = "get_weather"; c.arguments = {{"location","Paris"}};
    json d = q27::openai_tool_call_delta(0, "call_abc", c);
    CHECK(d["tool_calls"].size() == 1);
    CHECK(d["tool_calls"][0]["index"] == 0);
    CHECK(d["tool_calls"][0]["id"] == "call_abc");
    CHECK(d["tool_calls"][0]["function"]["name"] == "get_weather");
    json args = json::parse(d["tool_calls"][0]["function"]["arguments"].get<std::string>());
    CHECK(args["location"] == "Paris");
}

int main() {
    test_tools_passthrough();
    test_tools_absent();
    test_msgs_plain_roundtrip();
    test_msgs_content_parts_array();
    test_msgs_developer_role_maps_to_system();
    test_msgs_assistant_tool_calls_reconstructed();
    test_msgs_assistant_content_plus_tool_calls();
    test_msgs_malformed_arguments_string_kept_not_dropped();
    test_msgs_no_messages_key();
    test_msgs_content_less_message_no_crash();
    test_billing_header_2_1_220_shape();
    test_billing_header_legacy_cch_still_pinned();
    test_billing_header_leaves_other_prompts_alone();
    test_jread_null_reads_as_absent();
    test_jread_wrong_type_reads_as_absent();
    test_jread_present_values_win();
    test_jint_float_and_absurd_magnitude();
    test_anthropic_msgs_non_object_message_skipped();
    test_anthropic_msgs_bare_string_content_part_skipped();
    test_anthropic_msgs_system_array_of_strings();
    test_anthropic_msgs_tool_result_bare_string_content();
    test_anthropic_tools_non_array_and_non_object_entries();
    test_tool_choice_absent_is_auto();
    test_tool_choice_none();
    test_tool_choice_required();
    test_tool_choice_named_function();
    test_tool_choice_unknown_string_is_auto();
    test_end_to_end_chatml_prompt();
    test_chat_message_plain_text_no_calls();
    test_chat_message_empty_text_no_calls_is_empty_string_not_null();
    test_chat_message_call_no_leftover_text_content_null();
    test_chat_message_call_plus_leftover_text();
    test_chat_message_parallel_calls_indexed_and_ordered();
    test_chat_message_reasoning_content_included_when_present();
    test_chat_message_reasoning_content_absent_when_empty();
    test_chat_message_reasoning_content_with_tool_call();
    test_reasoning_delta_shape();
    test_stream_chunk_shape();
    test_stream_chunk_finish_reason();
    test_tool_call_delta_shape();
    if (failures) { fprintf(stderr, "%d FAILURE(S)\n", failures); return 1; }
    fprintf(stderr, "all tests passed\n");
    return 0;
}
