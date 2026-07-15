// Model-free unit tests for the Metal server's streaming helpers:
// UTF-8 boundary gating, stop-sequence holdback, and SSE event framing.
// These run without the model artifact (part of `make test-metal`) and are
// the regression net for the wire shapes the CUDA reference server defines.
#include "stream_format.h"

#include <cstdio>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); failures++; }
}

// ---- Utf8Gate ----
int test_utf8_gate() {
    // A three-byte em dash (E2 80 94) split across two token pieces: the first
    // feed must hold the incomplete lead+continuation back, the second complete
    // it. Emitting the partial would make json::dump throw.
    q27::Utf8Gate g;
    check(g.feed("ab") == "ab", "utf8: ascii passes through");
    check(g.feed("\xE2\x80") == "", "utf8: incomplete multibyte held back");
    check(g.feed("\x94") == "\xE2\x80\x94", "utf8: multibyte completes on continuation");
    check(g.flush() == "", "utf8: clean flush is empty");

    // A dangling partial at end of stream becomes U+FFFD.
    q27::Utf8Gate g2;
    check(g2.feed("x\xE2\x80") == "x", "utf8: emits valid prefix, holds tail");
    check(g2.flush() == "\xEF\xBF\xBD", "utf8: dangling tail flushes to U+FFFD");
    return 0;
}

// ---- StopBuffer ----
int test_stop_buffer() {
    // No stop sequences: identity passthrough, nothing held.
    {
        q27::StopBuffer sb;
        bool stopped = false;
        check(!sb.active(), "stop: empty is inactive");
        check(sb.feed("hello world", stopped) == "hello world", "stop: passthrough");
        check(!stopped, "stop: passthrough not stopped");
        check(sb.flush() == "", "stop: passthrough flush empty");
    }
    // Empty stop strings are dropped (would otherwise match everywhere).
    {
        q27::StopBuffer sb({"", ""});
        check(!sb.active(), "stop: empty strings dropped");
    }
    // Full match within one feed truncates at the stop and reports the index.
    {
        q27::StopBuffer sb({"STOP"});
        bool stopped = false;
        check(sb.feed("abcSTOPdef", stopped) == "abc", "stop: truncates before match");
        check(stopped && sb.matched == 0, "stop: reports stopped + index");
    }
    // Stop sequence split across two feeds: the first half is held back, not
    // leaked, and the match fires when the second half arrives.
    {
        q27::StopBuffer sb({"</s>"});
        bool stopped = false;
        check(sb.feed("hello</", stopped) == "hello", "stop: holds back partial prefix");
        check(!stopped, "stop: partial not yet stopped");
        check(sb.feed("s>world", stopped) == "", "stop: completes across feeds");
        check(stopped && sb.matched == 0, "stop: split match stops");
    }
    // A partial prefix that diverges is released, not swallowed.
    {
        q27::StopBuffer sb({"</s>"});
        bool stopped = false;
        check(sb.feed("a</b", stopped) == "a</b", "stop: diverging partial released");
        check(!stopped, "stop: diverging not stopped");
    }
    // Earliest match across multiple sequences wins.
    {
        q27::StopBuffer sb({"XX", "Y"});
        bool stopped = false;
        check(sb.feed("aYbXX", stopped) == "a", "stop: earliest position wins");
        check(stopped && sb.matched == 1, "stop: matched index is the Y sequence");
    }
    // Held-back partial with no completion is real output at flush.
    {
        q27::StopBuffer sb({"</s>"});
        bool stopped = false;
        check(sb.feed("text</s", stopped) == "text", "stop: holds trailing partial");
        check(!stopped, "stop: trailing partial not stopped");
        check(sb.flush() == "</s", "stop: flush releases held partial");
    }
    return 0;
}

// ---- SSE framing ----
int test_sse_framing() {
    check(q27::sse_data(json{{"a", 1}}) == "data: {\"a\":1}\n\n", "sse: data frame");
    check(q27::sse_done() == "data: [DONE]\n\n", "sse: done terminator");

    std::string ev = q27::sse_event("message_stop", json{{"type", "message_stop"}});
    check(ev == "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n", "sse: event frame");

    // OpenAI chat delta chunk shape: choices[0].delta.content, null finish.
    json chat = q27::openai_stream_chunk(true, "chatcmpl-metal", "chat.completion.chunk",
                                         1700000000, "q27-metal", "hi");
    check(chat["object"] == "chat.completion.chunk", "sse: chat chunk object");
    check(chat["choices"][0]["delta"]["content"] == "hi", "sse: chat delta content");
    check(chat["choices"][0]["finish_reason"].is_null(), "sse: chat chunk null finish");

    // Completions chunk shape: choices[0].text.
    json txt = q27::openai_stream_chunk(false, "cmpl-metal", "text_completion",
                                        1700000000, "q27-metal", "yo");
    check(txt["object"] == "text_completion", "sse: text chunk object");
    check(txt["choices"][0]["text"] == "yo", "sse: text chunk text");
    check(txt["choices"][0]["finish_reason"].is_null(), "sse: text chunk null finish");

    // Terminal chunk: real finish_reason, empty delta object (chat) / empty
    // text (completions) — server.cu's shape after security-review fix #7.
    json fchat = q27::openai_stream_final_chunk(true, "chatcmpl-metal", "chat.completion.chunk",
                                                1700000000, "q27-metal", "stop");
    check(fchat["choices"][0]["finish_reason"] == "stop", "sse: final chat finish_reason");
    check(fchat["choices"][0]["delta"].is_object() && fchat["choices"][0]["delta"].empty(),
          "sse: final chat empty delta object");
    json ftxt = q27::openai_stream_final_chunk(false, "cmpl-metal", "text_completion",
                                               1700000000, "q27-metal", "length");
    check(ftxt["choices"][0]["finish_reason"] == "length", "sse: final text finish_reason");
    check(ftxt["choices"][0]["text"] == "", "sse: final text empty");

    // Invalid UTF-8 must not throw through the serializer (replace backstop).
    std::string bad = q27::sse_data(q27::openai_stream_chunk(true, "id", "chat.completion.chunk",
                                                             0, "m", std::string("\xE2\x80")));
    check(!bad.empty(), "sse: invalid utf-8 serializes via replace handler");
    return 0;
}

} // namespace

int main() {
    test_utf8_gate();
    test_stop_buffer();
    test_sse_framing();
    if (failures) { fprintf(stderr, "%d stream-format check(s) failed\n", failures); return 1; }
    puts("Metal stream format: OK");
    return 0;
}
