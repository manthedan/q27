#!/usr/bin/env python3
"""CPU-only selftest for the experimental prefix installer/proxy."""

from __future__ import annotations

import http.server
import importlib.util
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

MODULE_PATH = Path(__file__).with_name("experimental_prefix_cache.py")
spec = importlib.util.spec_from_file_location("q27_experimental_prefix_cache", MODULE_PATH)
assert spec and spec.loader
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class FakeTarget(http.server.BaseHTTPRequestHandler):
    prewarms = []
    forwarded = []
    redirect_followed = False

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if self.path == "/experimental/prefix-cache/prewarm":
            assert self.headers.get("X-Q27-Admin-Token") == "test-token"
            payload = json.loads(body)
            self.prewarms.append(payload)
            if payload.get("request", {}).get("redirect") is True:
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/stolen")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if payload.get("request", {}).get("stream") is True:
                time.sleep(1.5)
            value = {
                "status": "ok", "experimental": True,
                "cache_key": "a" * 40, "prefix_tokens": 10,
                "request_prompt_tokens": 12, "already_cached": False,
                "snapshot_written": True,
            }
        else:
            assert self.headers.get("X-Q27-Admin-Token") is None
            assert self.headers.get("Authorization") is None
            request = json.loads(body)
            self.forwarded.append((self.path, request))
            if request.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b"data: first\n\n"); self.wfile.flush()
                time.sleep(0.5)
                self.wfile.write(b"data: second\n\n"); self.wfile.flush()
                self.close_connection = True
                return
            value = {"object": "chat.completion", "choices": []}
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/stolen":
            self.__class__.redirect_followed = True
        self.send_response(204); self.end_headers()

    def log_message(self, *_args) -> None:
        pass


def run() -> None:
    target_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeTarget)
    target_thread = threading.Thread(target=target_server.serve_forever, daemon=True)
    target_thread.start()
    target = cache.target_url(f"http://127.0.0.1:{target_server.server_port}")
    request = {"messages": [{"role": "user", "content": "hello"}]}
    result = cache.prewarm(target, "test-token", "chat_completions", request)
    assert result["status"] == "ok"
    assert FakeTarget.prewarms[-1] == {"api": "chat_completions", "request": request}
    # The direct path also accepts the Anthropic Messages API (Claude Code).
    claude_direct = {"system": "You are Claude Code.",
                     "messages": [{"role": "user", "content": "hi"}]}
    result = cache.prewarm(target, "test-token", "messages", claude_direct)
    assert result["status"] == "ok"
    assert FakeTarget.prewarms[-1] == {"api": "messages", "request": claude_direct}
    try:
        cache.prewarm(target, "test-token", "chat_completions", {"redirect": True})
        raise AssertionError("credential-bearing redirect was followed")
    except RuntimeError:
        pass
    assert not FakeTarget.redirect_followed
    # Count after the two direct prewarms (chat + messages) above.
    direct_prewarms = len(FakeTarget.prewarms)

    state = cache.ProxyState(target, "test-token", "capture-token")
    proxy_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), cache.PrefixProxy)
    proxy_server.state = state
    proxy_thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
    proxy_thread.start()
    url = f"http://127.0.0.1:{proxy_server.server_port}/v1/chat/completions"

    unauthorized = urllib.request.Request(
        url, data=json.dumps(request).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(unauthorized, timeout=5)
        raise AssertionError("unauthenticated capture request accepted")
    except urllib.error.HTTPError as exc:
        assert exc.code == 403
    assert len(FakeTarget.prewarms) == direct_prewarms

    # The first streaming harness request receives an SSE comment while its
    # cold prewarm is still running, then the actual upstream SSE response.
    cold_stream = {"stream": True,
                   "messages": [{"role": "user", "content": "install"}]}
    cold_req = urllib.request.Request(
        url, data=json.dumps(cold_stream).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer capture-token"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(cold_req, timeout=5) as response:
        first_line = response.readline()
        keepalive_elapsed = time.monotonic() - started
        cold_rest = response.read()
    assert first_line.startswith(b": q27 prefix prewarm")
    assert keepalive_elapsed < 1.3, f"cold prewarm was silent for {keepalive_elapsed:.3f}s"
    assert b"data: first" in cold_rest and b"data: second" in cold_rest

    for text in ("first", "second"):
        body = {"messages": [{"role": "user", "content": text}]}
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer capture-token"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            assert json.load(response)["object"] == "chat.completion"
    # The direct prewarms above plus exactly one proxy prewarm; subsequent
    # harness requests are forwarded without repeated multi-hundred-MiB saves.
    assert len(FakeTarget.prewarms) == direct_prewarms + 1
    assert len(FakeTarget.forwarded) == 3
    assert FakeTarget.forwarded[1][1]["messages"][0]["content"] == "first"
    assert FakeTarget.forwarded[2][1]["messages"][0]["content"] == "second"

    stream_body = {"stream": True,
                   "messages": [{"role": "user", "content": "stream"}]}
    stream_req = urllib.request.Request(
        url, data=json.dumps(stream_body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer capture-token"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(stream_req, timeout=5) as response:
        first = response.read(len(b"data: first\n\n"))
        elapsed = time.monotonic() - started
        rest = response.read()
    assert first == b"data: first\n\n" and b"second" in rest
    assert elapsed < 0.4, f"proxy buffered streaming response for {elapsed:.3f}s"

    responses_error = cache.sse_error_payload("boom", "responses").decode()
    response_rows = [json.loads(line[6:]) for line in responses_error.splitlines()
                     if line.startswith("data: ")]
    assert [row["type"] for row in response_rows] == ["error", "response.failed"]
    assert "[DONE]" not in responses_error
    chat_error = cache.sse_error_payload("boom", "chat_completions").decode()
    assert '\"error\"' in chat_error and "data: [DONE]" in chat_error
    # Anthropic /v1/messages streaming prewarm failures ride a named
    # "event: error" frame with the real API's error envelope, NOT the
    # OpenAI chat data-frame + [DONE] shape.
    anthropic_error = cache.sse_error_payload("boom", "messages").decode()
    assert anthropic_error.startswith("event: error\n"), anthropic_error
    anthropic_rows = [json.loads(line[6:]) for line in anthropic_error.splitlines()
                      if line.startswith("data: ")]
    assert anthropic_rows[0]["type"] == "error"
    assert anthropic_rows[0]["error"]["message"] == "boom"
    assert "[DONE]" not in anthropic_error

    rejected_listener = False
    try:
        cache.listen_address("0.0.0.0")
    except Exception:
        rejected_listener = True
    assert rejected_listener, "non-loopback listener accepted"

    # /v1/messages (Claude Code, Anthropic API) is an eligible capture path and
    # maps to api="messages" on the prewarm envelope.
    assert cache.ELIGIBLE["/v1/messages"] == "messages"
    state2 = cache.ProxyState(target, "test-token", "capture-token")
    proxy2 = http.server.ThreadingHTTPServer(("127.0.0.1", 0), cache.PrefixProxy)
    proxy2.state = state2
    proxy2_thread = threading.Thread(target=proxy2.serve_forever, daemon=True)
    proxy2_thread.start()
    prewarm_count_before = len(FakeTarget.prewarms)
    messages_url = f"http://127.0.0.1:{proxy2.server_port}/v1/messages"
    claude_request = {
        "system": "You are Claude Code.",
        "messages": [{"role": "user", "content": "fix the bug"}],
    }
    messages_req = urllib.request.Request(
        messages_url, data=json.dumps(claude_request).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer capture-token"},
    )
    with urllib.request.urlopen(messages_req, timeout=5) as response:
        assert response.status == 200
    assert FakeTarget.prewarms[-1] == {"api": "messages", "request": claude_request}
    assert len(FakeTarget.prewarms) == prewarm_count_before + 1
    # A second /v1/messages request is forwarded without re-prewarming.
    with urllib.request.urlopen(messages_req, timeout=5) as response:
        assert response.status == 200
    assert len(FakeTarget.prewarms) == prewarm_count_before + 1
    proxy2.shutdown(); proxy2.server_close(); proxy2_thread.join()

    proxy_server.shutdown(); proxy_server.server_close(); proxy_thread.join()
    target_server.shutdown(); target_server.server_close(); target_thread.join()
    print("experimental prefix cache selftest: PASS")


if __name__ == "__main__":
    run()
