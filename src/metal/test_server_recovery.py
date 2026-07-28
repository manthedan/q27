#!/usr/bin/env python3
"""Exercise streaming committed-command failure containment against a real model.

Usage: test_server_recovery.py SERVER MODEL TOKENIZER
"""

import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import tempfile


def fail(message, process=None, stderr_lines=None):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    if stderr_lines:
        sys.stderr.write("".join(stderr_lines))
    raise SystemExit(message)


def main():
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} SERVER MODEL TOKENIZER")
    server, model, tokenizer = map(os.path.abspath, sys.argv[1:])
    for path in (server, model, tokenizer):
        if not os.path.isfile(path):
            raise SystemExit(f"not a file: {path}")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    cache_dir = tempfile.TemporaryDirectory(prefix="q27-prefix-prewarm-")
    env = os.environ.copy()
    env.pop("Q27_API_KEY", None)
    env.update({
        "Q27_METAL_FAIL_FINISH": "1",
        "Q27_METAL_TEST_FAILPOINTS": "1",
        "Q27_METAL_ADMIN_TOKEN": "recovery-gate",
    })
    process = subprocess.Popen(
        [server, model, tokenizer, "--host", "127.0.0.1", "--port", str(port),
         "--ctx", "256", "--slots", "1", "--max-tokens-default", "1",
         "--experimental-prefix-cache", cache_dir.name, "--snapshot-max-mb", "256"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
    )

    def stop_process():
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    atexit.register(stop_process)
    stderr_lines = []
    ready = threading.Event()

    def drain_stderr():
        for line in process.stderr:
            stderr_lines.append(line)
            if "listening on http://" in line:
                ready.set()

    thread = threading.Thread(target=drain_stderr, daemon=True)
    thread.start()
    if not ready.wait(60):
        fail("server did not become ready", process, stderr_lines)

    url = f"http://127.0.0.1:{port}/v1/completions"
    def request_once(stream):
        payload = json.dumps({
            "model": "q27", "prompt": "Hello", "max_tokens": 1, "stream": stream,
        }).encode()
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                raw = response.read()
                if not stream:
                    return response.status, json.loads(raw)
                error_event = None
                for line in raw.decode().splitlines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    if "error" in event:
                        error_event = event
                return response.status, error_event
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def request_oversized_body():
        payload = b' ' * (2 << 20)
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    def request_tool_fallback(parallel=True):
        payload = json.dumps({
            "model": "q27", "input": "Run the fallback tool", "stream": True,
            "max_output_tokens": 8, "q27_test_tool_fallback": True,
            "parallel_tool_calls": parallel,
            "tools": [{
                "type": "function", "name": "fallback_tool",
                "description": "fallback regression fixture",
                "parameters": {"type": "object", "properties": {}},
            }],
            "tool_choice": {"type": "function", "name": "fallback_tool"},
        }).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/responses", data=payload,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                events = []
                for line in response.read().decode().splitlines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        events.append(json.loads(line[6:]))
                return response.status, events
        except urllib.error.HTTPError as error:
            return error.code, [json.loads(error.read())]

    def request_anthropic_prewarm():
        anthropic_request = {
            "model": "q27", "system": "Stable system prompt",
            "messages": [
                {"role": "user", "content": "Live request"},
            ],
            "max_tokens": 1,
            "tool_choice": {"type": "none"},
            "tools": [
                {"name": "alpha", "description": "first",
                 "input_schema": {"type": "object", "properties": {}}},
                {"name": "beta", "description": "second",
                 "input_schema": {"type": "object", "properties": {}}},
            ],
        }
        envelope = json.dumps({"api": "messages", "request": anthropic_request}).encode()
        prewarm = urllib.request.Request(
            f"http://127.0.0.1:{port}/experimental/prefix-cache/prewarm",
            data=envelope, headers={"Content-Type": "application/json",
                                    "X-Q27-Admin-Token": "recovery-gate"})
        try:
            with urllib.request.urlopen(prewarm, timeout=300) as response:
                prewarm_body = json.loads(response.read())
                prewarm_status = response.status
        except urllib.error.HTTPError as error:
            prewarm_status, prewarm_body = error.code, json.loads(error.read())
        live = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/messages",
            data=json.dumps(anthropic_request).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(live, timeout=300) as response:
                live_body = json.loads(response.read())
                live_status = response.status
        except urllib.error.HTTPError as error:
            live_status, live_body = error.code, json.loads(error.read())
        return prewarm_status, prewarm_body, live_status, live_body

    oversized_status = request_oversized_body()
    if oversized_status != 413:
        fail(f"oversized request returned HTTP {oversized_status}, expected 413",
             process, stderr_lines)

    first_status, first = request_once(True)
    second_status, second = request_once(False)
    if first_status != 200:
        fail(f"injected stream returned HTTP {first_status}, expected committed 200", process, stderr_lines)
    if not first or first.get("error", {}).get("type") != "api_error":
        fail(f"injected stream returned wrong error event: {first}", process, stderr_lines)
    if second_status != 200 or second.get("usage", {}).get("completion_tokens") != 1:
        fail(f"post-recovery request failed: {second_status} {second}", process, stderr_lines)
    if process.poll() is not None:
        fail("server exited during recovery", process, stderr_lines)
    if not any("Metal backend recovery: rebuilt 1 slot" in line for line in stderr_lines):
        fail("server did not report backend reconstruction", process, stderr_lines)
    prewarm_status, prewarm, live_status, live = request_anthropic_prewarm()
    if (prewarm_status != 200 or prewarm.get("status") != "ok" or
            not (prewarm.get("snapshot_written") or prewarm.get("already_cached")) or
            live_status != 200 or not live.get("q27_prefix_hit")):
        fail(f"Anthropic tool_choice prewarm mismatch: "
             f"{prewarm_status} {prewarm} / {live_status} {live}",
             process, stderr_lines)

    fallback_status, fallback_events = request_tool_fallback()
    fallback_items = [
        event.get("item", {}) for event in fallback_events
        if event.get("type") == "response.output_item.done"
    ]
    if fallback_status != 200 or not any(
            item.get("type") == "function_call" and
            item.get("name") == "fallback_tool" and
            item.get("arguments") == "{}"
            for item in fallback_items):
        fail(f"buffered tool fallback failed: {fallback_status} {fallback_events}", process, stderr_lines)
    if not any(event.get("type") == "response.completed" for event in fallback_events):
        fail(f"buffered tool fallback did not complete: {fallback_events}", process, stderr_lines)

    single_status, single_events = request_tool_fallback(False)
    single_items = [
        event.get("item", {}) for event in single_events
        if event.get("type") == "response.output_item.done"
    ]
    single_calls = [item for item in single_items if item.get("type") == "function_call"]
    preserved = [
        part.get("text", "")
        for item in single_items if item.get("type") == "message"
        for part in item.get("content", []) if part.get("type") == "output_text"
    ]
    if (single_status != 200 or len(single_calls) != 1 or
            not any("fallback_tool" in text for text in preserved)):
        fail(f"single-call fallback discarded extra output: "
             f"{single_status} {single_events}", process, stderr_lines)

    stop_process()
    atexit.unregister(stop_process)
    cache_dir.cleanup()
    print("Metal server recovery: PASS")


if __name__ == "__main__":
    main()
