#!/usr/bin/env python3
"""Exercise Metal server recovery and shared snapshot reuse against a real model.

Usage: test_server_recovery.py SERVER MODEL TOKENIZER
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request


def fail(message, process=None, stderr_lines=None):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if stderr_lines:
        sys.stderr.write("".join(stderr_lines))
    raise SystemExit(message)


def available_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def start_server(server, model, tokenizer, snapshot_dir, extra_env, context=256, extra_args=()):
    port = available_port()
    env = os.environ.copy()
    env.pop("Q27_API_KEY", None)
    env.update(extra_env)
    process = subprocess.Popen(
        [server, model, tokenizer, "--host", "127.0.0.1", "--port", str(port),
         "--ctx", str(context), "--slots", "1", "--max-tokens-default", "1",
         "--think-budget", "0", "--snapshot-dir", snapshot_dir,
         "--snapshot-max-mb", "256"] + list(extra_args),
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
    )
    stderr_lines = []
    ready = threading.Event()

    def drain_stderr():
        for line in process.stderr:
            stderr_lines.append(line)
            if "listening on http://" in line:
                ready.set()

    thread = threading.Thread(target=drain_stderr, daemon=True)
    thread.start()
    if not ready.wait(300):
        fail("server did not become ready", process, stderr_lines)
    return process, thread, stderr_lines, port


def stop_server(process, thread):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    thread.join(timeout=5)


def request_json(port, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if payload is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        raw = error.read()
        body = json.loads(raw) if raw else {}
        return error.code, body


def request_sse(port, path, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            events = []
            for line in response.read().decode().splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line[6:]))
            return response.status, response.headers.get_all("Content-Type") or [], events
    except urllib.error.HTTPError as error:
        raw = error.read()
        body = json.loads(raw) if raw else {}
        return error.code, error.headers.get_all("Content-Type") or [], [body]


def require_json_success(label, status, body):
    if status != 200:
        raise AssertionError(f"{label} returned {status}: {body}")
    return body


def exercise_recovery(server, model, tokenizer):
    snapshot_dir = tempfile.TemporaryDirectory(prefix="q27-shared-snapshots-")
    os.chmod(snapshot_dir.name, 0o777)
    process = thread = None
    stderr_lines = []
    try:
        process, thread, stderr_lines, port = start_server(
            server, model, tokenizer, snapshot_dir.name,
            {"Q27_METAL_FAIL_FINISH": "1"},
        )
        snapshot_mode = os.stat(snapshot_dir.name).st_mode & 0o777
        if snapshot_mode != 0o700:
            fail(f"snapshot directory mode is {snapshot_mode:o}, expected 700",
                 process, stderr_lines)

        models_status, models = request_json(port, "/v1/models")
        catalog = (models.get("data") or [{}])[0]
        if (models_status != 200 or models.get("object") != "list" or
                catalog.get("id") != "q27-metal"):
            fail(f"model catalog mismatch: {models_status} {models}", process, stderr_lines)

        first_status, content_types, first_events = request_sse(
            port, "/v1/completions",
            {"model": "q27-metal", "prompt": "Hello", "max_tokens": 1,
             "stream": True, "temperature": 0},
        )
        errors = [event.get("error", {}) for event in first_events if "error" in event]
        if (first_status != 200 or content_types != ["text/event-stream"] or
                not any(error.get("type") == "api_error" for error in errors)):
            fail(f"injected stream did not surface an API error: "
                 f"{first_status} {content_types} {first_events}", process, stderr_lines)

        second_status, second = request_json(
            port, "/v1/completions",
            {"model": "q27-metal", "prompt": "Hello", "max_tokens": 1,
             "stream": False, "temperature": 0},
        )
        if (second_status != 200 or
                second.get("usage", {}).get("completion_tokens") != 1):
            fail(f"post-recovery completion failed: {second_status} {second}",
                 process, stderr_lines)
        if not any("Metal backend recovery: rebuilt 1 slot" in line
                   for line in stderr_lines):
            fail("server did not reconstruct the failed Metal slot", process, stderr_lines)

        chat_status, chat = request_json(
            port, "/v1/chat/completions",
            {"model": "q27-metal", "messages": [{"role": "user", "content": "Hello"}],
             "max_tokens": 1, "temperature": 0},
        )
        require_json_success("chat completion", chat_status, chat)
        if chat.get("object") != "chat.completion":
            fail(f"chat response shape mismatch: {chat}", process, stderr_lines)

        responses_status, responses = request_json(
            port, "/v1/responses",
            {"model": "q27-metal", "input": "Hello", "max_output_tokens": 1,
             "stream": False},
        )
        require_json_success("Responses request", responses_status, responses)
        if responses.get("object") != "response":
            fail(f"Responses shape mismatch: {responses}", process, stderr_lines)

        messages_payload = {
            "model": "q27-metal",
            "system": "Stable shared prefix for two independent clients.",
            "messages": [{"role": "user", "content": "Return one short token."}],
            "max_tokens": 1,
            "temperature": 0,
            "snapshot": True,
        }
        first_message_status, first_message = request_json(
            port, "/v1/messages", messages_payload,
        )
        require_json_success("first Messages request", first_message_status, first_message)
        second_payload = dict(messages_payload)
        second_payload.pop("snapshot")
        second_message_status, second_message = request_json(
            port, "/v1/messages", second_payload,
        )
        require_json_success("second Messages request", second_message_status, second_message)
        if second_message.get("q27_prefix_hit", 0) <= 0:
            fail(f"shared snapshot was not reused by the second client: {second_message}",
                 process, stderr_lines)
    except AssertionError as error:
        fail(str(error), process, stderr_lines)
    finally:
        if process is not None:
            stop_server(process, thread)
        snapshot_dir.cleanup()


def exercise_post_publish_failure(server, model, tokenizer):
    snapshot_dir = tempfile.TemporaryDirectory(prefix="q27-publish-failure-")
    process = thread = None
    stderr_lines = []
    try:
        process, thread, stderr_lines, port = start_server(
            server, model, tokenizer, snapshot_dir.name,
            {"Q27_METAL_FAIL_AFTER_SNAPSHOT_PUBLISH": "1"},
        )
        status, body = request_json(
            port, "/v1/completions",
            {"model": "q27-metal", "prompt": "z " * 180, "max_tokens": 1,
             "stream": False, "temperature": 0, "snapshot": True},
        )
        if status != 200 or body.get("usage", {}).get("completion_tokens") != 1:
            fail(f"post-publication failure escaped inference: {status} {body}",
                 process, stderr_lines)
        deadline = time.monotonic() + 5
        while (time.monotonic() < deadline and not any(
                "injected failure after snapshot publication" in line
                for line in stderr_lines)):
            time.sleep(0.01)
        if not any("injected failure after snapshot publication" in line
                   for line in stderr_lines):
            fail("post-publication failpoint was not consumed", process, stderr_lines)
        snapshots = [name for name in os.listdir(snapshot_dir.name)
                     if name.endswith(".q27snap")]
        if not snapshots:
            fail("snapshot publication failpoint ran before publishing the shared artifact",
                 process, stderr_lines)
    finally:
        if process is not None:
            stop_server(process, thread)
        snapshot_dir.cleanup()


def main():
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} SERVER MODEL TOKENIZER")
    server, model, tokenizer = map(os.path.abspath, sys.argv[1:])
    for path in (server, model, tokenizer):
        if not os.path.isfile(path):
            raise SystemExit(f"not a file: {path}")

    exercise_recovery(server, model, tokenizer)
    exercise_post_publish_failure(server, model, tokenizer)
    print("Metal server recovery and shared snapshots: PASS")


if __name__ == "__main__":
    main()
