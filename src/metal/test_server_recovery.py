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

    env = os.environ.copy()
    env.pop("Q27_API_KEY", None)
    env.update({
        "Q27_METAL_FAIL_FINISH": "1",
        "Q27_METAL_TEST_FAILPOINTS": "1",
        "Q27_METAL_ADMIN_TOKEN": "recovery-gate",
    })
    process = subprocess.Popen(
        [server, model, tokenizer, "--host", "127.0.0.1", "--port", str(port),
         "--ctx", "32", "--slots", "1", "--max-tokens-default", "1"],
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

    stop_process()
    atexit.unregister(stop_process)
    print("Metal server recovery: PASS")


if __name__ == "__main__":
    main()
