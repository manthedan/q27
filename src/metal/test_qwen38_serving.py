#!/usr/bin/env python3
"""Exercise Qwen3.8 Metal serving across OpenAI and Anthropic tool workflows."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def available_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def request_json(port, path, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def openai_tool(name, description):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }


def require_openai_call(label, status, body, name, location):
    require(status == 200, f"{label} returned {status}: {body}")
    choice = (body.get("choices") or [{}])[0]
    calls = choice.get("message", {}).get("tool_calls") or []
    require(choice.get("finish_reason") == "tool_calls" and len(calls) == 1,
            f"{label} did not return one tool call: {body}")
    call = calls[0]
    function = call.get("function", {})
    require(function.get("name") == name, f"{label} called the wrong tool: {body}")
    try:
        arguments = json.loads(function.get("arguments", ""))
    except json.JSONDecodeError as error:
        raise AssertionError(f"{label} returned invalid arguments: {body}") from error
    require(arguments.get("location") == location,
            f"{label} returned the wrong location: {body}")
    return choice["message"], call


def exercise(port):
    weather = openai_tool("get_weather", "Get the current weather for one city.")
    forecast = openai_tool("get_forecast", "Get tomorrow's forecast for one city.")
    first_user = {
        "role": "user",
        "content": "Use get_weather to check the weather in Tokyo. Do not answer without calling the tool.",
    }
    first_payload = {
        "model": "q27-metal",
        "messages": [first_user],
        "tools": [weather],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "temperature": 0,
        "max_tokens": 128,
    }
    first_status, first_body = request_json(port, "/v1/chat/completions", first_payload)
    first_message, first_call = require_openai_call(
        "first OpenAI turn", first_status, first_body, "get_weather", "Tokyo")

    second_user = {
        "role": "user",
        "content": "Now call get_forecast for Tokyo tomorrow. Do not answer directly.",
    }
    second_messages = [
        first_user,
        first_message,
        {"role": "tool", "tool_call_id": first_call["id"],
         "content": "Tokyo is 24 C and clear."},
        second_user,
    ]
    second_payload = {
        "model": "q27-metal",
        "messages": second_messages,
        "tools": [weather, forecast],
        "tool_choice": {"type": "function", "function": {"name": "get_forecast"}},
        "temperature": 0,
        "max_tokens": 128,
    }
    second_status, second_body = request_json(port, "/v1/chat/completions", second_payload)
    second_message, second_call = require_openai_call(
        "second OpenAI turn", second_status, second_body, "get_forecast", "Tokyo")

    final_payload = {
        "model": "q27-metal",
        "messages": second_messages + [
            second_message,
            {"role": "tool", "tool_call_id": second_call["id"],
             "content": "Tomorrow: 26 C, light rain in the afternoon."},
            {"role": "user",
             "content": "Summarize today's weather and tomorrow's forecast in one sentence."},
        ],
        "tools": [weather, forecast],
        "tool_choice": "none",
        "temperature": 0,
        "max_tokens": 96,
    }
    final_status, final_body = request_json(port, "/v1/chat/completions", final_payload)
    require(final_status == 200, f"final OpenAI turn returned {final_status}: {final_body}")
    final_choice = (final_body.get("choices") or [{}])[0]
    content = final_choice.get("message", {}).get("content")
    require(final_choice.get("finish_reason") == "stop" and isinstance(content, str) and content,
            f"final OpenAI turn did not return text: {final_body}")

    anthropic_payload = {
        "model": "q27-metal",
        "max_tokens": 128,
        "temperature": 0,
        "messages": [{"role": "user", "content": "Use get_weather to check Paris."}],
        "tools": [{
            "name": "get_weather",
            "description": "Get the current weather for one city.",
            "input_schema": weather["function"]["parameters"],
        }],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }
    anthropic_status, anthropic_body = request_json(
        port, "/v1/messages", anthropic_payload)
    blocks = anthropic_body.get("content") or []
    require(anthropic_status == 200 and anthropic_body.get("stop_reason") == "tool_use" and
            len(blocks) == 1 and blocks[0].get("type") == "tool_use" and
            blocks[0].get("name") == "get_weather" and
            blocks[0].get("input", {}).get("location") == "Paris",
            f"Anthropic turn did not return the requested tool call: {anthropic_body}")


def main():
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} SERVER MODEL TOKENIZER")
    server, model, tokenizer = map(os.path.abspath, sys.argv[1:])
    for path in (server, model, tokenizer):
        if not os.path.isfile(path):
            raise SystemExit(f"not a file: {path}")

    port = available_port()
    process = subprocess.Popen(
        [server, model, tokenizer, "--host", "127.0.0.1", "--port", str(port),
         "--ctx", "2048", "--slots", "1", "--mtp", "4",
         "--max-tokens-default", "256"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
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
    try:
        deadline = time.monotonic() + 300
        while not ready.wait(0.1):
            if process.poll() is not None:
                raise AssertionError(
                    f"Qwen3.8 Metal server exited before readiness: {process.returncode}")
            if time.monotonic() >= deadline:
                raise AssertionError("Qwen3.8 Metal server did not become ready")
        exercise(port)
    except Exception:
        sys.stderr.write("".join(stderr_lines))
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        thread.join(timeout=5)

    print("Qwen3.8 Metal OpenAI and Anthropic serving workflow: PASS")


if __name__ == "__main__":
    main()
