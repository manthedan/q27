#!/usr/bin/env python3
"""Hermetic HTTP contract gate for an installed Qwen3.8 q27 package."""

import argparse
import json
import urllib.error
import urllib.request


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def request_json(base, path, payload, timeout=300):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"raw": raw.decode("utf-8", "replace")}
        return error.code, body


def get_json(base, path):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=30) as response:
        return response.status, json.loads(response.read())


def error_message(body):
    error = body.get("error", body) if isinstance(body, dict) else {}
    return error.get("message", "") if isinstance(error, dict) else ""


def require_rejected(label, status, body):
    require(400 <= status < 500 and error_message(body),
            f"{label} was not explicitly rejected: status={status} body={body}")


def weather_tools():
    schema = {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
        "additionalProperties": False,
    }
    return schema, [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for one city.",
            "parameters": schema,
        },
    }], [{
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for one city.",
        "parameters": schema,
    }]


def exercise_round_trips(base):
    schema, chat_tools, response_tools = weather_tools()

    status, chat = request_json(base, "/v1/chat/completions", {
        "model": "q27-metal",
        "messages": [{"role": "user", "content": "Reply with the word ready."}],
        "max_tokens": 256,
    })
    content = ((chat.get("choices") or [{}])[0].get("message") or {}).get("content")
    require(status == 200 and isinstance(content, str) and content,
            f"packaged Chat round trip failed: {status} {chat}")

    status, responses = request_json(base, "/v1/responses", {
        "model": "q27-metal",
        "input": "Reply with the word ready.",
        "max_output_tokens": 256,
    })
    output = responses.get("output") or []
    require(status == 200 and any(item.get("type") == "message" for item in output),
            f"packaged Responses round trip failed: {status} {responses}")

    status, anthropic = request_json(base, "/v1/messages", {
        "model": "q27-metal",
        "messages": [{"role": "user", "content": "Reply with the word ready."}],
        "max_tokens": 256,
    })
    require(status == 200 and any(
                block.get("type") == "text" and block.get("text")
                for block in anthropic.get("content") or []),
            f"packaged Anthropic round trip failed: {status} {anthropic}")

    status, chat = request_json(base, "/v1/chat/completions", {
        "model": "q27-metal",
        "messages": [{"role": "user", "content":
                      "Call get_weather for Paris. Do not answer directly."}],
        "tools": chat_tools,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "temperature": 0,
        "max_tokens": 192,
    })
    choice = (chat.get("choices") or [{}])[0]
    calls = (choice.get("message") or {}).get("tool_calls") or []
    require(status == 200 and choice.get("finish_reason") == "tool_calls" and
            len(calls) == 1 and calls[0].get("function", {}).get("name") == "get_weather",
            f"packaged Chat tool call failed: {status} {chat}")
    try:
        chat_arguments = json.loads(calls[0]["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise AssertionError(f"packaged Chat tool arguments are invalid: {chat}") from error
    require(chat_arguments.get("location") == "Paris",
            f"packaged Chat tool location is wrong: {chat}")

    status, responses = request_json(base, "/v1/responses", {
        "model": "q27-metal",
        "input": "Call get_weather for Paris. Do not answer directly.",
        "tools": response_tools,
        "tool_choice": {"type": "function", "name": "get_weather"},
        "temperature": 0,
        "max_output_tokens": 256,
    })
    calls = [item for item in responses.get("output") or []
             if item.get("type") == "function_call"]
    require(status == 200 and len(calls) == 1 and calls[0].get("name") == "get_weather",
            f"packaged Responses tool call failed: {status} {responses}")
    try:
        responses_arguments = json.loads(calls[0]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise AssertionError(
            f"packaged Responses tool arguments are invalid: {responses}") from error
    require(responses_arguments.get("location") == "Paris",
            f"packaged Responses tool location is wrong: {responses}")

    status, anthropic = request_json(base, "/v1/messages", {
        "model": "q27-metal",
        "messages": [{"role": "user", "content": "Call get_weather for Paris."}],
        "tools": [{"name": "get_weather", "description": "Get current weather.",
                   "input_schema": schema}],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "temperature": 0,
        "max_tokens": 192,
    })
    blocks = [block for block in anthropic.get("content") or []
              if block.get("type") == "tool_use"]
    require(status == 200 and anthropic.get("stop_reason") == "tool_use" and
            len(blocks) == 1 and blocks[0].get("name") == "get_weather" and
            blocks[0].get("input", {}).get("location") == "Paris",
            f"packaged Anthropic tool call failed: {status} {anthropic}")


def exercise_rejection_contracts(base):
    chat = {
        "model": "q27-metal",
        "messages": [{"role": "user", "content": "Reply briefly."}],
        "max_tokens": 16,
    }
    responses = {
        "model": "q27-metal",
        "input": "Reply briefly.",
        "max_output_tokens": 16,
    }
    for effort in ("low", "medium", "xhigh", "unknown", 7):
        for label, path, body in (
                ("Chat", "/v1/chat/completions", chat),
                ("Responses", "/v1/responses", responses)):
            payload = dict(body)
            payload["reasoning_effort"] = effort
            status, result = request_json(base, path, payload)
            require_rejected(f"{label} reasoning_effort={effort!r}", status, result)

    payload = dict(responses)
    payload["reasoning"] = {"effort": "medium"}
    status, result = request_json(base, "/v1/responses", payload)
    require_rejected("Responses reasoning object", status, result)

    for label, path, body in (
            ("Chat", "/v1/chat/completions", chat),
            ("Responses", "/v1/responses", responses)):
        payload = dict(body)
        payload["preserve_thinking"] = True
        status, result = request_json(base, path, payload)
        require_rejected(f"{label} preserve_thinking", status, result)

    ignored_thinking_controls = (
        ("Chat enable_thinking", "/v1/chat/completions",
         dict(chat, enable_thinking=False)),
        ("Chat thinking_token_budget", "/v1/chat/completions",
         dict(chat, thinking_token_budget=512)),
        ("Chat nested thinking budget", "/v1/chat/completions",
         dict(chat, chat_template_kwargs={"thinking_budget": 512})),
        ("Anthropic thinking disabled", "/v1/messages", {
            "model": "q27-metal", "max_tokens": 16,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Reply briefly."}],
        }),
    )
    for label, path, payload in ignored_thinking_controls:
        status, result = request_json(base, path, payload)
        require_rejected(label, status, result)

    historical = (
        ("Chat reasoning_content history", "/v1/chat/completions", {
            "model": "q27-metal", "max_tokens": 16,
            "messages": [
                {"role": "assistant", "content": "answer",
                 "reasoning_content": "private chain"},
                {"role": "user", "content": "continue"},
            ],
        }),
        ("Responses reasoning history", "/v1/responses", {
            "model": "q27-metal", "max_output_tokens": 16,
            "input": [
                {"type": "reasoning", "summary": []},
                {"role": "user", "content": "continue"},
            ],
        }),
    )
    for label, path, payload in historical:
        status, result = request_json(base, path, payload)
        require_rejected(label, status, result)

    unsupported = (("min_p", 0.1), ("presence_penalty", 1.0),
                   ("frequency_penalty", 0.5), ("repetition_penalty", 1.1))
    for key, value in unsupported:
        for label, path, body in (
                ("Chat", "/v1/chat/completions", chat),
                ("Responses", "/v1/responses", responses)):
            payload = dict(body)
            payload[key] = value
            status, result = request_json(base, path, payload)
            require_rejected(f"{label} unsupported {key}", status, result)

    media_cases = (
        ("Chat image_url", "/v1/chat/completions", {
            "model": "q27-metal", "max_tokens": 16,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": {"url":
                 "data:image/png;base64,iVBORw0KGgo="}},
            ]}],
        }),
        ("Responses input_image", "/v1/responses", {
            "model": "q27-metal", "max_output_tokens": 16,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "Describe this."},
                {"type": "input_image", "image_url":
                 "data:image/png;base64,iVBORw0KGgo="},
            ]}],
        }),
        ("Anthropic image", "/v1/messages", {
            "model": "q27-metal", "max_tokens": 16,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": "iVBORw0KGgo="}},
            ]}],
        }),
    )
    for label, path, payload in media_cases:
        status, result = request_json(base, path, payload)
        require_rejected(label, status, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("expected_profile")
    args = parser.parse_args()

    status, health = get_json(args.base_url, "/health")
    require(status == 200 and health.get("status") == "ok",
            f"packaged server health failed: {health}")
    # Servers that report a model identity must match the pack's profile;
    # the revival server's /health does not carry these fields yet.
    if "model_profile" in health:
        require(health.get("model_profile") == args.expected_profile and
                health.get("model_family") == "qwen38" and
                health.get("tool_dialect") == "xml",
                f"packaged server identity mismatch: {health}")
    exercise_rejection_contracts(args.base_url)
    exercise_round_trips(args.base_url)
    print(f"packaged API contracts: PASS ({args.expected_profile})")


if __name__ == "__main__":
    main()
