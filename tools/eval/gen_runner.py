#!/usr/bin/env python3
"""Generation runner: produce agreement.py-format JSONL from a running q27 server.

Reads prompts JSONL ({"prompt_id": str, "prompt": str}), POSTs each prompt
non-streaming to a q27 server, writes {"id", "prompt_id", "text"} JSONL.

Requests are strictly serial (house rule: one model consumer at a time; this
tool must only ever run inside a coordinated GPU slot, against a server whose
owner expects the traffic). One retry per prompt, then the prompt is recorded
with "text": "" and an "error" field — partial corpora fail loudly downstream
(agreement.py counts unextractable as disagreement, not missing data).

Endpoints: --api anthropic -> /v1/messages, --api openai -> /v1/chat/completions.
No third-party deps; stdlib urllib only.
"""
import argparse, json, sys, time, urllib.request, urllib.error


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code,
                                    "redirect refused: " + newurl, headers, fp)


DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def build_request(api, base_url, prompt, max_tokens):
    if api == "anthropic":
        url = base_url.rstrip("/") + "/v1/messages"
        body = {"model": "q27", "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
    else:
        url = base_url.rstrip("/") + "/v1/chat/completions"
        body = {"model": "q27", "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
    data = json.dumps(body).encode()
    return urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})


def extract_text(api, payload):
    if api == "anthropic":
        return "".join(b.get("text", "") for b in payload.get("content", [])
                       if b.get("type") == "text")
    choices = payload.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "") or ""


def extract_thinking(api, payload):
    """Concatenate thinking-block content (Anthropic) / reasoning_content
    (OpenAI). Returns "" when the response has none."""
    if api == "anthropic":
        return "".join(b.get("thinking", "") for b in payload.get("content", [])
                       if b.get("type") == "thinking")
    choices = payload.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("reasoning_content", "") or ""


def extract_stop_reason(api, payload):
    """The generation stop reason ('end_turn'/'stop' = complete,
    'max_tokens'/'length' = truncated). Drives the scorer's
    end_turn-only filter (the thinking-budget artifact of 2026-07-19)."""
    if api == "anthropic":
        return payload.get("stop_reason") or ""
    choices = payload.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("finish_reason") or ""


def run_one(api, base_url, prompt, max_tokens, timeout):
    """Returns (text, thinking, stop_reason)."""
    req = build_request(api, base_url, prompt, max_tokens)
    with DIRECT_OPENER.open(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode())
    return (extract_text(api, payload), extract_thinking(api, payload),
            extract_stop_reason(api, payload))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompts", help="input JSONL: {prompt_id, prompt}")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--base-url", default="http://127.0.0.1:8213")
    ap.add_argument("--api", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--tag", default="gen", help="id prefix in output rows")
    args = ap.parse_args()

    rows = []
    seen = set()
    with open(args.prompts) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "prompt_id" not in r or "prompt" not in r:
                sys.exit(f"line {ln}: missing prompt_id/prompt")
            if r["prompt_id"] in seen:
                sys.exit(f"line {ln}: duplicate prompt_id {r['prompt_id']!r}")
            seen.add(r["prompt_id"])
            rows.append(r)

    failures = 0
    with open(args.out, "w") as out:
        for i, r in enumerate(rows):
            rec = {"id": f"{args.tag}-{i:04d}", "prompt_id": r["prompt_id"]}
            err = None
            for attempt in (1, 2):
                try:
                    text, thinking, stop_reason = run_one(
                        args.api, args.base_url, r["prompt"],
                        args.max_tokens, args.timeout)
                    rec["text"] = text
                    rec["stop_reason"] = stop_reason
                    if thinking:
                        rec["thinking"] = thinking
                    err = None
                    break
                except (urllib.error.URLError, OSError, json.JSONDecodeError,
                        TimeoutError) as e:
                    err = f"attempt {attempt}: {e}"
                    time.sleep(2.0)
            if err is not None:
                rec["text"] = ""
                rec["error"] = err
                failures += 1
                print(f"FAIL {r['prompt_id']}: {err}", file=sys.stderr)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"[{i + 1}/{len(rows)}] {r['prompt_id']}"
                  + (" (failed)" if err else ""), file=sys.stderr)

    print(f"wrote {len(rows)} rows to {args.out}, {failures} failed",
          file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
