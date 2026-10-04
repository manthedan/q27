#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 /absolute/path/to/q27 PACK" >&2
  exit 2
fi
q27="$1"
pack="$2"
[[ "$q27" = /* && -x "$q27" ]] || {
  echo "packaged q27 launcher must be an absolute executable path" >&2
  exit 2
}
[[ -n "${Q27_HOME:-}" && -d "$Q27_HOME" ]] || {
  echo "Q27_HOME must name the fresh packaged-model cache" >&2
  exit 2
}
python="${PYTHON:-$(command -v python3 2>/dev/null || true)}"
[[ -x "$python" ]] || { echo "python3 is required for gate validation" >&2; exit 2; }

work="$(mktemp -d "${TMPDIR:-/tmp}/q27-packaged-agent.XXXXXX")"
cleanup() { rm -rf "$work"; }
trap cleanup EXIT
secret="Q27_PACKAGED_AGENT_READ_OK_38"
printf '%s\n' "$secret" >"$work/gate.txt"

home="${HOME:?}"
path="$(dirname "$q27"):/usr/bin:/bin:/usr/sbin:/sbin"
env -i HOME="$home" PATH="$path" TMPDIR="${TMPDIR:-/tmp}" \
  Q27_HOME="$Q27_HOME" Q27_RUN_DIR="$work/run" \
  Q27_AGENT_UI=classic Q27_AGENT_WORKSPACE="$work" \
  Q27_AGENT_CONTEXT=8192 Q27_AGENT_MAX_TOKENS=1024 \
  "$q27" agent "$pack" --prompt \
  "Use the read tool to read gate.txt. After the tool result, reply with the exact file contents and no other text." \
  --output-format jsonl >"$work/events.jsonl" 2>"$work/agent.err"

"$python" - "$work/events.jsonl" "$secret" <<'PY'
import base64
import json
import sys

path, secret = sys.argv[1:]
def require(condition, message):
    if not condition:
        raise RuntimeError(message)

events = []
with open(path, encoding="utf-8") as stream:
    for line in stream:
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
require(events, "native agent emitted no JSONL events")
types = [event.get("type") for event in events]
require("tool_done" in types, f"native agent did not complete a tool: {types}")
require("turn_done" in types, f"native agent did not complete its answer: {types}")
require(any(event.get("tool_kind") == "read" for event in events),
        f"native agent did not invoke read: {events}")
def event_payload(event):
    payload = []
    if isinstance(event.get("text"), str):
        payload.append(event["text"])
    encoded = event.get("data_b64")
    if encoded:
        payload.append(base64.b64decode(encoded).decode("utf-8", "replace"))
    return "".join(payload)

tool_done = next(i for i, event in enumerate(events)
                 if event.get("type") == "tool_done")
require(secret in "".join(event_payload(event) for event in events[tool_done + 1:]),
        "native agent did not use the tool result in its final answer")
require(not any(event.get("type") in ("error", "rejected", "generation_stalled")
                for event in events), f"native agent emitted a failure event: {types}")
PY

echo "packaged native-agent Qwen3.8 tool loop: PASS"
