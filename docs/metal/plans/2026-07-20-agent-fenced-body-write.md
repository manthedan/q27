# Same-turn fenced body for write/edit (2026-07-20)

Branch: `agent-fenced-body-write`

## Problem

Two observed failure modes on Bonsai-27B (T2/B1) under the native agent:

1. **JSON-embedded content** — model tries to put Python (f-strings, quotes)
   inside tool-call JSON → escaping death spiral / whitespace loops.
2. **Path-only + raw second turn** — control `write{path}` succeeds, then a
   free no-tools generation re-emits another `<tool_call>…write…` as the
   “file” (86 bytes in the operator trace), poisoning the workspace and
   cascading into edit/selection loops.

Prism’s published agentic scores are BFCL/τ²-style structured calls, not this
two-phase raw channel. Long-horizon coding is explicitly roadmap there.

## Design

Thin JSON header + **same-turn markdown-fenced body**:

```
<tool_call>{"name":"write","arguments":{"path":"hello.py"}}</tool_call>
```python
print(f"hi {name}")
```
```

| Tool | JSON args | After `</tool_call>` |
|---|---|---|
| `write` | `path` | required fence = new file (create only) |
| `overwrite` | `path` | required fence = whole file create/replace |
| `edit` | `path`, `old` (≤512 B) | required fence = replacement |
| `edit_selection` | `path`, `selection` | required fence = replacement |
| read/search/shell | as before | no trailing body |

### Engine

After `</tool_call>`, if the accepted tool name is a body tool, **continue**
free decoding until EOS (grammar already disengaged). Non-body tools still
stop at the closer.

### Control plane

- No second raw-payload chat turn for write/edit.
- Missing fence → tool_response error, no side effect.
- Body that looks like `<tool_call` / `{"name"` → reject, no side effect.
- Oversized `edit.old` → parse failure; steer to `overwrite`.

## Non-goals (this branch)

- Temperature 0.7 sampling (still greedy) — orthogonal next lever.
- Grammar-locking the fence opener after close (nice-to-have).
- Removing dead `generate_raw_payload` helpers from older session paths yet.

## Why not Rust for this

The change is a **protocol + control-loop** fix. The engine is Metal/C++ with
tight residency/KV contracts; rewriting the agent in Rust would add FFI
surface without fixing the mode-switch failure. Revisit language only if the
agent grows far beyond this harness, not for this patch.
