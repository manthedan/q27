# Frontend Protocol v1 — sketch

**Status:** P0 frozen · **P1–P5 landed** (`experiments/q27-tui`) · Issue 2 (seq@publish) fixed  
**Date:** 2026-07-21 (impl 2026-07-22)  
**Depends on:** `experiments/ds4-agent/` worker events + `--output-format jsonl`  
**Sibling:** `docs/metal/plans/2026-07-21-agent-tui.md`  
**Goal:** freeze an event-first boundary so a Rust (or any) TUI is a *client*, not a rewrite of the control plane.

---

## 1. Why this exists

Tau/pi’s useful idea is not “packages named `tau_ai` / `tau_agent`.” It is:

> **The agent brain emits a typed stream. Frontends only consume that stream and send small control messages.**

We already have most of the *outbound* half (`print_json_event` over stdout in jsonl mode). We do **not** yet have:

- a stable schema version
- an inbound control channel for a non-C frontend
- session/lifecycle events a UI needs (`hello`, `idle`, `queue`, slash results)
- text payloads that frontends can paint without base64-decoding every delta

This note freezes **Frontend Protocol v1 (FP1)** as the contract between:

| Process / role | Owns |
|----------------|------|
| **backend** (`q27-agent --frontend-proto` or equivalent) | Metal worker, transcript, tool loop, compaction, SNAP1 |
| **frontend** (C linenoise today; Rust Ratatui tomorrow; scripts forever) | layout, themes, editor, scrollback, user input |

The backend stays C/C++/Metal. Beauty lives on the other side of the wire.

---

## 2. Design principles

1. **Events are authoritative for UI state.** If it isn’t on the stream, the UI must not invent it (except pure presentation: theme, scroll position, collapsed widgets).
2. **One command at a time on the worker** (today’s admission rule). The UI may *queue* prompts; the backend owns the queue (§13 decision 1).
3. **Binary-safe payloads** where tools/files matter; UTF-8 text where chat matters. Prefer explicit fields over packing everything into `data_b64`.
4. **Additive evolution.** Unknown event `type`s and unknown fields are ignored by clients; unknown command `op`s are answered with an explicit `rejected` event (echoing `client_req_id` when parseable), never silent no-ops. Codes, features, and event types are additive — never renamed; clients handle unknowns generically.
5. **jsonl mode remains a valid client.** A frontend that only reads stdout and never sends structured control still works for one-shot and pipes.
6. **Every op failure is visible on-stream.** stderr is diagnostics for humans, never a channel a frontend must read (§6.1 `notice`).
7. **State changes are typed catalog events; notices are prose for humans.** Anything a frontend must *act* on (wipe scrollback, close a lifecycle, correlate an op) is a structured terminal (`turn_done`, `tool_done`, `session_done`, `rejected`, …) with optional machine `code`. `notice` carries human-readable nuance, never the sole signal for a state change.
8. **Clients key behavior off `hello.features`, not the spec’s final state.** Staged rollouts (e.g. P2 before queue) advertise only shipped capabilities; see §6.1 and §8.

---

## 3. Transport

### 3.1 v1 default: stdio NDJSON

```text
frontend ── stdin (NDJSON lines: ClientMessage) ──► backend
frontend ◄─ stdout (NDJSON lines: ServerEvent) ──── backend
backend diagnostics ── stderr ──► human / logs (not protocol)
```

- One JSON object per line, UTF-8, no pretty-print, `\n` terminated.
- Backend never mixes protocol frames with plain prose on stdout when FP1 is active.
- stderr stays free for ASan, “loading model…”, fatal diagnostics.
- **Frontend death:** EOF on stdin (frontend exited or crashed) requests shutdown — the backend stops accepting work, emits `bye` with `reason: "stdin_eof"`, and exits. A headless child must never outlive its frontend by accident.

### 3.2 Invocation sketch

```sh
# Headless brain for a GUI/TUI client
q27-agent MODEL.q27 MODEL.tok \
  --frontend-proto 1 \
  --auto-tools \
  --workspace "$PWD" \
  --session work.q27agent

# Existing machine stream (subset / compatibility — see §8)
q27-agent MODEL.q27 MODEL.tok --output-format jsonl --prompt '…'
```

Optional later (not v1): Unix socket or length-prefixed frames if stdio contention with child tools becomes real. Shell tools already isolate their own pipes; protocol stdio is the agent process, not the tool process.

### 3.3 Framing rules

| Rule | Detail |
|------|--------|
| Line size | Soft cap 1 MiB per line; backends may reject larger client messages |
| Binary in JSON | `text` fields are UTF-8 strings with JSON escaping; raw tool dumps use chunked `tool_output` with per-chunk `encoding` (§6.3) |
| Ordering | Server `seq` is strictly increasing per process (uint64) **and stream order equals seq order**. Worker-originated events receive `seq` at **dequeue/publish** (`next_event`), not at enqueue, so control-plane emits (`hello`, `idle`, `queue`, `bye`, `notice`, `rejected`, `tool_start`, …) that share the same counter cannot invert stream order under concurrent `reject_busy`. Selection/session synthetic builders also allocate at creation (immediate publish). Gaps or reordering mean bugs or a dropped reader |
| Correlation | Every work unit has `command_id`. Client may pass `client_req_id` on commands; server echoes it on related events when known |
| Malformed input | An unparseable NDJSON line produces `notice` with `severity: "error"` and the session stays up; a parsed message with `v != 1` is answered `rejected` with `code: "bad_version"` |

---

## 4. Envelope

### 4.1 Server → client (`ServerEvent`)

Common fields on **every** event:

```json
{
  "v": 1,
  "seq": 42,
  "type": "text_delta",
  "ts_ms": 1784678400123,
  "command_id": 7,
  "client_req_id": null,
  "state": "generating",
  "status": "ok",
  "code": null
}
```

| Field | Type | Notes |
|-------|------|--------|
| `v` | number | Protocol major version; always `1` for FP1 |
| `seq` | uint64 | Monotonic process-wide |
| `type` | string | Event discriminant |
| `ts_ms` | uint64 | Wall-clock milliseconds at publication. Informational only (UI clocks, log correlation); never used for correctness |
| `command_id` | uint64 | Worker command id when the event is tied to worker admission; **0** for control-plane-only work (lifecycle, queue, notice, compaction, `new`, many rejections). Compact never submits a worker command, so its terminals always use `command_id: 0` |
| `client_req_id` | string\|null | Echo of client’s id when known — the primary correlation key for control-plane ops |
| `state` | string | Worker admission phase: see §5. **Not** the same signal as the `idle` *event* (§5 disambiguation) |
| `status` | string | `ok` \| `cancelled` \| `rejected` \| `error` \| `stalled` |
| `code` | string\|null | Optional machine-stable reason (§4.2). Absent/`null` when not applicable |

Type-specific fields sit **alongside** the envelope (flat JSON, like today), not nested under `payload`, so existing jsonl parsers stay simple. A later v2 may nest if the flat surface gets messy.

### 4.2 Machine `code` vocabulary

Structured refusals and some notices carry a stable `code` string. **`text` is human-only** — clients must never parse prose for control flow.

| Rule | Detail |
|------|--------|
| Control-plane `rejected` | **Always** carries `code` |
| Worker-originated `rejected` | May omit `code` in v1 (e.g. generation reject with free-text `text`); codes can be added later without a version bump |
| `notice` | Optional `code`; default absent |
| Unknown codes | Clients handle generically: render `text`, do not fail the session |
| Evolution | Additive only — never rename or reuse a code for a different meaning |

**Initial control-plane codes (v1):**

| `code` | Typical event | Meaning |
|--------|---------------|---------|
| `busy` | `rejected` | Op needs worker/transcript exclusive access while a turn is live |
| `queue_full` | `rejected` | Backend prompt queue at cap (8) |
| `empty` | `rejected` | `prompt` with empty/whitespace-only `text` |
| `unknown_op` | `rejected` | Unrecognized `op` |
| `bad_version` | `rejected` | Parsed message with `v != 1` |
| `incomplete_turn` | `rejected` | Manual `compact` refused: open/incomplete root turn |
| `nothing_to_compact` | `rejected` | Manual `compact` refused: no eligible turns |
| `no_session` | `rejected` or `notice` | `save` (etc.) without a configured session file |
| `discard_failed` | `notice` | `/new`-style namespace discard partially failed (see §6.4) |

New codes may appear in later revisions; they do not require `v: 2`.

### 4.3 Client → server (`ClientMessage`)

```json
{
  "v": 1,
  "op": "prompt",
  "client_req_id": "ui-18",
  "text": "fix the failing test"
}
```

| Field | Type | Notes |
|-------|------|--------|
| `v` | number | Must be `1` |
| `op` | string | Command discriminant |
| `client_req_id` | string | Optional, ≤64 chars, for UI correlation |
| … | | op-specific fields |

Backend replies only via `ServerEvent` stream (no synchronous response objects). Each op has a defined reply set (§7): outcomes arrive as `turn_done` / `tool_done` / `session_done` / `rejected` / `error` / `notice`, echoing `client_req_id`.

**Feature-gated semantics.** Clients must key behavior off `hello.features`, not off this document’s final state. Example: until `features` contains `queue`, a busy backend answers `prompt` with `rejected` `code: "busy"` (P2); after `queue` is advertised, the same op enqueues (P4). That rule future-proofs any staged rollout beyond P4.

---

## 5. State vocabulary

Mirror of `q27_agent_worker_state` plus UI-facing phases. The envelope field `state` reports **worker admission** — roughly “will the worker accept the next command?”

| `state` | Meaning for UI |
|---------|----------------|
| `starting` | Model load / worker boot |
| `idle` | Worker free (no admitted command) |
| `generating` | Prefill or decode in flight |
| `tool_running` | Tool command admitted |
| `session_io` | Save/load/count |
| `compacting` | Control-plane compaction (synthetic — not a worker state; compaction’s internal generations stay off-stream, §6.4) |
| `stopping` | Shutdown requested |
| `error` | Worker poisoned or fatal |
| `stopped` | Clean exit soon |

**Disambiguation — `state: "idle"` vs the `idle` event.** These share a word and implementers will conflate them:

| Signal | Means |
|--------|--------|
| Envelope `state: "idle"` | Worker admission is free. May appear on a `turn_done` with `tool_call_complete: true` while the control plane is about to submit a tool — the *user* turn is not finished. |
| Event `type: "idle"` | Control plane is ready for the **next user prompt** (tool loop settled, cancel settled, etc.). This is what re-enables the input box. |

Footer chrome: use the latest event’s `state` for multiphase busy display, but treat **`type: "idle"`** as the “accepting prompts” signal. Token counters on the latest event still drive `ctx` display.

---

## 6. Server events (catalog)

### 6.1 Lifecycle (new relative to today’s jsonl)

#### `hello`

Emitted once after the worker is ready and before any turn.

```json
{
  "v": 1, "seq": 1, "type": "hello", "command_id": 0,
  "state": "idle", "status": "ok",
  "protocol": 1,
  "model_path": "…/model.q27",
  "tokenizer_path": "…/model.tok",
  "context": 32768,
  "workspace": "/path",
  "session_path": null,
  "auto_tools": true,
  "thinking": true,
  "max_tool_rounds": 8,
  "features": ["tools", "selections", "compact", "session", "queue"]
}
```

UI uses this for the header / about strip and to disable widgets for missing features. Top-level booleans report **session configuration**; `features` reports **protocol capabilities** — the two never overlap.

**Initial feature strings:** `tools`, `selections`, `compact`, `session`, `queue`.

Clients **must** gate behavior on this list (principle 8, §4.3). Until `queue` is present, busy `prompt` → `rejected` `code: "busy"`. When `queue` is present, busy `prompt` enqueues (cap 8) and emits `queue` events. Other staged capabilities follow the same pattern.

#### `idle`

Emitted whenever the control plane is ready for the next user prompt (including after `turn_done` + tool loop settled, and after cancel). This is the **control-plane readiness** signal — not the same as envelope `state: "idle"` (§5).

```json
{
  "v": 1, "seq": 99, "type": "idle", "command_id": 0,
  "state": "idle", "status": "ok",
  "ctx_used": 4200,
  "ctx_size": 32768,
  "queue_len": 0
}
```

**Post-terminal dequeue (normative).** After any command terminal settles and the control plane reaches readiness for the next user prompt, the backend dequeues the next queued prompt (if any), emits `queue`, and starts that turn — unless `queue_clear` arrived. This covers normal completion, cancel, stall, and error. It matches today’s linenoise TUI (`interactive_queue` pop at the top of every main-loop iteration, including after SIGINT-ack). **Cancel while already idle** is a no-op (the queue is empty by construction when `type: "idle"` was last emitted with nothing pending).

#### `queue`

Required when `hello.features` contains `queue`: the **backend owns the prompt queue** (§13 decision 1), so a reconnecting or scripted client can observe it. Emitted on every queue mutation: enqueue, dequeue-to-start, `queue_clear`.

```json
{
  "v": 1, "seq": 50, "type": "queue", "command_id": 0,
  "state": "generating", "status": "ok",
  "queue_len": 2,
  "items": [
    {"client_req_id": "ui-19", "preview": "also run cargo test"},
    {"client_req_id": "ui-20", "preview": "then commit"}
  ]
}
```

- `preview` is the first ≤80 chars of the prompt text, flattened to one line.
- Queue depth cap: 8 (mirrors the TUI’s `q27_tui_prompt_queue` default). A `prompt` beyond the cap is answered `rejected` with `code: "queue_full"`.
- The legacy linenoise TUI keeps its frontend-owned editor queue; that mode is not FP1 and emits no `queue` events.

#### `notice`

Generic control-plane **prose** for humans. On-stream replacement for user-facing `tui_diagf` diagnostics that are stderr-only today. A headless frontend must never need to parse stderr to explain a failed op — but **state changes still use typed catalog events** (§2.7); notices add nuance only.

```json
{
  "v": 1, "seq": 73, "type": "notice", "command_id": 0,
  "client_req_id": "ui-3",
  "state": "idle", "status": "ok",
  "severity": "error",
  "code": "no_session",
  "text": "save requires --session FILE"
}
```

- `severity`: `info` \| `warning` \| `error`. `status` keeps mirroring the worker; `severity` carries the message level.
- `code` optional (§4.2); default absent.
- Used for: `help` text, `session` report, malformed-input complaints (§3.3), warnings alongside a typed terminal (e.g. partial `new` discard — §6.4).

#### `bye`

Final event before process exit.

```json
{ "v": 1, "seq": 1000, "type": "bye", "state": "stopped", "status": "ok", "reason": "quit" }
```

`reason`: `quit` (client op) \| `stdin_eof` (frontend closed the pipe) \| `error` (fatal).

### 6.2 Generation (maps from existing worker events)

#### `state`

Phase transitions with no payload (or optional `detail` string).

```json
{
  "v": 1, "seq": 10, "type": "state", "command_id": 3,
  "state": "generating", "status": "ok"
}
```

#### `prefill_progress`

```json
{
  "v": 1, "seq": 11, "type": "prefill_progress", "command_id": 3,
  "state": "generating", "status": "ok",
  "prompt_tokens": 512,
  "cached_tokens": 400,
  "prefill_tokens": 96,
  "output_tokens": 0
}
```

UI: progress bar = `cached_tokens + prefill_tokens` vs `prompt_tokens` (same accounting as today).

#### `text_delta`

**Change from today:** prefer a UTF-8 `text` field for chat deltas; keep `data_b64` as an alternative for non-UTF-8 (should be rare for assistant text).

```json
{
  "v": 1, "seq": 12, "type": "text_delta", "command_id": 3,
  "state": "generating", "status": "ok",
  "text": "Here is the fix",
  "stream": "assistant"
}
```

| `stream` | Use |
|----------|-----|
| `assistant` | Main answer (default) |
| `thinking` | Optional later; collapse in UI (agent-tui plan item 5) |
| `raw` | Reserved |

v1 may only emit `assistant`. Thinking can stay mixed in transcript until we split streams.

#### `turn_done`

Terminal for one generation command.

```json
{
  "v": 1, "seq": 40, "type": "turn_done", "command_id": 3,
  "state": "idle", "status": "ok",
  "prompt_tokens": 512,
  "cached_tokens": 400,
  "prefill_tokens": 112,
  "output_tokens": 88,
  "tool_call_complete": true,
  "eos_reached": true
}
```

Notes:

- `tool_call_complete` drives whether the control plane will continue into tools (UI should show “calling tools…” not “done”). Envelope `state: "idle"` on that event only means the worker is free for the *next admitted command* (often the tool) — **not** that the user turn is finished (§5).
- **No full-text repeat on `turn_done`.** Clients concatenate `text_delta`s; a repeated full text can exceed the 1 MiB frame cap on long generations and duplicates bytes already streamed.
- Human-readable terminal *messages* (the reason string on `rejected` / `generation_stalled` / `error`) travel in `text`. Today those bytes ride in `data_b64`; FP1 may keep emitting `data_b64` with identical content during the transition window so v0 readers are unaffected.
- **Cancel is not a dedicated event type.** A cancelled command ends with its normal terminal — `turn_done` for a generation, `tool_done` for a tool, `session_done` for session I/O — carrying `status: "cancelled"`. This matches the worker’s existing terminal mapping (rejected→`rejected`, stalled→`generation_stalled`, error→`error`, otherwise `turn_done`/`tool_done`). After cancel settles to control-plane readiness, **queued prompts auto-dequeue** (§6.1 post-terminal dequeue).

#### `generation_stalled`

```json
{
  "v": 1, "seq": 41, "type": "generation_stalled", "command_id": 3,
  "state": "idle", "status": "stalled",
  "text": "generation stalled"
}
```

#### `rejected` / `error`

Control-plane refusal (always has `code`):

```json
{
  "v": 1, "seq": 42, "type": "rejected", "command_id": 0,
  "client_req_id": "ui-9",
  "state": "idle", "status": "rejected",
  "code": "queue_full",
  "text": "queue is full (max 8); send queue_clear or wait"
}
```

Worker-originated rejection (v1 may omit `code`):

```json
{
  "v": 1, "seq": 43, "type": "rejected", "command_id": 3,
  "state": "idle", "status": "rejected",
  "text": "context full; compaction failed"
}
```

```json
{
  "v": 1, "seq": 44, "type": "error", "command_id": 0,
  "state": "error", "status": "error",
  "text": "Metal device lost"
}
```

Control-plane `rejected` covers op-level refusals (`busy`, `queue_full`, `empty`, `unknown_op`, `bad_version`, `incomplete_turn`, `nothing_to_compact`, `no_session`, …): `command_id: 0`, `client_req_id` echoed when known, **`code` required**, `text` human-only (§4.2).

### 6.3 Tools

#### `tool_start`

```json
{
  "v": 1, "seq": 50, "type": "tool_start", "command_id": 4,
  "state": "tool_running", "status": "ok",
  "tool_kind": "read",
  "detail": "path=README.md",
  "preflight": false
}
```

New relative to the worker catalog: synthesized by the control plane at tool submission — exactly where the C TUI opens its tool card today. Preflight tools (`write_preflight`, `overwrite_preflight`, `edit_preflight`) set `preflight: true` so the UI can hide or de-emphasize them.

#### `tool_output`

Chunk of tool stdout/stderr (binary-safe).

```json
{
  "v": 1, "seq": 51, "type": "tool_output", "command_id": 4,
  "state": "tool_running", "status": "ok",
  "tool_kind": "read",
  "encoding": "utf8",
  "text": "# Quasar\n…",
  "data_b64": null
}
```

| `encoding` | Fields used |
|------------|-------------|
| `utf8` | `text` — emitted only when the chunk is valid UTF-8 |
| `base64` | `data_b64` only |

Encoding is chosen **per chunk**: chunks of one tool may mix encodings, and concatenation is byte-level after decoding each chunk. Never lossy-replace invalid bytes into `text` — corrupted bytes would flow into the model-visible transcript. Target ≤64 KiB decoded per chunk so lines stay well under the 1 MiB cap. Multiple `tool_output` events per command are normal; the collapse policy (`q27_tui_collapse_text`) moves to the frontend.

#### `tool_done`

```json
{
  "v": 1, "seq": 60, "type": "tool_done", "command_id": 4,
  "state": "idle", "status": "ok",
  "tool_kind": "read",
  "tool_exit_code": 0,
  "tool_flags": 0,
  "tool_output_bytes": 1204
}
```

#### `selection_handles`

```json
{
  "v": 1, "seq": 61, "type": "selection_handles", "command_id": 4,
  "state": "idle", "status": "ok",
  "parent_command_id": 4,
  "text": "…annotation blob as model sees it…",
  "count": 3
}
```

Structured handle arrays can wait for v1.1; v1 ships the model-visible annotation string only (enough to debug, optional to render).

### 6.4 Session / compaction

#### `session_done`

Typed terminal for **session mutations** — the session-mutation surface. Correlation for control-plane ops is **`client_req_id` echo**; `command_id` is a worker id only when a real worker session command ran (save/load/count that hit the engine). Compaction and `new` never submit a worker command → **`command_id` is always 0** for those actions.

```json
{
  "v": 1, "seq": 70, "type": "session_done", "command_id": 9,
  "client_req_id": "ui-3",
  "state": "idle", "status": "ok",
  "action": "save",
  "text": "saved work.q27agent",
  "prompt_tokens": 4200
}
```

`action`: `save` | `load` | `count` | `compact` | `new`.

##### `action: "new"`

Maps today’s `Q27_CMD_NEW`: discard session namespace, truncate transcript, recreate selection ledger. Frontend **wipes scrollback** on this event; a following `idle` arrives with `ctx_used` already reset.

```json
{
  "v": 1, "seq": 71, "type": "session_done", "command_id": 0,
  "client_req_id": "ui-5",
  "state": "idle", "status": "ok",
  "action": "new",
  "text": "new transcript; system retained"
}
```

Partial failure (`discard_rc == 2` today — namespace partially discarded, in-memory state cleared anyway):

```text
S: session_done  action=new  status=ok  client_req_id=…
S: notice        severity=warning  code=discard_failed  text="…"
S: idle          ctx_used=…   # reset
```

Typed event for the state change; notice for the nuance — the division of labor in §2.7.

##### Compaction lifecycle

Compaction is control-plane-only: **`command_id` is always 0**; client correlation is purely `client_req_id` on the manual `compact` op. Manual and auto compaction share one lifecycle that **must not leave `state=compacting` unclosed**.

```text
S: state          command_id=0  state=compacting
S: session_done   action=compact  status=ok|error  command_id=0
```

| Path | Terminal |
|------|----------|
| Manual `compact`, pre-flight refusal (incomplete turn, nothing eligible) | `rejected` with `code: "incomplete_turn"` or `"nothing_to_compact"` — **no** compacting lifecycle |
| Manual `compact`, mid-flight failure (empty summary, anchor exceeds context) | `state` `compacting` → `session_done` `action=compact` `status=error` + `text` |
| Manual `compact`, success | `state` `compacting` → `session_done` `action=compact` `status=ok` |
| Auto-compact failure (e.g. inside `prepare_turn`) | `state` `compacting` → `session_done` `action=compact` `status=error` `command_id=0` → then the **turn’s** `rejected` (e.g. `text: "context full; compaction failed"`) as the correlated terminal for the aborted user turn |
| Auto-compact success | `state` `compacting` → `session_done` `action=compact` `status=ok` `command_id=0`, then turn proceeds |

Compaction’s internal generations (the summary pass and the hidden acknowledgement) emit **no** `text_delta` / `prefill_progress` / terminals on FP1 — the same suppression as today’s internal path, which exists so no terminal can falsely claim durable publication early. The summary text never leaves the control plane. What changes versus today is that the lifecycle itself becomes visible: auto-compact mid-turn no longer looks like a prefill hang, and the `ctx_used` drop on the following `idle` is explained.

---

## 7. Client messages (catalog)

Minimum set for a beautiful TUI. Maps to existing slash/colon commands and stdin prompts.

**Admission while busy.** Today commands are only read between turns, which implicitly gates everything; with asynchronous NDJSON input the rule must be explicit:

| Op | While a turn/tool is running |
|----|------------------------------|
| `prompt` | If `features` has `queue`: accepted → backend queue (`queue` event); past cap → `rejected` `code=queue_full`. Without `queue`: `rejected` `code=busy` |
| `cancel` | Accepted; cancels the **active command only** — queued prompts survive and **auto-dequeue after settle** (§6.1). Cancel while idle is a no-op |
| `quit` | Accepted; drains to `bye` |
| `queue_clear` | Accepted only when `features` has `queue`; drops pending prompts (`queue` event). Else `rejected` `code=unknown_op` |
| `session`, `help` | Accepted anytime (control-plane queries; no worker admission). Reports are **point-in-time snapshots** and may be stale mid-turn; not a barrier |
| `save`, `compact`, `new`, `tool` | `rejected` with `code: "busy"` (need worker admission or mutate the live transcript) |

### 7.1 Required for interactive client

| `op` | Fields | Backend action / replies |
|------|--------|--------------------------|
| `prompt` | `text: string` | Enqueue or start a user turn → generation events. Empty/whitespace-only `text` → `rejected` `code=empty`. Multiline is plain `\n` inside the JSON string — no secondary framing |
| `cancel` | _(none)_ | Interrupt current generation/tool via the same alive-check path SIGINT uses today → normal terminal with `status: cancelled` → control-plane `idle` → optional queue dequeue |
| `quit` | _(none)_ | Request clean shutdown → `bye` |
| `queue_clear` | _(none)_ | Drop all pending queued prompts → `queue` `queue_len=0` (requires feature `queue`) |

```json
{ "v": 1, "op": "prompt", "client_req_id": "ui-1", "text": "explain src/server.cu" }
{ "v": 1, "op": "cancel", "client_req_id": "ui-2" }
{ "v": 1, "op": "quit" }
```

### 7.2 Session / maintenance (map from `/` commands)

| `op` | Slash today | Replies |
|------|-------------|---------|
| `save` | `/save` | `session_done` `action=save` \| `rejected` `code=no_session` \| `rejected` `code=busy` |
| `compact` | `/compact` | Pre-flight: `rejected` `code=incomplete_turn` \| `nothing_to_compact`. Else `state` `compacting` + `session_done` `action=compact` `status=ok\|error` (`command_id=0`) |
| `session` | `/session` | `notice` `severity=info` with report text (path, snapshot, ctx, turns). Point-in-time; may be stale mid-turn; not a barrier |
| `new` | `/new` | `session_done` `action=new` `status=ok` (`command_id=0`) + optional `notice` `code=discard_failed` \| `rejected` `code=busy` |
| `help` | `/help` | `notice` `severity=info` carrying the help text. Server-side and authoritative (static per protocol version); a frontend *may* render `/help` locally instead |

```json
{ "v": 1, "op": "save", "client_req_id": "ui-3" }
{ "v": 1, "op": "compact", "client_req_id": "ui-4" }
{ "v": 1, "op": "new", "client_req_id": "ui-5" }
```

### 7.3 Manual tools (map from `/read` `/search` `/shell`)

| `op` | Fields | Replies |
|------|--------|---------|
| `tool` | `kind`: `read`\|`search`\|`shell`\|… , plus kind-specific args | `tool_start` / `tool_output`* / `tool_done` \| `rejected` `code=busy` \| `notice` (invalid args) |

```json
{
  "v": 1,
  "op": "tool",
  "client_req_id": "ui-6",
  "kind": "read",
  "path": "README.md"
}
```

```json
{
  "v": 1,
  "op": "tool",
  "client_req_id": "ui-7",
  "kind": "search",
  "path": "src",
  "needle": "q27_agent_worker_submit"
}
```

```json
{
  "v": 1,
  "op": "tool",
  "client_req_id": "ui-8",
  "kind": "shell",
  "command": "make -n agent"
}
```

Manual tools share the same `tool_*` event lifecycle as model-driven tools.

### 7.4 Optional v1.1

| `op` | Purpose |
|------|---------|
| `set` | `thinking`, `auto_tools`, sampling — only if safe mid-session |
| `queue_cancel_item` | `client_req_id` of queued prompt |

---

## 8. Compatibility with today’s `--output-format jsonl`

Current emitter (`print_json_event`):

```json
{
  "seq": 1,
  "command_id": 1,
  "type": "text_delta",
  "state": "generating",
  "status": "ok",
  "data_b64": "…",
  "prompt_tokens": 0,
  "cached_tokens": 0,
  "prefill_tokens": 0,
  "output_tokens": 0,
  "tool_call_complete": false,
  "eos_reached": false,
  "tool_kind": "none",
  "tool_exit_code": 0,
  "tool_flags": 0,
  "tool_output_bytes": 0
}
```

**Migration rule:**

| Today | FP1 |
|-------|-----|
| no `v` | treat missing `v` as **legacy jsonl** (v0) |
| `data_b64` only | FP1 adds `text` where UTF-8; may still emit `data_b64` |
| no `hello`/`idle`/`bye` | FP1 requires them for interactive clients |
| op failures on stderr | FP1 surfaces them as `notice` / `rejected` with `code` where control-plane |
| no `code` field | FP1 control-plane refusals always include `code`; worker rejects may omit |
| stdin = raw prompt lines | FP1 stdin = NDJSON `ClientMessage` only when `--frontend-proto 1` |

**Feature advertisement, not doc finality.** A client must not assume the full §7 matrix just because `v: 1`. Behavior that is still staged (queue, …) is discoverable only via `hello.features`. Missing feature → the safer/P2 behavior (`rejected` `busy` for prompt-while-busy, etc.).

The in-repo v0 consumer base is small — the agent unit tests plus operator scripts; the eval/replay tooling reads server traces and claude stream-json, not agent jsonl. So v0 read-only support in the Rust client is a convenience for shipping against current builds, not a long-term compatibility burden.

A Rust client should support **v0 read-only** (watch jsonl from `--prompt` runs) and **v1 interactive** (full duplex). That lets us ship the TUI against current builds before the control channel exists.

---

## 9. Sequencing (normative examples)

### 9.1 Simple answer

```text
S: hello
S: idle  ctx_used=…
C: prompt text="ping"
S: state          command_id=1  state=generating
S: prefill_progress …
S: text_delta     text="pong"
S: turn_done      eos_reached=true  tool_call_complete=false
S: idle
```

### 9.2 Tool loop (model-driven)

```text
C: prompt text="read README and summarize"
S: state / prefill / text_delta…          command_id=1
S: turn_done  tool_call_complete=true     command_id=1
S: tool_start  tool_kind=read             command_id=2
S: tool_output …
S: tool_done   exit=0                     command_id=2
S: selection_handles                      command_id=2  (optional)
S: state / prefill / text_delta…          command_id=3
S: turn_done  tool_call_complete=false    command_id=3
S: idle
```

UI rule: **do not clear the turn chrome on `turn_done` if `tool_call_complete`** until a later `type: "idle"` (or until a following generation’s first `text_delta` if you prefer continuous transcript). Envelope `state: "idle"` on that `turn_done` is **not** sufficient (§5).

### 9.3 Cancel mid-generation (with queued follow-up)

```text
C: prompt text="first"     client_req_id=ui-1
C: prompt text="second"    client_req_id=ui-2   # queued (features includes queue)
S: queue  queue_len=1  items=[{client_req_id:"ui-2", preview:"second"}]
S: text_delta …            command_id=1
C: cancel
S: turn_done   status=cancelled   command_id=1
S: idle
S: queue  queue_len=0                              # post-terminal dequeue
S: state …  command_id=2                           # second starts
```

Cancel always surfaces as the command’s normal terminal type with `status: cancelled` — there is no dedicated `cancelled` event (§6.2). Queued prompts survive cancel and run after settle (same as today’s linenoise main loop). Cancel while idle: no-op.

### 9.4 Queue-while-busy

**Backend-owned queue (normative when `features` contains `queue`):**

```text
C: prompt text="first"    # starts immediately
C: prompt text="second"   # queued
S: queue  queue_len=1  items=[{client_req_id:"ui-20", preview:"second"}]
S: turn_done …            # first finishes (+ tools)
S: idle
S: queue  queue_len=0
S: state …                # second starts
```

```text
C: queue_clear            # any time
S: queue  queue_len=0
```

Without feature `queue`, the second `prompt` while busy is `rejected` `code=busy` instead.

The legacy linenoise TUI keeps its frontend-owned editor queue (`q27_tui_prompt_queue`, cap 8); that mode is not FP1 and emits no `queue` events.

### 9.5 Manual compact + new

```text
C: compact  client_req_id=ui-4
S: rejected  code=nothing_to_compact  client_req_id=ui-4   # pre-flight

C: compact  client_req_id=ui-4
S: state  state=compacting
S: session_done  action=compact  status=ok  command_id=0  client_req_id=ui-4
S: idle  ctx_used=…   # lower after success
```

```text
C: new  client_req_id=ui-5
S: session_done  action=new  status=ok  command_id=0  client_req_id=ui-5
S: notice  severity=warning  code=discard_failed   # only if partial discard
S: idle  ctx_used=…   # reset; UI wiped scrollback on session_done
```

---

## 10. What a Rust TUI actually implements

Thin client. Suggested modules:

```text
q27-tui/
  src/
    main.rs          // spawn or attach backend
    proto.rs         // ServerEvent / ClientMessage serde
    backend.rs       // child process + stdin/stdout lines + stderr logfile
    app.rs           // state machine: Idle | Streaming | Tool | Error
    ui.rs            // ratatui: scrollback, footer, tool cards, input
```

**App state** (derived only from events):

```text
struct Model {
  phase: Phase,
  ctx_used, ctx_size,
  prefill_done, prefill_total,
  gen_tokens, /* optional tps timers local */
  scrollback: Vec<Block>,  // User / Assistant / Tool / Notice
  active_tool: Option<ToolCard>,
  queue_len: u32,
  last_error: Option<String>,
}
```

**Render mapping:**

| Event | UI effect |
|-------|-----------|
| `hello` | header meta |
| `prefill_progress` | footer bar |
| `text_delta` | append to assistant bubble |
| `tool_start` | open card |
| `tool_output` | append/collapse body |
| `tool_done` | close card with exit |
| `turn_done` | finalize assistant block if `!tool_call_complete`; else keep “tools…” chrome |
| `type: idle` | enable input; footer “idle” (not merely `state: idle`) |
| `session_done` `action=new` | wipe scrollback |
| `session_done` `action=compact` | footer clear compacting; refresh ctx from following `idle` |
| `queue` | footer `queue N` |
| `notice` | toast / status line by severity; never sole state-change signal |
| `state` `compacting` | footer “compacting…” |
| `rejected` | toast from `code` + `text`; re-enable input if no turn active |
| `error` / `generation_stalled` | error strip |

**Input:** Enter → `prompt`; Ctrl-C → `cancel` (if busy) else clear line; `/save` etc. → structured `op`. Slash parsing lives in the frontend; there is deliberately **no `raw_line` op** in v1 — the backend surface stays the ten ops above. Frontend-local `/help` rendering is allowed, but the server `help` op is authoritative for the op list.

### 10.1 Process model

```sh
# Frontend owns the child
q27-tui -- MODEL.q27 MODEL.tok --auto-tools --session work.q27agent
# expands to:
#   spawn: q27-agent … --frontend-proto 1 …
#   wire:  TUI stdin←→ child, or TUI owns PTY-free pipes
```

- Redirect the child’s **stderr to a logfile** (ASan, “loading model…”, fatal diagnostics live there by design, §3.1) and surface the log path in the UI. If the child exits without `bye`, show the log tail.
- If the TUI is killed, its stdin pipe closes and the backend shuts down on `stdin_eof` (§3.1) — no orphan agents.

Alternate: user runs backend in one pane and TUI attaches later (v1.1 socket). Not required for first beauty milestone.

---

## 11. Explicit non-goals (v1)

- Replacing `Q27AGT2` / `Q27SNAP1` with Tau-style JSONL session graphs  
- Provider abstraction / multi-model router  
- Streaming partial tool-call JSON (we parse after generation today)  
- Extension plugins / npm-style pi packages  
- Multiplexed multi-agent tabs (one worker admission)  
- Making the C linenoise TUI speak FP1 before the headless path exists  

---

## 12. Implementation plan (suggested)

| Step | Work | Exit gate |
|------|------|-----------|
| **P0** | Spec freeze (this doc) | ✅ Reviewed; five pushbacks folded |
| **P1** | `--frontend-proto 1`: emit `v`, `ts_ms`, `hello` (features without `queue` until P4), `idle`, `bye`; add `text` on deltas/terminals (keep `data_b64`); envelope may carry `code` | ✅ `q27_agent_frontend.*` + unit tests; v0 jsonl path unchanged |
| **P2** | **stdin reader thread** + NDJSON ops `prompt`, `cancel`, `quit`; control-plane `rejected` always has `code`; prompt-while-busy → `code=busy` (no `queue` feature yet); stdin EOF → `bye` `stdin_eof` | ✅ Headless FP1 interactive path; cancel via alive-check |
| **P3** | `save` / `compact` / `new` / `session` / `help` / `tool`; `notice`; busy matrix; `session_done` `action=new`; compact pre-flight codes + mid-flight `session_done` | ✅ parse + handlers + emit; TUI slash → structured ops; busy matrix (session/help anytime) |
| **P4** | Advertise `queue`; backend queue + `queue_clear` + post-terminal dequeue; `tool_start`; compaction lifecycle events (`command_id=0`) | ✅ hello.features includes queue; mid-turn enqueue; tool_start; auto-compact lifecycle |
| **P5** | Rust `q27-tui` MVP + polish: scrollback, footer, tool cards, prompt, cancel; markdown; thinking collapse; themes; gates on `hello.features` | ✅ `experiments/q27-tui` (md/theme/app/ui); soak via `q27 agent b1` |
| **P6** | _(merged into P5)_ | — |

**Operator notes (post-P5):**

- **Cancel:** TUI `Esc` / `Ctrl-C` / `/cancel` → FP1 `cancel` + SIGINT dual-path; agent alive-check mid-decode. Terminal is `turn_done`/`tool_done` with `status: cancelled`.
- **Thinking UI:** `q27-tui` defaults to **collapsed** finished think blocks; empty-input `t` toggles full trace. Open (streaming) think is always fully visible.
- **Thinking token cap (agent-side, not a model feature):** `--max-think-tokens N` / env `Q27_AGENT_MAX_THINK_TOKENS` (default **off**). Counts tokens while `<think>` is open; on hit, **force-injects `</think>\n\n` into stream+KV and continues free generation** for the answer (same close the `--no-think` prefill uses). Hard-stops only if no room remains. Whole-turn bound remains `--max-tokens` / `Q27_AGENT_MAX_TOKENS` (`auto` ≤ 16384). Disable think with `--no-think` / `Q27_AGENT_NO_THINK=1`. Documented in `packaging/README.md` and `experiments/ds4-agent/README.md`.

**P2 is the risk step, not P5.** Today cancel is signal-driven (SIGINT → `interrupted` latch → alive-check fails) and the control plane reads stdin only *between* turns; the queue-while-busy machinery lives in the linenoise editor and is TTY-only. FP1 needs a **stdin reader thread** that parses NDJSON ops and posts them to a bounded pending list:

- `cancel` / `quit` set flags the alive-check consults — the same path the SIGINT latch takes — so they act mid-decode without waiting for the drain loop.
- All other ops are drained by the control plane between turns and, during long turns, from the event wait via the existing `q27_agent_worker_next_event_timeout`, so busy-rejections and queue accepts stay responsive.
- The reader thread never emits protocol events; all stdout emission stays serialized on the control-plane thread so `seq`/stream order holds (§3.3).
- EOF on stdin posts `quit` (`reason: stdin_eof`).

**Do not start P5 until P2 works.** A pretty TUI against an incomplete control channel is how you get two half-agents.

---

## 13. Decisions (resolved at P0 review)

1. **Queue owner: backend** for FP1 interactive (when `features` includes `queue`), so reconnecting a TUI can resync and scripts can observe the queue. `queue_clear` ships with the `queue` feature. **Post-terminal dequeue** after cancel/stall/error/success is normative and matches today’s linenoise main loop. Cancel while idle is a no-op. Legacy linenoise keeps its frontend-owned queue; that mode is not FP1.
2. **Cancel terminal: the command’s normal terminal with `status: cancelled`** (`turn_done` / `tool_done` / `session_done`). Matches the worker; no dedicated `cancelled` event.
3. **UTF-8 strictness on `tool_output`: per-chunk** — `utf8` when valid UTF-8, `base64` otherwise; never lossy-replace.
4. **`prompt` framing:** multiline via `\n` inside the JSON string; empty/whitespace-only → `rejected` `code=empty`. No secondary framing.
5. **Auth / untrusted frontend:** out of scope (single operator); protocol is local pipes only.
6. **Linenoise future:** subprocess-only FP1 for Rust; the C TUI may call control functions directly forever and keeps signal-driven cancel. FP1 cancel is message-driven through the reader thread (§12 P2).
7. **Machine `code` field** on control-plane `rejected` (required) and optional on `notice`. Initial vocabulary §4.2. Worker-originated `rejected` may omit `code` in v1. Unknown codes rendered generically via `text`. Additive only — never rename.
8. **State changes = typed events; notices = prose.** `new` → `session_done` `action=new` (optional `notice` `code=discard_failed` for partial discard). Manual compact pre-flight → `rejected` with codes; mid-flight / auto failure → `session_done` `action=compact` `status=error` then (auto) the turn’s `rejected`. Compact always `command_id=0`; correlate with `client_req_id`.
9. **Clients key off `hello.features`**, not the spec’s final matrix. Until `queue` is advertised, busy `prompt` → `code=busy`.

---

## 14. One-page cheat sheet

**Server types:**  
`hello` · `idle` · `bye` · `queue` · `notice` · `state` · `prefill_progress` · `text_delta` · `turn_done` · `generation_stalled` · `rejected` · `error` · `tool_start` · `tool_output` · `tool_done` · `selection_handles` · `session_done`

**Client ops:**  
`prompt` · `cancel` · `quit` · `queue_clear` · `save` · `compact` · `session` · `new` · `help` · `tool`

**`session_done.action`:** `save` · `load` · `count` · `compact` · `new`

**Control-plane `code`s (v1):**  
`busy` · `queue_full` · `empty` · `unknown_op` · `bad_version` · `incomplete_turn` · `nothing_to_compact` · `no_session` · `discard_failed`

**Invariants:**

- Terminals carry `status`; cancel = normal terminal + `status=cancelled`. No dedicated `cancelled` event.
- Control-plane `rejected` always has `code`; `text` is human-only.
- `state: idle` ≠ event `type: idle` (worker free vs accept next user prompt).
- After any terminal settles to control-plane readiness, dequeue next prompt (unless `queue_clear`).
- Compact / `new`: `command_id=0`; correlate with `client_req_id`.
- Internal compaction generations never emit deltas; lifecycle always closes with `session_done` `action=compact`.
- Gate client behavior on `hello.features`.
- `seq` strictly increasing; stream order == seq order.

**Golden rule:**  
Metal, tools, SNAP1, compaction stay in C.  
If the Rust side wants to know something, **put it on the event stream** — as a **typed** event when the UI must act, as `notice` when a human should read.

---

## 15. Relationship to Tau

```text
Tau:   provider stream → AgentEvent → harness loop → session JSONL → Textual
q27:   Metal sinks    → ServerEvent → C tool loop  → SNAP1         → Ratatui/linenoise
                 \________________ FP1 ________________/
```

Same *shape* of boundary. Different *brain*. We borrow Tau’s frontend isolation, not their Python stack or multi-provider core.
