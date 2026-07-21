# Abandoned-request cancellation (Metal server) — pre-registration (2026-07-17)

## The debt

Registered during the 2026-07-17 "not working" incident: a disconnected
client's non-streaming request has NO liveness check until its final write,
so it runs a multi-minute prefill + full generation for nobody, and every
request queued behind it on the single 128K slot waits (or times out and
retries, deepening the pile). Streaming requests probe `sink.is_writable()`
per emitted piece, but that starts only at the FIRST token — queue wait and
the whole prefill run blind on every endpoint shape. CUDA's non-streaming
path has the same blind spot (plain `set_content`, conductor cancel keys
off `on_token`), so this is a beyond-CUDA fix, like the chat tool_calls.

## Design

- **Vendored httplib patch** (2 lines, marked `q27 patch`): `Request::sock`
  carries the connection fd, set in `Server::process_request`. Probe =
  `httplib::detail::is_socket_alive(sock)` — zero-timeout select +
  MSG_PEEK; a pipelined follow-up request reads as alive (correct).
- **`Runtime::run()` gains a `live` callback**, probed at the two blind
  phases:
  - queue wait: `slot_free_.wait` becomes `wait_for` (250 ms tick); a dead
    ticket throws `ClientGone` — the TurnPass destructor already passes
    the turn, so the queue self-evacuates in arrival order.
  - prefill: probe before each chunk quantum. On death with an armed
    snapshot target (hint/auto — i.e. the big-prompt cases), save state at
    the current position FIRST (no-logits snapshot, same machinery as the
    boundary save), so a client-timeout retry resumes from disk instead of
    livelocking cold. Then throw `ClientGone`.
- **Generation**: non-streaming emit lambdas probe `is_socket_alive` every
  16 tokens (streaming already cancels via `is_writable` per piece).
- Handlers catch `ClientGone` and return without a body (the connection is
  gone; nothing to write). Metrics gain `cancelled_queue`,
  `cancelled_prefill` counters (existing `client_gone` decode-path
  accounting unchanged).

## Pre-registered gates (live T2 server, small prompts — no quiet needed)

1. **Prefill cancel:** POST a non-streaming ~2k-token prompt (multi-second
   T2 prefill), kill curl at ~1 s. `cancelled_prefill` increments and a
   probe request submitted immediately after answers without waiting for
   the dead request's full prefill+generation (wall-clock bound: probe
   completes in < the dead request's would-be runtime).
2. **Queue cancel:** occupy the slot with a long request, queue a second,
   kill the second while queued. `cancelled_queue` increments; the slot
   holder is unaffected.
3. **Snapshot-on-cancel:** big prompt (≥ snap_auto_min) killed mid-prefill
   → `disk_saves` increments; an identical retry restores the prefix
   (`prefix_hit` ≥ saved position) and completes fast.
4. **No-regression control:** agentic parity gate battery (G2–G7) green on
   the patched binary; pi read-tool smoke passes.

Kill criteria: false-positive cancels (gate 4 failing — live clients read
as dead) parks the probe behind an env flag and records why.

## RESULTS (2026-07-17 midday) — SHIPPED, all gates PASS

- Gate 1: 2 s kill of a ~90 s non-streaming prefill → `cancelled_prefill`
  +1 within one chunk; follow-up probe answered in **1.15 s** (previously
  it would have waited out the dead request's entire prefill+generation).
- Gate 2: queued request killed behind a live slot holder →
  `cancelled_queue` +1, holder unaffected.
- Gate 3: big-prompt kill banked a snapshot mid-prefill; the identical
  retry RESTORED it (disk_hits +1) and its own kill banked a deeper one —
  each timeout-retry cycle makes forward progress instead of livelocking
  cold. (Unplanned extra: gate 1's prompt crossed the 4096 auto threshold,
  so even its 2-second kill banked a prefix that gate 2's holder restored.)
- Gate 4: agentic parity battery ALL PASS on the new binary; pi read-tool
  smoke instant on the warm snapshot.

Codex round — 1 P1, 2 P2, 1 P3, all adopted:
- **P1 (real, caught pre-deploy): mid-queue cancel broke FIFO order** —
  TurnPass unconditionally advanced `slot_serving_`, so a cancelled ticket
  k with serving at k-2 skipped a live waiter and wedged the queue. Fix:
  `cancelled_tickets_` set + disarmed TurnPass; every wait predicate and
  new arrival fast-forwards serving past consecutively cancelled tickets
  (the arrival-time drain also keeps stale cancels from falsely 503ing on
  the QUEUE_MAX check). Re-gated with a 3-deep queue: mid-queue kill, the
  ticket behind it still served.
- P2: ClientGone in the non-streaming guards now sets **499** (client
  closed request) — a probe false-negative can never surface as an empty
  200. P2: `is_socket_alive` reads a half-closed write side as dead;
  accepted as documented (no real HTTP client half-closes; the 499 bounds
  the damage). P3: cancel-save made best-effort (logged, never converts a
  cancel into the generic error path).

**Rider (same deploy): `Q27_METAL_MAX_TOKENS_DEFAULT`** — pi sends
`max_tokens:null`, and the CUDA-parity 256 default truncated the operator's live
agent turn ("maximum output token limit"). The env knob overrides the
per-endpoint defaults for requests that omit max_tokens; explicit client
values and the context preflight clamp still win. Serving line now carries
`Q27_METAL_MAX_TOKENS_DEFAULT=16384` (= pi's advertised maxTokens);
verified 649 streamed tokens on a no-max_tokens request.

Registered residue: MTP/serial-path prefill (single coarse quantum) stays
unprobed — documented Phase 1 limitation; CUDA server still has the
non-streaming blind spot (port `Request::sock` + guards at next merge).
