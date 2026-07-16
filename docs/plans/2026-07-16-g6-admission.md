# G6 — multislot admission budget + dedicated 503 (last Phase 1 residue)

Status: PRE-REGISTERED before the gate run (results appended below).
Parent: 2026-07-15-multislot-phase1.md contract item 3 + gate G6
("configured budget that fits exactly one slot admits one and 503s the
overflow with the documented error body") and the review-2 residue list
(dedicated 503 for queue overflow; snapshot-capacity and GQA-partial-peak
terms in the admission budget).

## What the current admission misses (measured composition, T2, ctx 2048)

The engine ctor checks KV bytes only (~134 MB/slot fp16 @2048) against
`recommended_working_set_size()/2`. The contract's other terms, sized from
the ctor's own constants:

- **Fixed per-engine state ≈ 350 MB**: chunk buffers (96-wide, ~42 MB) +
  activation-quantized copies (~3 MB) + `clogits_` (47.7 MB at
  VERIFY_CHUNK_MAX=48, grown by lever 2) + gdn_replay parks (~95 MB) +
  GDN recurrent/ring live state (~157 MB) + logits/misc (~2 MB).
- **Snapshot capacity × snapshot bytes**: one snapshot at full context ≈
  GDN recurrent/ring copy (~157 MB) + attention KV copy (16×2×ctx×row) +
  hidden/logits (~1 MB) — ≈ 292 MB at ctx 2048 fp16.
- **GQA partial peak** (backend-shared, counted once, not per slot):
  PREFILL_CHUNK_MAX × N_HEAD × ceil(ctx/1024) × 258 × 4 — 4.8 MB at 2048,
  76 MB at 32K.

So a second slot at ctx 2048 with one snapshot really costs ~776 MB, not
~134 MB. The engine's KV-only check stays (defense in depth); the server
gains the full check before constructing each additional slot.

## Changes

1. `MetalEngine` exposes the accounting beside the allocations it mirrors:
   `kv_reserved_bytes()`, `snapshot_bytes()` (worst case at max context),
   `has_mtp()`, static `fixed_state_bytes()` and
   `gqa_partial_peak(context)`.
2. Runtime slot admission: additional slots admit only if
   `(n+1) × (kv + fixed + cache_entries × snapshot) + partial_peak ≤
   budget`, budget = `Q27_METAL_BUDGET_MB` env override (test hook,
   documented) or `recommended_working_set_size()/2`. Slot 0 is always
   constructed (a budget below one slot degrades to one, never to zero).
   Degradation logs the accounting.
3. Queue overflow throws a dedicated `ServerOverloaded`; the non-stream
   handlers return **HTTP 503** with body
   `{"error":{"message":"server overloaded: request queue is full",
   "type":"overloaded_error"}}`. (Streaming requests that hit overflow
   keep the SSE error-event path — status is already committed by then;
   recorded as known behavior, not a gap.)

## G6 gate (multislot_gates.py, own server launch)

- Launch with `Q27_METAL_BUDGET_MB=1000` (fits exactly one slot per the
  arithmetic above): `/stats` must report `slots == 1` — the degradation
  actually fired (anti-vacuous: the default suite asserts slots == 2, so
  both directions of the admission decision are exercised).
- Saturate: 12 concurrent requests against 1 slot × QUEUE_MAX=8 tickets:
  at least one request must receive **503 with `type ==
  "overloaded_error"`**, at least one must complete normally, and every
  request must terminate (no wedge) under a shared deadline.

---

## Results (appended post-gate, same day)

**G6 PASS — and the gate found a real hole before it passed.** First run:
slots=1 degradation fired correctly under the 1000 MB budget, but ZERO
503s appeared under 12-over-(1 slot + 8 tickets). Diagnosis: with the
httplib pool at 8 workers, at most 7 requests can wait on tickets while
one generates, so the ticket spread could never reach QUEUE_MAX=8 — **the
queue-overflow error path (the old 400 and the new 503 alike) was
unreachable dead code since the day it landed.** Fix: 16 workers
(> QUEUE_MAX + slots), keeping the 32-connection accept queue as the
outer bound. Second run: **3 × 503 with the documented overloaded_error
body, 9 requests completed with byte-identical greedy text, no wedge,
slots=1 confirmed**; the full suite then re-ran green
(G1/G1s/G2/G3/G4/Gq/G5/G6).

The falsifiability pattern held exactly as intended: the anti-vacuity
design (assert at least one 503, not just absence of errors) is what
surfaced the dead path — a gate that merely checked "no wrong statuses"
would have passed against broken load-shedding.
