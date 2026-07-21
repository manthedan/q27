# Multislot Phase 1 — two-slot serving on one mapping (2026-07-15)

Status: IMPLEMENTED same night — engine quantum surface + slot runtime
landed; gates G1/G2/G3/G5 (and a sampled-arm G1s) pass on the T2 artifact.
G4 (constraint isolation) landed the next morning: constrained tool decode
on one slot beside plain decode on the other, both byte-identical to solo;
constraint action is proven, not assumed — the declared tool name is one
the model never produces naturally (the few-shot demonstrates get_weather,
the grammar declares zz_paris_weather_probe; the probe name can only appear
via the mask), and server stderr must show the engage/close pairs
(vacuous-gate lesson applied). G6 (admission) landed the same day
(2026-07-16-g6-admission.md): full admission accounting (KV + fixed
engine state + snapshot capacity + GQA partial peak, Q27_METAL_BUDGET_MB
hook), dedicated 503 overloaded_error — and the gate exposed that the
queue-overflow error path had been unreachable dead code behind the
8-worker pool (now 16 > QUEUE_MAX + slots). Phase 1 contract: fully
discharged.
Prereqs met: round-2 P0s #1/#2/#3 fixed and gated (`dc8f017` + exit-code
hardening), triage ordering explicitly unblocks "v1 multislot
state/scheduling" as the next serving workstream (review-3 Q3: P0 fixes →
v1 multislot → N=2 slot-batched linears → verify_lanes → learned drafting).

## Implementation notes (added post-landing)

- Engine surface: `prefill_chunk` (prefill() now drives its own loop
  through it), `mtp_round` (extracted verbatim; `stream_mtp_batched` and
  `generate_mtp_batched` now sit on it, so whole-run and per-quantum MTP
  share one code path), `sample_from_logits` (request-owned RNG).
  Pre/post-refactor A/B: greedy fp16 and greedy turbo3 48-token runs
  byte-identical. **MTP arm NOT coverable on this rig** — the T2 Bonsai
  artifact carries no MTP layer (the first "identical" readout was two
  empty error outputs; vacuous-gate lesson re-learned, retracted). The
  measured MTP quantum gate (G2-MTP + G1-MTP) is queued for the 24 GB
  machine's artifact.
- Server: `Runtime::run()` signature unchanged (all six HTTP call sites
  untouched); slots + FIFO ticket-lock lease + width policy + /stats
  live inside Runtime. `host2dev` moved into Slot (per-engine mask pool);
  the host-side mask cache stays shared.
- **Fairness finding (measured):** with a plain `std::mutex` lease, the
  decoding slot's release/deliver/reacquire loop starved the competing
  slot for a whole generation — 5,491 ms gate wait vs the ~2 s quantum
  bound. The lease is a FIFO ticket lock for exactly this reason, and G5
  asserts max busy-arrival gate wait ≤ 3 s (bound proven able to fail by
  the pre-ticket measurement).
- Deviations from the sketch below: bounded queue = a ticketed FIFO
  admission (ticket spread caps at QUEUE_MAX=8 → error through the shared
  400 path; dedicated 503 is a follow-up) plus a bounded httplib task
  queue (8 workers, 32 queued connections) so excess load is rejected at
  the accept side rather than piling up unbounded (codex P1); admission =
  engine-ctor budget check with graceful slot-count degradation
  (snapshot-capacity and GQA partial-peak terms remain TODO with G6);
  MTP-warm prompt ingestion is one coarse quantum (engine-internal
  token-serial loop), documented and visible in the wait metrics.
- Codex round on the landing commit (1 P1 + 3 P2, all fixed same night):
  accept-side queue bounded (above); `--slots` capped at 2 — the
  one-competing-quantum guarantee doesn't extend to 3+ slots without a
  scheduler/width/stats redesign; slot handoff ticketed — a bare
  condition_variable lets a newcomer barge past an awakened waiter;
  constrained decode pre-materializes the advanced grammar state's mask
  outside the lease (ToolMaskCache::get simulates the whole vocabulary on
  a miss; the once-per-call engage path stays under the lease, bounded),
  with a dedicated innermost mask_mutex_ guarding the shared host cache.
- Second codex round: slot release now notify_all (notify_one could wake
  a non-serving ticket that re-sleeps, wedging the queue with an idle
  slot). The queue-pressure gate Gq (5 requests over 2 slots) covers the
  wedge class, and it exposed a metrics conflation fixed the same night:
  gate wait (admission → first lease; one-quantum bound, measured max
  83.5 ms under load) is now split from queue wait (arrival → admission;
  bounded only by QUEUE_MAX generations, measured max ~8 s under 5-over-2
  pressure, reported per arrival phase and deliberately not bounded).

## What Phase 1 is (and is not)

Phase 1 is **state isolation + scheduling**: two independent request slots,
each its own `MetalEngine` on one `MetalEngine::Shared` mapping (weights
mapped once — the two-engines-one-mapping pattern already proven by the
`--kl-kv` and `--chunk-parity` instruments), interleaved on the GPU at
scheduling-quantum boundaries. Throughput from batching the two slots'
linears (honest ceiling S₂ = 2/(α·f + 2(1−f)) ≈ 1.74× at f≈0.85, α=1) is
**Phase 2** — it lands on top of this scheduling substrate and is measured
by its own pre-registered gates. Phase 1's win is concurrency and latency
honesty, not aggregate tok/s: two clients make progress without
head-of-line blocking behind a long generation.

## Contract (review-2 finding 6, adopted verbatim)

1. **Prefill width is a runtime policy, not an engine constant.**
   `PREFILL_CHUNK_MAX=96` stays the engine ceiling; the server picks the
   dispatch width per chunk: 96 when the other slot is idle, 48 under
   two-slot load, 12–24 when a waiter exists (a slot with a runnable
   decode step or an admitted-but-unstarted prefill). A 96-token chunk at
   ~48.8 tok/s holds the GPU ~2 s — yield-after-chunk alone cannot deliver
   short waits, the width policy is what does.
2. **Honest guarantee: at most one active scheduling quantum.** The wait
   metric reported per request is gate-wait, split by arrival phase:
   arrival-during-decode, arrival-during-prefill, arrival-during-MTP-verify.
   No "~one decode token" claim anywhere.
3. **Admission budget** = fixed engine state (weights are shared; per-slot
   KV + GDN + logits + scratch) + configured snapshot capacity × snapshot
   bytes + GQA partials, checked against a configured device budget before
   constructing slot 2. Reservation accounting already RAII-safe (P0 #2).
   [E2, 2026-07-16: the GQA partial buffer moved from a backend-shared
   lazily-grown-to-peak allocation to per-engine eager allocation folded
   into each slot's `kv_reserved_bytes` — G6's `need` is now
   `(slots+1) × per_slot` with no shared term. At 2 slots the boundary is
   provably unchanged (old n·p + 2x = new n·(p+x) at n=2, with the peak
   halved since eager allocation has no grow-then-replace transient);
   >2 slots are now correctly charged per-slot. Sizing is
   threshold-independent (max_ctx, block, chunk-capability only) so the
   envelope instrument's runtime threshold flips cannot undersize it.
   Gates: unit suite + G1–G7 + A/B byte-identity (chunk/serial/gen) vs
   the pre-E2 binary — all pass, logs/e2_gates/.]
4. **Lock order: route mutex → GPU lease.** The route mutex protects slot
   assignment/admission/prefix-cache metadata and is never held across a
   GPU dispatch; the GPU lease serializes command-buffer submission and is
   taken per quantum. Documented order, asserted in debug builds.
5. **Cancel invariant.** Client disconnect mid-quantum finishes the quantum
   (bounded stall) and releases the slot; post-cancel MTP lane state is
   explicitly non-cacheable (`PrefixCache` insert is skipped for cancelled
   requests — cancellation can leave lanes at any position). Gate:
   cancel-at-every-lane-position stress.
6. **Two-thread stress gate.** Concurrent requests with interleaved
   cancels; byte-identical outputs vs the same requests run serially on a
   fresh server (greedy, temperature 0), plus no leak of constraint masks
   or MTP state across slots (P0 #1 class, now per-slot).

## Design

- `struct Slot { MetalEngine engine; PrefixCache cache; busy flags;
  per-slot constrainer state; }` — constraint mask cache and `host2dev`
  stay shared (read-mostly, mutex-protected init); the *applied mask* is
  per-engine already (engine-owned), so per-slot isolation is structural.
- `Runtime::run()` becomes `Runtime::run_on_slot(Slot&, ...)`; slot
  acquisition = route mutex, pick idle slot (or queue FIFO if both busy,
  bounded queue, 503 on overflow — matches CUDA server semantics).
- GPU lease: a second mutex owned for one quantum = one prefill chunk
  dispatch+wait, one decode step, or one MTP verify round. Between quanta
  the slot re-acquires, so two active slots alternate. Decode steps are
  ~87 ms (11.5 tok/s); prefill quantum at width 48 ≈ 1 s, at 12 ≈ 250 ms.
- Width policy hook: `uint32_t quantum_width() const` on Runtime reads the
  other slot's state under the route mutex — 96 idle / 48 both-prefill /
  12 when the other slot is decode-active (decode is latency-sensitive).
- Streaming: unchanged per request; emit callbacks never run under either
  mutex (already true — emit happens inside the engine sink, which will
  run under the GPU lease… **NO**: sink runs inside `stream_from_pending`,
  which would hold the lease across the whole generation). **Phase 1 core
  change**: replace the single `stream_from_pending(count)` call with a
  per-quantum loop — `stream_from_pending(pending, quantum_tokens=1|k)`
  per lease acquisition, preserving pending/MTP state across quanta via
  the engine's existing position/lane state. MTP verify rounds are one
  quantum each (lane state persists in the engine between rounds).
- Prefix cache: per-slot (snapshots are engine-shaped: KV lives in the
  engine's cache allocation). Cross-slot prefix sharing is Phase 3
  (needs movable/refcounted blocks — parked with the R1 layout notes).

## Engine-side deltas (small, gated)

- `ingest_prompt` gains a max-width parameter (default 96 = today's
  behavior) so the server can set the quantum width per chunk. Parity
  gate: `--chunk-parity` already proves widths {17,48,96} bit-identical,
  which is precisely this knob's quality certificate.
- `stream_from_pending` gains a resumable form (or a `step_quantum` that
  returns after k tokens with pending state intact). Gate: serial-vs-
  quantum-split generation byte-identical on greedy + MTP paths.

## Pre-registered gates (all must pass before the ledger entry)

- G1 two-slot isolation: interleaved A/B greedy generations byte-identical
  to solo runs (fp16 and turbo3 KV).
- G2 quantum split: k=1..3 quantum-split solo generation byte-identical to
  unsplit, greedy and MTP.
- G3 cancel storm: cancel at every lane position over an MTP generation;
  next request on that slot byte-identical to fresh-server run.
- G4 constraint isolation: constrained tool decode on slot A, plain decode
  on slot B, interleaved; B's outputs byte-identical to solo.
- G5 wait honesty: reported gate-wait percentiles under a two-client
  closed-loop load match wall-clock measurement within 10%.
- G6 admission: configured budget that fits exactly one slot admits one
  and 503s the second with the documented error body.

## Out of scope for Phase 1

Slot-batched N=2 linears (Phase 2, own bench + gates), >2 slots,
cross-slot prefix sharing, adaptive quantum tuning beyond the 3-level
policy, disk-KV snapshots (ds4 item 5).
