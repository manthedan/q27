# Metal multi-slot serving: slots, scheduling, backpressure

2026-07-15. Serving-closure item from METAL_PROGRESS ("backpressure,
multi-slot scheduling, and the CUDA server compatibility suite"). The Metal
server today is single-slot: one `MetalEngine`, one `std::mutex` held for an
entire generation, a 1-entry prefix cache. The CUDA reference (server.cu)
runs up to 4 engines behind a FIFO `GpuGate` with round-granularity
time-slicing (R1b); its lesson stands — see docs/multislot-throughput.md:
multi-slot on a bandwidth-bound decoder buys **capacity, admission latency,
and prefix-cache preservation**, not aggregate throughput. On Metal the
cache-preservation win is even larger than on CUDA: artifact prefill runs
~4–25 tok/s here, so a second interleaved conversation evicting the 1-entry
cache costs the first one **minutes** of re-prefill at 8K. That, plus
head-of-line admission (a queued request currently waits an entire
generation — 20–60 s at Metal decode rates), is what this buys.

Prereq already landed: `MetalEngine::Shared` (KL-gate refactor, 2026-07-15)
bundles the mmap, backend (whole-mapping buffer + residency set), and weight
wrap so N engines share **one mapping, one queue, one model load** — the
memory-safe policy's one-load rule holds for any slot count. Combined KV
budget across engines on a mapping is already enforced in the ctor, and the
self-KL gate proved two engines on one Shared produce bit-identical logits.

## 1. What a slot costs

From `metal_engine.cpp` allocations. Per-token KV row (K or V, one cache):
fp16 `N_KV·HEAD_DIM·2 = 4·256·2 = 2048 B`; turbo3 `N_KV·2·50 = 400 B`.
Total KV = `(16 base + 1 MTP if present) × 2(K,V) × ctx × row`. GDN state
(48 layers): recurrent `48·128·128·4 = 3 MiB/layer` → 144 MiB, conv ring
`3·10240·4 = 120 KiB/layer` → 5.6 MiB; ~149.6 MiB total, fixed. Activation
buffers (serial + chunk + verify/park + logits): ~45 MiB, dominated by
`clogits_` 11.4 MiB and the parked GDN inputs ~22.7 MiB (lazy). Call the
fixed per-engine cost **~200 MiB**; Metal buffers are physically lazy, so
these are worst-case-touched numbers.

Per-slot KV (official tier, 17 caches; T2 tier has no MTP layer → 16
caches, ~6% less):

| ctx    | fp16 KV   | turbo3 KV |
|--------|----------:|----------:|
| 4,096  | 272 MiB   | 53 MiB    |
| 8,192  | 544 MiB   | 106 MiB   |
| 32,768 | 2,176 MiB | 425 MiB   |

Per-slot total = KV + ~200 MiB. A slot that keeps a prefix-cache entry adds
a snapshot: ~151 MiB GDN-dominated fixed (144 recurrent + 5.6 ring + 0.95
logits + hidden) **plus active KV rows at the snapshot position** (full-ctx
fp16 8K: +544 MiB; turbo3 8K: +106 MiB). Budget a slot as
`KV + 200 + (151 + KV_at_position)` when it retains one cache entry.

**16 GiB mini (T2 artifact, 6.66 GiB wired, recommended working set
10.7 GiB).** Headroom after weights ≈ 4.0 GiB for all slots + caches. At
ctx 4096 turbo3 a slot-with-cache-entry is ~450 MiB → ~8 fit arithmetically;
at ctx 32768 turbo3 ~1.15 GiB → 3 fit. The engine's own budget check
(combined KV ≤ working-set/2 = 5.35 GiB) never binds at turbo3. **v1 caps
at 2 slots** anyway — decode is bandwidth-bound, extra slots add capacity
we cannot feed, and the mini has zero pressure incidents to date; keep it
that way.

**24 GB rig (official tier, 17 GiB mmap, unwired — exceeds maxBufferLength;
working set 17.8 GiB).** The mapping pages under pressure and this machine
has a hard-crash history. Nominal fits: 2 slots at ctx 8192 fp16 with cache
entries ≈ 2.9 GiB (fine); 2–3 slots at ctx 32768 turbo3 ≈ 1.2 GiB each
(fine); 2 slots at ctx 32768 fp16 ≈ 5.4 GiB (passes the budget check, but
leaves the pager ~5 GiB less slack under a 17 GiB unwired mapping — do not
default to it). **Recommendation: slot 0 at --ctx, slot 1+ turbo3 or
smaller ctx via a --slot1-ctx analog, exactly server.cu's shape.**

## 2. Scheduling: three options, one pick

The `Shared` contract is explicit: engines on one Shared alias one command
queue and one backend batching state, and must be driven from one thread
**or externally serialized**. Any design below must serialize every engine
call; the differences are granularity and where the calls run.

**(a) N engines + FIFO gate, whole-generation leases (CUDA R1).** Claim a
free engine, take a `GpuGate::Lease`, run `Runtime::run` unchanged, release.
Smallest diff; correct (mutex-class serialization satisfies the Shared
contract, and per-engine KV/GDN/activation isolation means no cross-slot
state hazard). But it re-creates R1's measured failure at 10–100× the
cost: CUDA's residual whole-generation queue wait was 24.4 s at ~200 tok/s;
Metal decodes at 4–13 tok/s, so a queued request waits **minutes** behind a
long generation. Acceptable only as a phase, not an endpoint.

**(b) N engines + FIFO gate, token/chunk-granularity yields (CUDA R1b).**
Same structure plus `maybe_yield()` at natural preemption points. Metal's
preemption points are *better* than CUDA's: every decode step already ends
host-synced (argmax/top-k readback closes the command batch), every prefill
chunk ends with a batch `finish()`, and every MTP verify round ends with
its one CPU sync. So the R1b "drain the stream before release" discipline
is free — at each yield point the queue is empty and the batching state
closed by construction. Yield points: (1) bottom of each decode token in
`stream_from_pending`/`stream_sampled_from_logits`, (2) after each
`encode_chunk` in `ingest_prompt`, (3) after acceptance in each MTP round.
`GpuGate` lives in `api_common.h`, pure host (mutex+condvar, zero CUDA
includes — verified) — metal_server includes it as-is. Uncontended cost is
one relaxed atomic load per token.

**(c) Single engine + capture_state/restore_state swapping.** Swap cost is
a 151 MiB + active-KV device copy each way (~300–800 MiB round trip at 8K),
plus snapshot allocation per swap, plus it either equals (a) at
generation granularity or pays that copy per time-slice. It buys nothing
over (b) — engines already share the one weight mapping, so per-slot state
is the *same memory* whether it lives in N engines or N snapshots — and it
adds a transient doubling of state during swap. **Rejected.**

**Decision: (b), phased through (a).** Concurrency contract, stated once:
*all* `MetalEngine` and `MetalBackend` calls happen inside a GpuGate
lease; the gate's mutex provides the external serialization and memory
visibility the Shared contract requires; engines are constructed at startup
on the main thread before `listen()`; yields occur only at the three points
above, where the engine is host-synced and no command batch is open.
Per-slot host state (PrefixCache, ToolMaskCache host2dev vector) is touched
only under the same lease, as today's mutex comment already documents.
Routing state (`busy`, LRU stamps) lives under a separate small route
mutex + condvar, copied from server.cu's `claim_slot`/`free_slot` including
the tier order (prefix-reuse > cache-empty > LRU) and stamp-on-free rule.
`Q27_NO_INTERLEAVE=1` keeps the whole-generation mode as the debug lever,
same escape hatch as CUDA.

One Metal-specific caveat carried forward: batched MTP commits the accepted
prefix before sinks fire (open codex finding, official tier only), so a
cancel mid-round leaves position advanced past the last emitted token.
Harmless for the *cancelled* request; under multi-slot it also cannot harm
siblings (per-engine state), but the slot's prefix snapshot must be
captured before generation (it already is), never after a cancelled run.

## 3. Backpressure

Fact first: **server.cu emits no 429 or 503.** Its admission is blocking —
`claim_slot` waits on `route_cv`, bounded by ≤4 slots and self-limiting
clients; the only load-relevant refusals are 400-class (empty prompt,
`context_length_exceeded`). So "matching server.cu" means the compat suite
must pass without ever seeing a shed request; anything we add is an
extension, and it must be additive.

v1 semantics:

- **Bounded admission.** Counter of in-flight + queued generations, cap
  `2 × slots` (default 4). At the cap, respond **503** immediately with
  `Retry-After: 2` and the per-API error body (Anthropic:
  `{"type":"error","error":{"type":"overloaded_error",...}}`; OpenAI
  routes: standard `error` object, type `"server_error"`, code
  `"overloaded"`). 503 over 429 because the condition is capacity, not a
  client rate policy; one status everywhere keeps the suite simple. Below
  the cap, behavior is server.cu's: block until a slot frees, FIFO at the
  gate.
- **Streaming keep-alives while queued.** A queued streaming request has
  sent headers but no events; agent clients time out on silent sockets.
  Emit an SSE comment line (`: queued\n\n`) every 10 s while waiting for
  engine claim + first lease. Comments are spec-invisible to every SSE
  parser, so this is safe on OpenAI, Anthropic, and Responses routes alike
  (no `ping` event before `message_start` — we stay inside each protocol's
  documented event sequence). Non-streaming requests get no keep-alive;
  they are covered by the admission cap.
- **Disconnect while queued.** The keep-alive write doubles as the probe: a
  failed write (or `sink.is_writable()` false) marks the request dead — it
  frees its admission count and never claims an engine. On lease grant,
  probe once more before touching the engine (the just-landed pattern:
  Anthropic/Responses streams gate the run on their opening writes and
  probe on empty pieces), so a client that vanished in the queue costs zero
  GPU work. Non-streaming queued requests keep today's behavior (detected
  at first emit).

## 4. Prefix cache under multi-slot

Snapshots are owner-bound (`snapshot->owner == this`; restore into a
different engine throws). Keep that. A shared cross-slot cache would need
snapshot portability (config-compatibility checks replacing the owner
check) and buys little: the CUDA design already showed per-slot caches +
reuse-aware routing capture the conversation-affinity win, because the
router sends a returning conversation back to the slot that holds its
prefix. **Decision: one PrefixCache per slot, capacity 1 (default), plus a
read-only `probe(prompt, mtp) -> match_len` used by `claim_slot`'s tier-2
test** — the Metal analog of `Engine::reuse_len`, safe because probing only
reads token vectors, never engine state, and only free slots are probed.

Memory: N slots × (151 MiB + KV-at-position) of snapshots. On the mini at
2 slots / ctx 4096 turbo3 that is ≤ ~2 × 204 MiB — fine. Eviction under
pressure: `prepare_insert` already releases the LRU entry before capturing
its replacement (peak never exceeds capacity), and capacity stays 1 per
slot in v1; a global snapshot-bytes ceiling (drop the LRU entry across all
slots when exceeded) is deferred until a measured configuration needs
capacity > 1. `--prefix-entries` becomes per-slot.

## 5. Phasing and gates

**Phase 0 — instrument (mini, no server changes).**
`tools/metal_multislot_bench.sh`: fires K concurrent streaming chat
requests (distinct prompts ~1K tokens, 128 out, greedy, `caffeinate -dims`)
and reports per-request TTFT, wall, aggregate committed tok/s, and byte
comparison of each response against its solo run. Baseline: current
single-slot binary at K=1 and K=2 (K=2 shows the serialized wait +
cache-eviction re-prefill we are fixing). This script is also the
acceptance instrument for every later phase.

**Phase 1 — slots + gate, whole-generation leases (mini, T2, 2 slots, ctx
4096 turbo3).** Restructure `Runtime` into `Slot {engine, cache, host2dev,
busy, last_used}` over one `Shared`; `claim_slot`/`free_slot` with the
CUDA tier order; GpuGate lease replacing the mutex; bounded admission +
503; queued keep-alives + disconnect handling. Gates: (1) `make
test-metal` green; (2) solo outputs byte-identical to the single-slot
binary, both tiers of sampling (greedy + seed-1 sampled); (3) two
alternating conversations each hit their own slot's prefix cache
(`q27_prefix_hit > 0` on turn 2 of both — RED on today's binary); (4)
concurrent pair completes with request 2's queue wait ≈ request 1's
generation (recorded, expected, removed by Phase 2); (5) 503 + error-shape
test at admission cap; (6) disconnect-while-queued frees the ticket
without an engine claim (log assertion).

**Phase 2 — token/chunk-granularity yields (mini).** `on_round_gap`-style
hook or direct `maybe_yield()` at the three preemption points;
`Q27_NO_INTERLEAVE=1` escape hatch; `gw`/`yields` telemetry appended to a
new `[req]` log line matching server.cu's format (the compat suite's
reqlog parsing is the eventual consumer). Gates: (1) interleave gate —
long streaming A + short B fired mid-A: B completes while A still streams,
both byte-identical to solo, A shows `yields>0`, `Q27_NO_INTERLEAVE=1`
restores Phase-1 serialization (RED against the Phase-1 build); (2) solo
regression ≤ 2% wall on a 128-token decode (the uncontended cost must be
one atomic load per token); (3) Phase-0 instrument A/B vs single-slot
baseline: TTFT for request 2 drops from ~full-generation to ~1 token
(~100–250 ms), aggregate tok/s within noise of solo (per
multislot-throughput.md, aggregate parity — not gain — is the pass
condition).

**Phase 3 — 24 GB rig, official tier.** Needs that machine: MTP-path
yields (verify rounds; also decide the cancel-mid-round position caveat),
mixed per-slot ctx (`--slot1-ctx`), fp16-slot + turbo3-slot pairing, and
the CUDA server compatibility suite run against the multi-slot Metal
server (endpoint shapes are already parity; the suite adds the [req]
telemetry and concurrency cases). Official-tier runs stay serialized with
everything else on that machine per the memory-safe policy.

**Kill criteria.** (1) Any committed-output divergence vs solo at any
phase — stop, this is the R1b correctness bet and it is non-negotiable.
(2) Solo-path wall regression > 2% with the gate uncontended. (3)
Contended 2-slot aggregate < 0.9× solo aggregate (scheduler tax; R1b on
CUDA measured ~0). (4) Any memory-pressure incident on the mini (swap
growth or jetsam during the instrument run) — drop to smaller ctx or 1
slot and record; the mini's clean pressure record is worth more than a
slot.

## Non-goals

- **Cross-user batched decode** (one weight stream feeding N users'
  tokens): P10-A priced and rejected it on CUDA; on Metal the chunk GEMM
  exists but per-lane sequence plumbing (separate KV bases, positions, GDN
  states per lane) does not. Aggregate throughput is explicitly not the
  goal (multislot-throughput.md).
- Priority/weighted scheduling, mid-generation preemption or eviction,
  > 4 slots, per-slot MTP-width policy: FIFO round-robin v1, revisit only
  on telemetry.
- Snapshot portability across engines (shared prefix cache): owner check
  stays; per-slot caches + routing capture the win.
- 429 rate-limiting semantics, request timeouts, auth: not in server.cu,
  not here.
- GPU-resident sampling (stage 3) interaction: lands independently; its
  per-token readback shrinkage only makes yield points cheaper.
