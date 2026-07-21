# I1 (strip-tier for DiskSnapshotStore) — mechanism design check before any code

**Status: DESIGN CHECK (2026-07-17, pi session).** Triage item I1
(`2026-07-17-ds4-product-triage.md`) adopts ds4's strip: demote cold
snapshots to text-only stubs, restore = re-prefill. Before building it I
traced the mechanism through OUR serving model, and I cannot make the win
survive contact with it. Recording the analysis so the verdict (build as
written / narrow / skip) is explicit — this is a question, not a refusal.

## The asymmetry with ds4

In ds4's agent the SERVER (session) owns the transcript. Strip drops the
heavy KV payload and keeps the rendered text; without the text stub the
session could not be rebuilt AT ALL — the transcript exists nowhere else.
The stub is the difference between "restore is possible by re-prefill"
and "the session is gone".

Our DiskSnapshotStore is a CACHE behind a request/response server: the
client owns the transcript and sends the full prompt on every request.
Every restore path a stub would enable already exists:

| scenario | today, evicted | with token-stub |
|---|---|---|
| same-prefix request | cold re-prefill of the request's own tokens (auto-snapshot re-saves if ≥4096) | cold re-prefill of the request's own tokens (identical — the stub's tokens are a prefix of the request body) |
| prefix divergence (P+U2 after P+U1) | cold re-prefill from the request body | identical |
| cancel-retry banking | banked snapshot if armed, else cold | identical (stub has no KV to bank) |

A tokens-only stub never saves one token of prefill: its token vector is
by construction a prefix of the request the client just sent. The
pre-registered gate ("stub restore byte-identical vs never-evicted")
passes trivially because restore ≡ cold — which is also the proof there
is no win to gate.

## What a stub COULD buy (the real design choice)

1. **Proactive re-materialization**: the stub is a re-prefill PLAN — on
   an idle slot the server rebuilds KV for a demoted-hot prefix before
   the next request. This is genuinely new capability (speculative
   warming), not "cheap": it spends GPU time speculatively on a
   single-slot box serving live traffic, needs a scheduling admission
   story, and its hit-rate economics depend on prefix recurrence we have
   not measured. (The trace stream from I2 now records prefix
   hit/miss/tier per request — a week of serving traces prices this
   properly. Recommend: measure first.)
2. **Hotness bookkeeping**: stubs preserve "this prefix family recurs"
   across budget pressure, feeding (1) or a smarter eviction order.
   Nearly free, but only useful once (1) exists.
3. **Skip I1 as written**: the lazy form is a no-op over cold.

## Recommendation

Skip the lazy strip tier (no mechanism). IF proactive warming is wanted,
it gets its own pre-registered round with: trace-measured prefix
recurrence rates, an idle-slot scheduling design (must not delay queued
requests; cancels instantly on admission), a disk-write budget for
re-saves, and a ship line on measured TTFT improvement vs the
auto-snapshot baseline. Until then the auto-snapshot tier (shipped,
gated) plus the Q4 chunk-GEMM port (held for the quiet window) covers
the same pain I1 was aimed at — the 5-minute cold prefill — with no
speculation.
