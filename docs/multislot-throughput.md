# How q27 multi-slot scales aggregate throughput (and what it cost)

2026-07-16, v0.2.0. Rewritten from the 2026-07-11 original ("Why q27
multi-slot doesn't scale aggregate throughput like vLLM") after the
continuous-batching campaign (P1-P3, BUILDLOG 2026-07-14..16) eliminated
the limitation that doc recorded. Its analysis and history are preserved
below -- it predicted the shape of the fix almost exactly.

> **SCOPE (2026-08-23): "extra slots buy fan-out, not tokens" is true for
> BIG-CONTEXT slots, and false as a general statement.** Everything below was
> measured at 2x32K/2x48K/2x96K, where union-cap trim binds and aggregate
> saturates around 250 t/s by two lanes. The C=4/C=8 E-series campaign
> (BUILDLOG 2026-08-18..19) runs **8 slots at 16K** and scales to **530.6 t/s
> at C=8 against 141.3 solo** (q4s, `Q27_DRAFT_CEIL1=1`; README cross-engine
> table) -- 3.75x, well past two lanes. Both hold: aggregate scales with lane
> count once per-slot context is small enough that the union cap does not
> trim. The Claude-Code serving shapes of record are still the big-context
> ones here, so the doctrine stands for agentic serving and needs the
> qualifier everywhere else. The C=8 ladder is a synthetic concurrency
> ladder, not one tenant's burst.

## Current reality (measured, batch_ab REPS=3, w16 2x32K, v0.2.0)

| KV | FIFO (old behavior) | batched + graphs | ratio | solo cost |
|---|---|---|---|---|
| fp8 | 168.9 t/s | **237.7 t/s** | **1.41x** | <=0.06% |
| turbo3 | 158.5 t/s | **224.2 t/s** | **1.41x** | <=0.07% |

Zero-config: a bare 2-slot `q27-server` spot-checks at 234-239 t/s
aggregate (90.9% graph-cache hits). ON by default since v0.2.0
(`Q27_BATCH=0` restores FIFO time-slicing; `Q27_BATCH_GRAPH=0` keeps
batching but drops graph replay; `Q27_PROFILE=ref` = conservative
reference). Serving shapes of record for CC traffic: fp8 2x48K, turbo3
2x96K (the w16 build caps at 2x32K on 32 GB). One edge (v0.2.0): `--slots N`
did not auto-size `--ctx`. Superseded by elastic windows (v0.11.2): with
`--slots N` and no `--ctx`, every slot may use the whole shared KV pool and
admission arbitrates; an explicit `--ctx` now also propagates to slots 1+
unless `--slot1-ctx` is given (`server.cu`).

Per-stream trade: each concurrent stream runs at ~63% of its solo rate
while the box produces ~41% more total tokens. Bursty agentic traffic
often does better than that arithmetic suggests -- tool-execution gaps
mean decode bursts frequently don't collide, and a stream that briefly
has the GPU alone runs the k==1 solo path (captured graphs, bitwise
identical to single-slot serving, 0.00% measured regression).

## The physics (unchanged from 07-11 -- what changed is who amortizes)

Decode on this box is bandwidth-bound on the 17.7 GB weight stream; every
round streams (nearly) the full weight set once. The only lever is
amortization -- more tokens through each weight read:

- q27's original bet amortized WITHIN one user: the MTP ladder + wide
  verify feed up to ~13 lanes of ONE sequence per weight read. That is
  what 200+ t/s single-stream is made of -- batching pointed at latency.
- The campaign added amortization ACROSS users ON TOP: a conductor thread
  fuses concurrent slots' verify rounds so one weight sweep serves the
  UNION of everyone's lanes (weight kernels were already per-lane pointer
  structs; cross-engine lanes are plumbing). Per-engine state work
  (attention on own KV, GDN chains) stays per-engine: attention shares
  nothing across sequences and is KV-saturated at depth (P4 measured
  no-go on further overlap), GDN chains fork onto side streams.
- Draft steps fuse the same way (one MTP head sweep per step for all
  active engines) -- overlap alone recovered ~nothing because drafts are
  weight-bound too. The campaign's physics triad: BW-bound work yields
  only to fusion; state-latency-bound work yields to overlap; saturated
  work yields to neither.

## The price P10-A named, and P3 paid

The 2026-07-11 P10-A rejection (docs/P10-decision.md) identified the
killer cost: the CUDA-graph zoo cannot bake per-perm role pointers across
independent mod-12 rotations, "forcing device-side indirection through
the round-critical path that every canonical bitwise gate protects."
That is, almost verbatim, what shipped: conv/delta TABLE TWINS resolve
role pointers through a device-resident table + per-engine perm scalar
(P10-spill-safe, ptxas spill 0, bitwise vs the direct kernels -- gated by
ninv's TWIN leg on both arches), making fused rounds perm-invariant so a
shape-keyed LRU cache can replay whole verify rounds as CUDA graphs
(~28-44 live shapes, 86-100% hits). The bitwise gates the 07-11 doc
worried about HELD: batched output is byte-identical to the pre-batching
references at every phase (master-gate refs, canonical, sampled-seed),
and solo serving is untouched by construction. [2026-08-16 amendment:
byte-identity now holds at k <= 2 members; k >= 3 unions default to the
vgemm tolerance class (rel ~1e-6) after the C-sweep found the GEMV family
caused the C=7 rolloff -- BUILDLOG (g)/(h), Q27_BATCH_GEMM=0 restores the
old policy.] What the 07-11 estimate
got right: ~1.9x was the no-graphs ceiling estimate at k=2; eager fusion
measured 1.21-1.31x and graph replay carried it to 1.41x. What it got
wrong: nothing material -- it priced the work as expensive (it was: P0-P3,
three days, ~15 gated commits) and capped at 2 big-context users by
per-slot GDN role VRAM. (07-17 amendment: turbo3 + the w8 build boots
FOUR 48K slots on 32 GB -- but the aggregate SATURATES at ~250 t/s by
2 lanes (3 slots matches, 4 regresses under union-cap trim), so the
2-slot cap is now a throughput fact rather than a VRAM one; extra
slots buy fan-out, not tokens. BUILDLOG 2026-07-17.)

## History: the 2026-07-11 observation this doc replaced

Two turbo3 slots at 131K, FIFO time-slicing, real CC sessions: solo T8
~205 t/s per-request median; concurrent T8 pair split to ~103.5 each
(engine busy-aggregate ~200 t/s -- same total, divided); bursty T2/T11
pairs rode near-solo walls. vLLM gained ~1.6x aggregate at c=2 where q27
gained ~1.0x -- by design, then. Full text in git history
(docs/multislot-throughput.md @ 361b0a2 and earlier); the decision record
it accompanied is docs/P10-decision.md; the campaign that changed the
answer is BUILDLOG 2026-07-14..16.
