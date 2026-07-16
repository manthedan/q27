# Long-context cache-block scheduling (critical-path item 4, last piece)

GQA KV reuse landed 2026-07-15: decode and causal chunk attention grid over
(KV head, 1024-position block), stage 8-row K/V tiles once for all six query
heads, merge `{m,l,acc}` block partials in index order — 6× KV-stream cut,
−7.8% wall at 8K, **−20.1% at 16K** at quality parity. What remains of item 4
is deciding what "cache-block scheduling" should mean beyond that. Attention
cost is still linear in depth: every decoded token streams the entire live KV
prefix through every attention layer. This doc prices that stream at
8K/32K/128K/262K, evaluates the four candidate responses, and picks a phased
plan with cheap Phase-0 kills.

Verified constants (metal_engine.cpp:211, metal_backend.mm:1266/1294, kernel
addressing at q27_kernels.metal:2089/2138): 16 attention layers (`layer%4==3`
of 64; the MTP layer's 17th cache only attends while drafting and is excluded
from the steady-state model). turbo3 K row = `N_KV·2·50` = **400 B/token**
(100 B per KV head: two 50-byte blocks), V identical → 800 B/token/layer K+V.
fp16 K row = `N_KV·256·2` = **2048 B/token** (512 B per head) → 4096
B/token/layer. Layout is token-major, head-interleaved:
`(pos·kv_heads + kvh)·per_head_bytes` — each GQA threadgroup (kvh, blk) reads
100 B (turbo3) or 512 B (fp16) segments at 400/2048 B stride.

## What already exists vs what is missing (candidate (a) inventory)

Block-count-aware dispatch **already exists**. Decode: `n_blocks =
ceil(seq_len/1024)` — only live blocks are dispatched, the last block clamps
`p1` to `seq_len` (metal_backend.mm:248). Chunk: grid is `(kv_heads,
n_blocks_max, tokens)`; a threadgroup whose block starts past its token's
visible sequence returns before any barrier (no KV traffic, ~free launch),
and the merge derives each token's live block count from `base_len + token`
(q27_kernels.metal:2223/2326). Threshold routing (`Q27_METAL_GQA_THRESHOLD`,
default 2048) splits straddling chunks into two dispatches so every row takes
the kernel decode takes at that length — the chunk↔decode bit-parity
contract. Merge cost is a non-issue at any depth we care about: 262K decode
folds 257 partials × 24 heads = 6.4 MB serially, sub-ms; the chunk partials
buffer peaks at ~77 MB at 262K×12 tokens (lazy, grows once).

Two things are actually missing:

1. **Cross-token tile reuse in the chunk kernel.** The causal grid has a
   token dimension: a 12-token chunk stages the *same* 8-row K/V tile twelve
   times, once per token's threadgroup. Chunk-mode attention therefore
   streams the full prefix per token, same as decode — a 12× redundancy the
   block structure was built to remove and didn't. (This is why NLL walls
   show attention shares far above decode shares.)
2. **Decode-grid occupancy at shallow depth.** With block=1024 fixed, decode
   at 2K–8K dispatches only 8–32 threadgroups (kv_heads=4 × 2–8 blocks) of
   192 threads — likely under-occupying the M4 exactly where the threshold
   hands over. Cheap to measure, and irrelevant above ~16K where blocks are
   plentiful; an adaptive block size would change merge summation order and
   need bucket-split machinery like the straddle split, so it only happens if
   Phase 0 shows real money there.

## Per-token attention cost model (this M4)

KV bytes per decoded token, 16 layers, K+V: **turbo3 12,800·L B; fp16
65,536·L B.**

Effective-bandwidth anchor, derived from the 16K NLL pair (T2 artifact,
caffeinated, mac-mini): legacy 75.7 ms/token vs GQA 60.5 ms/token at mean
depth ~8K. Legacy streams each row 6×, GQA 1×, so the 15.2 ms delta is 5
stream-units and GQA attention ≈ **3.0 ms/token at mean-8K → ~35 GB/s
effective** (legacy ≈ 18.2 ms, same ~35 GB/s per unit — consistent). That is
~40% of the 88–98 GB/s the select-form GEMV proves this machine sustains.
The suspect is the layout: turbo3 reads 100 B per (head, token) out of a
400 B interleaved row — ~25% dense, ~39% efficient against a ~256 B fetch
granule, which lands exactly on the observed 40%. fp16's 512 B contiguous
segments should fare much better; the derivation conflates chunk-kernel
occupancy and dequant ALU with pure stream cost, so Phase 0 pins all of this
with direct dispatch timings. Compute is not the physics: QK+PV at 32K is
~3.2 GFLOP/token → under 1 ms; bandwidth dominates everywhere below 262K.

Projected turbo3 GQA attention per decoded token (and measured denominators:
T2 resident decode ~80 ms/token weights, official ~218; chunk-mode weights
amortize 12× → T2 ~27 ms/token, official ~44):

| context | KV bytes/token | @35 GB/s (today) | @80 GB/s (post-R1) | share today: T2 decode | T2 chunk/NLL | official decode |
|---|---:|---:|---:|---:|---:|---:|
| 8K   | 104.9 MB | 3.0 ms  | 1.3 ms  | 3.6% | 10% | 1.4% |
| 32K  | 419.4 MB | 12.0 ms | 5.2 ms  | 13%  | 31% | 5.2% |
| 128K | 1.68 GB  | 48 ms   | 21 ms   | 37%  | 64% | 18%  |
| 262K | 3.36 GB  | 96 ms   | 42 ms   | 55%  | 78% | 31%  |

fp16 KV is 5.12× the bytes (32K: 2.15 GB/token, ~26 ms even at full stream
bound); 128K fp16 cache (8.5 GiB) barely passes the half-working-set budget
on the 24 GB machine, 262K fp16 is policy-refused. Long context in practice
means turbo3 KV, which is what this doc optimizes.

Reading the table: **decode below 32K needs nothing** (≤13% on T2, ≤5%
official). The money is (i) the chunk/NLL/prefill/verify path at any real
depth — 31% at 32K today — and (ii) decode at 128K+. And the absolute floor
at a perfect ~90 GB/s is still 4.7/18.6/37 ms at 32K/128K/262K: linear,
irreducible for dense attention; only fewer bytes (KV compression) or fewer
blocks (screening) go below it.

## Candidates

**(a) Live-block dispatch completion — token-tiled chunk kernel (R1b).**
Fold 2–4 chunk tokens into one (kvh, blk) threadgroup: stage each 8-row tile
once, run each resident token's per-simdgroup online softmax over it in
token order. Per-token arithmetic order is unchanged → **bit-identical** to
the current causal GQA output, so the chunk↔decode parity contract holds
untouched. Register cost is the limiter: m/l/acc = 10 floats/lane/token, so
tile factor 4 ≈ 40 extra floats/lane — Phase 0 measures where occupancy
breaks. Expected win: chunk-path attention stream ÷ tile factor (32K NLL
attention share 31% → ~10% at factor 4). Zero quality risk. Cheap: one
kernel + dispatch geometry, additive.

**(b) Two-pass block screening (ε-bounded, metadata-assisted).** Reject the
fixed top-k variant outright: selecting "top 32 blocks" changes which tokens
attend with no error bound — a ranking/semantic change, forbidden-class
under item 5's contract. The admissible variant is proof-carrying: per
(layer, kvh, block) keep dequant-domain per-dimension min/max of K (1
KB/block/head fp16, updated incrementally at `kv_store`; 16.8 MB total at
262K). Screening pass per query head: `ub_b = Σ_d max(q_d·kmin_d,
q_d·kmax_d)` (a rigorous bound on any score in the block, computed on the
values the kernel actually reads); always evaluate an anchor set exactly
(newest block + top-8 by bound) to get `M_lb, l_lb`; skip block b iff
`1024·exp(scale·(ub_b − M_lb)) < ε·l_lb` — dropped softmax mass < ε of kept
mass, output perturbation ≤ ~ε. At ε=1e-5 this sits below turbo3's own noise
floor (KL 0.0115 nats, ±0.17% PPL summation-reorder deltas), i.e.
numerical-tolerance class, but only because the bound is a proof, not a
heuristic. Two catches: skip granularity is per (kvh, block) — the tile is
staged if *any* of the 6 query heads needs it — and the win is entirely
distribution-dependent (wikitext attention is diffuse; needle-style is
peaky). Expected win: time × (1−s) for skip fraction s, metadata pass
negligible; s is unknown and must be measured before any kernel is written.
Complexity: kv_store changes, screening kernel, block-list indirection into
the GQA kernels — the most invasive candidate.

**(c) Block-level residency/layout.** Residency reordering (hot blocks
first, prefetch) is **dead on this hardware**: unified memory, KV buffers
device-resident, and the paging chase proved steady-state decode refaults
nothing — address order does not change stream cost. The live half is
**block-tiled head-major layout (R1)**: store KV as
`[block][kvh][pos-in-block]` so each (kvh, blk) threadgroup streams one
contiguous ~100 KB run instead of 100 B-per-400 B strides. Same values, same
arithmetic order, different addresses → **bit-exact**, and per-block
metadata for (b) later slots naturally into a per-block header. If the
granule hypothesis behind the 35 GB/s figure is right, this alone is a
~2–2.5× attention-wall cut and closes most of the gap to the GEMV-class
bound. Scope control: turbo3 caches only (fp16's 512 B segments have little
to reclaim; fp16 layout untouched halves the blast radius). Cost is real but
mechanical: both turbo3 store kernels, all four turbo3 attention kernels
(legacy per-qh kernels read the same cache below threshold), MTP-path calls,
snapshot copy switches from byte-prefix to whole-live-blocks (over-copy ≤ 1
block; rows are write-before-visible so copied garbage is inert), ABI bump.

**(d) Do nothing — and name the honest lever.** Below 32K decode, (d) *is*
the answer: ≤5% of official-tier wall. Everywhere else, (a)+(c) only move
effective bandwidth toward the ~90 GB/s bound; the 37 ms/token at 262K that
remains is the physics of dense attention over 3.36 GB of live KV. Going
below that floor means streaming fewer bytes: sub-2-bit KV compression,
which now has exactly the instrument it needs (the KL KV gate: turbo3's
0.0115 nats mean, depth-flat, is the bar any smaller codec must beat, plus
needle retention). That is a separate track and stays out of this plan's
scope; this plan's job is to stop wasting the bandwidth we have.

## Recommendation

Do (c)-layout and (a)-token-tiling — both bit-exact, both pure bandwidth —
gated by a synthetic Phase 0; hold (b) behind a measurement of attention
concentration, because its win is speculative and it is the only candidate
that touches quality at all. Decline residency games and sub-32K decode work
explicitly.

### Phases

0. **Synthetic long-context attention bench (mac-mini, no model).** Extend
   `metal_decode_bench`/`metal_prefill_bench` with a deep-KV attention mode:
   synthetic turbo3+fp16 caches at 8K/32K/128K/262K, per-dispatch GQA
   ms + effective GB/s (decode grid and chunk grid separately — the
   occupancy question falls out here). Prototype kernels: (i) contiguous
   head-major layout A/B (same math, synthetic relayout), (ii) token-tiled
   causal at factors 2/3/4. Gates: layout ≥1.5× effective GB/s on turbo3 at
   32K+ else park R1; token-tiling ≥2× chunk-attention throughput at factor
   ≥2 without occupancy collapse else park R1b. Budget: a day; kill cheaply,
   T3-style.
1. **R1 — turbo3 block-tiled head-major KV layout.** Store + all readers +
   snapshot block-copy, ABI bump. Gates: full `make test-metal` with the
   GQA/straddle parity suite re-pointed at the new layout (bit-exact CPU
   references), artifact committed tokens byte-identical fp16-KV and
   turbo3-KV, 8K NLL wall A/B + bucket parity (expect bit-identical NLL,
   lower wall), codex autoreview. Kill: if artifact wall win at 16K NLL is
   <5% despite the synthetic win, record and reassess (would indicate the
   share model is wrong, not the kernel).
2. **R1b — token-tiled causal GQA** at the Phase-0-chosen factor. Gates:
   memcmp bit-parity vs current causal GQA on device (the existing
   straddle-gate harness), chunked-vs-serial artifact A/B, 16K/32K NLL wall.
   Expected combined R1+R1b: 32K NLL attention ~12 → ~1.5–3 ms/token.
3. **Phase 0b for (b) — attention-mass instrument, then decide.** A stats
   variant of the GQA kernel dumps per-block `m_b`/mass for a real 32K run
   (T2 artifact, orchestrator-owned — the GPU is occupied by the needle run
   as of this writing; queue behind it). Offline: achievable skip fraction
   at ε=1e-5 with per-(kvh,block) granularity and the 6-head union rule, on
   both wikitext and the needle prompt. Gate to proceed: **s ≥ 30% at 32K**;
   below that the two-pass machinery cannot pay for itself post-R1. If
   proceeding: implement metadata + screening, quality gates = KL instrument
   vs unpruned (added KL target <0.001 nats, an order below turbo3's own),
   needle retrieval unchanged, 32K NLL bucket parity ±0.1%, committed-token
   A/B at short context (must be byte-identical — screening must never fire
   below its own anchor-set size). If wikitext says diffuse and needle says
   peaky, the honest disposition is park-with-data, same as T3.

### Machine matrix

Phase 0, R1/R1b builds and their synthetic + T2-artifact gates (16K/32K NLL,
KL, needle, 128K turbo3 = 1.68 GB cache): **mac-mini** (T2 artifact only —
the 17 GiB official tier does not fit its 8.0 GiB `maxBufferLength` or 16
GiB RAM). 262K T2 turbo3 passes the budget check on the mini (3.36 GB caches
vs 5.35 GiB half-working-set) but lands ~10 GB against 16 GiB with wired
weights — run 262K on the **24 GB machine**. Official-tier gates — canonical
16-token CUDA gate, fp16-KV A/Bs, any MTP-verify interaction at depth, 128K
fp16 (8.5 GiB cache, barely inside budget) — **24 GB machine only**, one
model load at a time, `caffeinate -dims` on every long run.

## Non-goals

- KV eviction / token dropping (H2O, StreamingLLM-class): semantic
  regression by construction, forbidden under the quality contract.
- Fixed top-k block selection (Quest-style budgets): same class; only the
  ε-bounded form of (b) is admissible here.
- Paged/virtual KV (vLLM-style block tables): single-slot engine with
  contiguous per-layer caches; indirection buys nothing until multi-slot
  scheduling exists (serving-closure item, not this one).
- Sub-2-bit KV codecs: separate track, owned by the KL KV instrument.
- fp16-KV layout changes and residency/prefetch machinery: no measured
  waste to reclaim.
- CUDA parity for any of this (Metal-first, as with the tier work).

## Phase 0 results (2026-07-15 night, mac-mini)

`build/metal_attn_bench` (synthetic caches, SLC defeated by cycling ≥256 MiB
of cache copies, probes memcmp'd bit-identical against the production GQA
kernels before timing; threshold forced to 1 so all depths route GQA):

| seq | decode gqa | decode hm | chunk12 gqa | chunk12 t2 | chunk12 t4 |
|---|---:|---:|---:|---:|---:|
| 4K   | 1.72 ms (1.9 GB/s) | 0.98× | 9.10 ms  | 1.75× | 1.13× |
| 8K   | 2.31 ms (2.8)      | 1.00× | 17.29 ms | 1.89× | 1.32× |
| 32K  | 6.44 ms (4.1)      | 1.00× | 66.54 ms | 2.01× | 1.44× |
| 128K | 23.38 ms (4.5)     | 1.01× | 263.4 ms | 2.06× | 1.51× |

fp16 at 32K: decode 6.75 ms, chunk12 67.6 ms — **the same wall as turbo3
while streaming 5.12× the bytes.**

**Verdicts:**

- **R1 (head-major layout): PARKED.** 1.00× at every depth including 128K
  fully-DRAM streams. The granule/stride hypothesis behind the 35 GB/s
  figure is refuted — the interleaved layout costs nothing, because the
  kernels are nowhere near the stream bound (risk #1 below fired).
- **The kernels are latency/occupancy-bound, not bandwidth-bound.** Turbo3
  logical bandwidth plateaus at 4.5–4.8 GB/s; fp16 pays no wall for 5× the
  bytes; back-of-envelope ALU is ~6% of the M4's issue rate. The suspect
  is residency: Kt+Vt tiles are 16 KB of threadgroup memory → 2
  threadgroups/core, ~120 resident simdgroups machine-wide, against a
  barrier-stage-barrier + 8-serial-rows structure (simd_sum + 2 exp
  dependency chains per row) that thin occupancy cannot hide.
- **Cost-model correction (the important number):** chunk attention at
  depth is ~10× this plan's estimate. At mean-8K depth the chunk kernel
  costs 17.3 ms/dispatch → 23 ms/token across 16 layers — 38% of the 16K
  NLL's measured 60.5 ms/token, which now adds up (weights ~27, GDN/other
  ~10). The "3.0 ms/token at mean-8K" derivation assumed byte-bound
  kernels; the 15.2 ms legacy-vs-GQA delta was an occupancy delta, not 5
  stream-units. R1b's prize is correspondingly ~10× larger than the table
  above suggested: ~−19% total NLL wall at 16K, more at 32K.
- **R1b (token-tiled causal): PROCEED at factor 2.** 1.89× / 2.01× / 2.06×
  at 8K/32K/128K — meets its ≥2× gate at 32K+. Factor 4 is strictly worse
  (1.32–1.51×; register pressure eats the staging amortization) and is
  rejected. The t2 probe kernel is bit-identical per token by construction
  and by memcmp, so graduation keeps the chunk↔decode parity contract.
- **Shallow-decode occupancy (missing item 2):** confirmed poor (1.9 GB/s
  at 4K vs 4.5 at 128K) but per the table above sub-32K decode remains
  ≤13% of T2 token time — still not worth an adaptive block size.
- **R2 candidate surfaced (not in the original plan):** halve the
  threadgroup-memory footprint to double residency — stage K, score all 8
  rows, restage V over the same 8 KB, apply. Preserves per-token arithmetic
  order exactly (bit-parity survives). Second candidate: raise
  tokens-per-stage beyond 2 without more registers. Both are post-R1b
  measurements; the ceiling here is large (fp16 parity says ~5× headroom
  before bytes matter even for fp16).

## Risks

- The 35 GB/s attribution is derived, not measured per-dispatch; if Phase 0
  shows the GQA kernels already near the stream bound (i.e. the 16K delta
  was occupancy or dequant ALU), R1's prize shrinks toward zero — that is
  exactly what Phase 0 exists to find out before ~10 kernels get touched.
  **[Fired, in the opposite direction: the kernels are so far from the
  stream bound that the layout is irrelevant and the tiling prize is ~10×
  the estimate. R1 parked, R1b proceeds, see Phase 0 results.]**
- R1 touches every turbo3 KV reader including the legacy sub-threshold
  kernels; a missed reader decodes garbage that only artifact gates catch —
  the parity suite must cover both sides of the threshold and the straddle
  (it does today; keep it pointed there), and the ABI tag guards stale
  binaries.
- Token-tiling multiplies per-lane register state; a spill turns a bandwidth
  win into a compute loss (the T3 lesson in miniature) — factor is chosen by
  measurement, not by argument.
- (b)'s bound is only as good as the metadata: min/max must be maintained in
  the dequant domain the kernel reads, and MTP/snapshot paths must keep
  metadata coherent with rows (restore must copy it). Any shortcut here
  converts "numerical tolerance" into silent ranking corruption.
