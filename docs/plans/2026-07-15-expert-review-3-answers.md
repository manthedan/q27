# Expert review round 3 — answers to the six post-measurement questions

2026-07-15. After the round-2 triage and the same-day measurement sprint
(Phase B kill, R1 park, R1b graduation, R2 kill, resident-greedy demotion)
we sent the expert six questions where outside knowledge exceeds what we can
measure locally. This round is design guidance, not a code review — there is
nothing to verify at file:line; statuses are ADOPT / RECORDED. Full response
text is with the operator. The expert confirmed every prediction outcome we
relayed (R1b "factor 2 may win" → measured 2.0×; Phase B premise → killed at
0.96×) and the feedback visibly sharpened this round — **keep sending
outcome reports with every future round.**

External evidence cited this round: arXiv 2606.12765 (M4 Max
characterization, again), 2603.27569 (M1 GPU microarchitecture — barrier
cost), 2601.19139 (Apple Silicon continuous batching), 2407.00088 (T-MAC
LUT), 2402.02750 (KIVI), 2604.16957 (Open-TQ-Metal); plus Kitty, AsymKV,
SVDq, H1B-KV for sub-2-bit KV.

## Q1: Are we at the M4's MMA ceiling? — "near THIS schedule's ceiling; unproven at the hardware's." ADOPT the A/B/C roofline as a one-day decision experiment

Their arithmetic (endorsed; goes in the ledger): closing the fork gap
(48.79 → 52.48 = 1.0756×) with GEMM at 94.3% of GPU time requires an **8.1%
GEMM-kernel speedup**; eliminating *all* non-GEMM work yields only ~51.7
tok/s. So no epilogue fusion or dispatch cleanup can close it — only a real
MMA-kernel improvement can. Also: "94.3% of time" proves Amdahl dominance
and "3.3 GB/s" proves DRAM irrelevance, but neither measures the fraction of
peak `simdgroup_multiply_accumulate` issue rate reached — effective model
FLOPs ≠ physical MMA instruction count (flushes, partial tiles, edge rows).

**The experiment** — three kernels at identical production geometry
(dispatch count, tile size, edge behavior), FLOP/s computed from *physical
MMA instructions*, plus occupancy/spills/scalar-share/stall counters:

- **A. MMA-core roofline**: operands already in half tiles, minimal
  plumbing, same physical MMA count, dependency-preserving accumulator
  updates + final store (defeat dead-code elimination).
- **B. Half-plumbing roofline**: half operands from device memory, same
  threadgroup staging, same barriers, same loads/MMA/accumulator stores and
  float scale-fold epilogue — **no packed unpack, no int8→half**.
- **C. Full production T2 G′.**

Pre-registered reads (use the lower 95% confidence bound, never best run):

| signal | verdict |
|---|---|
| B/C ≤ ~1.08 | no unpack/staging headroom to close the gap — **stop; classify M4 prefill MMA as mature** |
| B/C 1.08–1.15 | permit **exactly one** targeted kernel round |
| B/C > 1.15 | real implementation headroom; pursue |
| A/B small | at the shape-specific MMA ceiling |
| A/B large | barriers / tile movement / accumulator folding are the residual problem |

Lever ordering if (and only if) the roofline reopens work: K=128 flush
cadence probe only if plumbing accounts for >8–10% (K=32 already measured
−4%, so K=64 stands); prescaled-half operands (old variant A) reopen **only
if B/C implicates raw-int staging/flush machinery** — not merely because the
tolerance policy changed; barrier reduction only on a measured *production
arrival-stall*, not a synthetic empty barrier; **half accumulators stay
dead** — the measured +0.43% NLL was a real quality regression and the new
margin contract does not convert it into an optimization opportunity;
scalar/MMA interleave is unproven on Apple's shared issue machinery (MPP was
only 1.05–1.21× over `simdgroup_matrix`, fp8 no faster than fp16 — there is
no hidden 2× path to overlap against).

Bottom line adopted: **one attribution experiment is worth chasing; the 7%
itself is not** an open-ended optimization stream. 48.8 vs 52.5 while
carrying a different engine, contract, and serving goals is close enough to
call mature if the roofline says so.

## Q2: Latency-bound attention shape — ADOPT the barrier-free direct-read block-partial probe ("R3")

Phase 0's diagnosis inverts the old advice: with head-major at 1.00×, fp16
paying no wall for 5.12× bytes, R1b's 2× from amortizing barrier cadence,
factor 4 lost to registers, and R2's restage losing 0.76× to *extra*
barriers, the implicated cost is the **barrier-stage–serial-row cadence**,
not bytes.

**R3 kernel shape**: one simdgroup owns (one query head × one sequence
block × one-or-two query tokens); reads compressed turbo3 K/V **directly
from device memory**; dequantizes into registers; keeps per-token
`(m, l, acc)` online-softmax state; writes one block partial per query
token; reuses the existing block-order merge kernel. Six such simdgroups
are packaged per threadgroup (one per GQA query head) purely for dispatch
efficiency — **no shared threadgroup K/V, no cross-simdgroup barriers.**

- Variant order: **H1×T2 first** (direct-read analogue of the R1b win),
  **H2×T1 second** (two query heads sharing a KV head — tests whether
  modest head reuse beats staging); H2×T2 only if neither spills.
- Duplicated-read math (why 6× turbo3 reads are acceptable): ~200 B per KV
  head/position × 6 ≈ 1,200 B, vs the current shared fp16 path's ~1,024 B —
  and fp16 already measured at the same wall time as turbo3. The
  duplication lands in a byte range proven latency-tolerant; the risk is
  repeated dequant ALU, which Phase 0 estimated well under issue capacity.
- **Block-size sweep is first-class**: B = {128, 256, 512, 1024, 2048}.
  Smaller blocks → more independent simdgroups, shorter serial softmax
  chains, more merge partials; expert's prior: **256–512 wins**. Block-size
  changes fold order → gate under the margin-aware L2/L3 contract.
- Do NOT start from "more work per threadgroup" — factor 4 already showed
  the register cliff; the direction is more independent partials with less
  synchronization per partial.

**Barrier-cost calibration bench** (no published M4 constant exists; the M1
study's few-cycle intrinsic barrier number must not be used — the real cost
is *arrival skew* after unequal loads/dequant/`simd_sum`/exponentials).
Three otherwise-identical 8-row loops: (1) cooperative load + 2 barriers +
production arithmetic; (2) same, with balanced dummy arithmetic (uniform
arrival); (3) direct device reads + production arithmetic, no barriers.
Differences separate barrier floor / arrival skew / duplication cost /
serial-softmax dependency. Vary block size and token factor across it.

## Q3: Slot-batched decode — ADOPT; promoted above the generic verifier

Confirmed serving order: P0 fixes → v1 multislot state/scheduling (still
first: isolation, affinity, cancellation, the review-2 quantum contract) →
**N=2 slot-batched linears** → generic `verify_lanes` → learned drafting
only on a measured oracle budget. Rationale: two always-committing columns
have no acceptance probability, no drafter bytes, no rejection bookkeeping,
and work under ordinary mixed traffic.

**Honest ceiling (goes in the multislot doc — do not promise 2×):**
S₂ = 2 / (α·f + 2(1−f)) with f = batchable-linear share of solo decode and
α = batch-2 linear time / solo linear time. At our measured f ≈ 0.85 and
perfect reuse (α=1): **1.74×**. α=1.15 → 1.57×. Approaching 2× requires
also batching norms/elementwise/GDN/output-head work.

**Kernel structure**: adapt the select-form GEMV — keep 4 output rows per
simdgroup, hold *both* slots' activation slices in registers, unpack each
weight mask **once**, maintain paired (`float2`) accumulators, apply one
weight scale to two independent row outputs. Do **not** use the 8×8 matrix
path (badly underfills N=2). Q4/Q8: integer partials shared structurally,
`weight_scale × activation_scale` fold stays slot-specific. **B1 N=2 is
measured separately** — lower bpw goes issue-bound sooner; the T2 result
does not transfer.

**Pre-registered gates**: first an isolated projection-mix bench (N=1 twice
vs N=2 once, same matrices, bytes read, spills, aggregate rows/s), then a
true two-slot full-token bench: ≥1.70× aggregate → strong pass; 1.50–1.70×
with per-slot latency ≤1.35× solo → useful pass; 1.25–1.50× →
capacity-only; <1.25× or any single-slot regression → park. Mandatory:
instant fallback to the existing single-slot GEMV whenever only one slot is
runnable. (Published Apple continuous-batching scaling is at larger N —
confirmation of mechanism, not an N=2 promise.)

## Q4: B1 inner product — ADOPT the three-candidate Phase 0B (folded into the binary-tier doc)

Binary simplifies the select form: Σwᵢxᵢ = 2·Σ_{wᵢ=+1} xᵢ − Σx — **one**
conditional accumulation per element (T2 needs two). Candidates to bench:

1. **Positive-mask select (baseline)**: load 16–32 activations once,
   `sum_all`, per-row conditional accumulate on bit=1, return
   `scale·(2·sum_pos − sum_all)`.
2. **Float sign-bit XOR**: `bits ^= neg_mask & 0x80000000` — exact ±1
   multiply for finite float activations (signed zero included), no
   `sum_all` correction. Whether it beats `select()` depends on Apple's
   lowering of packed-bit → sign-mask; serious candidate, bench it.
3. **Int8 bitplane + popcount**: transpose each 32-activation group into 8
   bitplanes; S₊ = Σ_b 2^b·popcount(M ∧ P_b) − 128·popcount(M ∧ P₇). Turns
   32 conditional adds into 8 and+popcounts, **but preprocessing (int8
   quant + bitplane transpose + Σx, fused) MUST be inside the measured
   time** — activations amortize across thousands of rows, so it may pay,
   but a candidate that wins only with free pre-transposed activations is a
   kill, not a pass.

Deprioritized: generic LUT schemes (T-MAC is CPU-shaped; our T3 kill
independently demonstrated the Apple-GPU failure mode — dependent lookups
convert byte savings into an issue-bound regression); XNOR-popcount proper
(needs binary activations; for int8 it *is* candidate 3).

**Kill criterion is projection-mix wall time, never effective GB/s alone**
(a lower-bpw format can post low GB/s and still be faster, or high
elements/s while preprocessing eats the token). With T_B1 including
preprocessing and R the measured non-weight residual, predict
t_token = T_B1 + R. Pre-registered: **T_B1/T_T2 ≤ 0.60** and predicted
token ≤50 ms → strong GO (the ~20 tok/s thesis); **0.60–0.72** →
conditional GO, only if Phase 0A quality lands materially closer to T2 than
the 89.5% retention suggests; **>0.72** → kill the 20–23 serving headline.
Since B1 carries 0.529× T2's code bytes, ratio 0.60 needs ~85% of T2's
stream rate ≈ **79 GB/s**; the practical early kill line is **~70 GB/s
across the full production projection mix**. Expert's prior: sign-XOR or
positive-mask float wins on base M4; the popcount microbench is cheap
enough to run anyway.

## Q5: Sub-2-bit KV floor — "no universal floor is known; uniform sub-2-bit K+V is unsupported by the literature." ADOPT three instrument extensions

Every published sub-2-bit success changes the representation, not just the
bit count: KIVI (asymmetric K/V granularity at 2-bit), Kitty (2-bit hurts
long-context reasoning unless sensitive components stay high-precision),
AsymKV (1-bit only for selected layers), SVDq (~1.25-bit via decomposition
+ mixed treatment), H1B-KV (1-bit keys, higher-precision values, model
adaptation). Nothing supports "1.25 bits uniformly across K and V." Our
normalized-QK / (1/16)-scale regime is the favorable case (Open-TQ-Metal)
but guarantees nothing about the next 20–30%; no published evidence exists
for hybrid GDN/GQA at all — fewer attention layers inject less KV error,
but each error still propagates through later GDN/FFN layers.

KL-KV instrument extensions (adopt into the instrument plan):

1. **K-only / V-only / both arms**: fp16-K+cand-V, cand-K+fp16-V, both.
   K error perturbs rankings and softmax mass (nonlinear); V error is
   post-softmax and near-linear — **K-only reveals depth problems first.**
2. **Layer-local attention metrics** vs fp16 at selected layers/heads/
   positions: score RMS/max error, softmax forward KL, top-attended-token
   overlap, top-1/top-2 score margin, output relative-L2/cosine,
   post-attention residual error — find the *first* layer where error
   grows instead of waiting for final logits.
3. **Tails and persistence per depth bucket**, not just means: mean /
   median / p99 / p99.9-or-max / fraction above fixed KL levels / longest
   consecutive over-threshold run / margin-certified top-1 flip rate. Our
   own T2+turbo3 numbers (mean KL ~0.0115 nats, max ~2.83) already prove
   mean depth-flatness is not a serving-safety statement.

**Earliest depth-compounding gate** (cheap, fails before NLL): on one
teacher-forced stream, fit robust trends of K-only softmax KL /
attention-output error / final-logit KL vs position with whole-window
bootstrap CIs. Fail if deep-bucket K-only softmax KL > ~2× shallow with CI
excluding content noise, or margin-violating ranking flips rise
monotonically with depth, or p99/p99.9 final-logit KL grows while the mean
stays flat. Ladder afterwards: 8K KL → 32K paired → needle → on-policy →
128K only for clean codecs. **Every next codec is evaluated against both
fp16 AND turbo3** — turbo3 is the measured incumbent now, not a theory.

## Q6: Error-envelope construction — ADOPT per-class envelopes; CORRECTS the round-1 pooled-ensemble sketch

The round-1 integration doc sketched one envelope from {two Metal reduction
orders, serial-vs-batched, CUDA pair, repeats}. Superseded — those pairs
belong to different numerical classes and must not be pooled:

- **Classes**: GEMM (serial GEMV reference / float-staged chunk / G′
  half-input), attention (per-head online softmax / blocked GQA / one
  accepted alternate block size or merge order — **R1b is bit-identical to
  blocked GQA and is NOT an independent sample**), GDN (exact by
  construction — keep exact gates; never widen to manufacture envelope
  samples), KV codec (**turbo3-vs-fp16 is a quality comparison; its lossy
  distribution must never calibrate arithmetic-reorder envelopes**).
- **Minimum trustworthy ensemble per class**: deterministic repeat (must be
  exactly zero) + two independent accepted variants + the A-vs-B cross-pair
  as a holdout consistency check. With only one nonzero pair the envelope
  is *provisional* — keep stricter gates. **CUDA↔Metal is an external
  holdout, not a calibration member** (too many combined differences).
- **Stratify, never take a global max**: operation family × dtype ×
  token-width bucket (2–12 / 13–32 / 33–64 / 65–96) × K/row-shape ×
  context bucket (0–2K / 2–8K / 8–32K / 32K+) × attention/KV mode.
  Hierarchical fallback: a thin stratum inherits a conservative *parent*
  envelope, not the global worst. A 96-row output-head outlier never
  loosens a 12-row FFN gate — it gets its own stratum or a bug hunt.
- **Quantiles honest to corpus size**: at 2,048 positions p99.9 rests on
  ~2 tail points — use p99/p99.5 as the contract threshold, keep max as a
  separate hard alarm, bootstrap over contiguous *windows* (positions are
  correlated), reserve p99.9 for ≥~10K quasi-independent positions.
  Minimum corpus: 2,048 calibration + 2,048 disjoint holdout positions
  across prose / code / tool-shaped JSON / long-context.
- **Margin certificate on the relevant set**: over the union of reference
  top-64, candidate top-64, and both winners, record
  ρ(a,j) = (|e_a|+|e_j|) / (z_a−z_j); a top-1 flip is impossible when
  ρ < 1. Whole-vocab max/RMS stay as diagnostic alarms only — a huge error
  on an irrelevant tail logit must not drive the decision.
- **Thresholds**: per stratum, take the quantile of every accepted
  implementation pair and use the **median or upper-CB across pairwise
  quantiles** — not the single largest raw sample; separate hard alarm =
  calibration max + small numerical allowance; validate on the holdout;
  **leave-one-implementation-out** (build from two pairs, third must
  pass). A new kernel never automatically expands the envelope judging it;
  a repeat outlier implementation becomes its own numerical class with its
  own L2/L3 + task gates, or is rejected.
- **Contract storage per class**: normal envelope (p99/p99.5 RMS + KL,
  p99/p99.5 margin ratio, min top-k overlap) + hard alarms (max logit
  error, max KL, any certified-impossible flip, any NaN/Inf). Promotion =
  inside envelope + no alarm + holdout pass + free-running task comparison
  after first tolerance-legitimate divergence. Lift this verbatim when the
  envelope-instrument plan doc is written.

## Updated execution order (supersedes the round-2 triage §"Revised execution order")

1. **P0 fixes** (unchanged: constraint RAII, reservation guard, wide
   NLL/logit gate, screening-math doc fix).
2. **One-day A/B/C MMA roofline** at matched geometry; prefill declared
   mature and closed unless lower-CB headroom exceeds ~1.10–1.15×.
3. **R3 barrier-free block-partial attention probe** + block-size sweep +
   barrier-calibration bench.
4. **v1 two-slot state/scheduling** under the review-2 quantum contract.
5. **N=2 slot-batched linears** — now explicitly above the verifier.
6. **B1 Phase 0A/0B** (0B = three-candidate bench with wall-time kill
   lines; binary-tier doc updated).
7. **KL-KV instrument extensions** (K/V arms, layer-local metrics, tails,
   depth-slope gate).
8. **Per-class numeric envelopes** (this doc's Q6 rules).
9. **Generic `verify_lanes` + oracle/suffix** after the GEMM substrate
   settles.
10. **Learned drafting stays parked** unless the oracle leaves a measured
    budget.
