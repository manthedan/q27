# Expert review triage — review of EXPERT-BRIEF-2026-07-17 (received 2026-07-17)

House rule applied: every checkable claim verified against source/ledger
before adoption. Dispositions: **ADOPTED** / **VERIFIED-AND-ADOPTED** /
**CORRECTED-AND-ADOPTED** / **RECORDED** (no action) / **REJECTED** (none
this round).

## 0. The reviewer's two audit notes

- **"Census plan ends RESULTS (pending)"** — CONFIRMED, adopted as
  publication requirements. The 25-arm table, absolute NLLs, and the
  recovery denominator live on the mini (publication is its queued item).
  Until committed, the 114.4% figure is ledger-summarized, not
  independently auditable. Publication MUST include: absolute NLL per arm,
  the B1/T2 anchor pair on the same machine, per-arm paired ΔNLL in
  nats/token as the primary metric (recovery % secondary — unstable at
  modest denominators), all 25 arms reported (winner's-curse discipline),
  and resample-based intervals (§1 Phase A below).
- **"Decode-status discrepancy"** — RESOLVED, and the brief's Q3 premise
  was stale (my error, corrected in the brief by addendum). Ledger entry
  207 (Phase 4, 2026-07-15) already attributes the 8.2–9.4 vs 12.66 gap:
  (a) launch fault-in inside the timed window — fixed same-day (single
  MTLBuffer + residency set); (b) thermal-state mismatch — the ceiling
  itself re-measured 12.66 → 10.57 tok/s mid-session; GPU busy 96% of
  wall, T2 GEMV 84.7% at ~77 ms. "98–99.2% of resident GEMV ceiling" in
  the current-state table is the same statement at matched moment. No
  reopening, no definition change.

## 1. Grafting (the 114.4%)

**Framing CORRECTED-AND-ADOPTED.** The reviewer is right that >100% is not
intrinsically suspicious (T2 is not a mathematical upper bound; soups and
layer-swap literature support complementarity) — and right that the result
is currently a *grafting* result, not a quantization-sensitivity result.

**Their strongest single point, verified and STRENGTHENED with a premise
correction.** They wrote that `ssm_alpha/beta` are F16 in both packs, so
the 46.9% alphabeta recovery "cannot be caused by encoding." FORMAT.md
says otherwise: F16-both is the **official**-tier policy; our packs store
alpha/beta as **F32 (B1)** and **ternary (T2)**. The correction makes
their argument *stronger*, not weaker: the graft replaced **higher**-
precision F32 base values with **lower**-precision ternary donor values
and still recovered 46.9% of the gap. Encoding can explain none of it;
the T2 checkpoint's *trained values* are simply better there, even at
ternary precision. Conclusion stands: donor-weight/training/co-adaptation
signal concentrated in the recurrent pathway. The publication leads with
this arm. Census claim relabeled accordingly: "quality difference
concentrated in the GDN subsystem" is supported; "binary quantization
error concentrated in GDN" awaits the same-source control (Phase D).

**Adopted experiment design (amends the census plan):**

- **Phase A — holdout confirmation before combinations.** Frozen
  independent units: non-overlapping wikitext document groups, code
  continuation, instruction/chat, structured tool/JSON traffic, one long
  continuous-state stream. Absolute paired ΔNLL (nats/token) primary;
  recovery % secondary. Bootstrap documents (block bootstrap for
  continuous-state). Minimum set: B1, T2, B1+qkv, B1+alphabeta, B1+both.
- **Phase B — full 2^5 GDN factorial** (qkv, alphabeta, gate, out,
  GDN-only same-dtype cohort — split out of the broader shared_f32 arm),
  32 arms, pairwise interaction terms J_AB with resample CIs. Distinguishes
  additive repair / super-additive co-dependence / compensation collapse.
- **Phase C — reciprocal grafts** (T2 base + B1 gdn classes). Sign-pattern
  reciprocity discriminates "intrinsically important locus" from
  "checkpoint complementation."
- **Phase D — source × encoding control.** 2×2 (source × encoding) via
  dequant→re-encode; a common pre-QAT master is the gold standard and is
  unavailable to us — recorded as a **vendor ask** (they hold the masters).
- **Phase E — localize gdn_qkv.** q/k/v row split (validate vendor fused-
  row order first), depth bands, individual layers, head groups ranked by
  empirical retention half-life (log gate/decay values per head). Plus the
  state-drift experiment: identical recurrent-state snapshot, teacher-
  forced, compare one-step logits then free-evolved state at 16/128/1K —
  decomposes local projection repair vs prevented state drift.
- **Capability gate for the ship candidate.** NLL is discovery, not the
  serving gate: paired non-inferiority vs T2 on ground-truth success with
  the tolerated deficit and sample size fixed BEFORE running (U2 harness;
  ground-truth graded QA/reasoning, executable code, JSON/tool validity,
  long-context, small end-to-end agent set). Agreement-with-T2 alone is
  insufficient (shared wrong answers).

**GDN literature note (RECORDED):** Q-S5 and Quamba report recurrent /
selective-scan pathways unusually quantization-sensitive (sub-8-bit
recurrent weights degrade; activation outliers inside the recurrence) —
mechanistically consistent (48 GDN vs 16 attention layers; errors persist
in state). But "gdn_qkv alone recovers the whole gap" is not in the
literature: treat as empirical finding, not prior.

**Low-rank delta probe (ADOPTED, flagged high-value, CPU-only):** SVD of
`W_T2_donor − dequant(W_B1_base)` for gdn_qkv. If the delta is low-rank or
head-structured, B1 + a small F16/Q8 low-rank correction may recover most
of the gain for tens of MB instead of 315 — an unusually favorable trade
on a stream-bound device. Pre-register before running.

## 2. B1 "lower bandwidth" — accounting hypothesis VERIFIED-AND-ADOPTED

Recomputed their two-point fit from the ledger numbers (19.23 tok/s @ 69.3
GB/s; 11.54 @ 78.5): W_B1 = 3.604 GB/tok, W_T2 = 6.802 GB/tok →
**B ≈ 92.3 GB/s, c ≈ 13.0 ms/tok**, reconstructing both walls (52.0/86.7
ms). Arithmetic checks; two points always fit, so this is the leading
*hypothesis*, with the registered discriminating probe:

- **Projection-repeat sweep** N ∈ {1,2,4,8} in the exact decode bench:
  one command buffer/readback, N full production-shaped projection
  schedules at disjoint full-footprint addresses, non-projection work
  once, GPUStart→GPUEnd + wall recorded. Fit T(N) = c + N·W/B per tier.
  Same slope+intercept → close the residue as accounting; B1 slope lower
  → true per-byte efficiency difference; wall-slope match with physical-
  counter mismatch → transaction amplification.
- If B1's slope is lower: **32-bit-load-discard-half arm** (request
  granularity / outstanding-miss depth) and a **scale-free synthetic arm**
  (scale walk = 2× the logical-byte fraction at 1 bpw).
- **Unit debt adopted:** every byte count paired with its exact run and
  unit (the round's 78.5 GB/s implies 6.80 decimal GB = 6.33 GiB — the
  ledger's "GiB" usage there is loose). Normalize on commit of the probe.

## 3. Residency / decode turnaround

- **Comment overstatement VERIFIED, fix landed this commit:**
  metal_backend.mm said pages "stay wired" / the set "keeps pages wired" —
  requestResidency is preparatory and best-effort per Apple's docs.
  Comments now say residency request. (Two sites.)
- **Turnaround wrong-sign argument ACCEPTED:** the synthetic ceiling pays
  one command-buffer completion + readback *per token*; production K=8
  pays one per eight — turnaround has the wrong sign to explain
  artifact-vs-ceiling. Consistent with ledger 207's paging+thermal
  attribution; Phase-4 remainder stays "readback turnaround + the GEMV
  itself," with the reviewer's 4-interval instrumentation (encode /
  commit→GPUStart / GPUStart→GPUEnd / GPUEnd→unblock) adopted as the
  measurement shape.
- **Full-footprint ceiling control ADOPTED (the genuinely new experiment
  here):** the synthetic ceiling reuses one representative tensor per
  class — it reproduces bytes and dispatch shapes but NOT the 7 GB unique
  address footprint (page tables/TLB/file-backing/tensor order). 4-arm
  byte-identical bench: A compact-synthetic, B anonymous shared full
  offsets, C file-backed full offsets, D private full offsets. Readout
  table per the review (A>B≈C≈D → footprint/TLB; A≈B>C → pager; D>B≈C →
  private wins; all equal → benchmark/thermal mismatch). Pre-register as
  its own plan; run in the next GPU window.
- **MTLIOCommandBuffer load path RECORDED** for any future private arm
  (file→MTLBuffer direct; no 2× resident footprint; load gates on
  currentAllocatedSize / recommendedMaxWorkingSetSize / RSS / pressure).
- **CPU/GPU double-buffer pipelining RECORDED** as a few-percent lever
  (2 K-step rings, bounded speculative overrun for EOS/stop/grammar), not
  tens of ms. Not scheduled.
- **Thermal discipline re-affirmed:** interleaved A/B/A, power state
  recorded (the 12.66→10.57 lesson is already protocol; the probe plans
  above inherit it).

## 4. Speculation

- **Finite-class optimality certificate ADOPTED** into the suffix plan:
  conservative bounds (U on committed tokens from oracle replay, L on
  round cost, G⁺ on baseline) with session/prompt-level resampling
  (rounds within a continuation are correlated — the 1,093-round figure
  gets session-level bootstrap CIs as a **gate-6 amendment**).
- **DSpark closure noted as stronger than break-even:** the chained upper
  bound excluded drafter forward cost and still failed — no scheduling
  whose only benefit is reducing draft cost can rescue that drafter.
- **Next lever funded-in-principle:** adaptive multi-source zero-byte
  drafting (draft length by match length / source identity / verifier
  margin; suffix + prompt + session-history + retrieval sources, one
  chain). Exploits the flat 16-token tiles; preserves state semantics.
  Gate 6's histogram is the decider input, as already registered.
- **Tree speculation caution RECORDED:** GDN branches need recurrent
  state copies or replay — trees can spend the free tile on state
  bookkeeping. Shallow branching only after state-replication cost is
  measured.

## 5. CUDA↔Metal equivalence — margin-certified contract ADOPTED

Tiered per the review: (1) backend-internal byte-exactness (K=1 vs K=8,
chunked paths, snapshot round-trips, storage modes, buffer segmentation,
cold/warm — any diff = defect, not FP noise); (2) teacher-forced cross-
backend comparison on a fixed stream with margin/KL/top-k telemetry, no
free-running comparison past the first split; (3) the **exact argmax
certificate**: with d_i = z_M − z_C, c = (max d + min d)/2,
ε∞ = max|d−c|, margin m — if m > 2ε∞, identical argmax is mathematically
required (a disagreement there is a bug; m ≤ 2ε∞ is admissible
divergence). Replaces "low-margin token" hand-waving with a proof
obligation; (4) per-layer/state probes (residual, qkv, GDN ring state,
attn out, hidden, logits) — bounded per-layer perturbation healthy,
monotonic state drift or one-kernel-family discontinuity actionable;
(5) sparse FP64 offline arbitration on saved packed rows. Amends the
margin-aware-gates plan; release gate: keep the 16-token smoke, add
512–2K teacher-forced tokens across domains/depths, zero margin-certified
violations, calibrated envelopes, capability non-inferiority final.

## 6. "Conspicuously absent" — all eight ADOPTED

1. Full-footprint ceiling → §3 bench.
2. Same-source control → §1 Phase D (+ vendor ask).
3. Sub-class compression / low-rank delta → §1 probe (high-value).
4. **Direct mixed-pack speed:** the ~17–19 tok/s figure is inferred from
   pack size, not measured; the composed candidate gets its own decode
   bench incl. two-slot serving and 131K pressure before any claim.
5. **Session-reset vs continuous-state NLL:** document-reset and long
   recurrent-state arms both run; the difference itself is diagnostic.
6. **First live capability/agentic run ELEVATED** — the harness exists
   unrun; for a server whose differentiator is the agent lifecycle this
   is now a larger product risk than another perplexity refinement.
   Scheduled as the first GPU-slot item after the current window.
7. Physical-vs-logical accounting → §2 probe (+ unit debt).
8. **Execution-order artifact layout:** byte-identical repack in actual
   decode traversal order as a page/TLB/prefetch control — cheap, joins
   the §3 bench as an optional arm.

## 7. Recommended sequence — ADOPTED as ordered

1. Re-anchor decode physics (§3 bench + §2 sweep + interval timing).
2. Confirm the graft on independent holdouts (absolute NLLs + intervals).
3. 32-arm GDN factorial, then reciprocals.
4. Zoom q/k/v × depth × retention heads; low-rank/row-selective correction.
5. Build the best actual pack; gate NLL + capability + real decode +
   two-slot + 131K + pressure together.
6. Speculation: finite-class certificate, then adaptive multi-source
   n-gram drafting.
7. Margin-certified teacher-forced differential as the permanent contract.

Addition of ours: the low-rank SVD probe (§1) and all arithmetic
verifications above are CPU-only and can start without the GPU window.
