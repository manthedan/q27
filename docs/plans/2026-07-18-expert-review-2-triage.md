# Expert review 2 triage — "Deep Research Review" (received 2026-07-18)

Second independent review of the 2026-07-17 brief. Verified against source
per house rule; converges with review 1 on the large items and adds new
machinery. Dispositions per section below. **Convergence table first,
then what survived verification, then amendments.**

## Convergence (two independent reviews + the ledger agree)

| Item | Review 1 | Review 2 | Disposition |
|---|---|---|---|
| B1 bandwidth = dilution, not regression | B≈92.3, c≈13.0 | β≈95.3, F≈14.2 | SAME MODEL — constants differ by the GB/GiB unit mixing R1 flagged; reconciled below |
| GDN concentration has literature support | Q-S5, Quamba | classical RNN quant, MambaQuant, Quamba(2), outlier-channel PTQ | ADOPTED; R2 adds the delta-rule mechanism analysis |
| >100% recovery not intrinsically suspicious | soups/layer-swap | WiSE-FT endpoint overshoot, task-arithmetic, DARE | ADOPTED, probation battery stands |
| Dual gate (NLL + capability) | capability non-inferiority | dual conjunctive gate + KL tie-breaker | MERGED (A5/A6 below) |
| Tree speculation caution on GDN | branch state costs the free tile | +15% external hybrid datapoint (draft-ceiling bound) | CONSISTENT; narrow version only |
| Capability evals missing | elevated to product risk | "most conspicuous hole" | OVERTAKEN in part — cb099ee + 9649f2b landed before this review's basis snapshot |
| Thermal governance | interleave discipline (R1 §3) | powermetrics + admission rule, "not optional hygiene" | MERGED as harness amendment |
| Persistent drafter store / energy column | (our ideas list #1/#6) | independently proposed | CONVERGENT VALIDATION — promoted to plan candidates |

## Arithmetic verification (reproduced locally)

- **F cross-check — the review's strongest number:** ledger entry 207's
  profile (T2 GEMV 84.7% of a ~91 ms token ⇒ 13.9 ms non-GEMV) matches
  their fitted F = 14.2 ms within 2%, different machine, different
  instrument. VERIFIED — this is what elevates dilution from fit to
  finding.
- **γ\* ≈ 0.89 derivation VERIFIED internally consistent:** drafter cost
  per proposal = B_d/β + F ≈ 19.5 ms ≈ 0.24 serial-token-times (their
  numbers); 16 proposals ≈ 3.9 token-times; τ(16, γ) ≥ BE(16) + 16d
  ≈ 7.75 solves to γ ≈ 0.89 under the geometric acceptance sum — break-
  even, not the ship bar (ship needs γ ~ 0.93+). The DSpark τ = 3.209
  sits far below. The "closed condition" statement is sound.
- **BE(16) reproduction:** their R(w) flat model gives 3.9 vs measured
  4.13 (~6% under) — model family validated; see the w-peak conflict
  below for where the flat model loses to our measured tile curve.
- **F-halving → 22.2 tok/s for B1** (37.9 + 7.1 ms): arithmetic checks;
  becomes the motivation for the fixed-cost attack list (attention at
  latency-bound rates, GDN state ops, readback cadence).
- **Constants reconciliation with review 1:** R1 fit (92.3/13.0) used
  W_T2 = 6.802 GB decimal @ 86.7 ms (the 78.5/11.54 pair); R2 used
  6.33 GB @ 80.65 ms. Same model, different byte/unit pairs — exactly the
  unit debt R1 flagged. RESOLUTION: the per-token-bytes instrumentation
  (one-line addition, R2 §3.2) lands with the probe; both fits then
  re-derive from counted bytes.

## Dispositions per section

1. **Grafting — probation battery ADOPTED with one scoping correction.**
   The three artifact checks (paired per-position bootstrap CI; held-out
   code + multilingual replication; **KL(mixed ‖ official 17 GiB) <
   KL(T2 ‖ official) as the decisive quality gate**) join the census
   amendments. Correction: the "existing two-engine lockstep harness" is
   our `--kl-kv` family, which is a SAME-model KV-precision comparator —
   the cross-model paired-logit driver needs authoring (tokenizer is
   byte-identical across tiers, teacher-forced comparison machinery
   exists; estimate a driver + one session, not "one afternoon"). The
   shared_f32 zero-byte control already ran in the census (the
   publication carries its result). The A0–A7 skeleton (§8 of the
   review) is ADOPTED as the combination-arm sequence — cheaper than the
   full 2^5 factorial, which is retained as the escalation path if any
   J_AB/additivity signal appears (R2's own [0.8,1.2]×ΣΔ classification
   pre-registered). Per-layer bisection fallback on sub-additivity <
   0.8×ΣΔ adopted (mirrors the KV census's honest Σ-means precedent).
2. **GDN mechanism — the delta-rule analysis RECORDED as analysis (not
   citation) and its sharp falsifiable prediction ADOPTED into Phase E:**
   graft qkv but keep B1's β → recovery should collapse
   disproportionately; graft alphabeta INTO T2 → near-zero effect (T2's
   gates already near-exact). If both fire as predicted, the
   eigenvalue/addressing-sensitivity mechanism is confirmed on this
   family; the census becomes the first published evidence (publication
   item already on the mini; review endorses it as a contribution).
3. **Bandwidth — dilution ADOPTED as the primary model** (was "leading
   hypothesis" after R1; the 13.9-vs-14.2 independent cross-check
   promotes it). Probe ladder MERGED: (1) B1 decode-bench profiler shares
   (predicts GEMV ≈ 73% of token at 93–97 GB/s — one run, existing
   instrumentation) before (2) R1's N-sweep slope fit, before (3)
   footprint sweep (same kernel, 3.6 vs 6.3 GB) and scale-walk arms.
   Consequence recorded: B1's next 15% lives in the fixed ~14 ms, not
   the GEMV — the kernel rounds on B1's GEMV are done.
4. **Residency — toolbox MERGED:** R1's full-footprint 4-arm bench
   stands; R2 adds: setPurgeableState(NonVolatile) check (cheap audit
   item), the anonymous+mlock route as arm E, **chunked-staging private
   blit (256–512 MB staging pool with fences — resolves R1's 2×-footprint
   concern against MTLIOCommandBuffer direct loads; both recorded, bench
   decides)**, and the `iogpu.wired_limit_mb` sysctl for the official
   17 GiB tier → QA/ops docs. CB-turnaround verdict recorded: design
   artifact, already ~1–2% (K=8 + GPU argmax); wrong-sign argument (R1)
   and microseconds-scale evidence (R2) agree. **Thermal governance
   ADOPTED as harness amendment:** powermetrics thermal-pressure log line
   in every bench artifact + thermal-state admission rule (twice-burned:
   gate-3 wall anomaly, 12.66→10.57 ceiling).
5. **Speculation — the closed-condition formulation ADOPTED** (γ\* ≈
   0.89 at w=16; reopens iff π/β changes — cooperative tensors/M5, a
   block-parallel drafter that escapes the F floor with better acceptance,
   or trees). The **"cruel corollary" RECORDED:** every F reduction
   raises the learned-drafter bar further — engine improvement
   monotonically closes the window it doesn't reopen. **CONFLICT —
   R2's "S peaks at w ≈ 3–8" REJECTED for our engine:** our measured
   verify cost is flat per 16-token tile with sweet spots {16,32,48}
   (lever 2, S(48) = 3.94× measured); the flat R(w) model's w≈4
   crossover is superseded by that measurement wherever they disagree.
   The w=48 suffix widening stands. Tree version recorded narrow: branch
   only on multi-continuation suffix hits (external hybrid datapoint
   ~+15% vs +35–42% pure-attention, draft-ceiling bound).
   **Sampled-temperature coverage — VERIFIED in source
   (metal_server.cpp:661,660,952: sfx/mtp/constraint all gate on
   temperature==0.0).** Limitation, not a bug; the sampled-path
   acceptance rule (commit iff token equals the sampled target token;
   resample at first mismatch) and its battery are pre-registered as a
   suffix-plan amendment; serving docs note the temperature-0 gate
   explicitly until then.
6. **Equivalence — three amendments ADOPTED onto R1's margin-certified
   contract:** derive the byte-exact gate length from the measured
   per-token flip probability (per tier) instead of the fixed 16;
   compensated (Neumaier/Kahan) accumulation in the chunk-GEMM flush
   paths as the Δ-shrink lever (weight side already integer-exact);
   batch-invariant reduction discipline evaluated before any correctness
   claim under the two-slot scheduler. External fragility datapoints
   (batch-size/hardware sensitivity studies) recorded as corroboration.
7. **Absent items — dispositions:** capability battery (OVERTAKEN:
   cb099ee machinery + bands exist; maps to A6) · trees (recorded
   narrow, above) · thermal (adopted, above) · **training-side "B1+"
   vendor ask ADOPTED and consolidated with the pre-QAT-master ask — the
   census publication is the business case, quantified** · persistent
   per-tenant drafter store (convergent with our idea #1; promoted to
   plan candidate — snapshot-ride like the prefix cache) · energy
   tokens/joule column (convergent with idea #6; joins the bench-artifact
   amendment with powermetrics) · **feature-interaction matrix ADOPTED
   into QA_BEFORE_RELEASES.md** (bursts × snapshots × two-slot × KV
   exception cells — one seeded end-to-end matrix run; "pass alone, fail
   together" insurance) · census publication (already the mini's item;
   both reviews endorse).
8. **A0–A7 skeleton — ADOPTED as the combination-arm sequence** (amends
   the census plan; ≤ 10 NLL runs + one lockstep driver session).
9. **Overall assessment** — no disagreement; the three resolutions
   (probation battery / dilution identity / closed condition) match our
   triage-1 state with sharper instruments attached.

## Amended documents (this commit)

- Census plan: A0–A7 sequence + β-gate perturbation + KL-to-official
  tie-breaker + capability dual gate (maps cb099ee) + per-layer bisection
  fallback.
- QA checklist: feature-interaction matrix + thermal/powermetrics
  bench-artifact line.
- Suffix plan: sampled-path acceptance battery pre-registration note +
  temperature-0 gate documentation debt.
