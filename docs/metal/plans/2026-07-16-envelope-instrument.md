# Envelope instrument — `--envelope` mode (margin-aware Phase 1)

Status: PRE-REGISTERED before measurement (results appended below).
Parent: 2026-07-15-margin-aware-gates.md Phase 1 ("build the envelope
before using it") with the Q6 per-class rules from round 3 adopted
verbatim (per-class pairs, never pooled; p99/p99.5 as contract, max as
hard alarm; margin certificate ρ on the relevant set; repeats must be
exactly zero).

## Instrument

`--envelope MODE` rides the `--nll FILE [--nll-long N]` input path and
the `--kl-kv` lockstep structure: two engines on one Shared, teacher-
forced over the corpus, metrics per position. Modes = the legitimate
same-model reduction-order pairs on the T2 artifact:

- `half` — GEMM class: `Q27_METAL_GEMM_HALF` 1 (production G′) vs 0
  (float-staged chunk GEMM). Backend gains runtime setters
  (`set_gemm_half`, `set_gqa_threshold`) so one process can flip the
  knob between the two engines' lockstep passes — the env reads happen
  at backend init and both engines share one backend.
- `blocked` — attention class: GQA threshold forced 1 (blocked kernels
  at every depth) vs 0 (legacy per-head kernels). Forcing 1 rather than
  the 2048 default means the pair diverges at every position instead of
  only past 2 K context.
- `serial` — the fold-order class: chunked encode (width 12) vs serial
  encode (width-1 teacher forcing on the subject engine). The
  chunk-parity control measured this class's max at 0.646 over 384
  positions; this mode re-measures it at corpus scale.
- `repeat` — identical configs twice: every metric must be exactly zero
  (determinism gate rides along; any nonzero is a hard alarm).

Per position: whole-vocab max|Δlogit| and RMS (diagnostic alarms), KL
(base‖subject), top-1 agreement + baseline margin at every flip, and the
Q6 margin certificate ρ_max = max over j in top64(base) ∪ top64(subject)
of (|Δ_a|+|Δ_j|)/(z_a−z_j) — a flip is certified impossible when
ρ_max < 1, so a flip observed at ρ_max < 1 is a self-contradiction alarm
(instrument bug or state divergence, not rounding).

Output: p50/p90/p99/p99.5 quantiles + max for each metric, flip census
with margins, alarm lines (NaN/Inf, contradiction, repeat-nonzero), and
a CONTRACT block formatted for pasting into the margin-aware doc.

## Corpus and run plan

wikitext2 tokens (`data/wikitext2-test.tokens.bin`), 2,048 positions per
mode (the Q6 corpus floor for p99/p99.5; p99.9 deferred until a ≥10 K
multi-genre corpus run on the 24 GB rig — recorded as residue with the
disjoint-holdout requirement). Four modes ≈ tens of minutes total on the
mac-mini. Contract constants recorded per CLASS (GEMM / attention /
fold-order), never pooled, per Q6.

## Pre-registered reads

- `repeat` nonzero anywhere → instrument or determinism bug; STOP.
- A flip with ρ_max < 1 in any mode → contradiction alarm; STOP.
- Otherwise: p99/p99.5 of each metric per class become the margin-aware
  contract constants (this doc + the margin-aware doc's table); hard
  alarms = measured max + one order of magnitude, per the parent plan.
- Sanity anchor: the `serial` class p-max should sit near the known
  0.646 envelope (384-position sample); a large excursion above ~1.0
  suggests corpus-dependence worth a stratum, not a wider gate.

---

## Results (appended post-measurement, same evening, M4 16 GB)

**All four classes measured clean: zero contradictions, zero NaN, repeat
exactly zero at every position.** wikitext2, 2,048 positions per mode.

| class (mode) | max\|Δ\| p99 / p99.5 / max | KL p99 / p99.5 / max (nats) | flips | worst flip margin |
|---|---|---|---|---|
| determinism (repeat) | 0 / 0 / 0 | 0 / 0 / 0 | 0 | — |
| GEMM (half) | 0.486 / 0.670 / 1.163 | 0.0020 / 0.0031 / 0.0346 | 12 | 0.104 |
| attention (blocked) | 0.436 / 0.620 / 1.198 | 0.0020 / 0.0025 / 0.0074 | 14 | 0.031 |
| fold-order (serial) | 0.418 / 0.646 / 1.393 | 0.0017 / 0.0031 / 0.0266 | 11 | 0.145 |

- **Sanity anchor hit exactly**: serial-class p99.5 = 0.6461 vs the
  384-position chunk-parity control's max of 0.646 — the small-sample max
  was the large-sample p99.5, which is precisely the quantile story Q6
  predicted. The corpus max (1.39) shows why max is a hard alarm and
  never the contract.
- The three nonzero classes share one scale (they are all
  arithmetic-reorder classes on the same model); they remain stored
  per-class, never pooled, per Q6.
- **Every flip is certificate-consistent** (ρ ≥ 1 at all 37 flips across
  modes) and occurs at near-tie baseline margins (≤ 0.145) — the
  L3-legitimate shape. ρ p99 ≈ 1.1–1.7: ~1% of positions are
  flip-possible under these classes, ~0.6% actually flip.
- **Provisional status per Q6**: each class currently has ONE measured
  pair; the minimum trustworthy ensemble wants two independent accepted
  variants + a cross-pair holdout. Contract constants below are adopted
  as PROVISIONAL (stricter gates retained); the ≥10 K multi-genre corpus
  + holdout run is queued for the 24 GB rig.

### Adopted contract constants (provisional, T2, wikitext2 @ ctx ≤ 2048)

Per class: **normal envelope** = max|Δ| ≤ 0.7 (p99.5 class bound), KL ≤
0.004 nats (p99.5), flips legitimate only at baseline margin < 0.15 with
ρ ≥ 1. **Hard alarms** = max|Δ| > 2.0 (corpus max 1.39 + allowance), KL >
0.05, any flip with ρ < 1, any NaN/Inf, any repeat-mode nonzero.

### Immediate consequence applied

The oracle state gate's probe tolerance (0.7, set this afternoon from
the 384-position envelope) sits between its class's measured p99.5
(0.646) and corpus max (1.393) — a ~0.5–1% false-fail per probe.
Recalibrated to 1.5 (above corpus max, still well under corruption
signatures, which also flip argmax/position); the citation in the code
now points at this measurement.
