# KV-codec step 1 — kl-kv tail instrumentation + K/V attribution arms

Status: PRE-REGISTERED before measurement (results appended below).
Parent: 2026-07-16-paper-scan-triage.md §"New workstream: P1 KV codec",
step 1 (instrument prep). Calibration bar this feeds: turbo3 mean 0.0115
nats depth-flat, max per-position 2.83 (2026-07-15, 8K wikitext, mini).

## Instrument changes

- `--kl-kv` report gains a tail block: p50/p90/p99/p99.5/max per-position
  KL with the max's position, plus consecutive-high-KL run structure at
  two fixed thresholds (0.1 and 0.5 nats): positions above, number of
  maximal runs, longest run and its start. Same quantile convention as
  the envelope instrument (sorted index ⌊p·n⌋, clamped).
- New attribution arms `--kl-kv-k` / `--kl-kv-v`: the subject engine
  keeps the fp16 KV cache and fp16 attention kernels, but the store
  round-trips ONE side (K or V) through the exact turbo3 quantizer —
  normalize, ±1 sign flips, butterfly, 8-centroid nearest, half-rounded
  norm-correction scale — and back through the inverse transform
  (`q27_kv_store_f16_attrib_rows`; scale passes through half exactly as
  the packed block header does). The KL against the fp16 baseline is
  then attributable to that one side's quantization alone, with no mixed-
  format attention kernel. Engine knob `set_kv_attrib` (fp16-KV engines
  only, before any token); routed at all four fp16 store sites (decode,
  chunk, both MTP paths) so the semantics hold on any stream.

## Run plan

T2 artifact, wikitext2, 8,192 positions (`--nll-long 8192 --ctx 8192`) —
the exact config of the 2026-07-15 calibration — three arms: `--kl-kv`
(both sides), `--kl-kv-k`, `--kl-kv-v`. ~14 min each on the mini.

## Pre-registered reads

1. **Reproduction check (both-sides arm):** the GPU is deterministic
   (envelope repeat class = exactly 0), so the re-run must reproduce
   mean 0.0115 / max 2.83 exactly. A deviation is a regression in this
   change, not a measurement.
2. **Additivity:** K-arm + V-arm mean KL ≈ both-arm mean (smoke at 128
   positions: 0.00255 + 0.00541 = 0.00796 vs 0.00765, ~4% sub-additive).
   Strong super-additivity would mean joint K×V interaction dominates and
   step 4's allocation cannot treat sides independently.
3. **Tail ownership:** which side's arm carries the 2.83-class max and
   the p99. This orders step 2's scaling arms and step 3's census.
   (Smoke suggests V ≈ 2× K on the mean; the tail may differ.)
4. **Run structure:** longest run at 0.1/0.5 nats. Runs ≫ 1 ⇒ damage
   propagates through state (GDN carry), isolated spikes ⇒ per-position
   near-ties; the two point at different codec remedies.

No pass/fail gate — this is instrumentation; the numbers become the
attribution baseline for steps 2–4.
