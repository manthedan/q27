# Paper-scan triage — 3-month HF Daily Papers sweep, verified against repo state (2026-07-16 pm)

Status: TRIAGED. Source: external scan of Daily Papers 2026-04-16..07-16
(operator-provided). This doc records the verification and the reconciled
priorities; the scan itself is with the operator. **Verification result: the scan
is sound** — 3/3 load-bearing papers spot-checked real (arXiv links below),
and every project-facing claim checked traces to a real chronicle
measurement — except its P0 ("three confirmed correctness issues"), which
are the expert-review-2 P0s, all fixed the same night. The scan also
predates the same-day landings that reprice it: lever 2 (S(48) = 3.94×,
flat-per-16-tile), the R3 park (0.47–0.53×), and G6.

## Verified paper set (spot-checked)

- **Block-GTQ** — RoPE-aware bit allocation for KV-cache quantization,
  arXiv 2606.24033 (code: github.com/JIA-Lab-research/blockgtq). Per-layer/
  head/RoPE-frequency-block greedy bit allocation; NIAH 70.6 → 97.4 at the
  same average K budget on Llama-3.1-8B.
- **KVarN** — variance-normalized KV quantization, arXiv 2606.03458
  (Huawei; vLLM backend at github.com/huawei-csl/KVarN; llama.cpp research
  issue open). Hadamard rotation + dual-axis (token × feature) variance
  normalization, calibration-free; targets exactly the token-scale tail
  class.
- **BaseRT** — native Metal LLM runtime, arXiv 2607.00501 (source:
  github.com/basecompute/baseRT). Claims 1.56× vs llama.cpp / 1.35× vs MLX
  decode on M3/M4-Pro-class at q27-relevant sizes.
- Also real and already known-class: "Batch Speculative Decoding Done
  Right" (EXSPEC — same-length grouping; a warning we already lived through
  in the CUDA merge), plus a scan tail (Tangram, ModeSwitch-LLM, PBKV,
  LLM-Emu, VIA-SD, Mix-Quant, SPD) taken at abstract level, not verified
  individually — none of those drive a P1 decision here.

## New workstream: P1 KV codec (mini, `--kl-kv` instrument, T2)

Grounding measurement (already on the books, 2026-07-15 mini): turbo3 KL
mean 0.0115 nats depth-flat but **heavy-tailed, max per-position 2.83
nats**. The scan's thesis — treat KV precision as structured allocation
(K/V × layer × head × RoPE-pair × token scale) rather than another uniform
sub-2-bit alphabet — is the right frame for that tail. Order, each step
gated on the same instrument (the 0.0115/2.83 calibration is the bar):

1. **Instrument prep (small):** `--kl-kv` report gains p99 and max
   per-position KL + a consecutive-high-KL run counter (the tail is the
   target; mean+buckets can't see it). Also record per-position K-only /
   V-only attribution mode (quantize one side, fp16 the other — the Shared
   dual-engine harness already supports the arms).
2. **KVarN-style scaling on the existing turbo3 block** (no new packed
   format: same 50-byte block, scaling rule only). Arms: current / feature-
   group normalized / token-corrected / both; K-only / V-only / both.
   **Pre-registered read: an arm must cut max-KL ≥ 5× (2.83 → ≤ 0.57) or
   p99 ≥ 3× at unchanged mean (± 10%) to graduate.** Below that, record and
   stop — the tail is then not a scale artifact.
3. **Sensitivity census** — 16 attn layers × 4 KV heads × {K,V} = 128
   cells; quantize one cell, measure ΔKL/Δmax. Output: the allocation
   targets for step 4. CPU-driven loop, short runs.
4. **RoPE-pair-aware K allocation (Block-GTQ shape), simulation first:**
   capture fp16 K/V from a short stream, simulate allocations on CPU under
   average budgets {1.0, 1.25, 1.5} bpv, score query-key error + replayed
   KL. **No packed kernel unless non-uniform beats uniform at equal budget
   on the instrument.** M4 constraint from R3: attention is dequant-issue-
   bound — any eventual format must be static per (layer, head, RoPE-group)
   with regular strides; no per-token or intra-row variable widths.
5. Only then a sub-2-bit packed codec design, sized by 3–4's numbers.

Prize is capacity/long-context reach (128K–262K), not 32K speed —
consistent with attention being issue-bound, and with the standing quality
roadmap (needle at depth, then 128K/262K).

## N=2 slot-batched decode — ALREADY PARKED BY MEASUREMENT on M4

The scan promotes cross-user batched decode above generic verification.
Both halves are overtaken: generic verification shipped (lever 2), and the
N=2 kernel probe **already ran and parked** (2026-07-16 morning,
multislot-phase2-probe): slot-batched x2 GEMV aggregate s_k = 1.093 vs the
pre-registered 1.31 park line (implied e2e ceiling ~1.08×). The scan's
implicit premise — a second activation row costs ALU, not bandwidth — was
measured FALSE on base M4: the production select-form GEMV already streams
~89 GB/s at the compute/bandwidth balance point, and halving weight traffic
flips it issue-bound (α ≈ 1.83). The scan's grouping/scheduling advice
(EXSPEC same-shape pooling, position-advance correctness first) remains
good input for the multislot scheduler that DOES exist (Phase 1 + G6
discharged), where slots time-slice rather than batch. The probe surface
stays in tree for a one-shot re-run on the 24 GB machine (different
compute:bandwidth ratio) — that re-run is the only N=2 item left open.

## Adopted with existing owners / already answered

- **Stable-boundary snapshots** = ds4-item-5, motivation already measured
  (~23 min re-prefill per question at 31K; GDN-can't-rewind means snapshots
  are captured at boundaries during prefill — app hints + LRU is v1).
  Scan independently converges on it; priority affirmed (P3).
- **BaseRT source survey** — adopted as CPU-only parallel work, ds4
  discipline (import only bottleneck-matched designs). One of its questions
  is pre-answered: barrier-free attention loses 2× on M4 (R3 park) because
  it surrenders dequant sharing. Our decode is at 98–99.2% of own GEMV
  ceiling; only prefill-side imports can matter.
- **Mode controller / emulator** — good, cheap, premature until the modes
  exist; queued behind P1–P3 as the scan itself orders them.
- **VIA-SD slim verifier — park, now provably:** verify batch is 97.5% of
  round cost and a k-layer slim pass streams k/64 of the weights; strictly
  proportional economics, and S(48) = 3.94× already ships.
- SPD / Mix-Quant / sparse-attention family: dismissed for the scan's own
  reasons; all consistent with our measurements.

## DSpark repricing under lever 2 (decision moved, see phase0 doc)

Lever 2's measured curve: break-even ~4.1 tok/round at w=16 (the chained-
block width class), ~10.7 at w=48. Expected chained-block committed rate
5–6.5 → S ≈ 1.2–1.57×, **straddling the pre-registered 1.3× Phase-3 gate**.
The go/park is no longer decidable by arithmetic; it needs the fork's
measured per-block acceptance decay = exactly the Phase 2 fixture work.
Suffix bursts (no drafter, w=48 customer) are now the higher-EV speculation
item and proceed independently.
