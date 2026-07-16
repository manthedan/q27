# Multislot Phase 2, Phase-0 kernel probe — N=2 slot-batched T2 decode GEMV (2026-07-16)

Status: PRE-REGISTERED before measurement (this section written before the
kernel existed; results appended below the line).

## Question

When both slots are decoding, can one dispatch that reads the (bound)
weight stream once and computes two activation rows deliver enough kernel-
level speedup over two sequential GEMVs to justify Phase 2 integration?

## Candidate

`q27_matvec_t2_quantized_x2`: the production packed-dot ternary GEMV
(`q27_matvec_t2_quantized`) with two activation rows. Per-row K-striping,
scale-multiply order, and `simd_sum` reduction are kept **identical** to the
single-row kernel, so each row's output is bit-identical to the single
kernel — correctness is a byte-compare, not a tolerance argument
(margin-aware contract L1: identity where identity is free).

This is the select-form GEMV adaptation the round-3 review prescribed — NOT
the 8×8 MMA tile (`q27_matmul_t2_mm` at x_rows=2 wastes 14/16 token slots;
it is benched as a control arm, not a candidate).

Known risk (measured precedent): the fused *pair* kernel (two weights, one
x) LOST to back-to-back singles — 47–55 GB/s vs 67–69 GB/s — because
doubled register pressure hurt a weight-stream-bound kernel and the shared
activation bytes were already cached. The x2 kernel shares in the opposite
direction (one weight stream, two x rows): the shared resource is the bound
one. That asymmetry is the whole bet; the bench decides it.

## Arms (per shape, T2, quantized-activation path = production decode path)

- A: 2 × sequential `matvec_quantized` with two distinct activation rows —
  today's two-slot serial cost. Baseline.
- B: `matmul_quantized` at x_rows=2 (`q27_matmul_t2_mm`) — control.
- C: `matvec_quantized_x2` — candidate. Gate: rows byte-identical to arm A's
  two outputs before any timing is reported.

Shapes: ffn_gate/up proxy [17408×5120], ffn_down [5120×17408], gdn qkv
[10240×5120], ssm_out [5120×6144] (also the attn_output shape), output head
[248320×5120]. Aggregate s_k = Σbytes / Σ(bytes/s_shape) weighted by the
per-token production byte mix: ffn ×3×64, qkv ×48, ssm_out ×48, head ×1
(attention-layer q/k/v ≈ 6% of bytes, folded into the qkv/ssm proxies).

## Pre-registered decision lines (from S₂ = 1/(f/s_k + (1−f)), f = 0.85)

s_k = time(arm A) / time(arm C), bytes-weighted aggregate.

- s_k ≥ 1.80 → e2e ceiling ≥ 1.6×: **strong** — open Phase 2 integration.
- 1.65 ≤ s_k < 1.80 → e2e 1.50–1.60×: **useful** — integrate.
- 1.31 ≤ s_k < 1.65 → e2e 1.25–1.50×: **marginal** — park unless the
  integration is measured trivial after the design sketch; re-price then.
- s_k < 1.31 → e2e < 1.25×: **park Phase 2** (the pre-registered park line
  from the round-3 answers), record the numbers, move to the next
  workstream (verify_lanes / R3 attention probe).

f = 0.85 is the round-3 estimate of decode time in these linears; the bench
cannot refine f (kernel-level), so the integration gate re-checks e2e
against the same lines before Phase 2 is declared shipped.

---

## Results (appended post-measurement, same morning, M4 16 GB)

**VERDICT: PARK.** Aggregate s_k on the production kernel family = **1.093**
(post-codex corrected bench, see below), far below the 1.31 park line.
Implied e2e ceiling 1/(0.85/1.093 + 0.15) ≈ **1.08×**. Phase 2 slot-batched
linears are parked on this hardware; the probe cost one bench session and
zero integration work.

**Codex round on the probe commit (2 P2s, both fixed and re-measured):**
(1) the shape table substituted equal-byte orientations and a partial mix —
replaced with the exact production projection mix (8 shapes, true
orientations, per-token counts from the engine's weight table; gdn
alpha/beta [48×5120]×96 excluded explicitly, ~0.15% of bytes); (2) the
activation fill wrapped unsigned (`(i % 23) - 11` on `size_t`), so the
byte-identity gate had run with degenerate never-negative activations —
fixed with a signed cast (and the same-class pre-existing fill in the main
bench table fixed too). Corrected numbers below; the verdict did not move
(1.081 → 1.093).

### Baseline correction (recorded honestly)

The pre-registered arms benched the packed-dot *quantized* family — but
production serial decode routes T2 through the float-activation select-form
GEMV (`MetalEngine::project()` → `q27_matvec_t2_g128`); the quantized
packed-dot kernel serves the chunked/MTP paths. The decision baseline was
corrected to the select-form family (arms fA/fC added, same pre-registered
lines); both families are reported. `tools/metal_gemv_bench.cpp --slot2`
reproduces everything, including the byte-identity gates.

### Numbers (30 reps, ms/op; corrected bench — full production mix, signed activations)

| shape (per-token count) | qA 2×1 | qB mm2 | qC x2 | s_kq | fA 2×1 | fC x2 | s_kf |
|---|---|---|---|---|---|---|---|
| ffn gate/up [17408×5120] (×128) | 1.318 | 1.153 | 1.126 | 1.171 | 0.530 | 0.476 | 1.114 |
| ffn down [5120×17408] (×64) | 1.321 | 1.195 | 1.126 | 1.173 | 0.536 | 0.489 | 1.095 |
| gdn qkv [10240×5120] (×48) | 0.778 | 0.687 | 0.670 | 1.160 | 0.334 | 0.286 | 1.165 |
| gdn gate [6144×5120] (×48) | 0.478 | 0.439 | 0.404 | 1.183 | 0.199 | 0.185 | 1.079 |
| ssm/attn out [5120×6144] (×64) | 0.477 | 0.432 | 0.408 | 1.171 | 0.166 | 0.184 | **0.905** |
| attn q [12288×5120] (×16) | 0.940 | 0.833 | 0.798 | 1.177 | 0.383 | 0.342 | 1.122 |
| attn k/v [1024×5120] (×32) | 0.091 | 0.110 | 0.076 | 1.195 | 0.050 | 0.045 | 1.121 |
| output head [248320×5120] (×1) | 18.594 | 15.855 | 15.865 | 1.172 | 7.374 | 6.556 | 1.125 |
| **aggregate (byte-weighted)** | | | | **1.171** | | | **1.093** |

Identity: both x2 kernels byte-identical to their single-row kernels on all
shapes (memcmp'd before any timing was accepted).

Variant history on the quantized family: naive dual-dot 1.169; shared-unpack
(`q27_dot16_t2_dual`, one shift/mask/sub per code MAC'd into both rows)
1.176 — the compiler had already CSE'd the unpack.

### Why the ceiling's premise is false on M4

- The select-form production GEMV streams ~**89 GB/s** effective (23.7 MB
  ffn tensor in 0.265 ms/row) — near the M4 bandwidth roof, i.e. the single
  kernel sits at the compute/bandwidth **balance point**, not deep in
  bandwidth-bound territory.
- Batching two rows halves weight bytes per slot-token but doubles the
  select-adds; the kernel flips to issue-bound and gives back almost all of
  the bandwidth saving. On the smallest shape register pressure regresses
  it outright (0.905) — the fused-pair precedent repeating.
- The packed-dot quantized family is issue-bound outright (~36 GB/s
  effective single-row), which is why its s_k is also flat (1.18) and why
  the 16-token MMA tile at x_rows=2 (qB) beats two packed-dot singles but
  still loses to the select-form pair by ~2×.

S₂ = 2/(α·f + 2(1−f)) ≈ 1.74 assumed α ≈ 1 (batched linear ≈ one single).
Measured α ≈ 2/1.093 ≈ 1.83. The lever needs α ≲ 1.4 to clear the deploy
line; no candidate in this kernel family gets there.

### Residue (recorded, not scheduled)

- Batching only pays if per-row compute cost drops ~2× (an MMA/half-staged
  select-form hybrid). qB says the current tile is not that kernel.
- Prefill and MTP-verify already batch via the tile kernels; multislot
  prefill-prefill overlap needs nothing from this probe.
- Re-run the probe on the 24 GB machine only if its compute:bandwidth ratio
  differs materially (one `--slot2` invocation; the probe surface stays in
  tree, Metal-only, off the Backend interface).
