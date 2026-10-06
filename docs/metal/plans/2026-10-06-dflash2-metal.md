# DFlash2 speculative decoding on Metal — plan (2026-10-06)

Written with Astra (gpt-6-astra, session 01a11340), against master 5dcf4df.
Status: **plan; M1 is the go/no-go milestone.**

## Why

Upstream CUDA decodes Bonsai 2 T2 at 110 t/s plain and 233–346 t/s with a
DFlash2 drafter (5090, K=7; README "Bonsai 2" table). A drafter trained on the
ternary target exists: ProCreations/Ternary-Bonsai-2-27B-DFlash2 (Apache-2.0,
rev 4cfb6ad0; 3.80 tok/round vs 3.47 for the Qwen3.8 drafter on upstream's
campaign). Metal decode is ~80 ms/token, all of it GPU work, ~70 ms of it the
T2 matvec streaming 6.8 GB at ~95 GB/s (about the read roof). Bandwidth-bound
decode is exactly where verifying several drafted tokens per weight pass pays.

## What blocks it today (measured)

Verification runs drafted tokens through `chunk_forward(verify=true)`, i.e. the
prefill GEMM `q27_matmul_t2_mm_f`. Its cost is flat in the token count at small
widths (`tools/metal_t2_prefill_bench.mm`, full layer stack, mini M4):

| tokens | 1 | 4 | 8 |
|---|---|---|---|
| ms per layer stack | 292 | 299 | 300 |

against ~70 ms for one token through the matvec. Verify costs ~4.3 decode
steps, so speculation cannot break even at ~3.8 accepted tokens per round.
This is also why suffix bursts only win on long verbatim copies.

First prototype of the fix (`tools/metal_t2_multivec_bench.mm`, a K-vector
select-form matvec sharing each weight load; outputs **bitwise identical to
the production matvec per vector** at every K):

| K | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| layer stack vs 1 decode step | 0.91x | 1.38x | 65x | 136x |

K=2 already verifies two tokens for 1.38 decode steps. K>=4 collapses (register
pressure; full unrolling does not help). M1 is solving that.

## The drafter (upstream implementation)

- Config: 5 qwen3-style layers, hidden 5120, FFN 17408, 32/8 heads, head_dim
  128, sliding window 2048, non-causal, block_size 8, mask token 248070,
  target layers [5, 19, 33, 47, 61], conv groups 16 / kernel 2, selector rank
  256 / top-k 16, vocab 248320.
- Features: for each committed position before the pending token, the
  unnormalized residuals after those five target layers (after the FFN
  residual add), concatenated to 25600 floats. `ingest` projects them to 5120,
  RMS-normalizes, and writes each draft layer's context K/V; context rows skip
  convolution, Q and FFN (`src/engine.cuh:1869`, `src/dflash2.cu:613`).
- One draft forward embeds `[pending, MASK x K]` at positions F..F+K (W = K+1):
  five blocks of attention (normalized Q/K, NEOX RoPE, bidirectional within
  2048 over context + block) and SwiGLU, each with prepare/finish dynamic
  causal convolutions whose history restarts every block
  (`src/dflash2.cu:53`, `:157`, `:757`).
- Output: final-normalized mask rows give vocab logits and rank-256 selector
  features; top-16 per position; a sequential selector with
  predecessor/successor codebook terms yields K proposals; sampled mode keeps
  the 16-way proposal distribution. The target's embedding and head are
  reused; Bonsai needs the inverse embedding rotation and the head-input
  rotation (`src/dflash2.cu:829`, `src/dflash2.h:99`).
- Persistent state: five context K/V rings + absolute positions. Ingest only
  the verified prefix (not the new pending token); discard speculative block
  K/V; rewind on truncation; seed the prompt's last 2048 positions
  (`src/dflash2.h:141`, `src/engine.cuh:3482`, `:3559`).

## Cost model (M4, engineering projection, not measured)

A = committed tokens per round (including the anchor). Round time =
draft + ingest + verify + commit. Per round the drafter streams ~2.30 GB (Q8)
/ ~1.40 GB (Q4), floors 24 / 15 ms at 95 GB/s. Budget: verify+commit
`70 + 10W` ms with a bandwidth-bound small-W kernel (today `300 + 10W`),
draft+ingest Q8 `30 + W`, Q4 `20 + W` ms.

| K | verify ms | draft Q8/Q4 ms | t/s Q8, A=3–4 | t/s Q4, A=3–4 | today's GEMM t/s |
|---:|---:|---:|---:|---:|---:|
| 3 | 110 | 34/24 | 20.8–27.8 | 22.4–29.9 | 8.0–11.0 |
| 4 | 120 | 35/25 | 19.4–25.8 | 20.7–27.6 | 7.8–10.7 |
| 6 | 140 | 37/27 | 16.9–22.6 | 18.0–24.0 | 7.4–10.1 |
| 8 | 160 | 39/29 | 15.1–20.1 | 15.9–21.2 | 7.0–9.5 |

Against 12.6 t/s plain: **~1.6–2.4x if M1 lands**, a regression if it does
not. K=3 wins at fixed A (the model card also defaults to 3); the real optimum
needs measured A(K) on Metal, since acceptance does not transfer across widths
or hardware.

## Memory (16 GiB mini)

Pack 6.70 GiB + drafter Q4 1.13 / Q8 1.94 GiB + ring/scratch ~0.20 + engine
fixed ~0.34 GiB + target KV (64 KiB/token fp16, 34 KiB/token q8). At 16K:
~9.4 / 10.2 GiB (Q4 / Q8 drafter, fp16 KV), ~8.9 / 9.7 GiB with q8 KV, before
the OS and snapshots (one fp16 prefix snapshot ~1.15 GiB). 16K fp16 was already
the measured ceiling without a drafter, so nothing is proven swap-free yet:
start with Q4 drafter + q8 KV soaks at 4K and 8K, then 16K. Never duplicate the
target's embedding/head for the drafter.

## Milestones (sequential; each gate must pass before the next starts)

| | deliverable | gate | effort | main risk |
|---|---|---|---|---|
| **M1** | Small-W T2 projections + head (W=4..9), routed through suffix verify | per-lane bitwise parity with the matvec; long greedy identity incl. every partial-accept boundary; full verify <= 130 ms at W=4; suffix copy speedup measured | 3–5 d | register pressure (seen at K>=4), arithmetic-order changes |
| M2 | Tap capture at layers 5/19/33/47/61 in decode, verify and prefill; committed-prefix ingest; prompt-tail seeding | teacher-forced tap parity vs serial; position/rollback fixtures; < 1% capture overhead | 1–2 d | off-by-one positions, rotated residuals |
| M3 | `.d2w` loader + Metal drafter forward (Q8, then Q4) | intermediate parity vs CPU/upstream (rel RMSE <= 1e-3); selector goldens on non-tie cases; measured draft budget; Q4 acceptance loss <= 5% | 4–7 d | dynamic conv, RoPE, quantized-activation semantics |
| M4 | Round integration on the existing GDN replay (`metal_engine.cpp:2005`) | greedy identity vs plain decode across widths; sampled law (accept min(1,p/q), residual max(p-q,0)); EOS / cancel / truncation / continuation; >= 10% end-to-end gain or stop | 3–5 d | suffix acceptance assumes deterministic proposals |
| M5 | Serving/agent flags, per-session rings, admission/memory accounting, telemetry, fallback; Qwen3.8 on 24 GiB later | warm-prefix no-swap soak; agent wall-time improvement | 2–3 d | snapshots, constrained decoding, slot pressure |

Notes:

- M1 pays off on its own (suffix bursts get cheaper), so it is worth doing even
  if DFlash2 later stalls.
- Numerics contract: today's chunk path is tolerance-class against serial
  decode (`metal_engine.cpp:3222`), and upstream documents that greedy is not
  width-invariant on CUDA (README, Bonsai 2 section). Exact acceptance alone
  does not prove identity with plain decode; M1 must define the target
  numerics contract (the K-vector prototype is bitwise per vector, which is
  the strongest available option).
- `docs/dflash-block-verify-design.md` is superseded on S=16 assumptions; keep
  its measurement discipline. `docs/drafter-probe-plan.md`: test on real
  agent traffic first.
- The parked `q27_matvec_t2_g128_x2` s_k = 1.093 is aggregate speedup: two
  vectors cost ~1.83 single passes. The new K=2 prototype measures 1.38.
