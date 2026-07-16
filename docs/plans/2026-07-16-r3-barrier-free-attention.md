# R3 probe — barrier-free direct-read block-partial attention, H1×T2 (2026-07-16)

Status: PRE-REGISTERED before measurement (results appended below the line).
Source: round-3 answers Q2 (2026-07-15-expert-review-3-answers.md) — with
head-major at 1.00×, fp16 paying no wall for 5.12× bytes, R1b's 2× from
amortizing barrier cadence, and R2's restage losing 0.76× to *extra*
barriers, the implicated cost of deep-context attention is the
**barrier-stage–serial-row cadence**, not bytes.

## Kernel (H1×T2 first, per the prescribed variant order)

`q27_attention_turbo3_causal_gqa_bf2`: the R1b t2 template body with the
threadgroup staging and both barriers deleted. One simdgroup owns (one
query head × one sequence block × two chunk tokens); each lane dequantizes
its 8 K dims and 8 V dims per position **directly from device memory into
registers** (`turbo_dequant`, same element order d = lane + 32·i as the
staged kernels, so the arithmetic sequence per token is unchanged). Six
simdgroups per threadgroup — one per GQA query head — purely for dispatch
shape; they share nothing.

- Duplicated-read math: each KV row is read 6× (~600 B vs 100 B staged) —
  the byte range fp16 already proved latency-tolerant; the risk is
  repeated dequant ALU (Phase 0 estimated well under issue capacity).
- **Bit-identity at equal block size**: dequant produces the same floats
  in-register as staged; dot order, `simd_sum`, online-softmax updates,
  and partial slots are identical — so at B=1024 (production) the output
  must memcmp-equal the t2 kernel. The bench gates on this before timing.
- Block-size sweep is first-class: B ∈ {128, 256, 512, 1024, 2048}
  (expert prior: 256–512 wins). B ≠ 1024 changes the merge fold count and
  block boundaries → falls under the margin-aware L2/L3 contract if
  chosen for production.
- The bf2@1024-vs-t2@1024 A/B (same math, same fold, bit-identical output)
  IS the barrier-cost measurement the round-3 calibration bench asked
  for: any wall-time difference is staging + 2 barriers/8-row-tile +
  arrival skew, isolated from fold-order and duplication effects, with
  the B-sweep separating the serial-chain length term. The 3-loop
  synthetic calibration bench is superseded unless these numbers are
  ambiguous.

## Pre-registered decision lines

Metric: best-over-B bf2 chunk-dispatch time vs production t2@1024, same
synthetic deep-KV harness (`metal_attn_bench`, turbo3, chunk 12), at seq
16K and 32K; the deploy decision weights 32K (where attention share is
largest and R1b won 2×).

- **≥ 1.30×** → strong: route production (if best B = 1024: bit-identical,
  no tolerance debt; else through an L2/L3-gated block-size change).
- **1.10–1.30×** → useful: integrate only at B = 1024 (free identity);
  a B ≠ 1024 winner in this band must also clear the margin-gate cost
  argument before routing.
- **< 1.10×** → park H1×T2, record numbers, proceed to H2×T1 (two query
  heads sharing one simdgroup's KV reads) only if the bf2@1024 A/B shows
  the barrier term is real but duplication ate the win (i.e. bf2 loses at
  equal B but the sweep shows block-size sensitivity).
- Decode-shape (single-token) routing is out of scope for this probe; a
  win here re-opens it with its own A/B.

---

## Results (appended post-measurement, same morning, M4 16 GB)

**VERDICT: PARK R3 entirely.** bf2 runs at **0.47–0.53× of the production
t2 route** (i.e. ~2× slower) at every block size and both depths — far
below the 1.10 park line. The H2×T1 escape condition ("bf2 loses at equal
B but the sweep shows block-size sensitivity") is explicitly NOT met: the
sweep is flat (16K: 33.4–37.0 ms across B=128–2048; 32K: 66.3–73.1 ms), so
the serial-chain/barrier term the variant order was designed to chase is
~nil and H2×T1 does not proceed.

### Numbers (`metal_attn_bench`, turbo3 chunk 12, ms/dispatch)

| arm | 16K | 32K |
|---|---|---|
| untiled gqa (staged, barriers) | 34.325 | 68.343 |
| **t2 (production)** | **17.647** | **34.245** |
| t4 | 25.359 | 49.034 |
| bf2 B=128 | 33.426 (0.53×) | 66.550 (0.51×) |
| bf2 B=256 | 33.467 (0.53×) | 66.256 (0.52×) |
| bf2 B=512 | 33.996 (0.52×) | 66.879 (0.51×) |
| bf2 B=1024 | 35.106 (0.50×) | 69.245 (0.49×) |
| bf2 B=2048 | 36.967 (0.48×) | 73.110 (0.47×) |

Identity: bf2 @ B=1024 memcmp-equal to the staged kernels (both depths) —
the kernel is correct, just slow. Decode head-major control unchanged
(1.00–1.01×).

### The round-3 Q2 hypothesis inverts

bf2 lands at the **untiled** kernel's wall time almost exactly. Reading:

- **Staging + two barriers per 8-row tile cost ~nothing on M4.** bf2
  deletes them and gains zero; the block sweep says shorter serial softmax
  chains and more independent partials buy zero too (slightly negative —
  merge fold overhead grows).
- **R1b's 2× was dequant/stream sharing, not barrier amortization**: t2
  halves per-token dequant + KV traffic (each staged row serves 2 tokens ×
  6 heads). bf2 gives sharing up entirely — each simdgroup dequantizes
  every row itself (6× per token-pair vs 1× staged), and that duplicated
  dequant ALU costs exactly the factor the stream sharing had won.
- The "latency-tolerant byte range" premise held for *bytes* (fp16 5.12×
  bytes at no wall) but does not extend to *per-element dequant issue
  cost*, which scales with gqa × tokens/TF and is the actual wall.

### Residue (recorded, not scheduled)

- If more attention headroom is wanted, the direction is MORE sharing per
  dequant, not less synchronization: e.g. staging dequantized tiles in
  half (halves threadgroup traffic; the GEMM half-staging lever's sibling)
  or a register-disciplined t3 (t4 lost to spills, but t3 with 25% less
  register pressure than t4 is unexplored).
- The 3-loop barrier-cost calibration bench is superseded: bf2-vs-staged
  at equal B *is* that measurement, and it read ~zero barrier cost.
- Probe surface stays in tree (bench-only, `attention_turbo3_causal_gqa_bf`
  + `--seq`-driven sweep in `metal_attn_bench`) as the negative-result
  witness and for a one-shot re-run on other hardware.
