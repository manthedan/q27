# B1 serial-decode fusion round — pre-registration (2026-07-19)

Source: expert review 3, §1 (triage: `2026-07-19-expert-review-3-triage.md`).
The single funded kernel round from that review. This is a **small,
hard-gated** optimization, NOT a kernel survey. Two candidates, stop after
both if the wall gain is under 5%.

## The defect (verified against source)

For a pure-Bonsai pack (B1, T2, or mixed-Bonsai — `quant_policy`
`bonsai-{t2,b1,mixed}-v1`, metal_engine.cpp:96-102), the serial decode
path in `encode_token` (metal_engine.cpp:1256) computes activation
quantizations that are never consumed:

- `backend_.rmsnorm_quantized(h, attn_norm, x1, N_EMBD, EPS, q5120_)` —
  once per layer, line 1266.
- `backend_.rmsnorm_quantized(h, post_attention_norm, x1, ..., q5120_)` —
  once per layer, line 1269.
- `backend_.rmsnorm_quantized(h, output_norm, x1, ..., q5120_)` — once at
  the head when `produce_logits`, line 1274.

That is 2×64 + 1 = **129 calls per decoded token**, each producing an int8
quantized copy `q5120_` (scales + codes). But `project()`
(metal_engine.cpp:1147-1148) routes Bonsai dtypes to the FLOAT matvec
(`is_bonsai_dtype → backend_.matvec(w, x_float, out)`) and only official
Q4/Q8 to `matvec_quantized(w, xq, out)`. **For a Bonsai pack the int8 copy
is written 129×/token and read 0×/token.**

The int8 path exists for the official-tier (Q4/Q8) packs and for chunked
prefill (the Bonsai MMA kernels consume quantized activations there). So
the optimization must be **serial-decode-only and Bonsai-pack-only** —
official packs and all chunked execution keep the current route.

## Candidates (pre-registered, in order)

### Candidate A — pure-Bonsai float-only normalization

For Bonsai serial decode, call `rmsnorm()` instead of
`rmsnorm_quantized()` at the three sites (and the two MTP variants at
1492/1624 if in the serial path). Identical float output (`x1_`); the only
change is the int8 copy is not produced. Drop-in: same `(x, weight, out,
n, eps)` signature minus the `BackendQuantized&` (metal_backend.h:107-110).
Gate the swap on the engine's existing `bonsai` policy flag, so official
packs are byte-untouched. Scope: serial decode only; chunked prefill
unchanged.

### Candidate B — fused residual-add + normalization

The serial loop twice per layer does `add_inplace(h, y)` then
`rmsnorm[_quantized](h, x1)` (lines 1268/1269 region). Fuse into one
kernel that writes `h[i]+y[i]`, accumulates the norm over that exact
rounded value, and writes normalized `x1` (int8 copy only for official
Q4/Q8). Two such boundaries per layer → up to 128 fewer dispatches + one
fewer hidden-state pass per token. Run B only after A is measured; A and B
compose (B's fused kernel emits float x1 for Bonsai).

## Gates (both directions, exit codes)

- **G0 route liveness (hard, run first).** On a real B1 decode with
  profiling enabled, prove: `q27_matvec_b1_g128` calls > 0,
  `q27_matvec_b1_quantized` calls = 0, `q27_rmsnorm_quantized` calls =
  129/token. This validates the diagnosis and closes the round-2
  attribution error (see the CORRECTION in
  `2026-07-17-b1-select-round2.md`). No candidate is timed until G0
  passes.
- **G1 byte-identity (hard, byte-exact only — no envelope).** B1 committed
  output tokens byte-identical pre/post change over a fixed greedy
  continuation (≥256 tokens) at a fixed prompt; hidden rows byte-identical.
  Candidate B's fused kernel preserves the exact rounding order (explicit
  store + device barrier + reread of `h`, the same `reduce_sum` tree), so
  there is no mathematical reason the fusion must change rounding; if G1
  fails, fix or kill Candidate B rather than widen to an envelope. A later
  single-pass variant that drops the storage boundary is a separate
  experiment needing a margin-certified envelope, and must not rescue this
  exact-order candidate. Sabotage arm: a change that alters the float
  normalization or omits the store/reload barrier must FAIL byte-identity.
- **G2 no T2 regression.** T2 serial decode wall unchanged within noise
  (±1%) and T2 committed output byte-identical (T2 is also Bonsai-routed,
  so it must benefit or be neutral, never regress).
- **G3 official-tier untouched.** Official Q4/Q8 pack serial decode
  byte-identical; chunked prefill and verification untouched (they require
  quantized activations).
- **G4 wall measurement.** B1 resident decode ms/token, same machine, warm
  pack, ≥5 interleaved baseline/candidate pairs, same process where
  possible, matched thermal state (house rule); report GPU and wall time
  separately. Baseline is the current ~19 tok/s (52 ms/token, 3.36
  GiB/token @ ~69 GB/s — the memory-wall regime from
  `2026-07-17-b1-select-round2.md`).

## Ship / kill line

- **SHIP Candidate A/B only at ≥1.0 ms/token end-to-end B1
  resident-decode improvement** (~2% at the 52 ms/token wall) with
  G0/G1/G2/G3 all PASS (byte-identity holds) and the paired result
  consistently positive. Below that the gain is too vulnerable to
  machine-state noise and not worth splitting the execution path.
- **STOP after these two candidates** if the combined wall gain is under
  the line. Do not escalate into a broader survey — the parked ledger
  already holds the negative verdicts for the surrounding levers.

## Expected magnitude (honest, pre-registered)

**0–3%; more than 3% would be surprising.** The dead work is real (129
unused int8 quantizations/token) but it is not a memory-bandwidth lever:
at width 5120 the extra traffic per invocation is ≈ 5120×4 B read +
5120×1 B write + 160×4 B scale write = 26,240 B, so 129 calls ≈ **3.23
MiB/token, only ~0.09% of the 3.36 GiB/token projection stream.** The only
mechanism is latency removal on the dependency chain (the barrier, serial
per-simdgroup block iterations, and maxima reductions sit between the
RMSNorm float output and the next dependent GEMV). Decode can be
bandwidth-dominated while still containing additive serial latency before
and between weight streams. Candidate B's dispatch removal (up to 128
add_inplace dispatches) is a distinct lever; before funding it measure
`total q27_add_inplace GPU time/token + the pure-RMSNorm saving` — that
sum is the hard upper bound, and if it is under ~1 ms/token, kill the
fusion without writing the kernel. (Prior 2–8% estimate revised down per
expert review 4; it was too optimistic given the actual kernel.)

## Cost / risk / house rules

Emitter-level change in `metal_engine.cpp` + (Candidate B) one new fused
kernel in `metal_backend.mm`/`q27_kernels.metal`. No pack format, no API,
no serving change. One resident model; decode benches at matched thermal
state (the bench power-line instrumentation from fe11fc4/e0ba2f8 applies).
test-cpu must stay green; add a serial-decode byte-identity self-check to
the existing metal test battery if practical (G1 must fail-first on a
deliberately-wrong normalization to prove it bites).
