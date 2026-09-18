# mlx.fast + inferno trick survey (2026-07-31)

What the Laguna MLX Challenge leaderboard and nektarlabs/inferno know that
q27 can use. Sources, regime analysis, then a ranked transfer list with
expected values, then the dead ends (theirs and the cross-check against
ours), then methodology imports. Raw extractions: leaderboard data via the
public API (`api.mlx.fast/api/benchmarks/…/submissions`, 851 submissions,
473 scored, 96 promoted frontier entries 1.004 → 1.995), challenge source at
github.com/Layr-Labs/mlxfast-challenge (HEAD already contains the promoted
implementations), inferno at github.com/nektarlabs/inferno.

## The two regimes, and why it decides everything

**Their regime (mlx.fast frontier, M5 Max):** NVFP4 MoE, only 8 of 256
experts read per token ⇒ decode traffic ~1.05 GB/token on a 21.6 GB model.
Steady decode is 92.8–94% GPU-busy but the big movers stream at 445–544 GB/s
against a ~575 GB/s ceiling, and the dominant kernels are *latency-bound*
(93.6% busy at ~29% of peak bandwidth, 15 W). Their levers of record:
occupancy for latency hiding, fusion to kill wasteful kernels, async-eval
ladders to hide lazy-graph construction, and — above all — the lm_head,
which at 212–412 MB/token was **20–40% of their per-token bytes**.

**q27's regime:** dense hybrid, every weight read every token, decode at
98–99.2% of the resident-weight ceiling (METAL_PROGRESS Current state). We
are bandwidth-bound full stop; the per-op dispatch and latency tricks that
won their campaign are expected NULL here — consistent with our parked list
(slot-batched N=2 decode, head-major KV, barrier-free attention, K/V
restaging). What transfers 1:1 is their other category: **byte cuts and
certified-work cuts on passes that are truly bandwidth-bound**.

Their own meta-finding, measured repeatedly: "removing dispatches by itself"
is a myth (inter-CB gap was 0.34% of the step); wins came from cutting bytes
or replacing algorithmically wasteful kernels. That is exactly q27's
situation.

## Tier 1 — the one big transfer: certified lm_head prune

Their #1-scoring mechanism (frontier top, 1.9946; our extraction T5).

**Mechanism.** Greedy decode must read the whole lm_head every token.
Instead: keep a second, coarse copy of the head; each step (1) screen all
vocab rows against the hidden state with the coarse copy, (2) compute a
certified per-row error bound that reduces to one global threshold,
(3) exactly recompute only the surviving candidate rows with the stock
kernel. The argmax is provably the stock argmax; the emitted token is
bit-identical. The winning coarse format is **planar-packed symmetric int6**
(per 32-element group: 4-bit plane + 2-bit plane + one power-of-two scale
byte = 1600 B/row on their 2048-dim head): word-aligned loads, a few integer
ops hidden behind DRAM, and a *flat half-cell* error bound tighter than
MXFP8's float grid (uniform grid wins for an L1 bound; e4m3 wastes bits on
tiny magnitudes). Measured: −1.75% steady step, byte-proportional
(~480 GB/s implied on removed bytes), exact-pass candidates p90 = 7 rows.

**q27 fit.** `output.weight` is Q8_G128 ≈ 1.29 GB (248,320 × 5,120 ×
8.125 bpw) on the official 17 GiB tier — read in full every greedy step
(metal_engine.cpp:1341, `project(weight("output.weight"), …)`). That is
~7.4% of step bytes. An int6 planar screen is 4,000 B/row ≈ 0.99 GB ⇒
~−300 MB/step ≈ **−1.7% decode on the official tier**, exact, no envelope
risk. q27 even has the cleaner exactness story: candidate rows recompute
through the *same* `q27_matvec_q8*` PSO on the same bytes (bit-identity by
construction), where MLX had to mirror a stock-kernel replica. The screen
reuses the existing int8-quantized activation (`q5120_`), so the coarse dot
is int6×int8 integer work.

**Scope guards (from their dead ends):**
- Greedy serial path only. The certificate proves the *argmax*; sampled
  decode keeps the full GEMV (a top-k certificate is unsound even for them —
  their D26). Compose with `mask_logits` by disabling the prune under tool
  constraints, same gating pattern as `decode_resident`.
- int4/e2m1 screens are three orders too loose (>50% survivors — the exact
  pass then reads more than the screen saves); int5's p99 tail is
  disqualifying; group mean+radius screening is dead (vocab rows
  near-orthogonal, max cos 0.1–0.3, group bounds lose the √N cancellation).
  int6 is the priced sweet spot; do not re-litigate.
- **T2 tier: closed.** The ternary head is already 2.25 bpw (358 MB); a
  1-bit screen is int4-class loose and survivors eat the saving (their D17
  math). Official tier only.
- Format pricing must be done offline against *our* captured hidden states
  × our head before any GPU work (their protocol: 130 real activations,
  candidate-set percentiles per format, certificate emulation in fp32).

**Pre-registered experiment (repo discipline):**
- *Arm:* int6 planar screen + threshold + exact-candidate recompute on
  serial greedy decode, official tier, behind `Q27_METAL_HEAD_PRUNE`.
- *Gate:* 16-token canonical gate byte-exact (existing); greedy
  teacher-forced NLL envelope unchanged by construction (tokens identical).
- *Kill lines:* <1.0% steady-decode win on base M4 at 128-token decode, or
  any token mismatch, or p99 exact-pass reads > 5% of head bytes.
- *Expected value:* ~1.5–2% steady decode. Second-order: the exact pass
  currently reads 1.29 GB; post-prune it reads ~(screen + p90·7 rows), so
  head read drops ~23% even before candidate-set shrinkage.

## Tier 2 — small, cheap, or audit-only

- **Alpha-skip in online softmax (free, exact).** `q27_attention_f16`
  (q27_kernels.metal:959-968) rescales `l` and `acc` by `exp(m − m_new)` at
  every position. When the running max doesn't advance (their measurement:
  85.5% of keys) the correction is exp(0) = 1 exactly, so skipping the
  rescale is bit-identical. They measured it *neutral* and shipped it as
  exact+free. q27 expectation: neutral at long context (KV-bandwidth-bound),
  tiny at short. Bundle into the next attention-kernel touch; do not spend a
  dedicated experiment on it.
- **Fused QK-norm+RoPE, one head per simdgroup, no threadgroup broadcast.**
  Our attention path runs `rmsnorm_heads` then `q27_rope_neox` as separate
  dispatches (metal_engine.cpp:1254-1255). Their winning form: one SIMD per
  head, lane owns contiguous elements, rotary partner via `simd_shuffle`,
  `precise::rsqrt` from a local `simd_sum` — **the threadgroup-broadcast
  variant measured −0.19% (barrier cost) vs +1.19% barrier-free**. Fits
  q27's geometry exactly (HEAD_DIM 256 = 8 el/lane, N_ROT 64). But their win
  was on a latency-bound kernel; ours is bandwidth-bound ⇒ expect small.
  Low priority; only 16 of 64 layers have full attention.
- **Fused residual into GEMV epilogue (inferno `*_matvec_add`,
  q2_kernels.metal:137-172).** Our residuals are separate `q27_add_inplace`
  after each block. Fusing removes a kernel and a 20 KB vector round-trip
  per add — negligible bytes; expect ~null end-to-end. Note as a rider if a
  GEMV kernel is opened for other reasons.
- **Snapshot I/O hygiene (inferno, direct):** `fcntl(F_NOCACHE)` on
  one-time streaming reads so they don't evict the mmap'd weight pages the
  UBC is managing (q2.rs:750-754), `madvise(MADV_FREE)` on release
  (q2.rs:4353-4354). Apply to q27's snapshot load/save paths
  (disk_snapshot_store / save_state): snapshot traffic is one-time and
  currently competes with the weight working set. Small serving win on
  snapshot-heavy agentic workloads.
- **Per-command-buffer GPU timestamps (inferno):** read `GPUStartTime`/
  `GPUEndTime` on completed command buffers for always-on GPU-time
  attribution without Instruments (q2.rs:5197-5210). Near-free; complements
  Q27_METAL_PROFILE without its per-op-encoder barrier distortion.
- **Mach-native memory telemetry (inferno memory.rs:6-116):**
  `host_statistics64` + `task_info` + `sysctl vm.swapusage`, no
  subprocesses, plus their measured lesson that *more cache can lose*
  (30 expert slots beat 31–32 under unified-memory pressure). Import for
  `q27 report` and the admission-accounting docs; relevant to
  quant×context sizing on small Macs.

## Tier 3 — exploratory / gated

- **Metal-4 tensor (`metal::tensor`, "NAX") GEMM for the chunk path, M5+
  only.** Their prefill stack lives on NAX kernels (device-gated,
  `is_nax_available()` false on M3/M4); QK-loop `unroll_count(4)` alone was
  +12% kernel-level (upstream MLX PR #3843). q27 parked direct-RHS chunk
  GEMM at 0.78× *pre-Metal-4* ("needs Metal-4 cooperative tensors",
  METAL_PROGRESS parked list) — the campaign confirms that hardware path is
  real and where the headroom went. Counter-evidence to price first: their
  D16 — INT8 side layouts *lose* at M=512 (compute-bound; dequant is pure
  added work), win only at M=1. Our chunk width 96 sits between; no transfer
  without measurement. Base M4 has no NAX, so this is an M5-tier plan only.
- **DSA-style selected-row attention** (inferno: lightning-indexer scores +
  GPU bitonic top-k + attend gathered rows only). q27's DSpark port is
  already killed by measurement (τ = 3.209 vs break-even 4.13). Inferno
  changes nothing about that calculus for our attention pattern. **Stay
  parked.**
- **Adaptive hill-climb resource controller** (inferno
  runtime/cache_budget.rs: change one budget, measure an 8-token window,
  keep if ≥1%). q27's budgets are static admission accounting. Methodology
  worth borrowing if dynamic KV/cache sizing ever lands.

## Explicitly non-transferable (with the reason)

- **Everything MoE** (router top-8 bitonic, sorted gather-QMM, RUNSKIP/EG256
  expert-run elision, fused routed/shared SwiGLU, ExpertPack SSD streaming,
  SLRU expert cache, ready-wave IO pipelining): q27 is a dense model.
  Revisit only if a future tier goes MoE; inferno's ExpertPack principle
  (relayout the file to match the runtime's read granularity) is the one to
  remember.
- **MLX async-eval ladders / command-buffer sizing (200 MB/200 ops):** they
  pay per-op lazy-graph construction; q27 already encodes a whole token into
  one command buffer directly, and `decode_resident` chains K≤8 greedy steps
  per buffer. Strictly ahead; nothing to import.
- **Sliding-window mask elision, YaRN mscale round-trips:** q27 has full
  attention + fixed partial rotary (N_ROT=64, base 1e7); no SWA, no YaRN.
- **Wired-residency zero-headroom:** their poison was per-CB transient
  allocations falling out of an oversized residency set. q27 pre-allocates
  all activation/KV buffers at engine construction and already wires the
  weight buffer via MTLResidencySet (metal_backend.mm:573-586, 875-891).
  Audit-only: confirm the set's capacity covers the weight buffer.
- **PSO-warmup of the scored window:** their single biggest win (+3.14%)
  was an argmax PSO compiling inside the timed window. q27 builds all
  production PSOs eagerly in the backend constructor (metal_backend.mm:610-
  723); only bench/probe PSOs are lazy and they never fire in serving.
  Already covered. Related hazard worth keeping: Metal function constants
  are part of the PSO specialization key — flipping one mid-process forces a
  JIT compile that can land inside a measured window. Keep runtime knobs as
  kernel arguments (current pattern: env at ctor only).
- **Deferred min-correction dequant (inferno's Q2_K trick):** all q27
  formats are symmetric scale-only (FORMAT.md §Q4_G64/Q8_G128/T2/B1). No
  min term exists to defer.
- **Inferno's MTP negative result does NOT transfer.** Their MTP is slower
  because drafting/verification multiplies *unique expert SSD bytes* on an
  SSD-bound MoE. q27's verification re-reads the same resident weights, so
  verification is nearly free at batch 1→w. Our MTP gates stand.

## Dead ends worth importing (so we don't re-derive them)

| Their measured result | q27 relevance |
|---|---|
| Fused KV write: 17/17 local wins → **−18.9% ranked** | The canonical "local inverts on ranked" warning; also their qdot `uint2` load widening: −2.99% local vs −0.63% ranked, opposite-signed draws |
| GQA unpairing (pair-heads=1): +124 µs worse; pairing pays even at 32 W | Validates our factor-2 tiled GQA direction |
| Split-KV decode attention (FlashDecoding geometry): dead, +394 µs | Matches our parked attention splits |
| Head-packed KV cache: −0.12% | Matches our parked head-major KV (1.00×) |
| E2M1 constant-LUT dequant in QMV inner loops: −0.65% (register bit-unpack wins); same LUT *helps* in prefill staging | Context-dependent; our T2/B1 unpack is already mask/shift register code |
| Norm+QKV fusion with "improved" arithmetic: −3.4%; the stock-expression-mirroring form is what promoted | When fusing, mirror stock kernels expression-for-expression; A/B in isolation |
| Software-pipelined/double-buffered staging: +2–7% *regression* (register live-range crosses an occupancy step) | Matches our parked f16-accumulate MMA direction; registers are the currency |
| int4/int5/int7 head screens; mean+radius screening | Priced dead ends behind the int6 choice — see Tier 1 |
| Bare first-K-block peel is NOT exact (reachable −0.0/+0.0 divergence); fix = lane-0 `x + 0.0f` after simd_sum | Directly applicable the day we peel an accumulator seed in any K loop. `(a + 0.0f) * c` cannot contract into an FMA; fast-math already off here (MTLMathModeSafe) |
| Router rows-per-group < 8: NULL (per-TG barriered norm re-derivation is the floor) | Generic: re-derived per-TG state sets the floor, not TG count |
| 2-sample A/B flipped sign vs 8 ABBA pairs; single paired sample cannot resolve sub-1% | Already our pre-registration discipline; keep ≥6 interleaved pairs |
| Label-based profiler attribution *lies* with raw encoders (time glues to the preceding op); cross-check against bandwidth arithmetic | Applies to Q27_METAL_PROFILE readings; the lie detector is bytes/time vs the ~575 GB/s (M5 Max) / ~273 GB/s (base M4) ceilings |
| Reachability audit before working a lead (all their published leads targeted a kernel the scored path never dispatches) | Confirm the kernel fires on the measured path before optimizing it |

## Methodology imports (zero risk, high value)

1. **Regime diagnosis before remedy:** a kernel at ~94% busy but ~29% of
   peak bandwidth is latency-bound (fix: memory-level parallelism); at
   ~90%+ of the bandwidth ceiling it is byte-bound (fix: fewer bytes or
   certified less work). Choose the remedy by the diagnosis. q27 decode is
   the second case.
2. **Untouched-phase drift control:** measure the phase you did not modify
   as the control leg of every A/B (their drift floor ~0.4%).
3. **Byte-proportional expectation model:** at the ceiling, a pass that
   loses X MB/step should win ≈ X/(ceiling × step-time); if it wins more,
   look for the second mechanism (their int6 win slightly exceeded the byte
   model via exact-pass shrinkage).
4. **Offline format pricing:** price quantization/screens against real
   captured activations with the full certificate emulated before touching
   GPU code (their int6-vs-int4/5/7 table).
5. **Quiescence gating:** concurrent CPU work (shader compiles!) inflates
   GPU timings (+3.9% decode / +9.9% prefill measured). Their ranked box
   gates on a 40 °C thermal + telemetry quiescence; our A/B harness already
   interleaves — keep it.

## Tier 1 RESULTS (2026-07-31, commit a7958c33) — certified lm_head prune SHIPPED env-gated

**Pricing** (tools/price_head_prune.py, 256 real decode states across two
captures, official Q8 head; logs/headprune-20260731/): int4 dead (whole
vocab survives), int5 tail-disqualified (p90 126k rows), **int6 selected**:
p50 8 / p90 1404 / p99 14205 candidate rows; total head read p50 0.993 GB,
p99 1.067 GB vs 1.291 GB stock (−17% worst observed step). Certificate:
zero bound violations across both captures, argmax-in-set 256/256.

**Correctness gates.** Unit (test_metal_ops::test_head_prune): candidate
rows bit-identical to the production Q8 GEMV, certificate sound on the
GPU's own numbers, argmax unchanged. End-to-end: 128 wikitext greedy tokens
**byte-identical** to the full-GEMV baseline (`Q27_METAL_HEAD_PRUNE=1` vs 0).

**Performance.** Kernel isolation (tools/head_prune_bench.cpp, synthetic
resident [248320,5120] head, 3×30 interleaved iters): stock 15.30 ms vs
prune 13.07 ms = **1.171× on the head pass** (byte model ceiling 1.23×;
fixed costs: 3 extra dispatches, 2 MB C/δ traffic, threshold pass). Pack is
0.08 s one-time at load. At the head's 7.4% share of official-tier step
bytes this is ~1.2–1.3% end-to-end on resident-weight hardware. Engine A/B
on this 24 GB box (6 ABBA pairs, tools/head_prune_ab.sh): mean 0.995 —
unresolved at the session's ±15% drift floor (weights page here; late
stable pairs 0.968/0.991 lean ON).

**Ship state: `Q27_METAL_HEAD_PRUNE=1`, default OFF.** The +947 MB screen
bank pressures ≤24 GB boxes (the official tier already pages there); the
win is proven at kernel level and correctness is byte-exact, but the
end-to-end price needs resident-weight hardware (M5-class) — re-price
there per the experiment-queue doc. Scoped to greedy argmax consumers
(decode_resident, serial greedy loops); sampled/teacher paths keep the full
GEMV; tool-constraint steps take the full GEMV (certificate is relative to
the unmasked max); read_logits re-projects when the buffer is pruned.

## Tier 2 RESULTS (2026-07-31)

- **Alpha-skip: considered, not shipped.** Our decode softmax rescales by
  `exp(m − m_new)` per position (q27_kernels.metal:959-968); skipping when
  the max doesn't advance is bit-exact (×1.0 identity) but the frontier
  measured it *neutral* on a latency-bound kernel, and ours is
  bandwidth-bound — the effect sits below this box's measurement floor.
  Recorded as available for the next attention-kernel touch; not worth
  dedicated churn.
- **F_NOCACHE on snapshot I/O: SHIPPED** (metal_engine.cpp save/load
  fdopen sites). Snapshot payloads are one-time sequential I/O; keeping
  them out of the UBC protects the mmap'd weight working set on
  snapshot-heavy agentic workloads (inferno's pattern). Best-effort,
  never fatal.
- **Always-on command-buffer GPU timing: SHIPPED**
  (`MetalBackend::cb_gpu_time()`): GPUStartTime/GPUEndTime accumulated on
  every finished command buffer, no per-op encoder splits —
  wall-independent GPU-busy attribution for bench/report tooling, zero
  cost when unread.
- Fused QK-norm+RoPE and fused residual epilogues remain queued behind
  measurement (expected small/null in the bandwidth-bound regime; see the
  regime section above).

## Appendix — source artifacts

- Leaderboard dump: 851 submissions via `https://api.mlx.fast/api/benchmarks/1854efdf-feba-4773-bae9-b80520881a74/submissions` (public, notes included). Frontier: 1.004 → 1.995× in 96 promotions over ~8 days.
- Challenge repo: github.com/Layr-Labs/mlxfast-challenge — TASK.md (scoring, serial non-speculative rule), Sources/MLXFastModel (frontier implementations incl. `LagunaLmHeadPrune.swift`, the certified prune).
- Inferno: github.com/nektarlabs/inferno — Rust/Metal GLM-5.2 Q2 MoE, 64 GB target, SSD-streamed experts; kernel set in `crates/backend/src/metal/kernels/*.metal`.
- Full extractions (96-entry frontier stack + ~45 dead ends; T1–T23 techniques + D1–D33 negative results + H1–H23 hardware facts; inferno 8-area survey with file:line citations) are reproducible from the two sources above; the submission notes carry the raw detail.
