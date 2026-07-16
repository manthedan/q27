# BaseRT survey (draft) — evidence-driven design import vs q27 measured bottlenecks

Status: DRAFT. Survey only — nothing built, nothing run, no model loaded.

## What was surveyed (and the honesty caveat up front)

- Repo: `github.com/basecompute/baseRT` @ `67c2dac2d9ab5bdbbf7664d8f0f57050271dce62`
  (2026-07-07, "hub: retry transient HF downloads"), cloned to
  `~/.claude/jobs/625abbd9/tmp/basert`. Paper: arXiv 2607.00501.
- **The engine and all Metal kernel source are proprietary.** The repo is the
  "open ecosystem" only: CLI/converter (Rust), `.base` format spec, public C
  headers, bindings, docs (`docs/reference/engine-releases.md:3-8` — "engine
  binary itself is **proprietary**"). No other repo exists (`gh repo list
  basecompute` → only `baseRT` + a pi extension; `gh search repos basert` → noise).
- So evidence comes from two tiers:
  - **file:line** in the open repo (format spec, quant spec, C API headers, docs);
  - **symbol/string evidence** from the shipped release binaries
    (`basert-engine-macos-arm64-0.1.5.tar.gz`: `baseRT.metallib` 2.19 MB,
    `libbaseRT.0.1.0.dylib` 6.6 MB), extracted with `strings`/`otool` only —
    never executed. Metallib AIR metadata embeds source filenames, mangled
    kernel signatures, AIR intrinsics, and function-constant names, which is
    unusually complete for a closed binary. Cited below as `metallib:` /
    `dylib:` symbols. This is weaker than source (no loop bodies, no tile
    constants) — treat instruction-mix claims as high-confidence and tile-shape
    claims as unknown.

Version note: repo header says `BASERT_VERSION 0.4.1` (`include/baseRT/baseRT.h:65-67`)
while the latest engine release tag is v0.1.5 and the dylib is 0.1.0 — their
versioning is inconsistent; the binaries surveyed are the v0.1.5 release assets.

## Per-question findings

### 1. Command-buffer / encoder strategy

- Single global Metal state object. `dylib:` ObjC class `BaseRTMetalState`
  (instanceSize 88) with ivars `_queue, _cmdBuffer, _prevCmdBuffer, _encoder,
  _library, _pipelineEvent (MTLSharedEvent), _residencySet, _residencyDirty`.
  One live command buffer + one persistent `MTLComputeCommandEncoder`; the
  previous command buffer is retained and chained via
  `encodeSignalEvent:value:` / `encodeWaitForEvent:value:` (`dylib:` selectors;
  log string `MetalDevice::begin_encoding_after_event: failed to create compute
  encoder`) — i.e. CPU encodes buffer N+1 while N runs, ordered by a shared event.
- Command buffers created with `commandBufferWithUnretainedReferences`
  (`dylib:` selector) — the low-CPU-overhead path.
- **Prebuilt dispatch table**: log string `Dispatch table: %zu commands per
  token` (`dylib:`) — the per-token dispatch sequence is materialized once as a
  command list and replayed, not re-derived per token.
- **No indirect command buffers, no argument buffers**: zero hits for
  `indirectCommandBuffer`/`MTLArgumentEncoder`/ICB selectors in dylib strings.
  Argument binding is plain `setBuffer:offset:atIndex:` + `setBytes:length:atIndex:`
  + `setThreadgroupMemoryLength:atIndex:` (threadgroup tile sizes are set
  host-side, not compile-time arrays).
- CPU syncs: `waitUntilCompleted` present; per-step decode needs one sync to
  read the argmax token, but `baseRT_chain_decode` "generate multiple tokens in
  one GPU submission" (`include/baseRT/baseRT.h:589-592`) amortizes to ~1 sync
  per N tokens (GPU argmax at `argmax_f16` feeds next embedding lookup on-GPU).
  The profiling API confirms normal decode does NOT use per-layer command
  buffers: `baseRT_profile_decode_step` "Runs each layer in its own command
  buffer ... Much slower than normal decode" (`baseRT.h:468-471`).

### 2. Elementwise fusion around GEMV/GEMM; decode-step sequence

Kernel inventory (all from `dylib:`/`metallib:` symbol names):

- Fused residual+norm: `residual_rmsnorm_f16`, `rmsnorm_add_f16`,
  `rmsnorm_add_post_f16` (+`_wbf16` = bf16 norm weights), `rmsnorm_gemma_f16`.
- Fused activation GEMV: `gemv_q{2,3,4,5,6,8}_silu(_sbf16)(_l)` — gate+up are
  convert-time fused into one weight matrix (`base-convert/FORMAT.md:72` "Gate +
  up fusion for SwiGLU MLPs"), and the GEMV applies SiLU·gate in-kernel; dylib
  error string "Failed to get fused silu+down GEMV pipeline".
- Fused rope+KV-write (incl. quantized KV write): `rope_qkv_write`,
  `rope_neox_qkv_write`, `head_norm_rope_neox_kv_write` (QK-norm + rope + KV
  store in one kernel, Qwen3-style), `gemma4_norm_rope_kv_write`, each with
  `_k_q8/_kv_q8/_kv_q4` variants — so KV quantization happens inside the rope
  kernel, not a separate pass. QKV is one GEMV because Q/K/V are fused at
  convert time (`FORMAT.md:71`, header tensor name
  `self_attn.qkv_proj.weight` `FORMAT.md:454`).
- GPU tail: `logit_softcap_f16`, `argmax_f16(_batched)`, GPU
  temperature/repetition-penalty (`baseRT.h:479-483`).
- Per-phase profile labels give the decode sequence end to end (`dylib:`):
  `embedding` → per layer: `ffn_norm`/attn norm (`residual_rmsnorm`) →
  `qkv_proj` (one gemv) → `kv_proj_norm_rope_write` → `attn_decode_*` →
  `o_proj` (gemv + `residual_add_f16`) → `ffn_norm` → `ffn_gate_up_act` (one
  fused silu gemv) → `ffn_down` → … → `output_norm` → `logit_proj` → `argmax`.
  Rough count: ~8-9 dispatches/layer, same fusion set q27 already has. RoPE
  tables are precomputed host-side (`FORMAT.md:395` `RopeTables` slot; dylib
  "RoPE: precomputed %d frequencies").

### 3. simdgroup_matrix in prefill GEMM

- All quantized prefill GEMMs (`simd_gemm_q2..q8`, `_small_`, `_large_`
  variants) use **8x8 simdgroup_matrix with f16 accumulation**: AIR intrinsics
  `air.simdgroup_matrix_8x8_multiply_accumulate.v64f16.v64f16.v64f16.v64f16`
  and `..._load.v64f16.p3f16` (`metallib:` simd_gemm_q4 block). Loads are from
  **p3 = threadgroup address space only** — both operands are staged, no
  direct-device MMA loads. Staging is `llvm.memcpy.p3i8.p1i8.i64`
  (device→threadgroup bulk copy) for activations plus register dequant for
  weights (see Q6). This is the same staged family as q27 production — BaseRT
  provides zero support for the direct-RHS idea (consistent with q27's 0.78x
  probe result).
- **f32 accumulation is a named opt-in variant, not the default**:
  `simd_gemm_f16_f32acc`, `simd_gemm_q4_f32acc(_sbf16)` exist only for f16 and
  q4 (`dylib:`); every other quantized GEMM accumulates in f16. Attention
  prefill by contrast accumulates f32
  (`multiply_accumulate.v64f32.v64f16.v64f16.v64f32` in
  `metallib: flash_attention_prefill_v2_2sg`).
- Tile shapes are not recoverable from the binary (threadgroup memory is
  host-sized via `setThreadgroupMemoryLength`). Hints: `moe_simd_gemm_q4_tm8`
  ("tm8" ≈ M-tile 8), `simd_gemm_small_*` / `simd_gemm_large_*` size classes,
  `simd_gemm_f16_tailk` (K-tail handling), and function constants
  `FC_GEMM_K`, `FC_SMALL_GEMM_K`, `FC_TAILK_GEMM_K`, `FC_BATCH_M` — i.e. the K
  dimension and batch-M are **baked in as function constants at pipeline
  creation**, so inner loops compile with literal trip counts.
- **No Metal 4 cooperative tensors anywhere**: the metallib is
  `air64-apple-macosx15.0.0`, and the only matrix intrinsics are
  `simdgroup_matrix_8x8_*`. Nothing in BaseRT requires Metal 4 — everything
  here is expressible on base M4. (Flagged because the task asked; the flag is
  "all clear".)

### 4. Activation representation between layers

- **f16 everywhere.** All elementwise/norm/attention kernels are `*_f16`;
  logits scratch is f16 (`baseRT.h:558-563` "Source storage is f16 on GPU";
  `baseRT_read_batch_logits` returns f16 rows `baseRT.h:339-343`). `_wbf16`
  suffixes mean bf16 *weights* for norms, not bf16 activations
  (`CANONICAL_QUANT_SPEC.md:213-216`: scales/biases bf16, "Norms are stored at
  f16" / bf16 variants).
- **No activation quantization for GEMV inputs**: gemv signatures take
  `device const half*` activations (`metallib:`
  `gemv_q4_impl<...>(device const half*, device const char*, device half*, constant QuantParams&, ...)`);
  the only absmax machinery is convert-time AWQ calibration
  (`baseRT.h:521-539`), not runtime activation quant.

### 5. Heaps, residency sets, argument tables

- **MTLResidencySet: yes** — `newResidencySetWithDescriptor:error:`,
  `addResidencySet:`, `setResidencySet:`, ivar `_residencySet` +
  `_residencyDirty` (`dylib:` selectors). Format carries residency budget
  metadata and per-tensor hot/warm/cold hints for the planner
  (`FORMAT.md:158-182`, explicitly "budgets `MTLResidencySet` accordingly").
- **MTLHeap: no** (zero heap selectors). **Argument buffers/ICBs: no** (Q1).
- The closest thing to a prebuilt argument table is the CPU-side per-token
  dispatch table (Q1) plus function-constant-specialized PSOs cached by
  `get_pipeline` (`dylib:` "get_pipeline: no function named '%s'
  (cache_key='%s')") — specialization keys include model dims, so uniforms that
  q27 might pass per-dispatch are compile-time constants here.

### 6. Q2/Q4 dequant inner loop

- **GEMM path**: shared helpers `dequantize_q{2,3,4,5,6,8}_half4x4(device
  const uint*|uchar*, device const half*|bf16* scales, ..., 5×uint,
  metal::matrix<half,4,4>&)` (`metallib:` mangled names) — dequant granularity
  is **16 elements into a half4x4 register matrix per call**, written to the
  threadgroup tile that the 8x8 MMA fragments then load. Packing is MLX-affine
  lane-strided uint32 (lane i at bit i·bits; pack factor 32/bits;
  `CANONICAL_QUANT_SPEC.md:114-123`), asymmetric by default (dequant = q·scale
  + bias, `:91-99`), q2 at group_size 32, q4 at 64 (`:25-32`).
- **GEMV path** (decode): `gemv_q4_impl<4,4>` / `<8,4>` template instantiations
  (`metallib:` — two unroll widths, plain and `_l` "large"); instruction mix
  from AIR intrinsics: `air.simd_sum.f32` (cross-simd reduction),
  `air.convert.f.v4f32.f.v4f16` (half4-vectorized loads widened to f32
  accumulation in registers), `simd_broadcast_first`, **no threadgroup barriers
  in any gemv kernel** (no `air.wg.barrier` in gemv blocks) — pure
  simdgroup-level GEMV with f32 scalar accumulation, 4-wide vector unpacking.
- **No byte-LUT tricks**: zero `lut`/table strings in the metallib; unpack is
  shift/mask + fma (the AIR shows converts and fmas only). Scales/biases live
  in separate contiguous regions specifically "so GEMM kernels can prefetch
  them into registers once per group" (`FORMAT.md:96-102`).
- Elements/thread not recoverable exactly (loop bodies not visible); the
  `<4,4>`/`<8,4>` template ints and half4 vector ops bound it to 16-32
  elements per thread per group iteration.

### 7. Attention kernel shape

- **Staged-with-barriers everywhere, never direct-read.** Decode kernels carry
  threadgroup locals `tg_max`, `tg_sum`, `tg_out` (`metallib:` mangled locals in
  `flash_attention_decode_parallel_kv_q8_f16`) with `air.wg.barrier`; prefill
  loads K/V tiles from p3 (threadgroup) only. This independently corroborates
  q27's R3 result (direct-read 0.47-0.53x) — BaseRT made the same call.
- **Online softmax**: `air.fast_exp2` + `air.simd_shuffle_xor.f32` butterfly
  max + `air.simd_sum.f32` running rescale, f32 accumulators
  (`metallib: flash_attention_prefill_v2_2sg`).
- Variant zoo, selected per shape: decode = `flash_attention_decode`
  {base, `_2sg`, `_parallel`, `_fast`, `_split_kv`, `_split_kv_2sg`} +
  `flash_attention_merge` (split-KV two-pass with merge kernel); prefill =
  `flash_attention_prefill_v2` {base, `_2sg`, `_4sg_hdsplit`} — 2 or 4
  simdgroups per threadgroup, head-dim-split at 4sg. Unfused fallback exists
  (`attn_qkt_unfused`, `attn_softmax_unfused`, `attn_pv_unfused` labels).
- **GQA**: `AttentionParams {seq_len, kv_seq_len, head_dim, n_heads,
  n_kv_heads, scale, kv_stride, q_stride, sliding_window, q_pos_offset}`
  (`metallib:` struct_type_info) — classic head-ratio sharing via strides; no
  exotic dequant-sharing scheme visible.
- **KV quantization**: default Q8_0 when head_dim%32==0 ("1.88x smaller"),
  forceable to f16 (`baseRT.h:96-102`); kernels also exist for `kv_q4`
  (`block_q4_0` struct in AIR: `d` + `uchar qs`; `block_q8_0`: `d` + `char qs`)
  — llama.cpp-style 32-element blocks, dequantized inside the attention kernel.
  KV layout: contiguous slabs by default; optional paged mode = 16-token blocks
  (8 if head_dim≥256) addressed via CSR block table (`baseRT.h:104-112`), with
  radix-tree prefix cache over the block pool (`baseRT.h:122-127, 345-426`).
  Nothing here resembles q27's planned structured (layer×head×RoPE-pair)
  allocation — BaseRT's KV quant is uniform Q8_0/Q4_0.

### 8. Unified-memory specifics (the paper's claim)

- **mmap + zero-copy MTLBuffer**: weights blob starts at a 64 KiB boundary,
  GPU-region tensors page-aligned (16 KiB Apple) as "a precondition for
  zero-copy buffer creation via `MTLBuffer.makeBufferWithBytesNoCopy`"
  (`FORMAT.md:24-28`); dylib uses
  `newBufferWithBytesNoCopy:length:options:deallocator:`. Runtime is
  "mmap-only ... streaming-friendly post-convert" (`FORMAT.md:543-544`).
- **Zero per-forward transformation**: transpose, QKV/gate-up fusion, tile
  interleave, scale folding all baked at convert time (`FORMAT.md:60-89`).
- **Page residency / working set**: tensor ordering matched to OS page-cache
  readahead ("readahead naturally prefetches layer N+1 while layer N runs",
  `FORMAT.md:154-156`); per-tensor hot/warm/cold residency hints + header
  budget block (`total/hot/max_expert_bytes`, `recommended_min_device_mb`)
  driving MTLResidencySet budgeting or CPU fallback (`FORMAT.md:158-182`).
- UMA also exploited API-side: logits readback is "Pure UMA copy, no dispatch"
  (`baseRT.h:339-343`).
- Assessment: this is the paper's most distinctive engineering, but it targets
  load time, memory footprint, and MoE/big-model paging — not the prefill MMA
  issue bound or dequant issue cost. q27 already mmaps weights; the residency
  planner matters at model sizes q27 doesn't currently miss on.

### Benchmark context (their own numbers)

`benchmarks/results/m4-pro_baseRT.csv:1-30`: Qwen3-0.6B on M4 Pro (~273 GB/s
part) — pp512 ≈ 4400-4800 t/s, tg128 ≈ 450-470 t/s across Q2-Q5. Reference
only; not comparable to base-M4 q27 numbers without normalizing bandwidth, and
their CSVs are from converted GGUF checkpoints of a 0.6B model.

## Import candidates (bottleneck-matched)

q27's measured facts, restated as the filter: decode GEMV at ~89 GB/s = 98-99%
of own ceiling (decode imports near-worthless); prefill chunk GEMM
MMA-issue-bound at ~21-25 GB/s effective weight stream; direct-RHS probe 0.78x
(worse); barrier-free attention 0.47-0.53x (worse). Only prefill
instruction-issue reducers qualify. BaseRT's decode-side toys (dispatch table,
chain decode, shared-event pipelining, paged KV) target CPU overhead and
serving features — excluded. Attention and staging designs match q27 production
already — nothing to import, but two independent confirmations that q27's
staged choices are right.

1. **f16-accumulate MMA in the prefill chunk GEMM.** BaseRT's default for every
   quantized simd_gemm is f16×f16+f16 8x8 MMA (f32acc is the exception
   variant). If q27's chunk GEMM accumulates in f32 simdgroup matrices, f16
   accumulators halve accumulator register pressure and can raise MMA issue
   throughput — squarely the measured bottleneck.
   Probe (cheap): compile an f16-acc variant of the existing chunk-GEMM tile
   loop (accumulate per K-group in f16, optionally widen to f32 once per group
   boundary to bound error), run the standard prefill bench + logit-KL check.
   Kill line (pre-registered): <10% effective-weight-stream gain on the
   MMA-issue-bound chunk bench, OR logit KL/RMS regression beyond the existing
   quant-validation bar → park with numbers.

2. **Function-constant specialization of the dequant/GEMM inner loop.** BaseRT
   bakes `FC_GEMM_K`, `FC_GROUP_SIZE`, `FC_K_DIM`, `FC_BK_SIZE`, `FC_NQ_ROWS`,
   `FC_HEAD_DIM` into PSOs at load (per-model pipeline cache), so trip counts
   are literals and the compiler fully unrolls unpack loops — a direct
   per-element-issue-cost reducer. If q27 passes K/group_size as runtime
   uniforms, this is free instruction-mix improvement.
   Probe: specialize just the chunk-GEMM kernel's K and group_size via
   `MTLFunctionConstantValues` in the existing bench harness; diff instruction
   counts (compiler stats) + measured GB/s.
   Kill line: <5% on the chunk GEMM bench → not worth the PSO-cache complexity.

3. **half4x4-granularity register dequant feeding threadgroup tiles.**
   BaseRT's unpack unit is 16 elements → `matrix<half,4,4>` per helper call
   (shift/mask + fma, no LUTs, scales prefetched once per group from a separate
   contiguous region). Import only the *shape* of this: if q27's Q2/Q4 unpack
   emits finer-grained (scalar/half4) stores to threadgroup, batching to 4x4
   register tiles cuts store and address-arithmetic issue.
   Probe: rewrite the unpack inner loop of one production Q4 chunk kernel to
   16-element register-tile granularity; measure issue-bound GB/s.
   Kill line: <5% gain, or q27's current unpack already emits ≥16-element
   batches (check first — this may be a no-op) → drop.

Not imported, with reasons: staged-RHS and staged attention (BaseRT agrees with
q27 production; their existence *validates* the 0.78x and 0.47-0.53x parks);
Q8_0/Q4_0 KV quant (uniform blocks, strictly less structured than the P1 KV
codec plan); dispatch table / chain decode / residency sets (CPU-side or
footprint wins; decode is at 98-99% of bandwidth ceiling); MoE bucket kernels
(q27 is dense). Metal 4 cooperative tensors: not used by BaseRT at all —
no portability flag needed.

---

## Probe dispositions (mini, 2026-07-16 afternoon — appended to the draft)

**Probe 1 (f16-accumulate MMA): RAN and PARKED** — aggregate C/F = 1.065
[1.064, 1.066] vs the pre-registered 1.10 line, per-shape flat; full
record in 2026-07-16-f16acc-probe.md (numerics caveat included: the
synthetic bench never stressed the 5e-2 gate).

**Probe 2: MEASURED and PARKED. Probe 3: T2 no-op / Q4 deferred.**
(A first "parked by decomposition" framing for probe 2 was REJECTED in
review — codex P2, correct: loop/address overhead sits on BOTH sides of
C/Beq, so that ratio cannot bound specialization gains. Bounding-arm
re-reads kept for context: C/Beq = 1.081 [1.080, 1.082] post-byte-LUT,
C/Cx = 1.024.)

- Probe 2 (function-constant K/group-size baking) — measured as
  roofline arm K: the production mm_h with FC_COLS baked as a Metal
  function constant (per-shape specialized PSO, trip counts and all
  cols-derived address math become literals), bit-identity gate at
  tolerance 0 vs C — held exactly on every shape. **Aggregate C/K =
  1.010 [1.009, 1.011], 18 counterbalanced trials — PARK at the
  probe's own <5% kill line.** On q27's staged 64-K walk, outer trip
  count + cols-derived address arithmetic are ~1% of kernel time;
  BaseRT's baking presumably pays on their fully-unrolled unpack
  loops, not on this shape.
- Probe 3 (16-element register-tile dequant): split disposition. For
  the T2 production kernel the survey's own "may be a no-op" case
  holds — the byte-LUT unpack already stages 16 elements per thread
  per slab as 4× half4 vector stores. The probe's stated Q4 target
  (`q27_matmul_q4_mm` still expands 16 nibbles to scalar stores —
  codex file:line) is real but not on the T2 artifact's path at all;
  Q4 chunk GEMM only becomes production-relevant if DSpark Phase 3
  integrates the Q4_1 drafter. DEFER to the DSpark Phase-3 decision;
  revisit with the drafter's actual shapes if it graduates.

Net: the BaseRT survey is fully discharged on the mini side — probes 1
and 2 run and parked with measured numbers (C/F 1.066, C/K 1.010),
probe 3 a T2 no-op with its Q4 target deferred to the DSpark Phase-3
decision, all other imports pre-answered or adopted elsewhere (KV
codec, ds4 snapshot priority).
