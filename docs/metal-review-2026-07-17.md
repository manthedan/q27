# Metal implementation audit

**Date:** 2026-07-17
**Scope:** `src/metal/q27_kernels.metal` (4,299 lines), `src/metal/metal_backend.{h,mm}` (2,519 lines), engine, SPEC, METAL_PROGRESS, supporting plans and tests.
**Verdict:** Mature, unusually well-instrumented port. Strong barrier hygiene, careful edge-clamp conventions, a shader-ABI-tag guard born from a real stale-binary incident, and a high density of measured-then-banked alternatives. No race or barrier defects in the production attention/softmax/GEMM paths, no buffer lifetime leaks in steady-state decode, no numerics regressions vs. SPEC beyond the documented tolerance-gated classes. Findings below are severity-ranked.

---

## A. Bugs / Correctness issues

### A1 — `q27_b1_x_prep` shared-memory aliasing race (HIGH)

```metal
threadgroup float pmax[4], psum[4];
float amax = simd_max(fabs(xv));
if (lane == 0) pmax[simdgroup] = amax;
threadgroup_barrier(mem_flags::mem_threadgroup);
amax = max(max(pmax[0], pmax[1]), max(pmax[2], pmax[3]));
...
float sumu = simd_sum(float(u));
if (lane == 0) psum[simdgroup] = sumu;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) aux[group] = float2(s, psum[0] + psum[1] + psum[2] + psum[3]);
```

The comment explicitly claims "no barrier between the phases" works because "sum partials are written while other threads may still be reading the max partials." That is **wrong** as stated: `pmax` and `psum` are separate arrays, and the cross-simdgroup `amax` broadcast reads happen before the second barrier with no ordering against `psum` writes — benign only because `pmax != psum`. So the code is correct **by accident of layout**, not by the stated reason. If anyone later consolidates `pmax`/`psum` into one array (a natural optimization), this becomes a genuine data race. Add an explicit `threadgroup_barrier` between phases, or convert to a single `partial[8]` with one barrier.

### A2 — `q27_topk_logits` capacity and underflow handling (HIGH)

```metal
const uint slot = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
if (slot < args.capacity) { out_values[slot] = logits[i]; out_indices[slot] = i; }
```

The atomic counter is incremented unconditionally on every qualifying candidate; only the store is gated by `slot < capacity`. Two problems:

1. **Wraparound on degenerate tie storms.** `count` saturates at UINT32_MAX. Impossible for vocab=248K, but the safety comment about "degenerate tie storms" is not actually gated — `count > 1024` still trips, so this is a theoretical concern only.
2. **The host falls back to full readback when `count > capacity`, but there is no fallback when `count < k`.** Possible when the second-byte histogram walk finds `cumulative >= remaining` only at `bin == 0`, then the `key16 >= threshold` filter can include fewer than `k` items. The host must check `count >= k` and fall back, otherwise sampled output silently loses candidates. The code says "host truncates after an exact sort" but there is no min-k enforcement visible in `sampling.h`. **Add an explicit `if (count < k) fallback` on the host.**

### A3 — `q27_matvec_t3_g128` dead-code path for odd `nb` (MEDIUM)

```metal
const uint words_per_row = (row_bytes & 3) ? 0 : row_bytes / 4;
```

When `nb` is odd (`row_bytes = nb*26` is even but not /4 aligned for `nb ∈ {1, 3, ...}`), the entire word walk is skipped and every byte goes through the scalar tail loop. Comment claims "model shapes all have even nb and never do," but `cols=128*odd` (a future 1024-dim toy model or `cols=1152`) silently hits the slow path. **Add a startup assertion that `cols % 512 == 0` for T3, or fix the tail to be correct for any `nb`.**

### A4 — `q27_argmax` ties: stable but undocumented (LOW)

```metal
float best = -INFINITY;
uint best_i = 0;
for (uint i = tid; i < n; i += 256) {
    const float value = x[i];
    if (value > best || (value == best && i < best_i)) { best = value; best_i = i; }
}
```

Initial `best_i = 0` is fine for tid 0, but tid 1..255 start with `best = -inf, best_i = 0`. If lane k's first read equals -inf (a real logits value, e.g., grammar-masked tokens), the condition is false (`k < 0` is false), so a tied-at-`-inf` lane keeps `best_i = 0` rather than its own index. Still correct because the cross-simdgroup reduction compares `other_i < indices[tid]` and picks lower indices. The first-past-the-post tie semantics are stable. **Not a bug, but add a comment** — this is the load-bearing subtlety of deterministic argmax over masked logits.

### A5 — `q27_delta_step` silent no-op on shape mismatch (LOW)

```metal
if (head >= args.value_heads || args.head_dim != 128 || args.qk_heads != 16) return;
```

If a future model ever has different DeltaNet dimensions, the kernel silently produces zeros for every head (all threadgroups early-exit). Correct for the current artifact but should throw host-side rather than be a silent kernel no-op. The host `delta_step` does not validate this either; add an explicit `head_dim == 128 && qk_heads == 16` host check.

### A6 — Whole-mapping `tensor_limit` granularity regression (MEDIUM, acknowledged)

METAL_PROGRESS already notes this from codex review: when the 6.66 GiB T2 mapping is wrapped as one MTLBuffer, `check_range` uses `tensor_limit(buffer_size, offset, logical_size)` which is correct, but per-tensor extent is still validated against the whole mapping extent at loader open time, not at bind time. A corrupt tensor header whose declared extent is enormous but lies within the mapping would pass host validation and the kernel would read into neighbor tensors. The recorded fix is "logical lengths in `BackendTensor`" — not yet implemented. **Worth prioritizing** since it is the one path that can produce silently-wrong output on artifact corruption (exactly the class that burned two days on 2026-07-14).

---

## B. Robustness / resource issues (host)

### B1 — `gqa_partials` grows monotonically and is never shrunk (MEDIUM)

```objc
if (!gqa_partials || gqa_partials.length < partial_bytes)
    gqa_partials = [device newBufferWithLength:... options:MTLResourceStorageModePrivate];
```

Once a long context runs, the partials buffer stays at peak size for the backend's lifetime. On a multislot server cycling between a 32K and a 2K conversation, this holds ~26 MB × q_heads × n_blocks that is never reclaimed. Two fixes: (a) free when shrinking below 25% utilization; (b) make it per-engine, not per-backend (the codex review of `Shared` already moved KV budget to per-engine — extend the same to GQA partials).

### B2 — `attrib_dummy` and `dr_xt_scratch` are unconditionally Private (LOW)

Both are allocated with `MTLResourceStorageModePrivate` and never freed until `~MetalBackend`. The lifetime should be tied to the engine session, and `dr_xt_scratch` (bench-only, but allocated lazily on first `mma_roofline('d'|'2')`) leaks into production backends that never call the roofline. Add `dr_xt_scratch = nil` after the bench binary exits, or scope it behind `#ifdef Q27_METAL_BENCH`.

### B3 — `synchronize()` is a full command queue flush (MEDIUM)

Every `read()` issues a dedicated empty command buffer and `waitUntilCompleted`. For sampling, `read(out_index)` after argmax pays a full GPU pipeline stall per token. Recorded as Phase 4 leverage but worth surfacing in the roadmap explicitly: the GPU-assisted sampling stage 2 already cut the bytes read back, but the latency of the readback is unchanged. Pair this with stage 3 (on-GPU draw) or double-buffer the token index.

### B4 — `write()` and `zero()` reject batched context but `copy()` does not (LOW)

`copy()` is allowed during batching (it dispatches a kernel on the same encoder), which is correct, but the symmetry with `write`/`zero` is undocumented. Add a comment to `copy()` noting that it is the only host-mutation path safe during a batch (because it dispatches GPU work rather than touching shared-memory contents), otherwise a future maintainer will likely add the `if (batching) throw` check here too and break legitimate uses.

### B5 — `abort_commands() noexcept` does not call `[command cancel]` (LOW)

```objc
void MetalBackend::abort_commands() noexcept {
    @autoreleasepool {
        if (impl_->encoder) [impl_->encoder endEncoding];
        impl_->encoder = nil;
        impl_->command = nil;
        impl_->batching = false;
    }
}
```

Setting `command = nil` releases the objective-C reference but does not `[command cancel]`. If the command buffer was already `commit`-ed elsewhere (it should not be, since batching holds it uncommitted) and the server is shutting down under pressure, the buffer may still execute. The contract here is that `abort_commands` is only called between `begin_commands` and `end_commands` (no commit yet), so this is correct — but call `[command cancel]` defensively before dropping the reference.

---

## C. Performance observations

### C1 — `matvec_quantized` Q4/Q8 main loop is issue-bound at 4 ops/MAC (PARKED lever)

`q27_dot4` and `q27_dot8_q4` are scalar int8 dot helpers. METAL_PROGRESS already records 67–90 GB/s (70–85% of M4 stream) and explicitly parks further work. The Q4/Q8 path is the one place where the integer-dot technique is still issue-bound (Q4/Q8 didn't get the same select-form treatment T2 got). The roadmap item "remaining GEMV headroom toward ~100 GB/s" is in NOTEBOOK but not formalized — the natural next step is either (a) apply the T2 select-form trick to Q4 (4 codes × 16 elements, one scale per 64) or (b) stage a half-precision tile and use `simdgroup_multiply_accumulate` for the per-row reduction (the half-accumulator-exactness argument that landed for T2 chunk GEMM should apply to GEMV too).

### C2 — `matvec_quantized_x2` and `matvec_x2` are live despite probe being parked (LOW)

The kernels and host entry points are wired (`t2_quantized_x2`, `t2_x2`, `matvec_quantized_x2`, `matvec_x2`), but METAL_PROGRESS records them as parked by measurement (1.093 vs 1.31 line). If they are not engine-routed today, they are dead surface that increases shader compile time and ABI surface area. Either remove them or mark them with a clear `// PARKED` comment + assert they are not called.

### C3 — GQA blocked attention allocates `gqa_partials` per-dispatch (MEDIUM)

Every call to `attention_gqa_dispatch` re-checks `gqa_partials.length` and potentially reallocates. The allocation is in the steady-state hot path for long-context decode. Hoist the sizing to engine-construction time (or snapshot budget) so a 32K decode doesn't pay allocation jitter on the first block.

### C4 — `q27_attention_f16` (decode) pays a needless merge round for `n_blocks == 1` (LOW)

The threadgroup memory footprint of every attention merge is `4 * (2 + 256) * 4 = ~4 KB`. With head_dim=256 this is unavoidable, but the merge rounds have an `offset /= 2` loop that runs 3 iterations with a barrier pair each. For the common case of `n_blocks == 1` (single-block sequence), the merge could be skipped entirely with a `if (n_blocks == 1) { normalize-and-store; return; }` fast path. The current code does this implicitly (one iteration of the loop where `offset=1` only runs if `sg < 1`), but pays one barrier pair needlessly.

### C5 — `q27_matvec_t2_g128` 4-blocks-in-flight decision (LOW, confirms existing)

The T2 select-form GEMV is the production decode bottleneck. `ix = lane / 8` (4 blocks) with `il = (lane%8)*16` is the published structure. The amortization across rows is excellent (4 rows share the y-slice), but a `lane / 4` variant (8 blocks in flight) would double register pressure on `yl[]` from 16 to 32 floats — recorded as "tried and lost on register pressure" in METAL_PROGRESS. **No action**, just confirming the existing decision.

### C6 — `gemm_half` is backend-scoped, not engine-scoped (MEDIUM)

The half-staging GEMM is on by default; opt-out via `Q27_METAL_GEMM_HALF=0` or `set_gemm_half()`. The A/B lever (turning it off mid-session for KL attribution) mutates `impl_->gemm_half` without a memory barrier — fine for single-threaded engine use, but the `Shared` refactor that allows two engines on one mapping means two engines can share one backend and stomp on each other. Verify that `set_gemm_half` is engine-scoped, not backend-scoped. Looking at the header it is on `MetalBackend`, so it IS backend-scoped: if two engines on one backend ever flip this independently, KL measurements corrupt.

---

## D. Numerics / contract observations

### D1 — `q27_rmsnorm` uses `rsqrt(sum/n + eps)` (OK)

SPEC says rms eps is 1e-6. Kernel matches. Not a bug.

### D2 — `q27_gdn_gates` softplus branch points (OK, fixed in codex sweep)

```metal
const float softplus = value > 20.0f ? value
                     : (value < -16.0f ? exp(value) : log(1.0f + exp(value)));
```

The -16.0 cutoff matches CUDA's `log1pf` tail behavior. Documented in METAL_PROGRESS as a codex fix. Good.

### D3 — `q27_rope_neox` partial-RoPE position components (OK for text-only)

SPEC VERIFY-3 says IMROPE==Neox only when all position components equal. The Metal kernel uses NeoX directly. Correct for the text-only path but silently produces wrong results if anyone uses it for non-text RoPE (no MTP, no vision). Worth a kernel comment.

### D4 — `q27_matmul_t2_mm` and `q27_matmul_q8_mm` chunk GEMM are tolerance-gated, not bit-exact (OK)

The 3e-4 tolerance is recorded. The known consequence (MTP low-margin divergence ~token 35, "is known" vs "serves") is pre-existing and documented. Good.

---

## E. Suggested missing Roadmap items

These are gaps not formalized in `docs/plans/` or the "Active work" / "Known debt" sections of METAL_PROGRESS.md.

### E1 — Whole-mapping tensor-extent validation (P1, follows codex finding)
Replace the `check_range(... buffer_size ...)` family with logical-length enforcement in `BackendTensor`, so a corrupt tensor header cannot read into neighbors on the whole-mapping path. This is the one acknowledged unimplemented fix that could repeat the 2026-07-14 silent-corruption incident.

### E2 — Per-engine `gqa_partials` budget + lifetime (P2)
Move `gqa_partials` out of the backend `Impl` and into the engine (alongside KV budget), with shrink-on-low-utilization. Blocks multislot scenarios where slot context lengths vary.

### E3 — Engine-scoped `gemm_half` and other envelope-instrument knobs (P2)
`set_gemm_half` and `set_gqa_threshold` are backend-scoped today. If two engines on one backend (KL instrument, future multi-tier serve) ever flip them concurrently, instruments corrupt. Either make them per-engine or document a single-engine-mutation contract.

### E4 — Top-k host-side `count < k` fallback (P1, follows A2)
Explicit fallback in `sampling.h` / `sample_next` when the GPU's over-set underflows `k`. Today only the overflow case is covered.

### E5 — T3 / odd-`nb` startup assertion (P2)
Defensive: refuse to load T3 artifacts whose `cols % 512 != 0` until the T3 scalar tail is hardened for arbitrary `nb`. Cheap to add to `validate_architecture`.

### E6 — Q4/Q8 GEMV select-form or simdgroup-MMA variant (P3, perf)
The T2 select-form and T2 half-MMA rewrites each closed ~30% of the gap to stream bound. Q4/Q8 (still 67–90 GB/s vs ~120 stream) have not had the same treatment and are the dominant cost on the official tier (Q4 weights). Worth a `metal_gemv_bench` leg.

### E7 — `b1_x_prep` barrier hardening (P3, follows A1)
One explicit barrier between the max and sum phases; cheap insurance against future "consolidate the two arrays" optimizations.

### E8 — `q27_argmax` and `q27_topk_logits` correctness stress tests (P2)
Both have width-gated fast paths and tie-handling subtleties. Add explicit tests for: all-`-inf` input (grammar mask), all-tie input (degenerate tie storm), `n` not a multiple of 256/1024, and `count == k - 1` for topk (the A2 hazard).

### E9 — `synchronize()` avoidance on the sampling hot path (P2, perf)
Per-token `[queue commandBuffer]; commit; waitUntilCompleted]` for the readback is the structural latency floor of serial decode. Stage 3 of GPU-assisted sampling (on-GPU draw) closes this; without it, decode is hard-capped by the readback turnaround regardless of how fast the kernels get.

### E10 — Dead/experimental surface cleanup (P3)
`matvec_quantized_x2`, `matvec_x2`, the entire MMA roofline family (arms A/B/B_eq/C/Cx/F/K), the `mm_dr`/`mm_dr2` levers, and `attention_turbo3_gqa_hm`/`bf2`/`t4` are all bench-only probes that remain in the production backend binary. Each ships a PSO that costs startup compile time. Consider `#ifdef Q27_METAL_BENCH` gating around the bench-only pipeline creation in the constructor, so `q27-metal-server` doesn't compile probes it never dispatches.

### E11 — Explicit `head_dim == 128` DeltaNet host guard (P3, follows A5)
Mirror the kernel's early-exit conditions with host-side throws in `delta_step`/`delta_chunk`.

### E12 — `metal_backend.mm` host `copy()` batching semantics comment (P3)
Just a comment fix; documents that `copy()` is the only safe-during-batch mutation path.

---

## Summary

The Metal implementation is mature and unusually well-instrumented. The most consequential **open bug** is **A2** (topk `count < k` fallback), which can silently corrupt sampled output and is one cheap host check away from closed. The most consequential **structural debt** is **A6/E1** (whole-mapping tensor extent validation) — it is the one path that can reproduce the silent-corruption class that previously burned two days. The **perf roadmap gap most recommended for formalization** is **E6** (Q4/Q8 GEMV still has 30% headroom that the T2 rewrite closed for free on the other tier).

Everything else is either already acknowledged in METAL_PROGRESS (confirming priority), or is a small hardening/comment fix. The audit found **no race or barrier defects in the production attention/softmax/GEMM paths**, no buffer lifetime leaks in steady-state decode, and no numerics regressions vs. SPEC beyond the documented tolerance-gated classes.
