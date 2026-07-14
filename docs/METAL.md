# Metal backend

## Scope

This port preserves q27's model format and decoding algorithms while replacing CUDA execution with a Metal-native backend. CUDA remains the reference implementation. Cross-backend gates compare numerical tolerances, token rankings, and committed tokens rather than requiring bitwise equality.

The backend boundary is `src/backend.h`. It deliberately exposes model operations and opaque buffers, not CUDA streams or Metal encoders.

## Current status

Implemented:

- startup validation for `.q27` and `.tok` artifacts before backend upload;
- Metal device, command queue, runtime shader compilation, shared buffers, and mmap-backed weight views;
- F32, F16, Q8_G128, and Q4_G64 matrix-vector multiplication;
- decode primitives, one-token Gated DeltaNet, FP16 GQA attention, and state snapshots;
- complete 64-layer greedy decode plus the layer-64 MTP draft path;
- command-batched teacher-forced prefill;
- opt-in turbo3 KV (`--kv turbo3`) with WHT/codec/attention tests;
- CPU-reference and synthetic Metal tests that do not load the 17 GiB model.

Run on macOS from the repository root:

```sh
make test-cpu
make test-metal
```

`Q27_METAL_SOURCE=/path/to/q27_kernels.metal` overrides the runtime shader path.

`build/q27-metal` is the baseline Metal inference executable. It is functionally correct on the official artifact but not performance-competitive yet. The existing CUDA executable remains unchanged and is the production/reference backend. See `docs/METAL_PROGRESS.md` for checkpoint gates and measured status.

## Implementation order

1. **Decode primitives:** embedding lookup, RMSNorm, elementwise activations, RoPE, argmax, and FP16 KV storage.
2. **Single-token decode:** attention and one-token Gated DeltaNet recurrence, then layer/output wiring.
3. **MTP correctness:** widths 2, 4, 8, and 12 with state commit/rollback gates at every width.
4. **Long context:** turbo3 encode/decode and attention gates at 32K, 128K, and 262K.
5. **Performance:** mmap-backed no-copy weight buffers, command batching, fused GDN, and small-N verification GEMMs.

## Design constraints

- Threadgroup tiles must be selected from Metal device limits; CUDA shared-memory layouts are not carried over.
- Version-one weights retain the `.q27` layout so CUDA and Metal can consume the same artifact.
- The correctness kernels use safe Metal math. Relaxed math is a later, gated optimization.
- Model tensors are wrapped as page-aligned views of the model mmap; the `Model` must outlive every view.
- One command buffer per speculative round is the target. Recreating the CUDA graph zoo is not.
