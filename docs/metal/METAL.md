# Metal backend

## Scope

This port preserves q27's model format and decoding algorithms while replacing CUDA execution with a Metal-native backend. CUDA remains the reference implementation. Cross-backend gates compare numerical tolerances, token rankings, and committed tokens rather than requiring bitwise equality.

The backend boundary is `src/backend.h`. It deliberately exposes model operations and opaque buffers, not CUDA streams or Metal encoders.

## Current status

The Metal arm is the **serving stack and the active development backend**,
not a port-in-progress. It serves two slots on a base M4 with:

- full 64-layer decode, the layer-64 MTP draft path, and batched MTP;
- layer-major chunked prefill to width 96, factor-2 tiled GQA attention;
- suffix-burst speculation and batched verification to width 48;
- F32 / F16 / Q8_G128 / Q4_G64 matvec plus the ternary (T2) and binary (B1)
  bonsai tiers;
- opt-in turbo3 KV (`--kv turbo3`), with per-cell fp16 exception cells;
- disk prefix snapshots with token-exact verification, LRU + spine-pin
  eviction, and a fair two-slot scheduler with full memory admission
  accounting;
- an OpenAI/Anthropic/Responses-compatible server with tool-call streaming,
  optional API-key auth, and `--ctx auto` device-budget sizing.

Measured status, gates and the numbers behind each of those live in
[BUILDLOG.md](BUILDLOG.md); what was tried and rejected is in
[DECISIONS.md](DECISIONS.md).

Run on macOS from the repository root:

```sh
make test-cpu     # host suites + (on Darwin) the Metal server build
make test-metal   # Metal device tests
```

`Q27_METAL_SOURCE=/path/to/q27_kernels.metal` overrides the runtime shader
path.

`build/q27-metal` is the CLI; `build/q27-metal-server` is the serving
binary. The CUDA executable is unchanged and remains the numeric reference —
the 16-token canonical gate is byte-exact across CUDA and Metal.

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
