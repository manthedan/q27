# q27 weight format (version 1)

Offline-repacked weights for the q27 engine. Produced by `tools/repack.py` from the BF16 GGUF (plus `--mtp` for the companion MTP GGUF of current llama.cpp conversions), or from a PrismML Bonsai pack. Designed for mmap + single cudaMemcpy per tensor, and coalesced 128-byte warp reads in the fused-dequant GEMV.

## Container layout

```
[header]
  magic      u32   = 0x46373251  ("Q27F" little-endian)
  version    u32   = 1
  n_tensors  u32
  meta_len   u32
  meta       u8[meta_len]   JSON: arch config + layer map + quant policy
[tensor table]  n_tensors entries:
  name_len   u16
  name       u8[name_len]   GGUF tensor name, unchanged
  dtype      u8             0=F32  1=F16  2=Q8_G128  3=Q4_G64  4=T2_G128  5=T3_G128  6=B1_G128  7=FP4_G16
  n_dims     u8
  shape      u64[n_dims]    numpy row-major shape (outer first; innermost/contiguous LAST)
  data_off   u64            relative to data section start, 256-byte aligned
  data_size  u64
  scale_off  u64            0 if dtype has no scales
  scale_size u64
[data section]  256-byte aligned blobs
```

## Quantized types

All quantized types quantize along the **contiguous (innermost) axis**, which
is the GEMV reduction axis for every matmul weight in this model.

### Q4_G64 (bulk weights)
- symmetric, group size 64: `scale = max(|wـgroup|) / 7`, `q = clip(round(w/scale), -8, 7) + 8` stored as unsigned nibble
- packing: element `i` of a row -> byte `i/2`; **even index = low nibble**, odd = high nibble
- scales: fp16, shape `[rows, cols/64]`, separate contiguous blob
- effective 4.25 bpw
- a warp reading 128 B gets 256 consecutive weights = exactly 4 groups

### Q8_G128 (quality-sensitive weights)
- symmetric, group size 128: `scale = max(|w_group|) / 127`, int8
- scales: fp16, `[rows, cols/128]`
- effective 8.125 bpw

### T2_G128 (ternary weights, dtype 4)
- group size 128; element `i` of a row uses the 2-bit field at
  `(i % 4) * 2` in byte `i / 4` (sequential, LSB-first)
- code `c` must be in `{0,1,2}` and decodes to `(c-1) * scale`; code 3 is invalid
- row data is `cols/4` bytes; scales are fp16 `[rows, cols/128]`
- effective 2.125 bits per weight

### B1_G128 (binary weights, dtype 6)
- group size 128; element `i` of a row uses bit `i % 8` in byte `i / 8`
  (sequential, LSB-first)
- bit `b` decodes to `(2b-1) * scale`; every bit pattern is valid
- row data is `cols/8` bytes; scales are fp16 `[rows, cols/128]`
- effective 1.125 bits per weight
- emitted for v1 Bonsai binary packs (fork `Q1_0`, `quant_policy`
  `bonsai-b1-v1`); the Metal backend serves it, the CUDA loader rejects it
  (`cuda_weight_dtype_supported` in `loader.cpp`)

### T3_G128 (ternary, five trits per byte, dtype 5)
- group size 128; five ternary codes are stored per byte in base 3:
  `c0 + 3*c1 + 9*c2 + 27*c3 + 81*c4`, with every byte in `[0,242]`
- each group uses 26 bytes; byte 25 carries columns 125..127 and its unused
  slots 3 and 4 must contain code 1 (zero)
- row data is `(cols/128)*26` bytes; scales are fp16 `[rows, cols/128]`
- effective 1.75 bits per weight
- the loader rejects any byte above 242 and any final byte whose padding
  slots are not code 1 (`validate_tensor_payload`)
- emitted by `repack.py --bonsai2-container t3` for the Bonsai 2 8 GB-card
  packs (2026-09-20, every `blk.*` matrix; embeddings/head stay T2 under
  `--slim`). Metal has a kernel that reads these bytes directly
  (`q27_matvec_t3_g128`), though `MetalEngine` does not yet accept `bonsai2-*`
  packs; the
  CUDA engine relays them at upload into a window layout built for the
  decode GEMV (`kernels.cuh` T3 section: scaled base-3 bytes popped a digit
  per dp4a word, the same chunk order as `gemv_t2`, so the two GEMVs are
  bitwise), and its prefill MMA GEMM reads a per-matrix T2 conversion of
  it. Row bytes on the device: 5120 -> 1024, 6144 -> 1280, 17408 -> 3584.

### FP4_G16 (nvfp4 prefill sidecars, dtype 7)
- group size 16; e2m1 codes packed 2 per byte, even index = low nibble
- scales: one ue4m3 BYTE per group (not fp16), `[rows, cols/16]`
- emitted only by `repack.py --pf4` as `<name>.pf4` copies of the
  `blk.0-63` `ffn_gate/ffn_up/ffn_down/attn_q/attn_output` weights (header
  meta `pf4_sidecars`); read only by the fp4 prefill leg (`src/pf4.cu`,
  sm_120a). Metal does not accept it.

## Quant policy (v1)

| tensors | dtype | why |
|---|---|---|
| all `*_norm.weight`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, `output_norm` | F32 | tiny, numerically sensitive |
| `ssm_alpha.weight`, `ssm_beta.weight` | F16 | 48-wide heads, awkward group size, tiny anyway |
| `token_embd.weight`, `output.weight` | Q8_G128 | vocab quality; embed is row-lookup (not GEMV-read) |
| everything in `blk.64.*` (MTP layer) | Q8_G128 (matmuls) / F32 (norms) | draft/verify agreement must survive quantization or MTP acceptance craters |
| `blk.*.attn_k.weight`, `blk.*.attn_v.weight` | Q8_G128 | worst Q4 RMSE, and the error persists in the KV cache; ~84 MB |
| `output_q4.weight` (extra copy of `output.weight`) | Q4_G64 | v1.3: lm_head read by MTP draft passes only (`--q4-head` instead makes `output.weight` itself Q4 and drops the copy) |
| all other matmul weights (blk.0-63) | Q4_G64 | the ~14 GB bulk |

This table is the DEFAULT tier (`quant_policy` `v1.3`, or `v1.4` when `--q8`
promotes extra tensors; ~5.25 bpw overall). The q6 / q6k quality tiers
(2026-07-12, BUILDLOG) promote selected bulk tensors to Q8_G128 within the same
container and dtype set; the tier is recorded in the header meta as
`quant_policy` (e.g. `q6-v1`). No new dtypes, no version bump.

v1 Bonsai packs (PrismML fork `Q2_0` / `Q1_0`) are detected automatically and
repacked losslessly to T2_G128 / B1_G128 (`bonsai-t2-v1` / `bonsai-b1-v1`).

## Bonsai 2 packs (`quant_policy` `bonsai2-t2-v1` / `bonsai2-q4x-v1` / `bonsai2-t3-v1`, 2026-09-18)

PrismML's Ternary Bonsai 2 27B is Qwen3.8-27B with every projection ternary
(one fp16 scale per 128) in a Hadamard-rotated basis, and no MTP block
(`--mtp-safetensors` can append an external MTP head as an unrotated Q8
`blk.64`, setting block_count 65). The t2/q4x/t2+q4x containers need no
new dtype: the repack stores the ternary values EXACTLY in existing
containers (Q4_G64 nibble = trit+8 with the 128-group scale duplicated per
64; Q8_G128 int8 = trit for `token_embd`/`output`) with
`--bonsai2-container q4x`, or -- the default since Phase 3 -- in T2_G128
(`t2`, ~9 GB: the decode GEMVs and the prefill MMA GEMM both read it). The
`t2+q4x` layout is the Phase 2 one: T2 plus an exact-Q4 shadow named
`<name>.q4x` per `blk.*` rotated matrix, which the prefill GEMMs read only
under `Q27_T2_PF_SHADOW=1` (the bitwise A/B of the T2 prefill GEMM on a
served model). `t3` (2026-09-20) stores every `blk.*` matrix as T3_G128
above. `--slim` (t2/t3 only; repack refuses it with q4x) also stores
`token_embd`/`output` as exact T2_G128 and drops the `output_q4.weight`
copy -- the 12 GB-card t2 pack (~7.5 GB) and, with t3, the 8 GB-card pack.
Header meta additions:

- `"bonsai2": true`, `"bonsai2_container": "q4x" | "t2" | "t2+q4x" | "t3"`
  (each with a `-slim` suffix when `--slim` put `token_embd`/`output` in T2),
  `"qwen35.block_count": 64` (no `nextn_predict_layers`; 65 with
  `--mtp-safetensors`), `general.name`
  containing `qwen38` (the tool dialect and 3.8 template rules key on it).
- `"hadamard": {version 1, block_size 1024, transform
  "normalized-sylvester-walsh-hadamard", axis "input-last-dimension",
  sign_mode "explicit", sign_widths, sign_values, weight_names,
  inverse_weight_names ["token_embd.weight"], gdn_v_grouped}` -- the
  `prism.hadamard.*` GGUF keys verbatim.
- Sign vectors ALSO travel as F32 tensors `hadamard_signs.<width>` (5120,
  6144, 17408) so the engine loads them like any F32 tensor.

Runtime contract (engine.cuh `bonsai2`): before every rotated matmul the
activation is sign-flipped then transformed per contiguous 1024-block with
the normalized natural-order Walsh-Hadamard (`kernels.cuh hadamard1024`);
the embedding row gets the inverse after lookup (transform, then sign);
`ssm_out`'s input is permuted from the engine's tiled V-head order to the
grouped order first when `gdn_v_grouped` (`gdn_v_tiled_to_grouped`).
`ssm_alpha`/`ssm_beta` are unrotated and read the raw input.

T2_G128 on the CUDA device is stored in a dp4a-interleaved word order
(kernels.cuh `t2_interleave_device`, applied once at upload); the file and
the host copy stay in the sequential order above.

## Per-step read budget (decode)

Q4 bulk ~13.2 GB + Q8 lm_head ~1.3 GB + MTP layer ~0.4 GB + f16/f32 small tensors
=> ~14.8-15 GB per verify step. 5090 @ 1.79 TB/s => ~120 t/s ceiling before MTP amortization.
