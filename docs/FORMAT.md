# q27 weight format (version 1)

Offline-repacked weights for the q27 engine. Produced by `tools/repack.py` from the BF16 GGUF. Designed for mmap + single cudaMemcpy per tensor, and coalesced 128-byte warp reads in the fused-dequant GEMV.

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
  dtype      u8             0=F32  1=F16  2=Q8_G128  3=Q4_G64  4=T2_G128
  n_dims     u8
  shape      u64[n_dims]    numpy row-major shape (outer first; innermost/contiguous LAST)
  data_off   u64            relative to data section start, 256-byte aligned
  data_size  u64
  scale_off  u64            0 if dtype has no scales
  scale_size u64
[data section]  256-byte aligned blobs
```

## Quantized types

Both types quantize along the **contiguous (innermost) axis**, which is the GEMV
reduction axis for every matmul weight in this model.

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

### T2_G128 (ternary tier, `quant_policy: bonsai-t2-v1`)
- ternary codes, group size 128: element `i` of a row -> byte `i/4`, 2-bit field at
  bit offset `(i%4)*2` (**sequential, LSB-first** — no nibble split); code
  `c ∈ {0,1,2}` decodes to `(c-1) * scale`, i.e. `{-scale, 0, +scale}`. Code 3 is
  **forbidden** (the source format decodes it to +2; repack.py hard-fails if present).
- scales: fp16, `[rows, cols/128]`, separate contiguous blob (as Q8_G128)
- effective 2.25 bpw in-container (2 bits + fp16/128); a warp reading 128 B gets
  512 consecutive weights = exactly 4 groups
- produced losslessly (byte-copy of code bytes) from the PrismML fork's ggml type 42
  ("Q2_0", 34-byte `{fp16 d; u8 qs[32]}` blocks) — same within-byte order and decode
  formula; see docs/plans/2026-07-14-ternary-tier.md for the authoritative source
  reading. Ternary packs quantize `token_embd`, `output`, `ssm_alpha`, `ssm_beta`
  ternary too (unlike the official-tier policy); norms/`ssm_a`/`ssm_dt.bias`/
  `ssm_conv1d` stay F32. No MTP layer (`blk.64.*`) and no `output_q4.weight` alias —
  the loader must tolerate their absence (greedy + suffix drafting only).

### T3_G128 (experimental, parked — no artifacts produced)
- base-3 recode of T2: 5 codes per byte (`c0 + 3c1 + 9c2 + 27c3 + 81c4`,
  byte ∈ [0,242]), 26 bytes per 128-column group (byte 25 carries columns
  125–127; slots 3–4 must be code 1), scales as T2. 1.75 bpw effective.
- dtype id 5 is reserved and the Metal decode GEMV exists and is gated, but
  the Phase-0 bench killed the format on M4: decode-bound at ~50 GB/s vs
  T2's ~95 (see docs/plans/2026-07-15-t3-packing.md). `tools/repack.py`
  never emits it.

## Quant policy (v1)

| tensors | dtype | why |
|---|---|---|
| all `*_norm.weight`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, `output_norm` | F32 | tiny, numerically sensitive |
| `ssm_alpha.weight`, `ssm_beta.weight` | F16 | 48-wide heads, awkward group size, tiny anyway |
| `token_embd.weight`, `output.weight` | Q8_G128 | vocab quality; embed is row-lookup (not GEMV-read) |
| everything in `blk.64.*` (MTP layer) | Q8_G128 (matmuls) / F32 (norms) | draft/verify agreement must survive quantization or MTP acceptance craters |
| all other matmul weights (blk.0-63) | Q4_G64 | the ~14 GB bulk |

## Per-step read budget (decode)

Q4 bulk ~13.2 GB + Q8 lm_head ~1.3 GB + MTP layer ~0.4 GB + f16/f32 small tensors
=> ~14.8-15 GB per verify step. 5090 @ 1.79 TB/s => ~120 t/s ceiling before MTP amortization.
