# qwen35 forward-pass spec (ground truth for q27)

Extracted from mainline llama.cpp build 1491 by direct source read:
`src/models/qwen35.cpp` (graph: L129-228, attn: L257-336, deltanet: L338-470, ffn: L472-485, MTP: L488-644),
`src/models/delta-net-base.cpp` (AR recurrence: L289-371, conv state: L449+).
An earlier agent-generated summary had three material errors (K-head count, QK dims, attention scale);
everything below is from the source, not the summary.

Bonsai 2 27B (PrismML's ternary Qwen3.8-27B, 2026-09-18) runs this same
forward pass with two differences: no MTP layer (blocks 0..63 only, unless a
pack appends an external MTP head as an unrotated `blk.64`), and every
projection except `ssm_alpha`/`ssm_beta` is stored in a Hadamard-rotated
basis, so the engine sign-flips and Walsh-Hadamard-transforms each matmul
input per 1024-block and inverse-rotates the embedding row. Its weights are
T2_G128 or T3_G128 instead of Q4/Q8. The runtime contract is in FORMAT.md
("Bonsai 2 packs").

## Dimensions

| | |
|---|---|
| hidden (n_embd) | 5120 |
| layers | 0..63 main (48 DeltaNet + 16 attention at il%4==3), 64 = MTP |
| FFN | 17408, SwiGLU |
| full attention | 24 Q heads x 256, 4 KV heads x 256 (GQA 6:1) |
| DeltaNet | S_k=128, H_k=16 (q,k); S_v=128, H_v=48 (v); d_inner=6144 |
| DeltaNet state | per layer per seq: S[128,128,48] f32 (~3.1 MB); 48 layers ~151 MB |
| conv | kernel 4, channels 10240 (all of qkv), state = last 3 qkv columns/layer |
| rope | IMROPE, n_rot=64 of 256, sections [11,11,10,0], freq_base 1e7; text-only (all pos components equal) == neox partial rope, pair p freq = 1e7^(-2p/64) |
| rms eps | 1e-6 |
| vocab | 248320; embeddings/lm_head untied; no logit softcap; no embd scale |

## Block wiring (same for both types)

```
x1 = rms_norm(x, attn_norm) -> mixer (deltanet | attention) -> y
x2 = x + y                               # residual 1
x3 = rms_norm(x2, post_attention_norm) -> FFN -> z
out = x2 + z                             # residual 2 (from BEFORE post norm)
```

FFN: `down( silu(gate(x)) * up(x) )`.

## Gated DeltaNet block (48 layers)

1. `qkv = attn_qkv(x1)` -> [10240] layout **[q 16x128 | k 16x128 | v 48x128]**
2. `z = attn_gate(x1)` -> [6144] (this is the output gate, viewed [128,48])
3. `beta = sigmoid(ssm_beta(x1))` -> [48] per-head write strength
4. `g = ssm_a * softplus(ssm_alpha(x1) + ssm_dt_bias)` -> [48]; ssm_a stores -exp(A_log) so g<0; decay=exp(g) in (0,1)
5. conv: append qkv column to per-layer ring of last 3 columns; per channel c:
   `conv_out[c] = sum_{j=0..3} window[j][c] * conv1d[j][c]` (no bias); then **silu**
6. split conv_out -> q[128,16], k[128,16], v[128,48]
7. **L2-normalize** q and k per head (128-dim, eps 1e-6)
8. expand q,k from 16 -> 48 heads by **tile/modulo** (ggml_repeat: v-head h uses qk-head h%16) [VERIFY-1]
9. `q *= 1/sqrt(128)`
10. per v-head recurrence (state S[128,128]; ne0=k-index i, ne1=v-index j):
    ```
    S      = S * exp(g)                    # scalar decay per head
    pred_j = sum_i k[i] * S[i,j]           # k^T S
    d_j    = beta * (v[j] - pred_j)
    S[i,j] += k[i] * d_j                   # outer product update
    o_j    = sum_i q[i] * S[i,j]           # q^T S
    ```
11. `o = rms_norm(o, ssm_norm[128]) * silu(z)` per head (gated norm)
12. `y = ssm_out(o.flatten[6144])` -> [5120]

## Full attention block (layers 3,7,...,63 and MTP)

1. `qg = attn_q(x1)` -> [12288], **interleaved per head: [q0 256 | gate0 256 | q1 | gate1 | ...]**
2. `q = rms_norm(q, attn_q_norm[256])` per head, **before rope**
3. `k = attn_k(x1)` -> [1024] = 4x256; `k = rms_norm(k, attn_k_norm[256])` per head
4. `v = attn_v(x1)` -> [1024]; no norm
5. rope(q), rope(k): neox partial 64 dims (text-only path), pos = token position
6. attention: causal softmax, scale `1/sqrt(256)`, GQA 6:1 -> out [6144] (24 heads x 256)
7. `out *= sigmoid(gate)` elementwise (gate from step 1, per-head slots)
8. `y = attn_output(out)` -> [5120]

## Main sequence

```
h = embed[token]                       # row lookup, no scale
h = 65 blocks... (0..63)
h_final = rms_norm(h, output_norm)     # ALSO the "h_nextn" fed to MTP
logits = output(h_final)               # [248320]
```

## MTP layer (blk.64) — draft head

Input: token embedding of the NEXT (accepted/drafted) token + **h_nextn = post-output_norm hidden**
of the current position from the main pass.

```
e = rms_norm(embed[tok], nextn.enorm)
hn = rms_norm(h_nextn, nextn.hnorm)
c = concat(e, hn)                      # dim 0: embedding FIRST -> [10240]
x = nextn.eh_proj(c)                   # -> [5120]
x = <full attention block + FFN as above, blk.64 weights, OWN KV cache, same rope/pos>
x = rms_norm(x, nextn.shared_head_norm)   # (fallback output_norm; our GGUF has shared_head_norm)
logits = output(x)                     # our GGUF lacks shared_head_head/embed_tokens -> use main embed/output
```

MTP drafting loop (self-speculation, depth n): draft token t+1 from h_nextn(t); to draft t+2,
run MTP again feeding its own hidden state (llama.cpp feeds MTP output recursively) [VERIFY-2:
check llama.cpp speculative.cpp recursion before implementing multi-depth].

## KV/state budget (decode, f16 KV)

- 16 attention layers + 1 MTP layer: 17 x 4 heads x 256 x 2(K,V) x 2B = 68 KB/token
- 32K ctx: 2.2 GB f16, 1.1 GB fp8. The engine default is fp16 (`Q27_KV`
  unset); the server's default profile sets `Q27_KV=fp8` on sm_89+ and
  `turbo5k` on Ampere (`server.cu`), and `--kv-fp16` / `Q27_PROFILE=ref` keep fp16
- DeltaNet: 151 MB state + 48 x 3 x 10240 x 4B = 5.9 MB conv rings (constant, no growth)

## Verification flags (resolve during M1 validation)

- [VERIFY-1] ggml_repeat head expansion: tile (h%16) vs interleave (h/3). Assumed tile.
- [VERIFY-2] MTP multi-step drafting recursion: what h feeds draft step 2.
- [VERIFY-3] imrope==neox for equal position components (text-only).
- [VERIFY-4] ggml_ssm_conv window orientation (oldest-first assumed).
- Validation method: llama-eval-callback dumps per-layer activations on a fixed prompt
  (cb() names in source: "attn_norm", "linear_attn_out", "attn_gated", "ffn_out", ...).
