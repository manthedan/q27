# KV-codec step 3 — 128-cell sensitivity census

Status: PRE-REGISTERED before measurement (results appended after the
overnight batch). Parent: 2026-07-16-paper-scan-triage.md §KV codec,
step 3, reordered before step 2 because step 1's attribution (the tail
is K-owned, spikes not bursts — 2026-07-16-kv-codec-step1.md) makes the
census the direct next question: WHICH K cells own the tail. Step 2's
scaling arms then target measured cells instead of guessing.

## Instrument

`--kl-kv-cell N` (N = attn_idx·8 + head·2 + side, side 0=K 1=V; attn_idx
0..15 → absolute layer attn_idx·4+3): the subject round-trips exactly one
(attention layer, KV head, side) cell through the turbo3 quantizer at
store time; everything else stays clean fp16. Gates run while awake:
self-check exactly 0 through the extended path; full-side arms
bit-identical to pre-change (K-arm 128-pos mean 0.00255188 reproduced
exactly — the head-mask extension does not perturb existing arms); two
cells smoke nonzero-small (L3:h0:K 2.1e-4, L51:h0:V 1.7e-4 at 128 pos).

## Run plan

`tools/kv_census_overnight.sh`: 128 cells × 2,048 wikitext2 positions
(2,049 tokens — the KL path encodes n−1; codex P2), ctx 2048, one
process per cell, resumable behind a run fingerprint (HEAD + binary +
artifact + corpus + route pins; codex P1 — a resume after any identity
change refuses rather than mixing experiments), rebuild-at-top + all
four route knobs pinned with overwrite (Q27_METAL_GEMM_HALF=1,
GQA_TILE=2, GQA_THRESHOLD=2048, GQA_BLOCK=1024 — the current default
route, same as step 1's calibration), self-check canary, fail-loud per
cell. ~5.3 h ⇒ overnight batch per the standing rule. Output:
logs/kv_census/census_summary.tsv (cell, layer, head, side, mean, p99,
p99.5, max, max_pos).

## Pre-registered reads

1. **Concentration:** rank cells by max-KL. If ≤ 8/128 cells carry ≥ 80%
   of the both-sides tail mass (sum of cell maxima vs the 2.94 event
   class), step 4's allocation has real targets; if damage is flat
   across cells, Block-GTQ-shape allocation buys little and step 4's
   simulation bar should be treated as unlikely to clear.
2. **Side split:** K cells should dominate the top ranks (step 1: K max
   2.52 vs V max 0.44). A V cell in the top-4 contradicts step 1 and
   flags an interaction effect worth a dedicated pair run.
3. **The pos-1000 event:** which single cell (if any) reproduces a max
   at position 1000. If one K cell alone spikes there at the 1+ nat
   scale, the tail is a single-head phenomenon (strongest possible
   allocation signal). If no single cell exceeds ~0.3 at pos 1000, the
   event is a multi-cell sum and per-cell allocation has a weaker lever.
4. **Additivity at cell scale (diagnostic):** Σ cell means vs the 8K
   full-side means (K 0.00677, V 0.00526 — note census runs at 2048
   positions, so compare against step 1's 0–2k bucket instead). Strong
   super-additivity flags interaction; mild sub-additivity expected.
5. **Depth shape:** mean-KL by layer. GDN layers dominate the stack
   (48/64); whether early or late attention layers are KV-fragile
   decides where step 4 spends bits per layer.

Layer-only or head-only marginals are recoverable by summation; no
separate arms needed.
