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

## Run log (appended during the batch)

- 2026-07-16 ~18:25: run 1 ABORTED at cell 32 (L19:h0:K) — GPU
  command-buffer page fault at ~pos 1920
  (kIOGPUCommandBufferCallbackErrorPageFault). Cells 0–31 completed
  clean. Driver gained a retry-once-per-cell guard (a transient fault
  must not kill an unattended batch; two failures still abort loud).
- Identity migration at resume: the census launched from pre-rewrite
  HEAD 9d22394; the same-day history rewrite + hard reset moved HEAD to
  dc43c52 (tree additionally carries the step-2 instrument staging,
  unexercised at flags=0) and refreshed mtimes, so rebuild-at-top
  relinked the binary. Fingerprint refused the resume — correctly.
  Migration evidence: cell 0 re-run under the dc43c52 binary is
  DIGIT-IDENTICAL to cell_000.log (mean 0.000234749, max 0.00799207,
  all quantile/run-structure/bucket lines). Fingerprint rewritten to the
  new identity; full record in logs/kv_census/IDENTITY_MIGRATION.md.
- Resume relaunched 18:33 from cell 32 (~3.9 h remaining).
- Cell 32 verdict: clean on the resumed run's first attempt (mean
  0.000218842, max 0.0129 — unremarkable). The page fault was a
  transient, not an attn_idx>=4 instrument bug.
- 2026-07-16 22:30: run complete, 128/128 cells, no further failures.
  Summary: `logs/kv_census/census_summary.tsv` (per-cell logs alongside).

## Results (2026-07-16, pre-registered reads in order)

128 cells × 2,048 positions, ~2.4 min/cell, ctx 2048, pinned route
(gemm_half=1, tile=2, thr=2048, blk=1024). Each cell quantizes exactly
one (attn layer, KV head, K|V) to turbo3 against a full-fp16 baseline.

1. **Concentration: one head owns the extreme tail; the pooled 80% bar
   fails.** Top-8 cells carry 65.8% of Σ(cell maxima) and 80% needs 29
   cells — the bar as registered does not clear. But the structure is
   sharper than the bar anticipated: **L7 h1 is the only (layer, head)
   in the ≥0.3-nat class, and it is there twice** — its V cell (max
   1.204) and K cell (max 1.100), both @pos 641, together 54.6% of
   Σmax; no other cell exceeds 0.163. Verdict: step-4 allocation has
   exactly one glaring tail target (L7 h1, both sides), on top of a
   flat ~0.01–0.09 background. The pooled criterion failed because 126
   small maxima dilute the sum, not because damage is flat.
2. **Side split: V tops the ranking — step 1's K-dominance premise does
   NOT carry to single-cell sensitivities at 2K.** Top-4 sides are
   V,K,V,K; the single worst cell is a V cell (L7 h1 V, 1.204). Σ cell
   means split K 48.9% / V 51.1% — near-even, consistent with step 1's
   own note that V leads shallow contexts (this census is 0–2k only;
   K's 58% mean share was an 8K figure). Per the pre-registration this
   flags an interaction effect: a dedicated L7 h1 pair run (K+V jointly
   vs each alone) is the follow-up if step 4 wants to lean on this head.
3. **The pos-1000 event has no single-cell owner.** Largest single-cell
   KL near pos 1000 is 0.0285 (L27 h1 K) — two orders below the
   both-sides 2.94 event; nothing approaches the 0.3 screen. The step-1
   tail event is a multi-cell sum, and per-cell allocation has the
   weaker lever the pre-registration anticipated. Note the single-cell
   damage concentrates at pos 641 instead (5 of the top-5 cells max
   there or at 640) — position 641 is the globally KV-fragile position
   of this corpus window at 2K, not position 1000.
4. **Strong sub-additivity: Σ(128 cell means) = 0.0284 vs both-sides
   0–2k ≈ 0.0123 — parts sum to 2.32× the joint damage.** (Anchor
   caveat: 0.0122604 is the GEMM_HALF=0 calibration bucket; step 1's
   current-route per-side 0–2k buckets weren't preserved — same-route
   both-sides 8K overall was 0.01165, so the anchor is good to a few
   percent, far below the 2.3× signal.) Lone-cell perturbations
   overstate joint damage: a single quantized cell among fp16 peers is
   atypical, and jointly-quantized errors partially cancel. **Step-4
   consequence: allocation simulations must be validated jointly;
   summing per-cell sensitivities will overestimate savings ~2×.**
5. **Depth shape: means are flat-to-declining with depth on both sides,
   with two exceptions** — the L7 h1 fragile head (K and V cell means
   3–4× the background) and **a last-layer V bump: L63 V-mean 0.000606
   vs ~0.00014–0.00015 for L35–L59, driven by L63 h3 V (mean 0.00113 —
   the highest mean of all 128 cells) and L63 h2 V (0.000777)**. Early
   attention (L3–L11) is tail-fragile; the final attention layer is
   mean-fragile on V (closest to the output head, damage has no
   downstream layer to wash out). Step-4 bit allocation: protect L7 h1
   (both sides) for the tail, L63 V (h2/h3) for the mean; everything
   else is background.

**Net for step 4:** targets exist but are few and specific (one tail
head + one mean layer-side); the pooled-concentration bar failing means
Block-GTQ-shape *uniform-extra-bits-per-block* buys little — the win, if
any, is a tiny exception list (protect ~4 cells at higher precision,
~3% of KV bytes at 2K). Sub-additivity (read 4) says simulate jointly
before believing any projected win. Step 2's K-scaling arms are
unaffected (full-side treatment, same instrument as step 1) and run next.
