# KV exception cells in production: fp16 L7 h1 on the turbo3 engine

Status: PRE-REGISTERED before implementation (2026-07-17 ~02:30, mini).
Daniel approved the design step ("sure, try it"). Parent:
2026-07-17-kv-codec-step4-probe.md §4b — the L7 h1 pair is jointly
necessary (K alone: nothing; V alone: amplifies 1.35×; pair: full
2.20× tail cut) at +6.4% KV bytes. This lands the fp16-pair option:
it is the exact configuration the instrument measured, so it carries
no extrapolation risk; a mid-rate 2× code (~+3.2%) stays a follow-up
only if +6.4% ever hurts.

## Design: overwrite dispatch, zero attention-kernel changes

The turbo3 cache is token-major with heads interleaved
(`(p*kv_heads+kvh)*100`), so a head-subset offset dispatch cannot read
it — but the existing **f16 attention kernels index purely relatively**
(q + qh*q_stride; cache (p*kv_heads+kvh)*head_dim; out/partials by qh).
Therefore:

- **Side cache** per masked attention layer: private fp16 K/V buffers
  in kv_heads=1 layout (`p*256` halves), max_ctx rows, charged to
  `kv_reserved_bytes` (and thus G6) like any KV bytes.
- **Store**: the turbo3 store runs UNCHANGED (all 4 heads keep valid
  turbo3 blocks); additionally the masked head's K/V rows are copied
  fp16 into the side cache (serial: existing `kv_store_f16` with
  offset bases; chunked: a small additive kernel with a buf-stride arg
  if the rows variant cannot express stride 1024→256 — new kernel, no
  ABI bump; modifying existing bindings would require one).
- **Attention**: the production dispatch runs UNCHANGED over all 24
  query heads (the masked head's group computes valid-but-quantized
  output); a second dispatch of the **f16 kernel family** runs over
  the masked head's 6-query-head window (q/out/partials base offsets,
  kv_heads=1, q_heads=6) against the side cache and overwrites those 6
  output rows with fp16-exact attention. Metal buffer hazard tracking
  serializes the overlapping `out` writes. Cost: double attention for
  6/24 query heads on 1/16 attention layers + one row-copy — noise.
- **Interface**: `Q27_METAL_KV_FP16_CELLS` env (census cell numbers,
  e.g. "10,11"); unset (default) = zero new dispatches, byte-identical
  to today. Engine-wide; only meaningful on turbo3-KV engines (fp16
  engines are already fp16 — rejected loudly).
- **v1 exclusions (loud, recorded)**: save_state/load_state refuse
  when the mask is active (side cache not serialized yet); MTP KV
  never excepted (not a census cell; matches the instrument).

## Pre-registered gates (in order; each is a kill line)

1. **Default-off byte identity**: env unset → A/B greedy decode +
   chunked/serial NLL byte-identical pre/post commit.
2. **Unit parity**: mixed-path test in test_metal_ops — one layer,
   synthetic cache, masked head's output must match a CPU fp16
   attention reference while unmasked heads match the turbo3 path
   (both prefill-causal and decode shapes, above and below the GQA
   threshold).
3. **End-to-end graduation gate (the point of the feature)**:
   `--kl-kv` with the subject = turbo3 engine + `Q27_METAL_KV_FP16_CELLS=10,11`,
   8,191 positions, pinned route: expect the 4b l7h1 class — mean
   ≈0.0111, max ≈1.12 @7719 (vs 0.0119 / 2.478 @1000 unmasked).
   Accept band: max ≤ 1.30 (well under control's 2.478, allowing the
   attrib-vs-production route envelope), mean within ±5% of 0.01115.
   MISS = STOP and diagnose; do not ship on a partial cut.
4. **Wall neutrality**: 8K NLL wall within +2% of unmasked (the
   overwrite dispatch is 1 layer of 16 and 6 heads of 24).
5. Suite green (`make test-metal`), multislot G1–G7 (side cache in
   admission accounting).

Codex review after each landing commit, per standing practice.

## Gate results (2026-07-17 ~03:00)

1. Default-off byte identity: PASS (env-unset kl-kv smoke reproduces the
   pre-change value to every digit; make test-metal green).
2. Unit parity: PASS (new test_attention_fp16_window — side store exact
   at the bit level; decode + causal windows overwrite exactly the
   masked head's rows against a poisoned side cache, zero leakage).
   Bonus machinery probe: ALL 128 cells excepted collapses the KL to
   the envelope floor (mean 2.2e-4, max 0.0066, no pos-1000 spike at
   1,536 positions) — the window path is correct end to end.
3. **8K production graduation gate: FAIL on max — the kill line fires.**
   Cells 10,11: mean 0.011309 (IN band [0.0106, 0.0117]) but max
   2.265 @pos 1000 (band ≤ 1.30; the instrument's l7h1 arm read 1.124
   @7719). The pos-1000 event survives protection in production.

**Diagnosis (probe-supported): instrument→production transfer fails at
the knife-edge tail, not in the machinery.** The all-cells probe rules
out side-cache/window fidelity (event fully vanishes when everything
is protected), so the residual event is composed of the OTHER cells'
quantization as computed by the production turbo3 kernels. Attrib
semantics (round-trip + fp16 original-domain attention) and production
semantics (turbo3 encode + WHT-domain dequant attention) reconstruct
the same VALUES but sum differently, and at a near-tie position the
event's per-cell composition shifts: L7 h1's share is most of the
attrib event (2.478 → 1.124) but a minority of the production event
(2.937 → 2.265). The MEAN transfers (in band); the tail selection does
not. Program lesson: exception lists must be SELECTED under production
semantics — which gate 3's own tooling now makes cheap (the event is
at pos 1000, so 1,536-position production arms cost ~2 min).

## Production-domain re-selection (pre-registered before the sweep ran)

`tools/kv_except_layer_sweep.sh`: control + 16 layer-level arms (all 4
head-pairs of one attention layer protected, cells 8i..8i+7) at 1,536
positions. Reads: (1) which layers cut the production pos-1000 max and
by how much; (2) whether any single layer (or a small union) reaches
the ≤1.30-class kill of the event; (3) layer arms whose max argmax
moves off 1000 identify the event's production owners. Then head-pair
arms within the hot layer(s). Kill line: if no ≤4-pair union kills the
production event, targeted precision is PARKED for the tail (the
production event is irreducibly multi-cell) and the fp16-exception
machinery ships, if at all, as a mean lever only — priced against the
+12.9% uniform null.

### Sweep results (2026-07-17 03:10–03:45, 1,536 positions, control max 2.772 @1000)

Layer arms (all 4 head-pairs of one layer): **L7 kills the event —
max 0.672 @741** (the residual is the 641/741-class event, pos-1000
gone); L3 nearly does (0.879 @641); L19/L35 halve it (1.44 @641); the
other 12 layers do nothing (~2.7, incl. two slightly above control —
envelope at a near-tie). The production event is owned by early-plus-
sparse-mid attention layers, with L7 dominant.

Within-L7 decomposition: singles h0 2.160 / h1 2.265 / h2 2.608 /
h3 2.729 — none kills; **the h0+h1 union AMPLIFIES relative to its own
singles (2.347)**, the same anti-additive knife-edge behavior 4b saw
with l7h1v. Only the full layer (all four pairs) kills the event.
**Production exception list = L7 h0–h3, 8 cells, +25.8% of turbo3 KV
bytes** — exactly at the pre-registered ≤4-pair boundary, so it
formally survives, but at double the graduated price; the uniform null
(spend +25.8% on a richer global block code) is correspondingly
stronger and undesigned. Formal 8K gate-3 rerun band (pre-stated
before the run): max ≤ 1.30, mean ≤ control.

### 8K production gate, L7-full list: PASS (2026-07-17 04:00)

Cells 8–15, 8,191 positions: **max 0.7546 @2485 (band ≤1.30 — the
pos-1000 event is gone; the residual is a new mid-context event of the
sub-bar class), mean 0.0104 (−11% vs the 0.0116 both-sides class),
p99 0.0835 (−6%). Wall 639.4 s vs 634.7 s for the 2-cell arm (+0.7% —
gate 4 wall-neutrality holds; four window dispatches on one layer are
noise.** All five pre-registered gates now dispositioned: 1 PASS,
2 PASS, 3 FAIL-then-PASS under the production-selected list, 4 PASS,
5 (multislot) runs env-unset and is unaffected — the env knob is
default-off and byte-identical.

## Final disposition (handed to Daniel, not decided unilaterally)

The machinery ships default-off and gated; the SELECTION question has
three priced options:
1. **Fund L7-full** (`Q27_METAL_KV_FP16_CELLS=8,9,10,11,12,13,14,15`):
   +25.8% turbo3 KV bytes buys max 2.94→0.755 (3.9×), mean −11%,
   wall-neutral, snapshots refused (v1). At ctx 8192 this is ~13 MB
   per engine on this artifact — memory is cheap on the mini, and
   this is the only measured configuration that kills the event.
2. **Design the mid-rate code** for L7 (~2× rate on 8 cells ≈ +12.9%
   total): halves the price IF a 2× code preserves the kill —
   unmeasured, and 4b/sweep anti-additivity warns that partial
   fidelity can amplify; needs its own arm before any format work.
3. **The uniform null**: +25.8% spent on a richer global block code —
   also unmeasured, a larger design task, but principled.
Program note: the attrib instrument remains valid for MEANS (it
transferred within band twice) but NOT for knife-edge tail selection;
production arms at 1,536 positions (~2 min) are the standard tail
instrument from here.
