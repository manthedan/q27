# Mixed weight-tier census → mixed pack — pre-registration (2026-07-17, CENSUS + COMBO + SHIP GATES COMPLETE)

k3 roadmap item #2, adopted 2026-07-17 (Daniel: "sounds good, do it" —
queued behind the funded kernel rounds and the fp8-KV control arm). The
last artifact-level lever needing no kernel work: B1 (3.79 GB) carries
~1.05–1.16× PPL vs T2 (7.15 GB); if a small tensor set drives the gap, a
mixed pack (those tensors T2, bulk B1) plausibly lands ~4.3–5 GB at
~1.03–1.06× — a new serving point between the tiers.

## Method (amended from k3's sketch)

Census at tensor-CLASS × depth-band granularity FIRST, not per-tensor
(498 per-tensor arms is run-budget suicide; about 20 class arms of one 8K
NLL each is tractable, and mixed packs fit the mac-mini's 16 GB —
census arms are mini-parallelizable while the M4 holds kernel rounds).
Zoom to per-tensor only inside the classes the first pass indicts.

- Arm = one repacked artifact with a single class flipped B1→T2 (base:
  all-B1), measured on the standing 8K wikitext NLL protocol.
- Classes: attn {q, k/v, out} × depth bands {early, mid, late},
  ffn {gate, up, down} × bands, gdn {qkv, gate, out, alpha/beta},
  embeddings, output head, plus the zero-byte-delta shared-dtype cohort
  described below.
- Repack: `bonsai-mixed-v1` quant_policy — per-tensor dtype routing in
  repack.py; validate_architecture's matrix_dtype_ok admits {T2, B1}
  under the mixed policy with BOTH layout-meta code strings present.
  Engine dispatch is already per-tensor dtype (Phase 3) — no engine work.

## Pre-registered gates

- Census readout: a class is "gap-carrying" if flipping it recovers
  ≥ 15% of the B1→T2 NLL gap at ≤ 10% of the byte cost of full T2.
- Mixed-pack ship bands (carried from the tier ladder discipline):
  NLL mixed/T2 ≤ 1.06 at ≤ 5.0 GB → ship as a tier; 1.06–1.12 →
  conditional (report, Daniel decides); > 1.12 → the gap is diffuse,
  record and close (that is itself the answer: B1's 16% is not
  concentrated, and the two-tier ladder stands).
- Every shipped pack passes the suffix byte-identity battery and the
  behavioral probe set (4/4) before any serving claim.

## Pre-run implementation and method amendment (2026-07-17, before readout)

`tools/q27_mix.py` reassembles the two compatible containers byte-for-byte,
sets `bonsai-mixed-v1`, carries both layout declarations, and refuses
incompatible tensor inventories, non-Bonsai dtype mismatches, vacuous
selections, and source/output aliases. `validate_architecture()` admits
T2/B1 per tensor under only that policy; dispatch was already dtype-local.
A built mixed alpha/beta arm validated and generated `Paris.` on the mini;
a full `shared_f32` pack passed an 851-tensor blob-provenance hash check
(selected bytes = donor, every other byte = base); the full Metal suite is
green.

The source siblings share tensor names and shapes but are separately trained:
344 of their 353 same-dtype tensors differ in bytes. Their public model cards
call them companions derived from Qwen3.6-27B, but do not establish a shared
checkpoint, channel alignment, or independent quantization of one weight set.
This is therefore a **checkpoint-grafting search**, not a clean quantization-
sensitivity attribution. A positive arm is still actionable evidence that a
functional mixed checkpoint exists; a weak/negative arm can instead reflect
cross-checkpoint co-adaptation and cannot prove that the B1→T2 gap is diffuse.
Accordingly, the original >1.12 close band is amended before readout to “no
class-grafted candidate found; do not infer diffuseness.” The empirical ship
bands remain valid for any composed candidate that passes them end-to-end.

`--take` means **donor bytes even when dtype is unchanged**, and the first-pass
census adds one `shared_f32` arm covering output/block norms, GDN
conv/a/dt/norm tensors, and attention q/k norms. It costs zero bytes. Omitting
it would use the full B1→T2 gap as the recovery denominator while making that
part of the gap unreachable. If the cohort recovers ≥15%, the planned
zoom-by-class applies before composing a pack.

The sketch's “embeddings/head already tier-pinned — verify and exclude” is
corrected before readout: the actual artifacts carry B1 versus T2 copies,
each donor flip costs 158.9 MB (4.7% of the full 3,361.7 MB delta), and thus
each qualifies for the pre-registered ≤10% census gate. They run as separate
`embedding` and `output_head` arms rather than being silently unreachable.

`tools/mixed_census_batch.sh` runs the two same-machine baselines plus 25
serial class arms, one transient pack at a time. The sketch's combined FFN
`gate/up` classes cost 13.9–14.6% of the full tier delta and therefore could
never satisfy the registered ≤10% eligibility bar; they are split into
separate gate and up arms before readout. The driver self-caffeinates, retries
one transient failure, runs the engine under a clean environment with the
four numeric-route knobs explicitly pinned, and fingerprints HEAD, driver,
mixer, host binary, runtime Metal shader, both packs, tokenizer, corpus,
hashed machine identity, hardware model/GPU, macOS build, and all protocol
knobs before allowing a resume. Because the mixer, shader, and artifacts are
reopened by each child, that complete fingerprint is recomputed before and
after every measurement; a mid-run change aborts instead of splitting the
experiment. An atomic lifetime lock excludes a second driver and each run
uses a PID-unique transient pack. NLL output is written to a temporary log
and atomically published only after a zero exit, exactly one finite NLL, and
post-run global + exact-pack hash checks, so a crash cannot bless an unverified
completion. The 25-arm summary is validated and atomically published before
`COMPLETE` is printed. Results have not started.

## Execution registration (2026-07-17 midday, mini — recorded before the batch ran)

- B1 material: masters no longer on the mini — rebuilt from the pinned
  vendor GGUF (`prism-ml/Bonsai-27B-gguf` rev f10afb3,
  `Bonsai-27B-Q1_0.gguf`, sha256 17ef842e… verified against the LFS
  oid), repack.py lossless (498 byte-copies, RMSE 0.0000), engine
  validate + greedy smoke green. GGUF deleted after repack (13 GiB
  free box).
- Mixed packs assembled byte-level from the two q27 containers
  (`tools/q27_mix.py` — no quantization runs; policy/dtype/shape/name
  cross-validation, both tiers' layout meta merged, policy
  bonsai-mixed-v1). Engine admits {T2,B1} matrices under the mixed
  policy; embeddings/head and ssm alpha/beta accept either tier. This
  tests graft compatibility between trained companion checkpoints; it
  does not claim that class swaps isolate quantization error.
- Batch: `tools/mixed_census_batch.sh` — same-box baselines (all-B1,
  all-T2 — the recorded 1.055 ratio is a 24 GB M4 number, so the gap
  is re-anchored here), then 25 arms, one transient pack at a time
  (build → validate → 8K NLL → delete), fingerprinted for resume,
  caffeinated, serial. Arms: attn {q, k/v, out} × bands {early 0–20,
  mid 21–42, late 43–63}; ffn {gate, up, down} × bands; gdn {qkv, gate,
  out, alpha/beta} full-depth; embedding; output head; and the shared-
  dtype cohort. `gdn_out` fills the sketch's omitted `ssm_out` class;
  the final three arms are the pre-readout corrections above.
- Protocol: 8K wikitext NLL (`--nll-long 8192 --ctx 8192`), route pins
  GEMM_HALF=1 TILE=2 THRESHOLD=2048 BLOCK=1024, with a clean runtime,
  an exclusive lifetime lock, a run-unique pack path, and the complete
  file + hardware + OS/Metal identity rechecked around every arm.
- A premature launch was stopped during the incomplete B1 baseline and its
  directory quarantined: zsh changes `$0` to the function name inside
  `fingerprint()`, so the driver hash field was empty. The driver path is now
  captured before entering the function. No completed NLL/readout from that
  attempt is retained.

## RESULTS — first-pass census (2026-07-17 afternoon, mini)

Batch completed clean: 2 baselines + 25 arms, zero retries, zero aborts,
every log published through the atomic zero-exit/single-finite-NLL/hash
path, summary validated against all 25 arms before `COMPLETE`. Raw table:
`logs/mixed_census/census_summary.txt`.

Same-box anchor: base B1 2.6610, base T2 2.6081, gap 0.0529 nats
(ppl ratio 1.0543 — matches the recorded 24 GB M4 1.055 to 3 decimals).

```
gdn_qkv          NLL 2.6005  gap recovered 114.4%  bytes   +314.6 MB
gdn_alphabeta    NLL 2.6362  gap recovered  46.9%  bytes     +2.9 MB
attnq_mid        NLL 2.6517  gap recovered  17.6%  bytes    +39.3 MB
ffnup_mid        NLL 2.6541  gap recovered  13.0%  bytes   +245.1 MB
attnkv_late      NLL 2.6589  gap recovered   4.0%  bytes     +7.9 MB
attnout_mid      NLL 2.6623  gap recovered  -2.5%  bytes    +19.7 MB
shared_f32       NLL 2.6641  gap recovered  -5.9%  bytes     +0.0 MB
attnkv_early     NLL 2.6650  gap recovered  -7.6%  bytes     +6.6 MB
embedding        NLL 2.6678  gap recovered -12.9%  bytes   +158.9 MB
attnkv_mid       NLL 2.6686  gap recovered -14.4%  bytes     +6.6 MB
attnq_late       NLL 2.6747  gap recovered -25.9%  bytes    +47.2 MB
gdn_out          NLL 2.6757  gap recovered -27.8%  bytes   +188.7 MB
attnq_early      NLL 2.6778  gap recovered -31.8%  bytes    +39.3 MB
gdn_gate         NLL 2.6812  gap recovered -38.2%  bytes   +188.7 MB
ffngate_mid      NLL 2.6881  gap recovered -51.2%  bytes   +245.1 MB
ffngate_early    NLL 2.6895  gap recovered -53.9%  bytes   +234.0 MB
ffndown_mid      NLL 2.7051  gap recovered -83.4%  bytes   +245.1 MB
attnout_early    NLL 2.7163  gap recovered -104.5%  bytes   +19.7 MB
attnout_late     NLL 2.7201  gap recovered -111.7%  bytes   +23.6 MB
ffngate_late     NLL 2.7276  gap recovered -125.9%  bytes  +234.0 MB
output_head      NLL 2.7288  gap recovered -128.2%  bytes  +158.9 MB
ffnup_early      NLL 2.7299  gap recovered -130.2%  bytes  +234.0 MB
ffnup_late       NLL 2.7355  gap recovered -140.8%  bytes  +234.0 MB
ffndown_early    NLL 2.7400  gap recovered -149.3%  bytes  +234.0 MB
ffndown_late     NLL 2.8648  gap recovered -385.3%  bytes  +234.0 MB
```

### Pre-registered gate applied (≥15% recovery at ≤10% of the 3,361.7 MB delta)

Gap-carrying classes: **gdn_qkv** (114.4% at 9.4% of delta),
**gdn_alphabeta** (46.9% at 0.09%), **attnq_mid** (17.6% at 1.2%).
`ffnup_mid` (+13.0%) misses the 15% bar; nothing else is close. The
`shared_f32` zero-byte cohort lands at −5.9% and is NOT gap-carrying —
no zoom-by-class needed, and its |Δ| (~0.003 nats on an identical-cost
graft) doubles as the scale reference: single-arm readings inside about
±6% recovered should not be over-interpreted.

### Findings

1. The B1 quality deficit is overwhelmingly concentrated in the GDN/SSM
   block: `gdn_qkv` alone recovers 114% of the gap — the graft lands
   BELOW the pure-T2 baseline (2.6005 vs 2.6081) — and the 2.9 MB
   `gdn_alphabeta` decay parameters recover another 47% alone, the most
   byte-efficient result of the census by ~45×.
2. Because `gdn_qkv` alone beats all-T2, the remaining T2 tensors are
   net-negative GIVEN T2-tier SSM QKV. Single-class deltas are therefore
   NOT additive; composition must be measured, not summed. This is the
   expected signature of checkpoint grafting between separately trained
   companions (per the pre-readout amendment), not of quantization error.
3. Depth pattern in the transformer classes: early and late bands punish
   grafting across every module type; the only tolerant region is
   mid-depth (attn-Q +17.6%, ffn-up +13.0%). `ffndown_late` is the most
   graft-sensitive site in the network (−385%, +0.204 nats over base).
4. Per the amended close band: the 20 negative arms do NOT establish that
   the residual gap is diffuse — only that those single-class grafts fail.

## Combo phase — registration (2026-07-17 afternoon, recorded before running)

Additivity is broken (finding 2), so composed candidates are measured
directly by `tools/mixed_combo_batch.sh` — same fail-closed harness
(own lock, own `logs/mixed_combo` results dir, fresh same-box baselines,
fingerprinted resume, transient PID-unique packs), three registered arms
composed ONLY from gate-passing classes:

- `gdn_pair` = gdn_qkv + gdn_alphabeta (+317.5 MB → 4.11 GB pack)
- `gate_trio` = gdn_pair + attnq_mid (+356.8 MB → 4.15 GB pack)
- `cheap_pair` = gdn_alphabeta + attnq_mid (+42.3 MB → 3.83 GB pack)

(Sizes are decimal GB from the source-inventory byte deltas, cross-checked
by codex review; the driver's printed pack-size column in this first combo
run mixes GiB base with decimal-MB deltas and under-reports — the unit fix
is applied after the running batch completes, since editing the fingerprinted
driver mid-run would abort it. No band decision is affected: the worst-case
candidate is 4.15 GB against the 5.0 GB gate in either unit.)

### Combo RESULTS (2026-07-17 evening, mini — batch clean, zero retries)

Both baselines reproduced the census values exactly (B1 2.6610, T2 2.6081
— a full-protocol determinism check). Raw table:
`logs/mixed_combo/combo_summary.txt` (its pack-GB column carries the
unit bug above; corrected decimal sizes shown here):

```
gdn_pair   NLL 2.5885  gap recovered 137.1%  vs T2 0.9925  +317.5 MB  4.11 GB  SHIP-BAND
gate_trio  NLL 2.5982  gap recovered 118.7%  vs T2 0.9962  +356.8 MB  4.15 GB  SHIP-BAND
cheap_pair NLL 2.6355  gap recovered  48.2%  vs T2 1.0105  +42.3 MB   3.83 GB  SHIP-BAND
```

- **`gdn_pair` is the headline candidate**: 4.11 GB (57% of T2's bytes)
  and 0.0196 nats BETTER than all-T2 (ratio 0.9925). It also beats both
  of its own supersets tried so far — adding attnq_mid (`gate_trio`)
  costs 0.010 nats, and the all-T2 pack is another 0.010 behind that.
- Composition is consistently sub-additive and context-dependent:
  attnq_mid's solo +17.6% shrinks to ~+1 point on top of `gdn_alphabeta`
  (`cheap_pair` 48.2% vs 46.9%) and to negative on top of `gdn_pair`.
  The 2.9 MB alpha/beta graft does nearly all the work in `cheap_pair`.
- All three land in the pre-registered ship band; per the registration,
  a serving claim additionally requires the suffix byte-identity battery
  and the 4/4 behavioral probe set on the composed pack. That is the
  next gate for `gdn_pair` (and `cheap_pair` as the byte-cheap option).

## Ship gates run (2026-07-17 evening, mini — `logs/m1-ship-20260717/`)

Persistent `gdn_pair` pack was initially built under the provisional M1
name (4,110,049,792 bytes exactly as predicted, md5 in CHECKSUMS.md5): validate
OK, greedy smoke " Paris.", and an 8K NLL identity anchor under the census
route pins reproduced the combo measurement EXACTLY (2.5885) — the persistent
artifact is measurement-identical to the gated candidate.

Instrument fixes made before trusting the suffix battery (all in
`tools/suffix_burst_gates_2026-07-16.sh`, which also gained env-overridable
MODEL/OUT): (1) gate 3b's awk compared the literal words `accepted`/`live`
(fields 5/3) instead of their values (fields 6/4) — the leg was
unsatisfiable on ANY pack, so no historical battery can have passed it;
(2) a standing `rep3` incrementing-chapter arm was added because the
canonical repetition prompt legitimately all-accepts (periodic greedy
continuation) and cannot exercise rejection — rep3 forces live lane
rejections (observed `live 16 accepted 3`) with committed bytes identical;
(3) gate 4's "zero bursts on neutral" assertion is an economics prior, not
a correctness contract — T2's own greedy continuation of the neutral prompt
goes periodic on the mini and fires 3 bursts with bytes still identical —
so silence is now WARN and byte-identity remains the hard assertion.
Under the fixed instrument: the provisional `gdn_pair` 8/8 PASS, and the
parents re-run clean same-box (B1 8/8; T2 8/8 + the documented WARN).

**Behavioral probes: `gdn_pair` passes only 3/4 — ship gate FAIL (4/4 required).** The
constraints leg collapses at greedy into a thinking-mode repetition loop
("water bottle (2) -> " …) to the full 6144-token cap with empty final
content. Same-box, same-server, same-protocol controls: B1 PASS, T2 PASS,
alphabeta-only graft PASS, qkv-only graft PASS. The loop appears ONLY in
the qkv+alphabeta composition — an emergent cross-checkpoint co-adaptation
pathology of exactly the class the pre-readout amendment warned about, and
one wikitext NLL cannot see (the same pack beats all-T2 on NLL while
failing an agentic-collapse probe). json/codeedit/toolcall legs PASS.
Registered rescue residue (not run): band-restricted partial qkv grafts;
vendor-B.1 sampling variant (temp 0.7/top-p 0.95/top-k 20 — NOT the
registered greedy protocol, so it can inform but not substitute); per-layer
loop localization.

**`cheap_pair` passes the COMPLETE ship gate — first mixed pack to do so.**
Rebuilt pack (md5 91db7fdd368ba3558e59bb5e111bbd07): 8K NLL identity anchor
2.6355 EXACT; suffix battery 8/8 PASS; probes 4/4 PASS (json exact
keys/types; constraints all honored, `finish stop`; codeedit boundary fix
exact; toolcall get_weather city=Taipei with schema-valid optional
unit="c"). Ship band: 1.0105 vs T2 at 3.83 GB. Serving point: ~48% of the
B1→T2 gap closed for +42.3 MB over B1 (+1.1% bytes).

**Promotion decision (2026-07-17): `cheap_pair` is the M1 serving tier.**
The gated bytes now live at
`models/bonsai-27b-m1/bonsai-27b-m1.q27`; the sibling `CHECKSUMS.md5`
records md5 91db7fdd368ba3558e59bb5e111bbd07. M1 means the B1 base with T2
`ssm_alpha`/`ssm_beta` and attention-Q in blocks 21–42. The NLL-stronger
but probe-failing `gdn_pair` is retained explicitly as a non-serving
experiment at
`models/bonsai-27b-gdn-pair-experimental/bonsai-27b-gdn-pair-experimental.q27`
(md5 107647e9cba0f01a003934011644c2fe); its rescue investigation is parked
until separately funded. This naming keeps the serving path fail-closed:
only the artifact that passed every registered gate carries the M1 tier
name.

Ship bands are unchanged from the pre-registration: NLL mixed/T2 ≤ 1.06
at ≤ 5.0 GB → ship candidate (then suffix byte-identity battery +
behavioral probes before any serving claim); 1.06–1.12 conditional;
worse → record. All three candidates are inside the size band by
construction; `gdn_pair` and `gate_trio` test whether the all-T2
baseline can be beaten outright at ~57% of its bytes, `cheap_pair`
whether ~half the gap closes for ~1% extra bytes.
