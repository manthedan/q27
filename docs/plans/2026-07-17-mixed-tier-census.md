# Mixed weight-tier census → mixed pack — pre-registration (2026-07-17, INSTRUMENT READY; measurement pending)

k3 roadmap item #2, adopted 2026-07-17 (operator: "sounds good, do it" —
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
  conditional (report, the operator decides); > 1.12 → the gap is diffuse,
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

## RESULTS

(pending)
