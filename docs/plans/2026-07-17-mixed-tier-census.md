# Mixed weight-tier census → mixed pack — pre-registration (2026-07-17, QUEUED not started)

k3 roadmap item #2, adopted 2026-07-17 (operator: "sounds good, do it" —
queued behind the funded kernel rounds and the fp8-KV control arm). The
last artifact-level lever needing no kernel work: B1 (3.79 GB) carries
~1.05–1.16× PPL vs T2 (7.15 GB); if a small tensor set drives the gap, a
mixed pack (those tensors T2, bulk B1) plausibly lands ~4.3–5 GB at
~1.03–1.06× — a new serving point between the tiers.

## Method (amended from k3's sketch)

Census at tensor-CLASS × depth-band granularity FIRST, not per-tensor
(498 per-tensor arms is run-budget suicide; ~15–20 class arms of one 8K
NLL each is 2–3 machine-days, and mixed packs fit the mac-mini's 16 GB —
census arms are mini-parallelizable while the M4 holds kernel rounds).
Zoom to per-tensor only inside the classes the first pass indicts.

- Arm = one repacked artifact with a single class flipped B1→T2 (base:
  all-B1), measured on the standing 8K wikitext NLL protocol.
- Classes: attn {q, k/v, out} × depth bands {early, mid, late},
  ffn {gate/up, down} × bands, gdn {qkv, gate, alpha/beta}, embeddings/
  head (already tier-pinned — verify and exclude).
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

## RESULTS

(pending)
