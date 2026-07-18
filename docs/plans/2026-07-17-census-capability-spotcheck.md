# Census capability spot-check — pre-registration (2026-07-17)

Gate amendment to the mixed-tier census (2026-07-17-mixed-tier-census.md),
adopted from the roadmap review the operator ran 2026-07-17 evening
("go for it"): the census's headline — gdn_qkv alone recovers 114.4% of
the B1→T2 NLL gap — is a single-arm, single-metric (8K wikitext NLL)
number, and the e4m3 round just demonstrated how knife-edged
partial-fidelity effects are. NLL recovery does not automatically mean
capability recovery. **No mixed-pack ship claim goes out on NLL alone:**
this spot-check must run on the census's decision arms first. U2's own
triage line applies — question sets only when a decision needs them; the
mixed-pack ship claim is that decision.

## Method

Weak-vs-strong output agreement (tools/eval/agreement.py, ds4's
sufficient first gate) on a fixed 120-item prompt set
(`tools/eval/prompts/{choice,numeric,freeform}.jsonl`: 60 anchored
multiple choice across logic/reasoning/math/science/code, 40 numeric
word problems, 20 freeform short answers; every item's gold round-trips
through the real extractors — `tools/eval/promptlint.py`, which proves
failure on duplicate ids, missing anchors, and unextractable golds).
Greedy decoding (gen_runner sends no temperature; the server's
missing-temperature default is 0.0), max_tokens 1024, serial requests
per COORDINATION.md via `tools/eval/run_arm.sh <arm> <base-url>`.

Arms (all served from the mini's packs, one resident model at a time):

- `t2-base` — pure T2 pack: the REFERENCE (the tier whose quality the
  mixed pack claims to match; its outputs define agreement).
- `b1-base` — pure B1 pack: the FLOOR.
- `gdn-qkv` — the census's 114.4% single arm (informational: does the
  NLL overshoot transfer? no ship weight on its own).
- `combo` — the combination arm the mini is measuring (B1 +
  gdn_qkv + gdn_alphabeta ± attnq_mid, whichever the mini's NLL run
  selects): the SHIP-DECIDING arm.

Decision metric, mirroring the census's own currency
(`tools/eval/spotcheck_verdict.py`):

    capability gap recovered = (A(combo) − A(b1-base)) / (1 − A(b1-base))

where A(x) = x's pair-weighted overall agreement rate with `t2-base`
across the three modes; unextractable output counts as disagreement
(a capability signal, not missing data).

## Pre-registered bands

- **≥ 50% of the capability gap recovered → ship-supporting.** The
  NLL claim survives its first capability cross-check; the mixed-pack
  writeup may cite both numbers together.
- **0–50% → CONDITIONAL.** The pack ships (if NLL holds) but the
  capability transfer is named as partial in the writeup — no
  "T2-quality" language, "recovers X% of the capability gap" only.
- **< 0 (combo agrees less than pure B1) → the mirage is caught.**
  No ship claim; the census's NLL metric is recorded as
  non-transferring for this graft and the combination arm goes back
  for per-mode diagnosis.

Honesty notes, registered before any run: (1) agreement-with-T2 is not
correctness — a shared wrong answer agrees; that is the right metric
for a "matches T2" claim and the golds are still recorded for a
secondary accuracy read. (2) 120 items gives ±~4.5 points of binomial
noise at rate 0.5 (1σ); the 50% band is deliberately coarse — this is
a spot-check, not a benchmark. (3) The mini's box serves all four
arms with the same binary and protocol pins fingerprinted in the
census harness; arms are not comparable across differing binaries.

## RESULTS

(pending — runs on the mini after its census publication + combination
arm; ~1 h/arm at mini decode rates, 4 arms)
