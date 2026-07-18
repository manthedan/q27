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
per COORDINATION.md via
`tools/eval/run_arm.sh <arm> <base-url> <local-model-path>`. The runner
requires a localhost server, checks `/health?identity=1` for SHA1 over the
resident mmap plus its artifact filename (the absolute deployment path is not
exposed), verifies the local artifact's full MD5/size against
`tools/eval/arms.tsv`, and writes a provenance sidecar containing the server
binary SHA1, immutable compiled-shader SHA1/ABI, tokenizer SHA1, normalized
numeric-path/protocol configuration, hardware/OS identity, and an install-local
`~/.q27/eval-host-id` fingerprint kept out of unauthenticated health, plus a
fresh run ID and server boot ID. Each arm generates into a temporary workspace, rechecks
model/runtime/boot immediately before publication, publishes provenance last,
and binds every output-row ID to that run; verdict tooling rejects stale/mixed
mode files, revalidates the artifact, and requires runtime + machine identity
to match exactly across all arms.

Arms (all served from the mini's packs, one resident model at a time):

- `t2-base` — pure T2 pack: the REFERENCE (the tier whose quality the
  mixed pack claims to match; its outputs define agreement).
- `b1-base` — pure B1 pack: the FLOOR.
- `m1-candidate` — the selected `cheap_pair`: B1 + T2 gdn_alphabeta +
  attention-Q blocks 21–42, md5
  `91db7fdd368ba3558e59bb5e111bbd07`. This is the ONLY candidate arm.
  The NLL-winning `gdn_pair` contains gdn_qkv, failed the constraints
  probe, and must not be substituted. The earlier informational gdn-qkv
  arm is dropped from the product gate rather than spending another model
  hour on a non-decision arm.

Supporting agreement metric (`tools/eval/spotcheck_verdict.py`):

    capability gap recovered =
        (A(m1-candidate) − A(b1-base)) / (1 − A(b1-base))

where A(x) = x's pair-weighted overall agreement rate with `t2-base`
across the three modes; unextractable output counts as disagreement
(a capability signal, not missing data). This is diagnostic evidence,
NOT the amended A6 ship gate: agreement can preserve a shared wrong answer.

## Agreement bands (supporting read only; superseded as a ship gate)

- **≥ 50% of the capability gap recovered → supporting agreement read.**
- **0–50% → partial transfer.** No ship implication and no "T2-quality"
  language; report only "recovers X% of the agreement gap."
- **< 0 (candidate agrees less than pure B1) → mirage caught.** No ship
  claim; diagnose by mode.

## Amended A6 ground-truth gate (registered after candidate selection,
before any capability run)

A6 must grade each saved answer against the existing gold, not merely
against T2. The fixed decision arm is `m1-candidate`; sample size is the
frozen 120 items. Pre-registered non-inferiority margin: candidate minus T2
accuracy ≥ −5 percentage points, with the lower bound of a paired 95%
bootstrap confidence interval ≥ −0.05. Unextractable answers are wrong.
This is conjunctive across the overall set and the three reported modes;
mode intervals are diagnostic because their n is small, while any mode
with candidate point deficit worse than −10 points blocks. The ground-truth
verdict driver and its must-fail selftest still need authoring. Until they
land, `spotcheck_verdict.py` can report agreement but CANNOT clear A6 and
its zero exit status means only that the report completed.

Honesty notes, registered before any run: (1) agreement-with-T2 is not
correctness — a shared wrong answer agrees; the later expert-review
amendment therefore demotes it beneath the ground-truth A6 gate above.
(2) 120 items gives ±~4.5 points of binomial
noise at rate 0.5 (1σ); the 50% band is deliberately coarse — this is
a spot-check, not a benchmark. (3) The mini's box serves all three arms with the same binary and protocol
pins fingerprinted in the census harness; arms are not comparable across
differing binaries. Artifact provenance is mandatory, not inferred from the
caller-supplied arm label.

## RESULTS

(pending — first author the ground-truth verdict driver/selftest, then run
`t2-base`, `b1-base`, and the selected `m1-candidate`; ~1 h/arm at mini
decode rates)
