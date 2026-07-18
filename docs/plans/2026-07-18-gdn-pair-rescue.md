# gdn_pair rescue: localize the composition-only repetition loop — pre-registration (2026-07-18)

Status: FUNDED by the operator (2026-07-18 roadmap: "fund item 4"). This is
the rescue investigation registered as residue in
`2026-07-17-mixed-tier-census.md` ("Registered rescue residue (not run):
band-restricted partial qkv grafts; vendor-B.1 sampling variant … ;
per-layer loop localization"). It is a **mechanism investigation**, not a
serving claim. Nothing here ships a pack.

## Why this is still worth funding after the A5 KILL

The A5 KL gate (`2026-07-18-kl-pair-a5.md`) killed the mixed-pack **serving
thesis**: the eligible M1 candidate (`cheap_pair`) is FARTHER from the
official 17 GiB reference than T2 on both corpora (wikitext +0.0458, code
+0.1753 nats). `gdn_pair` was explicitly **ineligible** for A5 (it had
already failed its own ship gate). So A5 says nothing about `gdn_pair`
directly — and `gdn_pair`'s open question is not a serving question.

The open question is a **mechanism anomaly** the census itself flagged as
the interesting scientific object: `gdn_pair` (B1 base + T2 `attn_qkv` +
T2 `ssm_alpha`/`ssm_beta`) **beats all-T2 on wikitext NLL** (2.5885, ratio
0.9925) yet **collapses at greedy** on the constraints leg into a
thinking-mode repetition loop (`water bottle (2) -> …`) to the 6144-token
cap with empty final content. The controls isolate the cause precisely
(mini, `logs/m1-ship-20260717/`, same box / server / protocol):

| arm | completion_tokens | finish | constraints leg |
|---|---|---|---|
| alphabeta-only | 2242 | stop | exact pass |
| qkv-only | 2211 | stop | finite near-pass (5 items, `done` not `done.`) |
| B1 control | 2932 | stop | finite near-pass |
| T2 control | 4069 | stop | finite near-pass |
| **gdn_pair (qkv+alphabeta)** | **6144 (cap)** | **length** | **catastrophic loop, empty content** |

Each parent alone is fine. The **composition** of two individually-safe
grafts produces an emergent cross-checkpoint co-adaptation pathology that
per-token NLL is blind to (the same pack is NLL-*better* than T2). That is
a concrete, reproducible instance of "checkpoint grafting between
separately trained companions breaks in a way quality metrics do not
surface" — exactly the class the pre-readout amendment warned about.
Understanding **where** the loop lives (which layers, which graft class)
is the funded deliverable: it either validates the band-restricted-graft
hypothesis or falsifies it, and it sharpens every future mixed-tier claim
with a probe that catches what NLL cannot.

This investigation ALSO serves the standing "not a valid tool call" /
agentic-collapse watch: a repetition loop that empties final content is
the same surface shape as an agentic client hang.

## The frozen probe (all legs reuse it byte-for-byte)

Constraints prompt (reconstructed exactly from the failing probe's
thinking echo, `logs/m1-ship-20260717/probe-constraints.json`), greedy,
thinking mode, max 6144:

> List exactly 5 hiking essentials as a numbered list. Each item exactly
> 3 words. No item starts with the letter A. Include one item containing
> the word "map". All lowercase. No punctuation at the end of any line.
> The final line must read: done.

**Loop detector (deterministic, no judge):** a leg LOOPS if
`finish_reason == "length"` (hit the 6144 cap) OR the reasoning stream
contains ≥8 consecutive repeats of the same `(text (n) -> )` token span.
A leg is FINITE otherwise. (The 4 controls all finish `stop` ≤4069
tokens; gdn_pair finishes `length` at 6144 — the detector separates them
with a wide margin.)

## Arms (band-restricted partial qkv grafts on the B1 base)

`gdn_pair` = alphabeta (all blocks) + qkv (all blocks). The localization
holds **alphabeta fixed at all blocks** (it is the tiny, exact-pass graft)
and bisects **which qkv band** triggers the loop when composed. All packs
are B1 base, built with `tools/q27_mix.py --take` (regex on tensor names),
mini-side, transient PID-unique, same fingerprinted fail-closed discipline
as the combo driver. Block range is the transformer's 48 blocks (0–47).

Registered arms (each = `ssm_(alpha|beta)` on all blocks + `attn_qkv` on
the named band):

- **P0 full (control)** = gdn_pair rebuilt, qkv on 0–47. Must reproduce
  the LOOP. (Fail-closed: if P0 does not loop, the harness/prompt drifted
  and every other arm is void.)
- **P1** qkv blocks 0–15 + alphabeta all
- **P2** qkv blocks 16–31 + alphabeta all
- **P3** qkv blocks 32–47 + alphabeta all
- **P4** qkv blocks 0–23 + alphabeta all (escalation bisection, run only
  if P1–P3 are not decisive — see decision rule)
- **P5** qkv blocks 24–47 + alphabeta all (escalation bisection, same)

Each arm runs the frozen constraints leg **3×** (greedy is deterministic
given an identical prefix; 3 runs guard against prefix-cache /
snapshot-restore nondeterminism flipping the verdict). An arm's verdict is
the majority of its 3 runs.

## Gates (both directions, exit codes)

- **G1 control reproduction (fail-closed).** P0 LOOPS on ≥2/3 runs. If
  P0 is finite, ABORT: the probe or harness drifted — re-anchor before any
  band verdict. This is the sabotage arm proving the detector can fire.
- **G2 band localization.** P1/P2/P3 each classified LOOP / FINITE.
  Decisive patterns and their readings:
  - Exactly one band LOOPS → the pathology is **depth-localized** to that
    band; record which. That is the headline mechanism result.
  - All three bands LOOP → the pathology is **distributed** (any
    sufficient qkv mass composed with alphabeta triggers it); band
    restriction cannot rescue the pack.
  - No band LOOPS → the pathology requires the **full-depth joint** graft
    (interaction needs qkv across all depths simultaneously); also
    decisive against a band-restricted rescue.
- **G3 escalation (conditional).** If exactly one of P1–P3 LOOPS, run the
  two half-bands P4/P5 that partition the looping third to sharpen
  localization to ~12 blocks. If P1–P3 are unanimous (all loop or all
  finite), G3 is skipped as non-informative.
- **G4 byte-sanity.** Every built pack `--validate-only` clean and its
  NLL measured once (single 8K wikitext run) — recorded as the
  quality/behavior pair, confirming each band pack still sits near the
  gdn_pair NLL point (sanity, not a gate).

## Ship / kill line (for the INVESTIGATION, not a pack)

- **RESOLVED (the funded outcome):** G1 passes AND G2 yields a decisive
  pattern (depth-localized / distributed / full-depth-joint). The
  mechanism write-up records which, with the per-arm loop verdicts and
  NLL pairs. If a single band carries the loop, that names the
  co-adaptation site and is a publishable negative-space result about
  cross-checkpoint grafting on this family.
- **UNRESOLVED:** G1 fails (control won't reproduce → probe drift; the
  loop is not stable enough to localize, which is itself recorded) or the
  pattern is internally contradictory across reruns (nondeterminism
  dominates → the detector/protocol is refined and the arm reruns once
  before declaring UNRESOLVED).

There is **no pack ship line** in this investigation. A band-restricted
pack that avoids the loop is, at most, a hypothesis for a future
separately-funded serving gate (it would still have to clear A5-class KL,
which `cheap_pair` already failed). This work answers *where/why*, not
*what ships*.

## Vendor-B.1 sampling variant (informs-only, explicitly NOT a rescue)

The registered vendor-B.1 settings (temp 0.7 / top-p 0.95 / top-k 20) are
NOT the greedy protocol and cannot substitute for it. Run ONCE on P0 as a
single diagnostic leg: if B.1 sampling breaks the loop, that says the
pathology is a greedy-attractor (sharp, low-entropy basin) rather than a
broad degradation — a useful mechanistic hint, recorded as a footnote. It
changes no gate.

## Cost / risk / house rules

Mini-side (has B1, T2, the gdn-pair-experimental pack, and the probe
harness that produced the 5 controls). One resident model at a time,
enforced by the existing combo-driver lock + pgrep fail-closed guard; the
operator's no-live-traffic ruling permits server teardown but does NOT
permit two concurrent 10.9 GB models on this box. Estimated: 4–6 transient
pack builds (~4 GB each, deleted after use) + ≤20 greedy constraints runs
(≤6144 tokens each) + ≤6 single 8K NLL runs. Fits in one mini session.
Risk is a nondeterministic loop verdict, mitigated by 3× runs + the G1
fail-closed control + one refine-and-rerun allowance before UNRESOLVED.
