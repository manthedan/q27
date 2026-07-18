# Handoff — mixed-tier census → ship gates (2026-07-17, mini)

Read this + `docs/plans/2026-07-17-mixed-tier-census.md` (the plan doc is the
authoritative record; its RESULTS / combo / ship-gate sections are complete
through tonight).

## Where things stand

1. **First-pass census (25 arms) + combo phase (3 arms): DONE, committed
   `1890f67`.** Same-box anchors B1 2.6610 / T2 2.6081 (gap 0.0529 nats).
   Gate-passing classes: gdn_qkv (114.4%), gdn_alphabeta (46.9%), attnq_mid
   (17.6%). Non-additivity proven both directions. Raw:
   `logs/mixed_census/census_summary.txt`, `logs/mixed_combo/combo_summary.txt`.

2. **Ship gates: RUN and committed (the commit that adds this handoff;
   evidence in `logs/m1-ship-20260717/`).**
   - `gdn_pair` = B1 + T2{gdn_qkv, gdn_alphabeta}. Persistent pack
     `models/bonsai-27b-m1/bonsai-27b-m1.q27` (4,110,049,792 bytes, md5
     107647e9…, CHECKSUMS.md5 written). NLL anchor 2.5885 EXACT (beats
     all-T2, 0.9925), suffix battery 8/8 — but **behavioral probes 3/4 =
     SHIP GATE FAIL**: constraints probe collapses at greedy into a
     thinking-mode repetition loop to the 6144 cap. Controls all pass
     same-box/server/protocol: B1, T2, alphabeta-only, qkv-only. The loop
     is EMERGENT from the qkv+alphabeta combination (cross-checkpoint
     co-adaptation; NLL is blind to it).
   - `cheap_pair` = B1 + T2{gdn_alphabeta, attnq_mid(blk 21–42)}.
     **PASSES THE COMPLETE SHIP GATE — first mixed pack to do so.** NLL
     anchor 2.6355 EXACT (1.0105 vs T2), 3.83 GB, suffix 8/8, probes 4/4.
     Artifact currently `models/probe_cheap.q27` (md5 91db7fdd…).

3. **Suffix battery instrument fixes (committed, in
   `tools/suffix_burst_gates_2026-07-16.sh`)**: gate 3b awk compared the
   words `accepted`/`live` (fields 5/3) not the numbers (6/4) — leg was
   unsatisfiable on any pack ever; added standing `rep3`
   incrementing-chapter arm (forces live lane rejections, bytes identical);
   gate 4 neutral-silence demoted to WARN (economics prior — T2's greedy
   neutral continuation is periodic on the mini, bytes identical);
   MODEL/OUT now env-overridable. Under the fixed instrument all four
   packs run clean batteries same-box (B1 8/8, T2 8/8+WARN, m1 8/8,
   cheap 8/8).

## Immediate next steps

1. **Daniel decisions pending**: (a) tier name/home for the gate-passing
   `cheap_pair` artifact (rename `models/probe_cheap.q27`, write its
   CHECKSUMS.md5, note in plan doc); (b) fund the `gdn_pair` rescue
   investigation or park it.
2. Codex review of the ship-gate changes already ran clean (one wording
   fix adopted); everything through the handoff commit is in the tree.
   No commit trailers, ever.

## Registered residue (not run)

- gdn_pair rescue: band-restricted partial qkv grafts; per-layer loop
  localization; vendor-B.1 sampling variant (informative only — the
  registered probe protocol is greedy).
- Combo summary's published `combo_summary.txt` pack-GB column carries the
  (fixed-in-script) GiB/MB unit bug; plan doc records corrected values.
- METAL_PROGRESS.md has no entry yet for tonight's census→ship-gate arc.

## Gotchas that cost time today

- `codex exec` without `< /dev/null` blocks on stdin forever.
- Committing (HEAD change) or editing a fingerprinted driver while its
  batch runs trips `check_fingerprint` and aborts the run — hold commits
  until batches complete.
- `q27-metal-server` needed a rebuild before it accepted the mixed policy
  (stale-binary rule applies to the SERVER binary too).
- Both batch drivers now share `logs/q27_metal_batch.lock` (post-codex fix).
