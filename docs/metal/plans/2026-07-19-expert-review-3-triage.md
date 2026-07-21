# Expert review 3 triage — "engine near-completion" review (received 2026-07-19)

Third-party code review asserting the single-M4 Metal engine is "roughly
90–95% complete" under exact-forward / one-base-M4 / batch-1 constraints,
with a recommended sequence. Reviewed against source per house rule (the
review inspected `d792f07`, Jul 18). **Convergence first, then what is
stale, then what is superseded, then the dispositions we adopt.**

## Verdict in one line

Technically strong and mostly accurate; the load-bearing source claims
verify. But two of its "to-do" sections are stale relative to HEAD, and its
freshest result (the M1 A5 KILL) is superseded by the stronger result we
produced the same week (gdn_pair exoneration + A5-on-gdn_pair KILL).

## Verified correct against source (the claims that matter)

| Claim | Verification | Where |
|---|---|---|
| B1/T2 serial decode computes 129 unused activation-quantization outputs/token | `encode_token` calls `rmsnorm_quantized` 2×64+1 = 129×; `project()` routes Bonsai dtypes to the float matvec and never consumes the int8 copy | metal_engine.cpp:1266/1269/1274 (calls), 1147-1148 (`is_bonsai_dtype → matvec`) |
| Constrained decode is serial-only (no resident K-step under tool mask) | `active_mask_ >= 0 → throw` in the resident path; MTP masks unwired on Metal | metal_engine.cpp:1454, 1279 |
| Snapshot identity = whole-artifact SHA1, cached per-process only | `snapshot_identity()` SHA1s `mapping_base`/`mapping_size`, ~whole pack | metal_engine.cpp:803-817 |
| The don't-refund kill list (direct-RHS, barrier removal, function constants, T3, learned drafters, wider verify) | consistent with the parked ledger | — |

**The B1 dead-quantization find is the one genuinely new, mechanistically
justified engine item.** It is correct, it is small, and it is the only
candidate in the review worth funding. Pre-registered separately
(`2026-07-19-b1-serial-decode-fusion.md`).

## Stale relative to HEAD (already landed after d792f07)

- **"Put tokenizer coverage into the standard gate" — DONE.** `make
  test-cpu` runs `test_tokenizer` with the `.tok` (416ec14, the
  vacuous-pass trap closure). The Makefile target the review cites is
  already fixed.
- **Snapshot metadata rehash — partially addressed.** The full-peek
  `meta_` cache (fbce23d, autoreview round 2) means repeat `best_match`
  calls no longer re-peek. The remaining cost is the **per-cold-process
  whole-weight rehash** — the review's "trusted manifest identity"
  (SHA-256 at repack + verification cache) is still valid and unbuilt.
  Half-done; the durable-cache half is the real remaining lever.
- **trace_gate cancel-probe race — DONE** (630d1e5). **T1 eviction
  classes — SHIPPED** (3d4f451). **Streaming parity — SHIPPED** (c5150dc).

## Superseded by our own results (the review is behind us)

- **The M1/A5 framing.** The review reports the Jul 18 cheap_pair A5 KILL
  as the freshest state. We have since gone further: the gdn_pair rescue
  exonerated the PACK (the repetition loop was a serving-binary artifact,
  6abf45f), then A5-on-gdn_pair KILLED the strongest arm on both corpora
  (55ac29d). The review's closing hedge — "a future B1→T2 bridge probably
  requires training/distillation/a learned adapter, not byte-level
  grafting" — is correct and is now a MEASURED result on two arms, not a
  conjecture.
- **"De-stale README/METAL_PROGRESS to remove M1" — partially overtaken.**
  The 7ca3238 merge already relabeled M1 "candidate, not shipped." README
  and METAL_PROGRESS.md still predate the merge and are stale — the review
  is right that they need a pass, but it should be written against TODAY's
  state (mixed thesis permanently closed, CUDA dropped, no live traffic),
  not the reviewer's Jul-18 snapshot.

## Dispositions

1. **B1 serial-decode fusion (review §1) — ADOPT, the single funded kernel
   round.** Pre-registered at `2026-07-19-b1-serial-decode-fusion.md` with
   the reviewer's own ship line (B1 byte-identical output, ≥5% end-to-end
   B1 decode or STOP after two candidates). Expect 2–8%, not 20% (the
   documented dilution: B1's 2.36× GEMV win moved the wall 3%).
2. **Durable snapshot-identity cache (review §3) — ADOPT as a small
   high-UX-value item, separately scoped.** SHA-256 manifest at repack +
   inode/size/mtime verification cache so a warm restart does not reread
   7 GB to rediscover a digest already verified at install. NOT bundled
   with the B1 round (different subsystem, different risk).
3. **Profile-first B1 residue round (review §2) — DEFER until §1 lands.**
   §1's dead-quantization removal changes the very profile §2 wants to
   measure; profiling first measures the wrong baseline. Sequence matters.
4. **Constrained-generation perf (review §5) — PARK behind the native
   agent graduating, exactly as the review gates it.** Measure constrained-
   token share first; do not build a GPU grammar engine speculatively.
5. **Two-Mac pipeline prefill (review §6) and B1 two-slot fused probe
   (review §7) — DO NOT FUND now.** Both are large lifts against saturated
   ceilings with negative priors (the T2 N=2 fused kernel was already
   killed). Revisit only if a factor-level cold-TTFT win justifies a
   distributed engine, or if aggregate concurrency becomes a real product
   requirement.
6. **README/METAL_PROGRESS de-stale — DO, but against current HEAD,** in
   the same pass that records the gdn_pair exoneration + A5 KILL + the
   CUDA-dropped and no-live-traffic house rules.

## What this review changes

One funded kernel round (B1 fusion), one small durability item (snapshot
identity cache), one doc pass. Everything else is confirmed-parked or
already-done. The review's own closing is the correct frame: the large
remaining gains are stateful (restore / append / compact / pipeline), not
kernel — so the B1 round is gated hard and stopped if it misses, and it
must not become another survey.
