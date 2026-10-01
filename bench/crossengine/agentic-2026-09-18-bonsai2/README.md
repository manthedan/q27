# Bonsai 2 27B on the agentic campaign (2026-09-18)

One leg, `bonsai2`, of `../agentic-2026-09-07/campaign.sh`: the master
worktree's `q27-server` serving PrismML's Ternary Bonsai 2 27B
(`bonsai2-27b-t2q4x.q27` -- T2 decode plus exact-Q4 prefill shadows, the
Phase 2 layout; `docs/plans/2026-09-18-bonsai2-ternary.md`) with the Qwen3.8
DFlash2 Q8 pack, prefix cache on a fresh tmpfs root, KV fp8, medium effort,
the 12 pinned SWE-bench instances under Claude Code. Compared against the
same-day-config Qwen3.8 legs of 2026-09-17 (`../agentic-2026-09-17-seed`:
`q27seed` = the same binary family with DFlash2, `q27ladr` = MTP ladder) and
the 09-09 `ninferd2` reference, with `../agentic-2026-09-09-echo/turns_cmp.py`
and `../agentic-2026-09-17-turns/depth_think.py`.

| leg | gold | turns/inst | think K/inst | out tok/inst | wall s/inst | agg t/s | tok/round |
|---|--:|--:|--:|--:|--:|--:|--:|
| bonsai2d2b (pure T2, Bonsai-trained drafter) | 10/12 | 37.2 | 66.1 | 22414 | 138 | 227.7 | 3.80 |
| bonsai2mtp (T2+MTP pack, MTP ladder, no DFlash2) | 10/12 | 32.8 | 60.2 | 21106 | 150 | 178.8 | 2.85 |
| bonsai2 | 11/12 | 37.3 | 74.8 | 26214 | 182 | 205.1 | 3.47 |
| q27seed (3.8 default, DFlash2) | 11/12 | 22.0 | 34.5 | 12564 | 78 | 222.0 | 4.02 |
| q27ladr (3.8 default, MTP ladder) | 11/12 | 19.9 | 32.5 | 12244 | 92 | 176.0 | 3.31 |
| ninferd2 (09-09) | 11/12 | 14.0 | 15.5 | 6246 | 36 | -- | -- |

Reading: the ternary checkpoint lands the same patches (gold 11/12, the same
miss set as the q27 legs) but reasons about twice as long per instance and
runs 1.7x the API turns, so wall is 2.3x. Per-token serving is close: 205 vs
222 t/s aggregate over the whole run, 3.47 vs 4.02 accepted tokens per
DFlash2 round (the drafter was trained on Qwen3.8's distribution; the
ternary target accepts a little less of it), prompt reuse 0.969 vs 0.962.
This is the #49 finding's axis with the sign flipped: ninfer's NVFP4 arm
reasons shorter than the model, Bonsai 2 reasons longer, and the engine
mechanics are the same on both sides. Thinking by turn depth (`depth_think.py`)
is longer at every depth from turn 3 on (medians 250-680 chars vs 150-670
for q27seed, with a heavier tail: 24% of turns over 2K chars vs 18%).

Second leg, same afternoon: `bonsai2d2b` is the pure-T2 pack (Phase 3, no
Q4 shadows) with the third-party Bonsai-trained DFlash2 drafter
(ProCreations/Ternary-Bonsai-2-27B-DFlash2, repacked Q8 by
`tools/dflash2_pack.py`, identity-gated on the 3090). Same trajectory shape
(37 turns), +9.7% tokens per round and +11% aggregate decode over the
Qwen3.8 drafter; the gold flip on requests-1921 is one sampled trial, the
verify is exact in distribution. BUILDLOG (as).

Third leg: `bonsai2mtp`, the T2+MTP pack on the MTP ladder alone (BUILDLOG
(au)): 2.85 tokens per round and 178.8 t/s aggregate -- the ladder loses
to DFlash2 single-slot by 27%; its win is multi-slot.

Files: `results.<leg>.jsonl` (harness rows), `<leg>.log` (run.sh output). The server journal and the request-body log
(`REQBODY_LOG=/mnt/ai/data/reqbody/2026-09-18-bonsai2`) are session content
and stay out of the repo.
