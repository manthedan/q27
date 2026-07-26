# Merge-back plan (2026-07-25)

Supersedes [plans/2026-07-16-upstream-prs.md](plans/2026-07-16-upstream-prs.md),
whose queue is folded in below and whose scope numbers are stale.

**Status.** The maintainer approved the *idea* in principle on 2026-07-16
(operator-relayed). The hold has always been ours: "do not open PRs yet —
wait until our project is further along." The unhold criteria recorded then
are now essentially met (suffix-burst gate 6 PASS, official-tier multislot
run, Homebrew formula tagged at metal-v0.6.1). Decision 2026-07-25: **prepare
everything, then ask** — stage every branch locally, then send the issue.

## Blocking prerequisite — CLEARED

`src/engine.cuh` had silently reverted 162 lines of the maintainer's own
CUDA work (`splitk_ws`, `sampled_graphs`, `capture_constrained`,
`pf_batch_min`, `Q27_PF_T/PF_SB`), dropped by the resolution in `df34c67`
and carried forward by two later merges. **Any PR opened from this branch
would have read as deleting his features.** Restored byte-identical to
upstream and audited every other shared file for the same failure — no other
drops (audit A9). This is why nothing ships before a clean-diff check.

## Hard constraint: we cannot build CUDA

There is no `nvcc` on either of our machines (both Apple Silicon). Every
CUDA-touching PR below is **untestable by us** and must be labelled as such
in its description, with the reasoning and the evidence we do have. This is
a reason to keep our CUDA surface *small and obviously-correct*, not to
hide it.

## Real scope (measured, not estimated)

`git diff upstream/master..HEAD` is 973 files / 164,932 insertions, which
overstates the review burden badly:

| area | insertions | files | goes upstream? |
|---|---|---|---|
| `logs/` | 86,882 | 708 | **no** — never in a path-scoped PR |
| `experiments/` | 23,413 | 48 | probably not (agent harness, TUI) |
| `src/metal/` | 20,563 | 13 | **yes** — the actual port |
| `docs/` | 16,852 | 97 | condensed subset |
| `tools/` | 11,707 | 67 | selective (gates worth having) |
| `packaging/` | 1,811 | 10 | no (our Homebrew tap) |
| shared `src/*` | ~2,900 | 20 | **yes** — the seam + hardening |

So the genuinely proposed surface is roughly **24k lines**, not 165k, and
the single biggest reviewable chunk is `src/metal/` at 20.5k.

`LICENSE` carries our fork-additions copyright line. Already acknowledged by
the maintainer in the 07-16 exchange; it must **not** appear in any PR.

## Can the Metal PRs avoid CUDA entirely? — YES, measured

Tested rather than assumed: a worktree at pristine `upstream/master`, with
**only our new files** copied in (`src/metal/*`, `backend.h`, `sampling.h`,
`kl.h`, `strip_ctrl.h`, `tool_preamble.h`) and every shared file left
untouched, then compiled.

Result: 48 errors, and **every one is a missing addition — not a single CUDA
dependency.** `blocks.cu`, `server.cu`, `test_kernels.cu` and `inspect.cpp`
are not required by Metal at all; they are independent optional
contributions that can be held back or dropped without affecting the port.

The shared-file surface Metal genuinely needs is small, and almost all of it
is *additive* (nothing existing is changed, so his CUDA build is unaffected
by construction):

| file | what Metal needs | shape |
|---|---|---|
| `src/loader.h` | `DType` values T2_G128/T3_G128/B1_G128 for the bonsai tiers, `Model::mapping_base()` | +6/-1 — existing enum values untouched |
| `src/tokenizer.h` | `vocab_size()` | +10/-1 |
| `src/stream_split.h` | streaming split additions | +19/-2 |
| `src/api_common.h` | `initial_harness_prefix`, `ToolCallStreamer` | +461/-73 — **the one to split up** |
| `third_party/httplib.h` | `Request::sock`, so a handler doing long GPU work before its first write can probe client liveness | +7/-0 |
| `Makefile` | metal build rules | additive rules only |

Two consequences worth acting on:

**The `-73` in `api_common.h` is avoidable.** It is our extraction of
`strip_ctrl` / `tools_preamble` into their own headers. Upstream's
`api_common.h` *already defines both* — so Metal can simply use his, the
extraction drops out of the merge-back entirely, and with it the only place
we restructure his code rather than adding to it. (It also causes the
redefinition errors in the test above.) Keep the extraction in our fork if
we want; do not propose it.

**`api_common.h`'s +461 is not one thing.** It bundles what Metal *needs*
(`initial_harness_prefix`) with an independent feature that benefits both
arms (`ToolCallStreamer`, the incremental tool-call argument streamer). Split
it: the streamer is its own PR with its own motivation, and the Metal
dependency shrinks to a few functions.

**The `httplib.h` patch needs an explicit conversation** — patching a
vendored dependency is a maintainer-preference call, not a technical one.
Options: propose it, upstream it to yhirose/cpp-httplib first, or find a
liveness probe that does not need the fd.

So the achievable shape is: **Metal lands as new files plus roughly a dozen
additive lines across four shared headers**, touching zero `.cu` files and
restructuring nothing.

## Stages

Each stage is a branch off `upstream/master`, not off `metal`, so its diff
contains only that stage.

### PR 1 — `server.cu` CUDA 12.0 compat  *(smallest, safest, send first)*
Lambda-captured structured bindings are rejected by CUDA 12.0's nvcc;
behavior-identical rewrite to named tuple references. ~10 lines.
*Untestable by us* — but it is a compile fix for a toolchain he may not
have, and the failure mode is a build error, not a behavior change.

### PR 2 — README note for 24 GB cards  — **DROPPED, already upstream**

The 07-16 queue listed this as ready-to-send. It is now redundant and must
not be proposed: `upstream/master`'s README already documents
`make build/q27-server-w8` / `Q27_W_MAX=8`, that the default width-12 build
"OOMs at graph setup on 24GB", turbo3-by-default on Ampere, the q4s
recommendation, and further `Q27_MAXD=4` / `Q27_SAMPLED=0` knobs with
field measurements from a 22.6 GiB A10 (issue #1). That is more complete
than our note would have been.

Lesson recorded rather than just fixed: the queue was nine days stale and
one of its three ready-to-send items had been overtaken. **Re-check each
remaining item against current upstream immediately before sending.**

### PR 3 — argmax tie-break parity  *(offer, flag the risk)*
`blocks.cu` + `test_kernels.cu`: exact-value ties resolve to the **lowest**
index, so CUDA agrees with Metal argmax and CPU `max_element`. This is the
single most valuable thing we have for him *if* he ever wants a second
backend — but it **changes CUDA kernel behavior**, and his canonical
anchors are byte-exact hashes. Ties are rare, and our 16-token canonical
gate is byte-exact across both arms, but he must re-bless anchors himself.
Send with that caveat stated up front, or hold until after PR 5.

### PR 4 — `Q27_GRAPH_TRACE=1` instrument  *(offer, not push; value reduced)*
Per-family graph-memory attribution in `build_spec_graphs`; prints the table
on instantiate failure so an OOM report self-attributes. Measured:
`verify_w` dominates at 855 MB/w8, sampled family 577 MB. Value is LOWER
than when this was queued: upstream has since done its own attribution and
now documents the zoo's cost in the README (~600 MB sampled set on sm_86,
~280 MB for `Q27_MAXD=4`). What the instrument still adds is
self-attribution *on failure*, which a documented constant cannot do.
Preserved at
`patches/0001-cuda-graph-trace-ORIGINAL.patch`; **needs re-applying onto his
current `engine.cuh`**, which has moved since (the graph-zoo capture gates
landed in that same function).

### PR 5 — the backend seam  *(the real conversation)*
`src/backend.h` plus the `loader` / `tokenizer` / `sampling.h` / `kl.h`
generalizations that let a non-CUDA backend exist at all. No Metal code.
This is where he decides whether q27 is CUDA-only or multi-backend.
Everything after is additive; if he declines, we stop here and stay a
labelled downstream port.

Additive only. Does **not** include the `strip_ctrl` / `tools_preamble`
extraction — see the section above: his `api_common.h` already defines both,
so Metal uses his and the extraction stays a fork-local convenience. That
keeps this PR free of any restructuring of upstream code.

### PR 6 — Metal core
`metal_backend.{h,mm}`, `q27_kernels.metal`, `metal_engine.{h,cpp}`,
`metal_cli.cpp`, `test_metal*`. ~14k lines, self-contained, touches no CUDA.
Fully testable by us.

### PR 7 — Metal serving
`metal_server.cpp`, `stream_format.h`, `disk_snapshot_store.h`,
`snapshot_evict.h` and their gates. Separable from core: it is the HTTP
layer, and he may want the engine without our serving opinions.

### PR 8 — shared serving hardening
Our `api_common.h` / `stream_split.h` additions that benefit both arms.
Note much of this file already flows the other way (his tolerant readers,
drift modes, billing-header fix came to us) — so this PR is small and should
be diffed carefully against what he already has.

### Probably never
`experiments/` (ds4-agent, q27-tui), `packaging/` (our tap), `logs/`,
`docs/metal/` beyond a short pointer. Fine — they are ours.

## Staged branches (local, not pushed)

| branch | stage | diff vs `upstream/master` | state |
|---|---|---|---|
| `pr1-cuda12-compat` | PR 1 | `server.cu` +8/−2 | ready |
| `pr3-argmax-tiebreak` | PR 3 | `blocks.cu` +15/−3, `test_kernels.cu` +11/−1 | ready, carries the anchor-reblessing caveat |
| `pr5-backend-seam` | PR 5 | 3 new headers + 15 added lines across `loader.{h,cpp}` / `tokenizer.h`; **1 deletion total** (the DType enum line, extended in place) | ready |

Each is a worktree off `upstream/master`, so its diff contains only that
stage. Verified for PR 5: every new header is self-contained, and
`loader.cpp` / `tokenizer.cpp` still build clean at `-Wall -Wextra`.

Two things the staging turned up:

**Extending `DType` is not free.** It makes `dtype_name()`'s switch
non-exhaustive and adds a `-Wswitch` warning to *his* build. Three case
labels fix it; caught by compiling his sources against the new header rather
than assuming additive meant safe.

**A separate bug worth its own PR.** Upstream's `Tokenizer` owns a raw
`Impl*` with **no destructor and no deleted copy constructor** — it leaks,
and copying it would double-free. Our fork fixed this incidentally; it
should be offered on its own merits, independent of any Metal work, rather
than smuggled in as part of the seam.

## Sequencing note

PRs 1–2 are free goodwill and cost him minutes. PR 5 is the decision point;
there is no value in preparing 6–8 for review until he answers it, though
having them *staged* is exactly what "prepare everything, then ask" means.

## Before sending anything

- [x] A9 clean-diff check (`engine.cuh` restored, all shared files audited)
- [ ] Each stage branched off `upstream/master` and its diff eyeballed for
      unintended carry-over
- [ ] Refresh the 07-16 issue draft: scope numbers are stale (it says
      "~160 commits / +30k lines")
- [ ] Every CUDA-touching PR body states we could not compile it
