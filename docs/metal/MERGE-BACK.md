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

## CUDA verification: yukon (RTX 3090, CUDA 12.0)

**Correction (2026-07-27).** An earlier revision of this doc asserted "we
cannot build CUDA." That was wrong — `yukon` (RTX 3090, nvcc 12.0.140) is
reachable over Tailscale and is already the repo's CUDA oracle
(`tools/yukon_regate_2026-07-16.sh`, `metal_cuda_gate.py --cuda-ssh yukon`).
The claim should have been checked against the repo before being written
into a plan.

What that changes: **compiling needs no GPU**, so every CUDA-touching PR can
be compile-verified at effectively zero contention. All three staged PRs
were verified on nvcc 12.0 (sm_86; note nvcc 12.0 predates sm_120, so drop
that `-gencode` when building there):

| PR | result |
|---|---|
| PR 1 | `upstream/master` **FAILS**: `src/server.cu(2331): error: structured binding cannot be captured` ×3. With the patch: **compiles**. The bug is real and reproduced on the exact toolchain. |
| PR 3 | `blocks.cu` and `test_kernels.cu` both compile |
| PR 5 | upstream's own `loader.cpp`, `tokenizer.cpp` and `engine.cu` all build clean against the new headers, `-Wall -Wextra`, no new warnings |

### PR 3 gate-verified on hardware (2026-07-27)

The tie-break needed no weights after all: `test_kernels` loads a model for
its GEMV/attention cases, but the argmax cases are synthetic (`rand_vec`).
Extracted them into a standalone model-free harness,
`tools/argmax_tie_gate.cu` (43 lines, builds against `blocks.cu` alone, runs
in seconds), and ran it on the 3090 against both arms:

| case | `upstream/master` | PR 3 | CPU truth |
|---|---|---|---|
| random ×3 (no ties) | 86094 / 224571 / 160077 | **identical** | same |
| all-equal ties | 32767 ✗ | **0** ✓ | 0 |
| max@0, max@last | ok | ok | — |
| dup max (far apart) | 200000 ✗ | **100** ✓ | 100 |
| triple tie, adjacent | 248318 ✗ | **7** ✓ | 7 |

Upstream fails 3 of 8; PR 3 passes 8 of 8 (`worst |idx − lowest| = 0`).

### The canonical anchor does NOT move (measured 2026-07-27)

The artifact fetch on yukon finished and verified bit-identical to ours
(md5 `39fd7244…` for the `.q27`, `bb95b3ca…` for the `.tok` — matching
`models/qwen36-27b-mtp/CHECKSUMS.md5` despite a resumed, retried download).
That unblocked the two gates that actually settle PR 3:

**Full CUDA kernel battery**, `build/test_kernels` against the real artifact,
both arms: **ALL PASS**, with every reported error value identical between
them (`h16 vs fd2 rel t3 ntok=8 seq=4096` → `2.297e-03` on both, and so on
down the list). PR 3 perturbs no other kernel. Upstream passes its own
battery because the tie assertion is precisely what PR 3 adds.

**The canonical bitwise gate** — upstream's own, from
`tools/constrain_gate.sh:89`:
`./build/q27 $MODEL --tokens "760,6511,314,9338,369" --ctx 2048 -n 128 --spec`

| arm | canonical md5 |
|---|---|
| `upstream/master` (`c2d2116`) | `a2982c5197c627551b27d76a0a94b220` |
| `pr3-argmax-tiebreak` (`dc05889`) | `a2982c5197c627551b27d76a0a94b220` |
| upstream's **published** anchor | `a2982c5197c627551b27d76a0a94b220` |

**Identical, and equal to the published anchor.** So PR 3 is not a
"behavior change that may require re-blessing" — on the reference tier's
canonical sequence it is bit-for-bit inert, and it only differs where the
old code was demonstrably wrong.

Stated precisely, because it is one sequence and not a proof: this shows the
*published* anchor is unaffected — which is the one he would have had to
re-bless. It does not prove no prompt anywhere can hit an argmax tie; the
128-token greedy canonical simply does not. Combined with the non-tie cases
being bit-identical, the residual risk is confined to a generation landing on
exact float equality at the argmax.

**Earlier framing of the anchor question.** The three non-tie
cases are bit-identical between arms, so the change is confined to exact
float-equality ties. An anchor can only shift if a real generation hits an
exact logit tie at the argmax — rare enough that we did not observe one, but
still his call to re-bless. Lead the PR with this table: it shows both that
the bug is real and that the blast radius is bounded.

Ship the harness with the PR. It gives him a model-free reproduction, which
`test_kernels` cannot be for someone without the artifact.

### What is still blocked: runtime gates

**yukon had no model artifacts** (`/mnt/ai/models` is gone; no `.q27`
anywhere). An authorized fetch of the official artifact is in
progress there (`~/q27-artifacts/`, resumable `fetch.sh`; the Tailscale path
from the Mac measured ~1 MB/s, so pulling from HF on the box is the faster
route). PR 3 no longer needs it — the model-free harness above covers it —
but the **canonical/NLL gates still do**, which is what would settle anchor
drift end-to-end rather than by argument.

### Shared-machine etiquette

yukon is used by other projects. Rules, mirroring the local
GPU-exclusive-slot rule:

- **Query before running anything**: `nvidia-smi --query-compute-apps=...`;
  if non-empty, do not start GPU work.
- Re-check immediately before launch, not just at plan time.
- Builds are CPU-only — `nice` them and cap `-j`; they do not touch the GPU.
- **Never disturb `~/projects/q27`** on yukon. It sits on an unrelated
  branch (`audit/fable-codebase-review`) with uncommitted changes, last
  touched 2026-07-13. Work in a throwaway clone instead; the verification
  clone is at `/tmp/q27-prverify`.

### Incidental portability find

The Makefile hardcodes `NVCC ?= /usr/local/cuda/bin/nvcc`, which misses a
distro-packaged CUDA (yukon's is `/usr/bin/nvcc`). Falling back to `nvcc`
from `PATH` when that path is absent is a one-line courtesy PR.

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

**The `httplib.h` patch needs an explicit conversation** — see the dedicated
section below; the ask has since been shrunk to a 7-line field with no
dependency on httplib internals.

So the achievable shape is: **Metal lands as new files plus roughly a dozen
additive lines across four shared headers**, touching zero `.cu` files and
restructuring nothing.

## The `third_party/httplib.h` patch — disclose, do not smuggle

`diff` against stock cpp-httplib **0.18.3** is exactly **7 insertions, 0
deletions**: a `socket_t sock = INVALID_SOCKET` field on `Request`, and
`req.sock = strm.socket()` in `Server::process_request`. Everything else in
the header is stock.

**Why.** The Metal server does minutes of GPU work before it writes
anything — queue wait, then prefill (measured 322 s for an 8 K prompt on a
base M4). Without a liveness probe it holds a slot and burns the GPU
producing output for a client that already hung up. With the fd: 2 s kill of
a ~90 s prefill, follow-up answered in 1.15 s instead of waiting the dead
request out.

**The gap is real and checkable in three sentences.** `DataSink::is_writable`
exists only *after* headers are written. `Stream&` never leaves
`process_request`. `set_socket_options` fires at accept with no per-request
correlation. There is no supported path to the fd before the first write.

### Two corrections to an earlier draft of this section

*Both were wrong and are worth recording, because the wrong versions would
have weakened the ask.*

1. It claimed `is_writable` "may not detect a half-closed socket as reliably
   as `is_socket_alive`." **False** — in 0.18.3 `SocketStream::is_writable()`
   is `select_write(...) > 0 && is_socket_alive(sock_)` (`httplib.h:5886`),
   and `DataSink::is_writable` is wired straight to `strm.is_writable()`
   (`:4479, :4525, :4577`). They are the same probe, which is exactly why our
   streaming path already cancels correctly.
   **The real reason restructuring fails**: `write_response_core` emits the
   status line and headers *before* invoking the content provider (status at
   relative line 45, provider at 60). Moving queue-wait and prefill inside a
   provider commits to `200` before the work starts, so overload `429`/`503`
   and `EngineError`→`500` become in-band errors no OpenAI/Anthropic client
   parses. That argument survives a maintainer who knows the header.

2. It claimed the patch "fails silently" on an httplib upgrade. **False, and
   this is our best defense.** All **12** call sites are unguarded — 0
   `#ifdef`s. Drop the patch and the build fails at those exact lines:
   loud, self-localizing, 7 lines to repair. **Discipline to state and keep:
   never wrap these in `#ifdef` for vanilla-httplib compatibility — that is
   the only thing that would make the failure silent.**

### The ask has been shrunk (`socket_alive`, this round)

We no longer touch httplib internals at all. `httplib::detail::is_socket_alive`
was an internal-namespace dependency, so the ask was really "a field *plus* a
promise about `detail::`". Replaced with a self-contained ~20-line POSIX probe
in `metal_server.cpp`; all 9 internal call sites repointed.

Verified identical, not assumed: `tools/socket_alive_diff_test.cpp` runs both
implementations over six socket states — idle, peer-wrote, peer-closed-with-
buffered-data, drained EOF, EBADF, and peer half-close — and they **agree on
all six**. The two intended edge cases are covered there: a half-closed peer
reads dead (no real HTTP client does that), and a pipelined follow-up reads
alive (correct).

So the ask is now precisely: *7 lines, one borrowed fd, no use of your
internals.*

### Upstream in parallel, not as a fallback

`Request` already carries `remote_addr` / `remote_port` / `local_addr` /
`local_port` (`httplib.h:619-622`), so a borrowed server-side fd defaulting
to `INVALID_SOCKET` sits squarely inside the struct's existing idiom —
upstream acceptance at yhirose/cpp-httplib is plausible. **File that issue
the same week we talk to Gabe**, not after he answers: if he declines, the
clock is already running; if upstream accepts later, our vendored patch
becomes a dated backport with a deletion ticket. It also strengthens the ask
— "upstream path started, this is the bridge."

### What goes in the message

Numbers first (322 s prefill, 2 s kill vs ~90 s wait-out, 1.15 s follow-up)
— the justification is the behavior, not the patch. Then the three-sentence
gap proof. Then the compile-loud failure mode. Then the edge cases, which
show the semantics were reasoned about rather than bolted on. Then explicit
deference: *if you'd rather carry this differently — a different accessor
shape, a per-request hook — say so; the diff is 7 lines precisely so any
alternative is cheap to adopt.*

And ask **before** PR 7 lands, plainly preferring a "no" now. A disclosed
7-line patch with a compile-loud failure mode is a minor governance
question; the same patch found during review reads as a smuggled fork.

## Stages

Each stage is a branch off `upstream/master`, not off `metal`, so its diff
contains only that stage.

### PR 1 — `server.cu` CUDA 12.0 compat  *(smallest, safest, send first)*
Lambda-captured structured bindings are rejected by CUDA 12.0's nvcc;
behavior-identical rewrite to named tuple references. ~10 lines.
**Verified on nvcc 12.0**: upstream fails with three `structured binding
cannot be captured` errors at `server.cu(2331-2332)`; with the patch it
compiles. Lead the PR with that reproduction.

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

### PR 3 — argmax tie-break parity  *(risk measured away)*
`blocks.cu` + `test_kernels.cu`: exact-value ties resolve to the **lowest**
index, so CUDA agrees with Metal argmax and CPU `max_element`. This is the
single most valuable thing we have for him *if* he ever wants a second
backend. It changes CUDA kernel behavior in principle — but **measured on
his own canonical gate, the anchor does not move** (see above), and the full
kernel battery is identical. Send it with the evidence table, not with a
warning.
Compile-verified AND gate-verified on the 3090 — see the table above.

### PR 4 — `Q27_GRAPH_TRACE=1` instrument — **DROPPED, overtaken upstream**

Upstream added `inst_or_advise()` to `engine.cuh` on 2026-07-18 for the same
issue #1 / A10 case, and it is **better than our instrument for the person
who hits the problem**: a graph-zoo OOM now produces an actionable refusal
naming the exact levers (`--ctx`, `Q27_MAXD=4`, `Q27_SAMPLED=0`,
`build/q27-server-w8`) plus free-VRAM, instead of a raw CUDA abort.

`Q27_GRAPH_TRACE` prints per-family byte attribution. That is a developer's
view, and he has already published the numbers it would produce (~600 MB
sampled set on sm_86, ~280 MB for `Q27_MAXD=4`). Proposing ~79 lines of
diagnostic into his boot path to re-derive documented constants is not a
good trade. Keep it fork-local.

**That is two of the three items in the 07-16 ready-to-send queue now
overtaken** (PR 2 by his README, PR 4 by `inst_or_advise`). Only the
CUDA-12.0 compat fix survived — and it survived *because* it is a
toolchain-compat bug he has no way to hit. Re-checking each item against
current upstream immediately before sending is not optional.

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
