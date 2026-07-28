# PR playbook (2026-07-27)

How to actually send the merge-back. [MERGE-BACK.md](MERGE-BACK.md) is the
*plan and its evidence*; this is the **runbook** — order, gates, copy-paste
commands, prepared answers, and abort conditions.

Everything named here is staged and pushed to `origin` (`manthedan/q27`).
Nothing has been sent. The first send is the issue, not a PR.

---

## Rule 0 — re-check every item against upstream immediately before sending

The 2026-07-16 queue had three ready-to-send items. Nine days later **two of
the three had been overtaken** by the maintainer's own work (PR 2 by his
README, PR 4 by `inst_or_advise`). Sending either would have cost credibility
on the ones that matter.

So before *every* send, not once per campaign:

```bash
git fetch upstream
git log --oneline metal..upstream/master          # what landed since we staged
git diff upstream/master..<pr-branch>              # does our diff still apply cleanly?
```

If upstream moved under a branch, rebase it and re-run its gate before
sending. A stale diff is worse than a late one.

---

## Rule 1 — GitHub cannot express this stack

The branches stack locally (`pr6` off `pr5`, `pr7` off `pr6`+`pr8`). A
cross-fork PR's base **must be a branch in the base repo**, so there is no
way to open `pr6` against `pr5` on `signalnine/q27`. Opening PR 6 early would
show PR 5's 730 lines *inside* it.

**Therefore the stack is serialized, not parallel.** Each Metal PR opens only
after its parent has landed on `upstream/master`, and is rebased onto master
first:

```bash
git fetch upstream
git rebase upstream/master pr6-metal-core     # after PR 5 merges
git push --force-with-lease origin pr6-metal-core
```

This is not a preference — it is the reason the waves below exist.

---

## The waves

| wave | contents | gate to open it |
|---|---|---|
| **0** | the issue | none — send first |
| **1** | PR 1, tokenizer, PR 3, boundary fix | none — these stand on their own merit; send whether or not he wants Metal |
| **2** | PR 5 — the backend seam | he answered the issue with interest |
| **3** | PR 6 → streamer → PR 7 | PR 5 **merged**, and the httplib question answered |

Wave 1 is deliberately independent of the answer. Four small fixes to his own
code, each with a reproduction, is also the most honest possible cover letter:
it demonstrates the fork found real bugs before it asks him for anything.

---

## Wave 0 — the issue

Send [upstream-issue-draft.md](upstream-issue-draft.md) (rewritten
2026-07-27; the pre-07-25 version asserted "I cannot compile CUDA" and still
listed the two dropped PRs — **do not send that text**).

```bash
gh issue create --repo signalnine/q27 \
  --title "Interested in a Metal backend merge-back?" \
  --body-file docs/metal/upstream-issue-draft.md
```

Lead with **23 files, 21,692 insertions, 3 deletions, zero `.cu` files**.
That single line is the whole argument: it is the difference between "a 20k
fork wants merging" and "a 20k addition that deletes three lines of your
code."

Then stop. Wave 2 does not move until he answers.

---

## Wave 1 — four standalone fixes

Open all four; they are independent of each other and of the Metal question.

```bash
gh pr create --repo signalnine/q27 --base master \
  --head manthedan:<branch> --title "<title>" --body-file <file>
```

### 1a. `pr1-cuda12-compat` — +8/−2, `src/server.cu`

**Title:** `fix: CUDA 12.0 compat — nvcc rejects lambda-captured structured bindings`

Body must open with the reproduction, because it is the only thing he cannot
check himself on a modern toolchain:

> On CUDA 12.0.140 (`nvcc` from the Ubuntu package, RTX 3090, `sm_86`),
> `master` fails to build:
> ```
> src/server.cu(2331): error: structured binding cannot be captured  (×3)
> ```
> C++20 relaxed this; 12.0's nvcc predates the relaxation. The fix binds
> `make_item_cbs`'s results as named tuple references instead. Behaviour-
> identical — no codegen change, just names the lambda can capture.
> With the patch applied the same toolchain compiles clean.

Re-verify before sending: `ssh yukon`, throwaway clone, `NVCC=/usr/bin/nvcc`,
drop the `sm_120` gencode (12.0 predates it).

### 1b. `pr-tokenizer-lifetime` — +9/−1, `src/tokenizer.{h,cpp}`

**Title:** `tokenizer: add a destructor and delete copies (Impl* was leaked)`

The commit message is already the PR body — use it verbatim. The argument is
in one line: `impl_` is an owning raw pointer with no destructor, and the
implicit copy would double-free it once a destructor exists, which is why the
`= delete`s ship *with* the destructor rather than after.

Expect "it's a single long-lived instance, so who cares." Prepared answer is
in the objection handbook below.

### 1c. `pr3-argmax-tiebreak` — +22/−4, `src/blocks.cu`, `src/test_kernels.cu`

**Title:** `argmax: resolve exact-value ties to the lowest index`

This is the highest-value item we have and the only one that touches kernel
behaviour. **Send it with the evidence table, not with a warning.** The old
issue draft's "you'd want to re-bless your anchors yourself" framing is now
obsolete — we measured it, and the anchor does not move:

| arm | canonical md5 |
|---|---|
| `upstream/master` (`c2d2116`) | `a2982c5197c627551b27d76a0a94b220` |
| `pr3-argmax-tiebreak` (`dc05889`) | `a2982c5197c627551b27d76a0a94b220` |
| your published anchor | `a2982c5197c627551b27d76a0a94b220` |

Paste in the PR body, in this order:

1. The tie table (upstream fails 3/8, PR 3 passes 8/8, `worst |idx − lowest| = 0`).
2. The anchor table above — run on a 3090 against the official artifact,
   md5-verified against `CHECKSUMS.md5`.
3. Full `build/test_kernels` battery: **all pass on both arms, every reported
   error value identical** (`h16 vs fd2 rel t3 ntok=8 seq=4096` → `2.297e-03`
   on both).
4. The honest limit, stated by us before he asks: this shows the *published*
   anchor is unaffected; it does not prove no prompt can ever hit a tie. The
   three non-tie cases being bit-identical bounds the residual risk to a
   generation landing on exact float equality at the argmax.

**Ship `tools/argmax_tie_gate.cu` with it** (43 lines, builds against
`blocks.cu` alone, no model needed). `test_kernels` cannot be a reproduction
for anyone without the 17 GB artifact; this can.

> ⚠ `tools/argmax_tie_gate.cu` is not currently on the `pr3` branch — it
> lives on `metal`. Cherry-pick it before sending, or the PR claims a
> reproduction it does not ship.

### 1d. the adjacent-tool-call boundary fix — `src/stream_split.h`, +21/−2

**Currently bundled** into `pr8-toolcall-streamer` with the 268-line
`ToolCallStreamer`. Recommend splitting:

- The boundary fix is **a bug in his code today** — `</tool_call><tool_call>`
  emits no separating segment, so a consumer buffering one TOOL segment at a
  time folds two calls into one buffer and silently loses a malformed second
  call. Same consumer pattern exists CUDA-side. It belongs in wave 1.
- `ToolCallStreamer` is 268 lines that **nothing upstream calls**. On its own
  merits it is speculative; alongside PR 7, which uses it, it is motivated.
  Move it to wave 3.

Cost of splitting: one rebase of `pr7`, which merges `pr8` today. Do it before
sending, not after.

---

## Wave 2 — PR 5, the backend seam

**Do not open until the issue is answered.** This is the decision point: it is
where he decides whether q27 is CUDA-only or multi-backend. Everything after
it is additive; if he declines here, we stop and stay a labelled downstream
port. That outcome is fine and should be said out loud in the PR body.

`pr5-backend-seam` — +730/−1 across `backend.h` (new, 312), `sampling.h`
(new, 370), `kl.h` (+34), and 15 additive lines in `loader.{h,cpp}` /
`tokenizer.h`. **The single deletion is the `DType` enum line, extended in
place.** No Metal code. No `.cu` file touched.

Evidence for the body: his own `loader.cpp`, `tokenizer.cpp` and `engine.cu`
build clean against these headers under `-Wall -Wextra` with no new warnings,
verified on nvcc 12.0.

Explicitly **not** included: our `strip_ctrl` / `tools_preamble` extraction.
His `api_common.h` already defines both, so Metal uses his and the extraction
stays fork-local. That is what keeps this PR free of any restructuring of his
code — say so in the body, because "additive only" is the whole ask.

---

## Wave 3 — Metal

Gate: **PR 5 merged** (not just approved — Rule 1 means PR 6 needs master to
contain it), and the httplib question answered.

### 3a. The httplib conversation — before PR 7, not during review

Ask in the issue thread, not in the PR. A disclosed 7-line patch with a
compile-loud failure mode is a minor governance question; the same patch
*found during review* reads as a smuggled fork.

Order the message this way — the justification is the behaviour, not the patch:

1. **Numbers first.** 322 s prefill for an 8 K prompt on a base M4. Without a
   liveness probe the server holds a slot and burns GPU producing output for a
   client that already hung up. With it: 2 s kill of a ~90 s prefill,
   follow-up answered in 1.15 s instead of waiting the dead request out.
2. **The gap, in three sentences.** `DataSink::is_writable` exists only
   *after* headers are written. `Stream&` never leaves `process_request`.
   `set_socket_options` fires at accept with no per-request correlation.
   There is no supported path to the fd before the first write.
3. **Why restructuring doesn't fix it.** `write_response_core` emits the
   status line and headers *before* invoking the content provider. Moving
   queue-wait and prefill inside a provider commits to `200` before the work
   starts, so overload `429`/`503` and `EngineError`→`500` become in-band
   errors no OpenAI or Anthropic client parses.
4. **The failure mode is loud.** All 12 call sites are unguarded — zero
   `#ifdef`s. Drop the patch on an httplib upgrade and the build fails at
   those exact lines: self-localizing, 7 lines to repair.
5. **The ask is now smaller than it was.** We no longer touch httplib
   internals at all — `httplib::detail::is_socket_alive` was replaced by a
   self-contained ~20-line POSIX probe in `metal_server.cpp`, verified
   identical over six socket states by `tools/socket_alive_diff_test.cpp`.
   So: *7 lines, one borrowed fd, no use of your internals.*
6. **Deference, explicitly.** "If you'd rather carry this differently — a
   different accessor shape, a per-request hook — say so; the diff is 7 lines
   precisely so any alternative is cheap to adopt."

Prefer a "no" now over a surprise later. And **file the parallel issue at
yhirose/cpp-httplib the same week**, not after he answers: `Request` already
carries `remote_addr`/`remote_port`/`local_addr`/`local_port`, so a borrowed
server-side fd defaulting to `INVALID_SOCKET` sits inside the struct's
existing idiom. If he declines, the clock is already running; if upstream
accepts, our vendored patch becomes a dated backport with a deletion ticket.

**Standing discipline: never `#ifdef` those call sites for vanilla-httplib
compatibility.** That is the only change that would make the failure silent,
and the loud failure is our best defense of the patch.

### 3b. `pr6-metal-core` — +16,136/−0

`metal_backend.{h,mm}`, `q27_kernels.metal`, `metal_engine.{h,cpp}`,
`metal_cli.cpp`, `test_metal{,_ops}.cpp`, plus additive `Makefile` rules.
Zero deletions. Self-contained; touches no CUDA. Builds and passes on an M4.

> ⚠ **Gap: no documentation is staged.** Every PR branch is code-only — no
> README, no `docs/`. Metal would land invisible. Write a short README
> section (how to build, what tiers work, what the M4 numbers are) and add it
> to this PR before sending. Not our lab notebook — a paragraph and a table.

### 3c. `ToolCallStreamer` — +268/−0 in `api_common.h`

Split out of `pr8` per 1d. Purely additive, nothing existing calls it, so it
changes no behaviour on its own. Motivate it by what it enables: agent clients
render tool calls incrementally, and buffering means the user watches nothing
happen for the whole argument-generation window. `JsonQuoteContext` is the
part that makes it safe — a fragment boundary landing mid-escape would
otherwise put invalid JSON on the wire.

Verified: `api_common.h` compiles standalone; his own
`tools/test_openai_bridge.cpp` passes unchanged.

### 3d. `pr7-metal-serving` — +4,826/−0 of new files

`metal_server.cpp`, `stream_format.h`, `disk_snapshot_store.h`,
`snapshot_evict.h`, `test_metal_stream.cpp`.

Its entire dependency on shared files is **two things** — worth stating in the
body, because it is much smaller than the raw diff suggests:

- `api_common.h` + `initial_harness_prefix` (23 additive lines)
- `third_party/httplib.h` + the 7-line `Request::sock` patch

Separable from PR 6 on purpose: it is the HTTP layer, and he may want the
engine without our serving opinions.

---

## Objection handbook

Prepared answers, each backed by something already measured. Do not improvise
these — the evidence exists precisely so we don't have to.

| he says | answer |
|---|---|
| "The tie-break changes kernel behaviour; I'd have to re-bless anchors." | Measured: the published canonical md5 is identical across both arms and equal to your published value. The full kernel battery reports the same error values on both. It only differs where the old code was demonstrably wrong. |
| "The tokenizer leak doesn't matter — one long-lived instance." | Correct today, which is why it hides. Anything constructing tokenizers repeatedly leaks the vocab + merge table + special-token list each time. The copy-delete is the load-bearing half: without it, adding the destructor introduces a double-free. Three lines and a `delete`. |
| "20k lines is too much to review." | The seam is 730 lines and that's the only decision. 16k of the rest is a self-contained directory that cannot affect your build — it compiles only under a Darwin branch in the Makefile, and touches zero `.cu` files. Review the seam; the port is take-it-or-leave-it. |
| "I don't want to maintain a backend I can't test." | Say yes plainly: we maintain it, we run the gates, we carry the on-call. Offer a `CODEOWNERS` entry for `src/metal/` and a stated policy that a Metal break never blocks a CUDA release. |
| "Why is there a patched httplib in third_party?" | Use the wave-3a script. Never let this be discovered — ask first. |
| "Can you split the Metal PR further?" | Yes, and offer it before he asks: engine / shaders / tests are separable inside PR 6. |
| "Does this slow down or complicate the CUDA path?" | Measured, not asserted: built Metal against pristine `upstream/master` and every one of the 48 errors was a missing addition — zero CUDA dependencies. `blocks.cu`, `server.cu`, `test_kernels.cu`, `inspect.cpp` are not required by Metal at all. |

---

## Abort protocol

Each wave has a defined stop, and stopping is a normal outcome — not a
failure to argue harder.

- **Issue declined** → send wave 1 anyway (the four fixes stand alone), then
  stop. Keep the fork clearly labelled a downstream Metal port, keep pulling
  from him. Say this in the issue itself so the no is easy to give.
- **PR 5 declined** → stop at wave 2. Waves 3's branches stay staged; they
  cost nothing parked. Do not re-litigate the seam in a Metal PR.
- **httplib patch declined** → PR 7 does not ship as-is. Fallback is not
  "restructure into a content provider" (that breaks error status codes — see
  3a.3). Fallback is: hold PR 7, ship PR 6 alone, and wait on the
  cpp-httplib upstream issue.
- **Any single PR declined on its merits** → close it, record the reason in
  [DECISIONS.md](DECISIONS.md), and do not resubmit in another shape.

---

## Invariants — check before every `gh pr create`

- [ ] `LICENSE` appears in no PR. It carries our fork-additions copyright
      line, acknowledged in the 07-16 exchange, and must never be in a diff.
- [ ] No `logs/`, `experiments/`, `packaging/`, or `docs/metal/` in any PR.
- [ ] `git diff --name-only upstream/master..<branch>` matches the file list
      this playbook states for that branch. Anything extra is carry-over.
- [ ] Deletion count matches: 3 across the entire stack, and every one is
      accounted for (`loader.h` −1 enum line extended in place, `stream_split.h`
      −2 for `emit_head`'s new `bool` return).
- [ ] Every CUDA-touching PR names the toolchain it was verified on
      (nvcc 12.0.140, sm_86, RTX 3090) — and no longer claims we cannot
      compile CUDA, which was wrong.
- [ ] yukon etiquette if re-verifying: `nvidia-smi --query-compute-apps=pid
      --format=csv,noheader` immediately before launch, not at plan time; work
      in a throwaway clone; **never touch `~/projects/q27` on yukon**.

---

## After each send

Update the status table in [MERGE-BACK.md](MERGE-BACK.md) with the PR number
and outcome, and add a row to [DECISIONS.md](DECISIONS.md) for anything
declined — including *why*, because a parked item with a recorded mechanism is
what stops it coming back.
