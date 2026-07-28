# PR playbook (audited 2026-07-27)

Runbook for the prepared merge-back branches. The evidence and architecture
rationale live in [MERGE-BACK.md](MERGE-BACK.md). Everything named here is
pushed to `origin` (`manthedan/q27`). Nothing has been sent upstream.

## Rule 0: refresh before every send

The old queue proved that a prepared PR can become obsolete within days.
Before every issue or PR:

```bash
git fetch upstream
git log --oneline metal..upstream/master
git diff --check upstream/master..<branch>
git diff --stat upstream/master..<branch>
```

If upstream moved under the branch, rebase and rerun the branch's own gate.
Record the exact tip tested. A result from an earlier tip is evidence, not a
substitute for the final pre-send run.

## Rule 1: serialize the Metal stack

Cross-fork PRs cannot use a contributor branch as the base in the upstream
repository. The prepared stack is therefore serialized:

```text
wave 1 PR 9 -> upstream/master
upstream/master -> PR 5 -> PR 6 -> PR 7
```

Open PR 6 only after PR 5 and PR 9 merge, then rebase PR 6 onto upstream so
its temporary PR 9 dependency disappears. Open PR 7 only after PR 6 merges,
then rebase PR 7 onto upstream. Never open a stacked branch early and ask the
maintainer to mentally subtract its parent.

## Waves

| wave | contents | gate |
|---|---|---|
| 0 | upstream issue | first send |
| 1 | CUDA 12 compat, tokenizer ownership, argmax ties, stream boundary | independent of Metal answer |
| 2 | backend seam | maintainer expresses interest in a multi-backend q27 |
| 3 | Metal core, then Metal serving | parent merged; httplib question answered before serving |

## Wave 0: issue

Send [upstream-issue-draft.md](upstream-issue-draft.md), not an older copy.

```bash
gh issue create --repo signalnine/q27 \
  --title "Interested in a Metal backend merge-back?" \
  --body-file docs/metal/upstream-issue-draft.md
```

The measured Metal-only stack, excluding standalone PR 9, is **36 files,
+25,293/-91**. Do not repeat the old "zero `.cu` files" claim: PR 6 core
touches no CUDA, but PR 7 serving includes a deliberate `src/server.cu`
OpenAI-parity change.

## Wave 1: four standalone fixes

Create each PR against `master` from the named branch.

```bash
gh pr create --repo signalnine/q27 --base master \
  --head manthedan:<branch> --title "<title>" --body-file <file>
```

### `pr1-cuda12-compat` (`7c45d7c`)

Scope: 1 file, +8/-2, `src/server.cu`.

Title:

```text
fix: CUDA 12.0 compat — nvcc rejects lambda-captured structured bindings
```

Lead with the reproduction: nvcc 12.0.140 on RTX 3090 / `sm_86` rejects three
lambda captures of structured bindings in pristine upstream; named tuple
references compile on the same toolchain without changing behavior.

Pre-send gate: throwaway clone on yukon, `NVCC=/usr/bin/nvcc`, remove the
unsupported `sm_120` gencode, compile pristine upstream and final branch tip.

### `pr-tokenizer-lifetime` (`380d56c`)

Scope: 3 files, +113/-15.

Title:

```text
tokenizer: make PImpl ownership construction-failure-safe
```

The destructor, deleted copies, and constructor-local RAII are one fix. The
important regression is not only the normal lifetime leak: a parse exception
must free the partially built `Impl` too.

Gate:

```bash
make build/test_tokenizer
./build/test_tokenizer --selftest
```

Required output includes `tokenizer failure lifetime: PASS`.

### `pr3-argmax-tiebreak` (`eefd494`)

Scope: 4 files, +86/-7.

Title:

```text
argmax: resolve exact and signed-zero ties to the lowest index
```

The PR body should contain:

1. Exact-value tie behavior: lowest index, matching CPU and Metal.
2. Signed-zero contract: `-0.0f` and `+0.0f` are IEEE-equal and must not be
   ordered by their sign bit.
3. Earlier hardware evidence from `dc05889`: upstream failed 3/8 synthetic tie
   cases, patched tip passed 8/8; full `test_kernels` numerical output matched;
   canonical md5 stayed `a2982c5197c627551b27d76a0a94b220`.
4. Final-tip evidence from `eefd494` on nvcc 12.0.140, `sm_86`, RTX 3090:
   both signed-zero orders pass in plain and fused paths, `test_kernels`
   reports `ALL PASS`, and the same canonical md5 is preserved.

Ship `tools/test_argmax_tie.cu`. The final tip is gated; rerun its model-free
harness, full kernel battery, and canonical anchor only if Rule 0 changes the
tip before sending.

### `pr9-stream-boundary` (`2b2b74d`)

Scope: 3 files, +109/-34.

Title:

```text
stream_split: emit a boundary between adjacent tool calls
```

Explain the observable failure: `</tool_call><tool_call>` emits no separator,
so a consumer buffering one TOOL segment can merge two calls or lose the raw
text of a malformed second call.

Gate:

```bash
make build/test_stream_split
./build/test_stream_split
```

All full, bytewise, empty-think, text-between, and flush cases must pass.

## Wave 2: backend seam

Branch: `pr5-backend-seam` (`3e43d4b`).

Measured scope: 11 files, +1,145/-19. No Metal implementation and no `.cu`
change. The branch adds `backend.h`, sampling and packed-model contracts,
format documentation, and validation fixtures in `inspect`, `loader`, and
sampling tests. It also adds the required four-line `engine.cuh` guard that
rejects non-Q8 token embeddings before CUDA's Q8-only row lookup can launch.

This is the decision point. If the maintainer wants q27 to remain CUDA-only,
accept the answer and stop the Metal stack.

Local gates:

```bash
make build/inspect build/test_sampling build/test_tokenizer
./build/test_sampling
make test-inspect
```

The tokenizer binary's artifact-backed suite currently inherits an upstream
`billing-header cch normalize` failure with the available model fixture; do
not misreport that baseline failure as a PR 5 regression or as a green gate.
The PR-specific sampling and packed-format gates pass. Sampling coverage rejects
NaN, positive infinity, and fully masked rows across full/candidate paths while
preserving mixed negative-infinity masks.

On the prepared tip, nvcc 12.0.140 compiled `build/q27` and
`build/test_kernels`; the host-built inspect fixtures passed, and the invalid
embedding fixture hit the new CUDA guard before launch. Re-run that exact gate
if Rule 0 changes the tip.

## Wave 3a: Metal core

Branch: `pr6-metal-core` (`36ff230`), currently stacked on PR 5 and carrying
PR 9 as a temporary test dependency.

The prepared ancestry is 15 files, +16,809/-35. After PR 9 merges and PR 6 is
rebased, its intended review scope returns to 13 files, +16,701/-2: the
Objective-C++ backend, shaders, engine, CLI, README, KL support, device tests,
and real-artifact engine-contract gate. It touches no `.cu` file.

After PR 5 and PR 9 merge:

```bash
git fetch upstream
git rebase -i --onto upstream/master 3e43d4b pr6-metal-core
# In the rebase todo, drop 3fe89c7 (the temporary PR 9 commit).
git diff --exit-code upstream/master..pr6-metal-core -- src/stream_split.h tools/test_stream_split.cpp
git diff upstream/master..pr6-metal-core -- Makefile # inspect: only Metal targets remain
git push --force-with-lease origin pr6-metal-core
make test-metal build/q27-metal build/test_metal_engine_contracts
./build/q27-metal /path/to/official.q27 /path/to/model.tok --validate-only
./build/q27-metal /path/to/bonsai.q27 /path/to/model.tok --validate-only
./build/test_metal_engine_contracts /path/to/official.q27 /path/to/bonsai.q27
```

Required evidence on Apple Silicon:

- `build/q27-metal` compiles and `--validate-only` accepts both official and
  Bonsai artifacts
- Bonsai stays on serial float-activation prefill; forced chunking rejects
- cancelled batched MTP clears speculative position and resident logits
- Metal matvec and post-commit poison PASS
- decode, FP16/turbo3 attention, GQA KV reuse, GDN, and chunked prefill PASS
- installed shader-path precedence and exact ABI-line validation PASS
- profiled explicit-batch overflow rejects before the 2049th operation

## Wave 3b: httplib decision

Ask in the issue thread before opening PR 7. The vendored change is seven
additive lines exposing a borrowed request socket before the first response
write. The behavior argument comes first:

- queue wait and prefill can take minutes before headers
- a disconnected client otherwise burns a slot and GPU work
- moving work into a content provider commits HTTP 200 before overload or
  engine errors are known
- call sites are intentionally unguarded, so removal fails loudly at compile
  time rather than silently disabling cancellation
- the server uses its own POSIX liveness probe, not httplib internals

If the maintainer declines, ship PR 6 alone and hold PR 7.

## Wave 3c: Metal serving

Branch: `pr7-metal-serving` (`779ef04`), stacked directly on PR 6.

Measured stage scope: 18 files, +7,462/-85. There is no standalone streamer PR:
`ToolCallStreamer` is integrated here because this server is its caller.

The branch intentionally includes shared serving changes:

- `src/metal/metal_backend.h` / `.mm`

- `src/api_common.h`
- `src/server.cu`
- `third_party/httplib.h`
- bridge/integration tests

Do not describe it as new-files-only or zero-CUDA. Its architectural boundary
is HTTP serving, not file novelty.

PR 7 contains no stream-boundary delta relative to PR 6. After PR 6 lands,
run `git rebase --onto upstream/master 36ff230 pr7-metal-serving`, then verify
that remains true.

Required Apple Silicon and host gates:

```bash
make test-metal \
  build/q27-metal-server \
  build/test_openai_bridge \
  build/test_stream_split \
  build/test_sampling \
  build/inspect
./build/test_openai_bridge
./build/test_stream_split
./build/test_sampling
make test-inspect
bash tools/build_chat_completions_integration.sh
make test-metal-contracts MODEL=/path/to/model.q27
make test-metal-recovery MODEL=/path/to/model.q27 TOKENIZER=/path/to/model.tok
```

Observed on the prepared tip: every command above passes on Apple M4. The
chat-completions extraction check reports byte-for-byte matches for
`build_prompt()` and `handle()` and all integration cases pass. Live Anthropic
smoke requests also prove 400s for missing/empty/wrong-typed messages and bad
tool choices, named-tool forcing, `none` suppression, and single-call
enforcement for `disable_parallel_tool_use`.
The recovery gate requires an oversized body to return HTTP 413, reconstructs
the backend after a committed command failure, streams a split noncanonical
tool-call head through the Responses route, and requires a completed
structured function call. It also prewarms an Anthropic `tool_choice:none`
prefix and requires the corresponding live request to report a nonzero exact
prefix hit. The model contract gate proves scheduler chunk prefill invalidates
stale logits until a forward pass refreshes them.
The final review additions also prove that forced tool choices cannot disappear
at the exact token limit, externally replaced snapshot metadata is re-peeked,
hosted Responses shell tools receive model-facing `exec_command` /
`write_stdin` schemas, and fractional token limits return HTTP 400 while
integral JSON floats remain valid. Final branch autoreview reports no
accepted/actionable findings.

Required final-tip CUDA gate on yukon, because this stage changes
`src/server.cu`:

```bash
NVCC=/usr/bin/nvcc \
NVCCFLAGS='-O2 -std=c++17 -gencode arch=compute_86,code=sm_86 -Xcompiler -Wall' \
make build/q27-server
```

The current `779ef04` tree compiles on nvcc 12.0.140, `sm_86` after applying
the planned PR 1 compatibility prerequisite. Run the command again after the
final upstream rebase and record that exact tip. The host extraction gate
supplements this compile; it does not replace compiling the shipped CUDA
server.

## Objection handbook

| objection | answer |
|---|---|
| "The argmax change moves anchors." | The final signed-zero tip preserved the published canonical md5 and full kernel numerics on nvcc 12.0.140. Repeat only if Rule 0 changes the tip. |
| "One tokenizer lives for the process lifetime." | Normal lifetime is only half the bug. A throwing constructor leaked its partially built PImpl; the regression repeats malformed construction and checks resource stability. |
| "The Metal diff is too large." | The decision is staged: +1,145/-19 seam, +16,701/-2 core after PR 9 lands, +7,462/-85 serving. Review or decline each layer independently. |
| "I cannot maintain Metal." | We own the Metal path and gates; a Metal break should not block a CUDA release. Offer `CODEOWNERS` for `src/metal/`. |
| "Why patch httplib?" | Seven additive lines expose the only pre-header liveness signal. Ask before review; if declined, hold serving and land core alone. |
| "Does Metal complicate CUDA?" | PR 6 core does not touch `.cu`. PR 7 does touch `server.cu` for shared OpenAI parity; it is disclosed and can be declined independently. |

## Abort protocol

- Issue declined: wave 1 may still proceed; stop the Metal stack.
- PR 5 declined: stop. Do not repackage the seam inside Metal code.
- httplib declined: hold PR 7; PR 6 may still land.
- Any individual PR declined: record the reason in `DECISIONS.md`; do not
  resubmit the same change under another title.

## Invariants before `gh pr create`

- [ ] Fetch upstream and inspect new commits immediately before the send.
- [ ] `LICENSE`, logs, experiments, packaging, and `docs/metal/` are absent.
- [ ] Branch tip, diff stat, and file list match this playbook after rebase.
- [ ] The exact tip named in the PR body is the tip that passed its gate.
- [ ] CUDA-touching PRs name nvcc 12.0.140, `sm_86`, RTX 3090.
- [x] PR 3 final signed-zero tip has fresh CUDA evidence; reset if rebased.
- [ ] PR 7 final tip compiles `build/q27-server` on yukon and passes the
      byte-for-byte extracted chat-completions integration gate.
- [ ] PR 6 opens only after PR 5 and PR 9 merged; its rebased diff contains no
      stream-boundary patch.
- [ ] PR 7 parent is merged upstream, httplib is agreed, and its diff contains
      no duplicate boundary patch.

## After each send

Update the status table in [MERGE-BACK.md](MERGE-BACK.md). Record declined
items and reasons in [DECISIONS.md](DECISIONS.md).
