# Merge-back plan (audited 2026-07-27)

This supersedes the earlier merge-back drafts. The prepared branches have been
rebuilt from `upstream/master` (`c2d2116`), audited, tested where the local
hardware permits, and pushed to `origin` (`manthedan/q27`). Nothing has been
sent upstream. The first send remains the issue, not a PR.

## Decision

Ask whether the maintainer wants the Metal backend. Send four independent bug
fixes regardless of that answer. If the answer is yes, serialize the backend
stack: backend seam, Metal core, then Metal serving.

Do not present the fork's raw `metal` diff as the contribution. It includes
logs, experiments, packaging, and private project history. The prepared
branches below are the review surface.

## Audit corrections

The branch audit found and repaired issues that made the previous plan unsafe:

1. `src/engine.cuh` had once dropped 162 lines of upstream CUDA work. That was
   already restored before branch preparation; every prepared branch is now
   based on current `upstream/master` rather than `metal`.
2. The first tokenizer fix added a destructor but did not make a throwing
   constructor safe. `pr-tokenizer-lifetime` now owns `Impl` with RAII during
   construction, deletes copies, and includes a repeated-failure lifetime gate.
3. The argmax patch did not define IEEE signed-zero ties. The final branch
   canonicalizes `-0.0f` and `+0.0f` to the lowest index in both argmax paths
   and ships a model-free CUDA gate.
4. The adjacent-tool-call boundary bug was bundled with a much larger streamer.
   It is now the standalone wave-1 branch `pr9-stream-boundary` with direct
   bytewise and flush regressions.
5. The backend seam accepted malformed packed shapes, overflowing sizes, and
   NaN sampling inputs. `pr5-backend-seam` now validates those contracts and
   includes CPU fixtures.
6. Metal command failures could leave engine state reusable after a partially
   committed operation. `pr6-metal-core` now preserves or poisons state at the
   ownership boundary and tests post-commit failure.
7. The old `pr7-metal-serving` was not actually stacked on the repaired PR 6;
   diffing it against PR 6 showed it reverting loader, sampling, and engine
   fixes. It was rebuilt directly on `pr6-metal-core`, so it no longer reverts
   parent work and contains no boundary patch. Its `metal_backend.h/.mm` delta
   is intentional: serving recovery needs an atomic poison flag plus a
   `healthy()` query, and that lower-layer dependency is disclosed below.
8. The standalone `pr8-toolcall-streamer` was speculative because upstream did
   not call it. It is retired. The streamer now lives in PR 7, where the Metal
   server actually uses it.
9. The Metal Anthropic route accepted missing or empty messages, ignored
   `tool_choice`, and could expose undeclared model calls as executable tools.
   PR 7 now validates both message endpoints, normalizes `auto`/`none`/`any`/
   named-tool choices, enforces the selected registry and
   `disable_parallel_tool_use`, and has unit plus live endpoint evidence.
10. Metal no-copy uploads passed logical file lengths to an API that requires
    page-aligned lengths. PR 6 now tracks the page-rounded mapped span for
    whole mappings and EOF views; both the sub-page fixture and the 17.7 GB
    official artifact path pass.
11. Disk snapshots recorded whether logits were resident, but the engine did
    not enforce that flag after restore. PR 6 now tracks logits validity across
    reset, disk and RAM restore, prefill, decode, and teacher forcing; PR 7's
    model contract gate proves stale logits reject before a forward pass and
    become usable after one.
12. GPU top-k could sort a negative-sign NaN below its candidate threshold and
    bypass full-row validation. The Metal kernel now emits the existing
    fallback sentinel for either NaN sign, with direct positive- and
    negative-NaN regressions.
13. Installed Metal binaries searched the process working directory before the
    build-time shader path, so an unrelated `src/metal/q27_kernels.metal` could
    override the packaged shader. PR 6 now gives the baked path precedence,
    keeps intentional overrides explicit, and matches the ABI tag as a whole
    logical line; a hostile-working-directory regression passes.
14. Per-dispatch profiling had a fixed 2048-operation counter buffer without a
    safe explicit-batch overflow contract. PR 6 now rejects the 2049th
    operation before encoding it, preserving batch abort semantics, and tests
    the boundary.
15. Scheduler `prefill_chunk()` advanced recurrent and KV state without
    invalidating the previously resident output row. PR 6 now clears logits
    validity after every successful chunk; PR 7's model contract gate proves
    derivation rejects until a new forward pass refreshes the row.
16. The Responses streaming fallback flushed a deviated tool head as soon as
    the first fragment triggered fallback, fragmenting recoverable mode-6/7/8
    bodies into text. PR 7 now buffers raw TOOL bytes until the channel closes;
    the live recovery gate streams a split wrong-order body and observes a
    completed structured function call.
17. Backend-neutral sampling rejected NaNs but allowed `+inf` and fully
    `-inf` rows to collapse through zero weights into a deterministic argmax,
    potentially selecting a forbidden token. PR 5 now rejects positive
    infinity and rows with no finite value while preserving mixed `-inf`
    masking; all four full/candidate distribution paths are covered.
18. Adding packed dtypes made a non-Q8 `token_embd.weight` loadable even
    though every CUDA embedding path decodes Q8 rows. PR 5 now rejects that
    model contract before any CUDA kernel launch, and the inspect fixture
    enforces the documented Q8 embedding policy.
19. Bonsai artifacts have no MTP layer and must not use the
    activation-quantized chunked-prefill path. PR 6 now defaults them to the
    serial float-activation path, rejects attempts to enable chunking, and
    validates both the official and Bonsai artifacts on Apple M4.
20. Batched MTP committed an accepted speculative prefix before delivering it
    to the stream sink, so cancellation could leave reusable engine state
    ahead of the client-visible prefix. PR 6 now resets that speculative state
    on cancellation; a real-model gate proves position and logits are cleared.
21. cpp-httplib's default request-body limit is unbounded. PR 7 now caps bodies
    at 1 MiB..64 MiB according to configured context, and the live recovery
    gate requires an oversized request to return HTTP 413.
22. Automatic context sizing re-statted the model pathname instead of charging
    the artifact already mapped by the engine. PR 7 now uses the resident
    mapping size, so an atomic deployment rename cannot undercount weights.
23. Anthropic prefix prewarming ignored `tool_choice`, producing snapshots
    that ordinary serving could never hit. PR 7 now runs the same choice and
    registry normalization as `/v1/messages`; the live gate prewarms a
    `tool_choice:none` request and requires a nonzero exact prefix hit.
24. Forced tool choices were bypassed when decoding ended exactly at the token
    limit. PR 7 now rejects a forced request with no eligible call regardless
    of whether generation ended on EOS or `max_tokens`; non-stream and SSE
    integration regressions pass.
25. Snapshot metadata was cached by pathname even though multiple server
    processes atomically replace the same token-key file. PR 7 now keys cached
    metadata by device and inode, re-peeks external replacements, and tests
    resident-to-stale and stale-to-resident publication.
26. A hosted Responses `shell` declaration admitted `exec_command` and
    `write_stdin` outputs without giving those function schemas to the model.
    PR 7 now injects the model-facing Codex shell definitions while retaining
    hosted output validation and forced/allowed selection semantics.
27. Fractional `max_tokens` / `max_output_tokens` values were silently
    truncated. PR 7 now accepts integral JSON numbers such as `4096.0` but
    returns the native HTTP 400 invalid-request envelope for values such as
    `1.9`; a live server smoke confirms both paths.

## Prepared branches

All hashes below are pushed to `origin`.

| branch | base | tip | measured diff | verification |
|---|---|---:|---:|---|
| `pr1-cuda12-compat` | upstream | `7c45d7c` | 1 file, +8/-2 | previously reproduced on nvcc 12.0.140: upstream fails, patch compiles |
| `pr-tokenizer-lifetime` | upstream | `380d56c` | 3 files, +113/-15 | `build/test_tokenizer --selftest`: UTF-8, GPU gate, API shapes, repeated construction failure all PASS |
| `pr3-argmax-tiebreak` | upstream | `eefd494` | 4 files, +86/-7 | nvcc 12.0.140, `sm_86`, RTX 3090: signed-zero harness PASS, full CUDA kernel battery ALL PASS, canonical md5 unchanged |
| `pr9-stream-boundary` | upstream | `2b2b74d` | 3 files, +109/-34 | `build/test_stream_split`: adjacent/full, bytewise, empty-think, text-between, and flush cases PASS |
| `pr5-backend-seam` | upstream | `3e43d4b` | 11 files, +1,145/-19 | CPU sampling and packed-format fixtures PASS; nvcc 12.0.140 builds the affected CUDA executable/tests and rejects a non-Q8 embedding before launch |
| `pr6-metal-core` | PR 5 + PR 9 temporary dependency | `36ff230` | 15 files, +16,809/-35 | Metal CLI, device/poison/profile/operator gates, concat-overflow regression, official+Bonsai validation, Bonsai prefill policy, and batched-MTP cancellation invalidation PASS on Apple M4 |
| `pr7-metal-serving` | PR 6 | `779ef04` | 18 files, +7,462/-85 | Full host/Metal gates PASS, including HTTP 413 bounding, external snapshot replacement, forced-call token-limit enforcement, hosted-shell prompt schemas, integral token-limit validation, OpenAI single-call enforcement, and Anthropic prewarm/live prefix parity; final autoreview is clean; `server.cu` also compiles on nvcc 12.0.140 when the planned PR 1 prerequisite is applied; final rebased-tip CUDA compile remains a pre-send gate |

After PR 9 merges and its temporary dependency disappears from PR 6, the
Metal-only stack is **36 files, 25,293 insertions, 91 deletions**. That is the
honest number for the issue. The current prepared ancestry including PR 9 is
38 files, +25,401/-124. Do not count the standalone fix twice.

It is no longer correct to claim "zero `.cu` files": PR 6 core touches no CUDA,
but PR 7 deliberately changes `src/server.cu` for shared OpenAI serving parity.
Disclose that distinction rather than hiding it in the aggregate.

`pr6-metal-core` temporarily carries the exact PR 9 boundary change so its
descendant can be tested before wave 1 lands. PR 7 is directly based on PR 6
and has no boundary delta. Before opening PR 6, merge PR 9 upstream, rebase PR
6, and verify the temporary commit disappears; then rebase PR 7 onto the
merged PR 6.

## CUDA evidence and remaining pre-send gate

The CUDA oracle is `yukon`: RTX 3090, nvcc 12.0.140, `sm_86`. Evidence on
2026-07-27 established:

- PR 1: pristine upstream fails with three lambda-captured structured-binding
  errors; the patch compiles on the same toolchain.
- PR 3 at `dc05889`: upstream fails 3/8 synthetic tie cases, the patch passes
  8/8, the full `test_kernels` battery reports identical numerical results,
  and the canonical generation md5 is unchanged:
  `a2982c5197c627551b27d76a0a94b220` on upstream, the patch, and the published
  anchor.
- PR 5 at `3e43d4b`: nvcc 12.0.140 compiles `build/q27` and
  `build/test_kernels`; the host-built inspect fixture gate passes and a
  non-Q8 embedding fixture is rejected before any CUDA kernel launch.

The final PR 3 tip `eefd494` was then rerun on the same host: both signed-zero
orders pass in plain and fused paths, `test_kernels` reports `ALL PASS`, and
the 128-token canonical md5 remains
`a2982c5197c627551b27d76a0a94b220`.

PR 7 changes `src/server.cu`. Its current `779ef04` tree plus the planned PR 1
compatibility prerequisite compiles `build/q27-server` on yukon with nvcc
12.0.140, `sm_86`. After the final upstream rebase, compile the exact rebased
tip again and rerun the extracted chat-completions integration gate; the
prerequisite simulation does not replace that final-tip check.

Yukon etiquette: query active compute processes immediately before launch,
use a throwaway clone, cap build parallelism, and never touch
`~/projects/q27` there.

## Stage 1: independent fixes

These stand on their own and may be sent whether or not Metal is wanted.

### CUDA 12.0 compatibility

`pr1-cuda12-compat`: replace lambda-captured structured bindings in
`server.cu` with named tuple references. Behavior is unchanged; the patch is
toolchain compatibility for nvcc 12.0.

### Tokenizer ownership

`pr-tokenizer-lifetime`: make the PImpl ownership explicit and
construction-failure-safe. The destructor, deleted copy operations, and RAII
construction must ship together; splitting them recreates either the leak or a
double-free/partial-construction hazard.

### Argmax tie contract

`pr3-argmax-tiebreak`: exact IEEE-equal maxima, including opposite signed
zeros, resolve to the lowest index in plain argmax and argmax-with-margin. The
final signed-zero tip preserved the published non-zero anchor; repeat the CUDA
gate only if Rule 0 changes that tip before sending.

### Adjacent tool-call boundary

`pr9-stream-boundary`: `</tool_call><tool_call>` must emit a boundary segment so
a consumer buffering one TOOL segment cannot merge two calls or lose a
malformed second call's raw text.

## Stage 2: backend seam

`pr5-backend-seam` is the actual architecture decision. It adds the
backend-neutral interfaces and packed dtype/sampling contracts without Metal
implementation code or `.cu` changes. Its only CUDA-engine edit is a four-line
`engine.cuh` guard required by the new dtype set: `token_embd.weight` must stay
Q8 before the existing Q8-only row kernel can launch. It also documents the
packed format and adds overflow, truncation, alignment, shape, and finite-logit
validation.

If the maintainer wants q27 to remain CUDA-only, stop here. Do not smuggle the
seam in through a later Metal PR.

## Stage 3: Metal core

`pr6-metal-core` is stacked on PR 5 and temporarily includes PR 9 solely so
the prepared stack is testable before wave 1 lands. After PR 9 merges, rebase
PR 6 and verify that dependency disappears, restoring the intended 13-file,
+16,701/-2 Metal core scope. The core contains the Objective-C++ backend,
Metal shaders, engine, CLI, README, KL support, device tests, and the
real-artifact engine-contract gate. It touches no `.cu` file.

## Stage 4: Metal serving

`pr7-metal-serving` is stacked on PR 6. It contains the HTTP server, snapshots,
stream formatting, incremental tool-call streaming, shared OpenAI compatibility
work, recovery gates, the backend health/atomic-poison support used by server
reconstruction, and the vendored httplib field described below.
Unlike PR 6, PR 7 is not a new-files-only change. Its shared surface is
intentional and must be reviewed as such:

- `src/api_common.h`: parsing, tool-choice, usage, and incremental streaming
- `src/metal/metal_backend.h` / `.mm`: backend health query and atomic poison

- `src/server.cu`: shared OpenAI compatibility behavior
- `third_party/httplib.h`: seven-line borrowed request socket
- integration and bridge tests

The server layer remains separable from the Metal engine: if the maintainer
wants the engine but not these serving opinions, PR 6 can land alone.

## The `third_party/httplib.h` question

The vendored cpp-httplib change is exactly seven additive lines: a borrowed
`Request::sock` field initialized to `INVALID_SOCKET`, populated from the
connection stream. The Metal server may spend minutes in queue wait and
prefill before its first response byte; without a pre-header liveness probe it
can hold a slot for a disconnected client.

The server uses a self-contained POSIX probe rather than
`httplib::detail::is_socket_alive`. The patch remains compile-loud: call sites
are unguarded, so removing the field on a dependency refresh fails at build
time rather than silently disabling cancellation.

Ask about this before PR 7. If declined, ship PR 6 alone and hold PR 7 rather
than restructuring work into a content provider that would commit HTTP 200
before overload or engine errors are known.

## Send order

1. Open the issue from `upstream-issue-draft.md`.
2. Send the four independent fixes after refreshing/rebasing each one.
3. If the maintainer is interested, send PR 5.
4. After PR 5 and PR 9 merge, rebase PR 6 with old PR 5 tip `3e43d4b` as the
   replay boundary, explicitly drop temporary PR 9 commit `3fe89c7`, verify no
   boundary delta remains, rerun its gates, send.
5. After PR 6 merges and the httplib question is answered, rebase PR 7 with
   old PR 6 tip `36ff230` as the replay boundary, verify it still contains no
   boundary patch, rerun every serving/device gate, send.

## Pre-send invariants

- [ ] Fetch upstream immediately before every send; stale evidence is not a
      campaign-wide exemption.
- [ ] Each branch diff contains only the files listed for that stage.
- [ ] `LICENSE`, logs, experiments, packaging, and `docs/metal/` appear in no PR.
- [x] PR 3 final tip `eefd494` has fresh CUDA harness, kernel battery, and
      canonical-anchor evidence; reset this item if the branch is rebased.
- [ ] Every CUDA-touching PR names nvcc 12.0.140, `sm_86`, RTX 3090, and the
      exact tip tested.
- [ ] PR 7 final tip compiles `build/q27-server` on yukon and passes the
      extracted chat-completions integration gate.
- [ ] PR 6 is opened only after PR 5 and PR 9 merged, was rebased to upstream,
      and no boundary patch remains in its diff.
- [ ] PR 7 is opened only after PR 6 merged, the httplib question is answered,
      and its diff against current upstream contains no duplicate boundary patch.
- [ ] Any declined item is recorded in `DECISIONS.md` with the reason and is not
      resubmitted under another label.
