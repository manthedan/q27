# Merge-review triage — mac-mini codex pass over the merged tree (2026-07-16)

After merging origin/metal (the yukon Gate-0/upstream-CUDA stream, 33
commits) into the mini's stream, the whole incoming diff was codex-reviewed
(`codex exec review --base <pre-merge HEAD>`). Six P2s. Four were mechanical
and verifiable on the mini — fixed in this commit. Two are CUDA runtime
logic the mini can neither compile nor test (no nvcc/GPU) — **deferred to
yukon**, recorded here so they are not lost:

## Fixed in this commit (mini)

1. `Makefile`: `build/q27-server` and `build/q27-server-w8` now list
   `src/conductor.h` (server.cu includes it; incremental builds could go
   stale — the exact stale-binary class the ABI-tag lesson exists for).
2. `tools/yukon_regate_2026-07-16.sh`: build targets corrected to
   `build/q27 build/q27-server` (bare `q27 q27-server` have no rules —
   the staged re-gate would have failed before its comparison).
3. `tools/yukon_regate_2026-07-16.sh`: the canonical gate's exit status
   now fails the script; `DONE` says PASS/FAILED instead of being written
   unconditionally (a failed gate must not read as success).
4. `tools/official_tier_gates_2026-07-16.sh`: the EOS rerun gate aborts if
   the server never becomes healthy or either request fails/returns empty —
   two empty files cmp as identical (the vacuous-empty class the script's
   own header warns about).

## Deferred to yukon (CUDA runtime, needs compile + measurement)

5. **`src/engine.cuh:1985-1992` — DepthCtl fed a batching-trimmed cap.**
   With `Q27_MAXD=auto` at level 6/7, when the conductor trims a member
   whose original `gate_cap >= md_used`, the clamped `cap < md_used`
   reaches `DepthCtl::update`, which decays `fired_ema` on every such
   round — repeated trimmed rounds can demote the drafter even though it
   keeps hitting its ceiling (exactly the spurious evidence the adjacent
   comment says to avoid). Fix direction per codex: clamp only the
   histogram input, or preserve the original cap for the controller
   update. Needs a CUDA build + an auto-depth run to verify.

6. **`src/conductor.h:476-479` — fused union accepts mixed fast-head
   modes.** Two engines sharing a `DeviceModel` but with different
   `fast_head` settings pass the shared-model assertion, yet the union
   selects the output tensor from `es[0]` alone — other members get
   logits from the wrong head vs their solo path. Fix direction: require
   identical head modes when forming a union (assert), or exclude such
   members from fusion. Needs the CUDA lockstep tests.

Both deferred items sit in the continuous-batching path that
`tools/yukon_regate_2026-07-16.sh` re-gates; the 16-token byte-exact gate
would NOT catch either (single engine, no union, fixed depth), so they
need their own checks — noted for the yukon queue.
