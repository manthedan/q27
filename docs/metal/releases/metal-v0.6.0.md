# q27 metal-v0.6.0 — sampled MTP, fenced-body agent writes, FP1 + Ratatui TUI, think budget

First **prebuilt macOS arm64** release of the Metal line: `brew install` no
longer compiles anything at install time (previously a full C++ build on the
user's machine). Ships the Metal CLI, server, native agent, the new Ratatui
TUI (`q27-tui`), benches, and the corpus tokenizer. Built with
`MACOSX_DEPLOYMENT_TARGET=13.0`; requires Apple silicon, macOS 13+.

## Shipped since metal-v0.5.2 (68 commits, two feature branches)

- **Sampled serving + sampled MTP** (`agent-fenced-body-write` lane):
  mask-verified sampling plumbing, exact rejection walk, end-to-end
  distribution gate (`tools/metal_sampled_mtp_dist.sh`, TV floor vs greedy),
  integer-contract request validation for `top_k`/`seed`, greedy path bitwise
  when no sampling flags are set (rule zero, distribution-gated).
- **Fenced-body agent write protocol**: full-file writes ride CommonMark
  fences with EOS-gated recovery, transport-LF normalization, tick-gated
  legacy unwrap (≤3-tick transports only), bare tool-call JSON fail-closed by
  shape (package.json content stays legal), longer-outer-fence escaping
  taught to the model. 8-round branch review to clean (9 fixes, 2 documented
  rejections).
- **FP1 frontend protocol v1 + Ratatui TUI** (`agent-tui` lane, merged as a
  union with the MTP decode architecture — 24 codex review rounds to
  zero-P1): NDJSON control plane (queue/cancel/session ops, correlated
  request ids on every event, ordered quit with drain-before-bye and a
  post-quit rejection barrier, cancel/failure transcript rollback, strict
  JSON gate with surrogate-pair and UTF-8 validation, chunked history replay
  with think-split, gapless visible sequences via quiet internal dequeues);
  Ratatui client with pending-prompt reconciliation, terminal-control
  sanitization, binary chunk placeholders, 5-minute quit drain, 0600 logs.
  Classic linenoise path kept as fallback (`Q27_AGENT_UI=classic`).
- **Think budget** (`--max-think-tokens`): force-closes an over-long thinking
  span and continues the answer instead of abandoning the turn; MTP is gated
  out of budget-armed spans; the tracker never observes post-tool-engagement
  tokens, so literal `<think>` in file payloads is content, not a span
  (live-verified).

## Gates (evidence: /tmp/q27-release-v0.6.0/ on the release machine)

- Build: zero C/C++ warnings (9 pre-existing Rust dead-code warnings);
  `make test-cpu`, `make test-metal`, tokenizer self-tests: ALL PASS.
- Canonical: `--chunk-parity 384 --ctx 4096` widths 48/96 bit-identical to
  width 12 (exit 0); `snapshot_gate.sh` ALL PASS; `failpoint_gate.sh` PASS
  (N=4). yukon CUDA byte gate: N/A — trigger files (loader.cpp, tokenizer,
  api_common.h, stream_split.h, tool_preamble.h) untouched in the delta;
  yukon also unreachable. `ckpt_gate.sh`/`constrain_gate.sh` are CUDA-side
  scripts (nvcc/`[gen]` log lines); Metal coverage via multislot G7 prefix
  legs, test-cpu toolconstrain, and the agentic battery.
- Live server: `agentic_parity_gate.sh` ALL PASS (G2–G7);
  `responses_parity_gate.sh` PASS except G8d (recorded skip below);
  `trace_gate.sh` ALL PASS (5 API families, 7 errors, 2 recoveries,
  4 cancels, monotonic tms). Cancel smoke covered by the trace gate's
  forced-cancellation legs. pi smoke: a week of pi dogfooding against this
  stack (fenced writes, think budget, FP1 lifecycle) plus the G2–G7 tool
  loops.
- Self-launching: `multislot_gates.py` greedy ALL PASS (G1–G7);
  `multislot_gates.py mtp` PASS (live speculation counters: 391 rounds,
  945 committed); `suffix_burst_gates` PASS (benign periodic-greedy WARN).
- Quality: official 8K NLL **1.8349** — exactly the recorded release
  baseline (Δ 0.000%, band ±0.5%). Decode/prefill frontier spots:
  decode 4.25 tok/s at ~2k (ctx 4k, fp16 KV, resident greedy) and 12.07 tok/s wall on the teacher-forced 8K pass, both measured on a USER-ACTIVE machine (contended; quiet-machine references from the chronicle: 11.5–11.75 tok/s short-ctx decode class). 4k/8k decode frontiers deferred to a quiet window.
- Mechanics: SHADER_ABI 13 → 13 (kernels untouched); docs sweep spot-checks
  pass (README env/endpoint tables current); LICENSE carries upstream
  attribution; CHECKSUMS.md5 present per published artifact.

## Recorded skips / known limitations

- **G8d (Responses custom tool within 400 output tokens)**: the qwen36
  ternary under greedy rides the known think-forever pathology on this
  prompt — it reasons past the token budget instead of calling the tool
  (the same pathology `--max-think-tokens` exists to bound). Not a protocol
  regression: the custom_tool_call shape path is unit-covered and unchanged;
  the leg passed/fails purely on model verbosity. The `/v1/responses`
  endpoint also does not yet honor per-request `enable_thinking:false`
  (chat completions does) — tracked.
- TUI backlog (P2 cosmetics, recorded in METAL_PROGRESS): Ratatui live
  stream merges tool-call markup into the answer; `/new` does not drain
  queued prompts; residual envelope-state cosmetics.
- Weights are not bundled: fetch a pack per
  https://github.com/manthedan/q27#weights.
