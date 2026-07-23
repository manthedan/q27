# q27 metal-v0.6.1 — upstream parity: no-think profile, per-request thinking everywhere, --ctx auto, max_tokens 8192

Serving-behavior release. Brings the Metal server to functional parity with
upstream v0.4.0 (CUDA arm) and folds in the upstream tool-call hardening
series. **Two default behaviors change** — read the notes before upgrading.

## Behavior changes (defaults)

- **No-think is now the server default** (upstream v0.4.0 parity). Prompts
  render a closed empty think block and the model answers directly. The
  qwen36 think-forever pathology makes forced traces a serving hazard;
  upstream documents the same default for speed. Opt back in three ways:
  `--think` at boot, or per request on any API shape — `enable_thinking`
  (OpenAI/Qwen), `chat_template_kwargs.enable_thinking` (llama.cpp/GLM),
  `thinking: {"type": "enabled"|"disabled"}` (Anthropic, Claude Code's
  toggle). Thinking streams route through the pre-seeded splitter into
  `reasoning_content` / thinking blocks / reasoning items with a paired
  lifecycle. Give thinking requests enough max_tokens for trace + answer.
- **`--ctx auto` is the default** (was: constant 8192). The server sizes
  each slot's window from the measured free envelope (recommended working
  set minus weights and live allocations, bounded overcommit, activation
  reserve), split across `--slots`, floored at the legacy 8192, capped at
  262144. On the 24 GB M4 with the official tier: 8241 tokens × 2 slots.
  Explicit `--ctx N` overrides; `--ctx 0` is rejected.

## Also shipped

- Upstream merge `df34c67`: tool-drift recovery modes 10/11/12/12b,
  fence-skip guard, streaming stray-close strip, ChatML-injection
  sanitizer — all verified live on Metal call paths; upstream's host
  parser suites (`test_tool_drift`, `test_think_resolve`,
  `test_stream_split`) are wired into `make test-cpu`.
- Unified `max_tokens` default **8192** across completions/chat/messages/
  responses (was 256/256/1024/4096; clamped to the remaining window;
  explicit oversize still 400s; `Q27_METAL_MAX_TOKENS_DEFAULT` override).
- `--nll` token files are validated against the vocab with a diagnostic
  (the CUDA guard's twin).
- Release process: `docs/QA_BEFORE_RELEASES.md` v2 (risk tiers A/B/C) +
  `tools/release_gates.sh` runner (one command, dated evidence dir,
  ledger). This release ran **Tier B** (serving delta; no kernel/engine
  numerics changes, SHADER_ABI 13 unchanged).

## Gates (Tier B; evidence: /tmp/q27-v061-gates5 on the release machine)

`test-cpu` PASS (incl. upstream parser suites) · chunk-parity 384 sentinel:
widths 48/96 bit-identical · `agentic_parity_gate` ALL PASS (G2–G7) ·
`responses_parity_gate` PASS except G8d (recorded skip: qwen36 think-forever
verbosity on the custom-tool prompt; the shape path is unit-covered and
G8d passes intermittently — it passed under the no-think default during
bring-up) · `trace_gate` ALL PASS · auto-ctx smoke PASS (resolves
consistently, serves). yukon/CUDA-side legs: N/A (trigger files untouched
in the serving delta; host unreachable).

Autoreview (codex): parity patch clean (0.91); `--ctx auto` clean after
one fix round (0.86). The no-think-profile + splitter pre-seed changes
were gate-driven (Tier B failures → root cause → fix → full-green rerun).

## Known limitations

- G8d (Responses custom tool within 400 tokens) remains a model-verbosity
  lottery under greedy; pass `--think`-free direct prompts or raise the
  budget. The `/v1/responses` endpoint now honors `enable_thinking` (it
  did not at v0.6.0).
- The default window on 24 GB + official tier is conservative (8241 × 2
  slots) by design; explicit `--ctx` sizes past it at your own OOM risk.
- Weights are not bundled: https://github.com/manthedan/q27#weights
