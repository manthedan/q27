# Responses-API parity + Anthropic error-class split (residue round, 2026-07-17)

Closes the two registered residues from the agentic-parity round while the
Q4-port ship bench waits for a quiet machine.

## 1. /v1/responses full CUDA-style port (Codex CLI endpoint)

Today's Metal endpoint feeds RAW flattened text (no chat template) with a
tools_preamble hack and emits text-only output — no function_call items, so
a Codex agent loop cannot execute tools. Port src/server.cu:1441-1882
wholesale onto Runtime::run():

- Request mapping: `instructions` → system; `input` string or item array
  (message / function_call / custom_tool_call / function_call_output →
  tool_call_text / tool_response_text reconstruction; reasoning items
  dropped); `custom` freeform tools bridged to one-string-param functions;
  hosted tool types skipped, never rejected; consecutive same-role merge;
  `chatml_prompt(merged, tools, think)` — replacing the raw-text encode
  (kills the preamble hack; mini's fail-loud empty-input P1 is preserved as
  an explicit empty-messages check before the template renders).
- Output: StreamSplitter routing into reasoning / message / function_call /
  custom_tool_call items; arguments as JSON-encoded STRING; malformed
  wrapped calls surface as text; bare-call recovery runs EVEN WITH EMPTY
  tools (codex registers its shell tool as a hosted type — CUDA comment).
- Streaming: codex 0.143 item lifecycle (output_item.added +
  content_part.added before the first output_text.delta; done triplet on
  flush; one item open at a time), response.output_item.done per item,
  response.completed{response:{id, output, usage}} terminator.
- Keeps: per-API context-preflight 400 (context_length_exceeded,
  fatal-class for codex — correct), 4096 default max_tokens (env override
  rides), cancellation probes, snapshot hint.

## 2. Anthropic error-class split (+ OpenAI 500 twin)

Engine failures during generation currently answer 400
invalid_request_error everywhere; CUDA's convention is "400 is fatal to
codex, 500 retries — 500 on bugs", and the Anthropic envelope defines
api_error for them (the Metal STREAM path already emits api_error events —
only the pre-commit non-stream path misclassifies). Fix: Runtime::
EngineError wrapper; non-stream handlers wrap ONLY the runtime.run() call
(parse/validation/overflow all throw before it); anthropic_guarded maps
EngineError → 502-class Anthropic api_error 500, guarded maps it → OpenAI
{"error":{type:"api_error"}} 500. Known coarseness, recorded: the rare
post-restore "prompt exceeds context" inside run() rides the 500 class —
the normal overflow path preflights to the proper 400 long before.

## Pre-registered gates (live T2 server)

1. G8a non-stream: weather-tool request → output contains a function_call
   item, name correct, `arguments` is a JSON-encoded string.
2. G8b stream: full wire shape — output_item.added(message) +
   content_part.added precede the first output_text.delta;
   output_item.done carries the function_call; response.completed carries
   output items + usage.
3. G8c round-trip: input array with message + function_call +
   function_call_output → the model consumes the tool result (answer
   references it), no spurious new call required.
4. G8d custom tool: `custom` tool declaration → custom_tool_call item with
   bare-string `input`.
5. G9 controls: invalid JSON, empty input, oversize prompt still answer
   400 in their registered shapes (the split must not widen the 400 class);
   agentic battery G2-G7 green (no regression on the other endpoints).
   HONESTY NOTE: the 500 leg has no forced-engine-failure negative control
   (no failpoint on this path yet) — code-review-only, recorded as a weak
   gate like the EOS-rerun precedent.

Kill: none — this is a wiring port with a verbatim reference; failures are
bugs to fix, not levers to park.

## RESULTS (2026-07-17 midday, 24 GB M4)

Both work items landed; SIX codex rounds to clean.

**Implementation** — the /v1/responses port + EngineError split as
pre-registered, plus review-driven hardening:
- Round 1 (no P1): P2 text→think transitions corrupted output_index /
  item order (both routes now flush pending text BEFORE think
  accumulates, the TOOL-branch rule); P2 non-stream bare-call recovery
  double-emitted raw JSON + call; P3 a streaming engine failure left an
  unterminated lifecycle (item state + machinery hoisted above the try;
  the failure path closes open items, keeps the api_error event, and
  ends the turn with response.failed carrying last_error + partial
  output).
- Round 2 (no P1/P3): P2 a COMPLETE bare call committed at a transition
  flush was never recovered → recovery moved INSIDE flush_text
  (per-segment, tx=pre, model-order items).
- Round 3 (no P1): P2 truncation repair false-fired at mid-turn segment
  boundaries → parse_bare_tool_calls gains a defaulted allow_trunc_repair
  (api_common.h; every prior caller unchanged), transition flushes pass
  false; P2 malformed-wrapper raw committed with no recovery → folded
  into the text path (chat/Anthropic convention).
- Round 4 (no P1/P3): P2 the fold let tx=pre drop prose that followed a
  wrapper → malformed raw now commits as its OWN segment immediately.
- Round 5 (no P1/P3): P2 adjacent wrapped calls (the multi-call batch
  shape) folded into one tool_buf → **StreamSplitter emits an empty
  {TEXT,""} boundary segment between adjacent calls** (shared component:
  every handler on BOTH backends gains call separation; the canonical
  non-tool path is segment-identical; test_splitter() pins it); P2 the
  stream path lacked end-of-turn truncation repair → stream
  flush_tool(final) rescues via the repair path (beyond the CUDA
  reference, which commits the fragment; matches the non-stream rescue).
- Round 6: **CLEAN** (no P1/P2/P3).

**Gates (live T2 server, final binary):**
tools/responses_parity_gate.sh — G8a function_call item + arguments as a
JSON-encoded string; G8b(i) stream lifecycle ordering (added +
content_part.added before the first delta, completed terminator);
G8b(ii) output_item.done carries the function_call, completed carries
output + usage; G8c tool-result round-trip consumed, no spurious new
call; G8d custom_tool_call with bare-string input; G9 invalid JSON /
missing input / oversize prompt all answer 400 in their registered
shapes — ALL PASS (logs/q4port-20260717/responses-gates-p2fix6.log).
tools/agentic_parity_gate.sh G2–G7 ALL PASS — the 400 class is not
widened (logs/q4port-20260717/agentic-gates-p2fix6.log).
make test-cpu + test-metal green (incl. the new splitter suite).

**Honesty notes:** the 500 leg and the response.failed failure path have
no forced-failure failpoint (code-review only, as pre-registered);
natural-segment tx=pre trimming is the house convention (identical in
the chat/Anthropic handlers) — prose after a recovered call within one
natural segment is trimmed because parse_bare_tool_calls reports no
suffix.

**Operational incident (recorded):** two OOM crashes on this box during
the session, both = two 17 GB servers resident (wrapper-pid kill
orphaning the server; a concurrent agent launching without a health
check). Coordination protocol committed separately:
logs/q4port-20260717/COORDINATION.md + tasks/lessons.md.
