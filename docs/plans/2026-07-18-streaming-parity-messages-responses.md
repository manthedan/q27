# Streaming parity: /v1/messages + /v1/responses incremental tool-call argument deltas — pre-registration (2026-07-18)

Registered residue from `2026-07-17-incremental-tool-call-streaming.md`
("follow-up rounds: /v1/messages input_json_delta fragments, /v1/responses
function_call_arguments deltas"). The ship/kill discipline below mirrors
that round's. Operator-directed sequencing: pre-register first, then build.

## The gap

`/v1/chat/completions` already streams tool-call argument fragments
incrementally via `q27::ToolCallStreamer` (api_common.h), proven through
four review rounds on live pi traffic. `/v1/messages` and `/v1/responses`
still buffer a wrapped `<tool_call>` body WHOLE and emit one block at
segment close:

- **/v1/messages**: `emit_tool_block` emits `content_block_start` (empty
  `input`) + ONE `input_json_delta` carrying the full re-serialized
  arguments + `content_block_stop`, all at segment close.
- **/v1/responses**: `push_call` emits a complete `function_call` /
  `custom_tool_call` item at flush, no `function_call_arguments` deltas.

For a multi-minute write call (~2,000 tokens at ~8 tok/s) this is a silent
stretch every agent client renders as a hang — the same UX lie the
chat/completions round closed on pi's wire. Production Anthropic streams
`input_json_delta` partial JSON; production OpenAI responses-streams
`function_call_arguments` deltas. Wire parity is the standing value.

## Mechanism (reuse the proven streamer, change only the emitter)

The same `q27::ToolCallStreamer` state machine (HEAD/ARGS/DONE/
INVALID_DONE/FALLBACK), the same inline mode-3/mode-5/mode-10 sanitizer,
the same `trail()` packed-call recovery. Only the chunk shape differs per
endpoint:

- **/v1/messages**: on `opened(name)` emit `content_block_start`
  (`tool_use`, id, name, `input: {}`). On each ARGS fragment emit
  `content_block_delta` with `delta.type = "input_json_delta"`,
  `partial_json = fragment`. On close emit `content_block_stop`. This
  replaces the single full-args `input_json_delta` with N fragment deltas
  whose concatenation is parse-EQUAL to the buffered arguments.
- **/v1/responses**: on `opened(name)` emit `response.output_item.added`
  (`function_call`, call_id, name, empty arguments). On each ARGS fragment
  emit `response.function_call_arguments.delta` with `delta = fragment`.
  On close emit the `response.output_item.done` with the complete
  arguments. Custom-tool (`custom_tool_call`) names keep whole-item
  emission (their input is a plain string, not streamed JSON) — recorded
  deviation.

FALLBACK (deviant head / mode-6/7/8) and bare (wrapper-less) calls are
UNCHANGED: buffered parse_tool_call + recovery chain, raw byte-exact. The
mid-stream cancellation, keepalive, and liveness-probe machinery is already
shared and stays.

## The pre-registered trade (restated, unchanged from the chat round)

Streaming commits the call at head-parse: a valid-head call whose body ends
broken reaches the client broken (their parse errors, the agent loop
retries with the error in context). That is production behavior — the
model's bytes are the model's bytes. The high-frequency drift modes (5, 10,
3) are repaired INLINE; residual exposure is the rare structurally-
unbalanced body. Bare calls keep full end-of-turn recovery. Non-streaming
endpoints unchanged.

## Adjacency to the "not a valid tool call" watch (explicit)

This change touches the wire format on the endpoints the unreproduced
client report concerns. Per the operator (2026-07-18): there is NO live pi
traffic and no live server to protect — breaking changes are fine, the
:8213 server may be torn down at will, and the quiet-gap / never-restart-
during-a-live-turn rules are suspended. The trace stays armed during the
battery so any "not a valid tool call" can be pulled by boot + tms; if such
a report recurs, the first diagnostic is whether the failing request
consumed a fragmented tool call the client could not reassemble.

## Gates (pre-registered, both directions, exit codes)

- **G1 — unit (test_tokenizer.cpp api_common battery, extended).** For the
  new emitter lambdas: well-formed wrapped call streams opener + N
  fragments whose concatenation is parse-EQUAL to the buffered arguments
  (byte equality NOT required — the buffered path re-serializes);
  multiline-content and in-string-quote bodies sanitize inline to valid
  JSON; mode-6 head falls back raw byte-exact; truncated ARGS reports
  incomplete; two adjacent calls stream as two independent blocks. Sabotage
  arm: the pre-change buffered emitter must FAIL the "fragments
  concatenate" assertion (proves the gate can fail).
- **G2 — live wire-shape battery (agentic_replay / G5-class).** For each of
  /v1/messages and /v1/responses: opener block carries id/name/empty
  arguments; N ≥ 1 fragment deltas concatenate to valid JSON equal to the
  non-streaming endpoint's arguments for the same frozen prompt; terminal
  stop_reason / finish is `tool_use`/`tool_calls`; full agentic battery +
  `tools/trace_gate.sh` green on the deployed binary.
- **G3 — non-streaming parity invariant.** The non-streaming
  /v1/messages and /v1/responses paths produce byte-identical arguments to
  before the change (they share the buffered emitter, which must not
  change). Sabotage arm: a change that alters buffered output fails this.

## Ship / kill line

- **SHIP:** G1 + G2 + G3 ALL PASS on the deployed binary; a real wrapped
  write call streams its arguments as ≥2 `input_json_delta` fragments on
  /v1/messages and ≥2 `function_call_arguments` deltas on /v1/responses;
  non-streaming paths byte-identical. (Deploy is unconstrained — no live
  traffic; the server may be torn down for testing at will.)
- **KILL:** the emitter cannot produce parse-EQUAL fragments without
  diverging from the buffered sanitizer (i.e. streaming and buffered would
  emit different JSON for the same bytes), or the live battery shows a
  client-visible reassembly break the chat round did not. On KILL, the
  buffered behavior stands and this residue re-parks with the mechanism
  recorded.

## Cost / risk

Emitter-only change inside two handlers that already have the streamer,
splitter, keepalive, and recovery wired. No engine/GPU change. The risk is
wire-format, addressed by the gates + quiet-gap deploy + armed trace.
Estimated under a session, plus battery time.
