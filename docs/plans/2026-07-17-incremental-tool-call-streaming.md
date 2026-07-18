# Incremental tool-call argument streaming — pre-registration (2026-07-17 night)

The operator's call, from live pi traffic: "production APIs stream their
tool calls as they are writing them." They do — OpenAI streams
`delta.tool_calls[].function.arguments` string fragments, Anthropic
streams `input_json_delta` partial JSON — and wire parity with
production behavior is this project's standing value. Our servers
buffer a wrapped `<tool_call>` body WHOLE and emit one block at close:
a multi-minute silent stretch for a big write call (~2,000 tokens at
~8 tok/s), which every agent client renders as a hang. Tonight's
keepalives stop the disconnects (shipped, live); they do not stop the
UX lie. The StreamSplitter already delivers TOOL-channel bytes
incrementally — buffering is purely the handler's choice.

## Mechanism: ToolCallStreamer (api_common.h, shared with CUDA at next merge)

Per TOOL segment run, a small state machine:

- **HEAD**: accumulate until the call head parses —
  `{"name": "<name>", "arguments":` followed by an object opener
  (tolerating the mode-9 missing quote on the arguments key). On
  match: report `opened(name)`; the handler emits the production-shape
  opener chunk (id + name + empty arguments). If the head exceeds a
  bound or deviates (mode-6/7/8 shapes: name-dropped, alias-named,
  shell-wrapped), fall back to FALLBACK.
- **ARGS**: incremental brace/string scanner over the arguments object
  bytes — the SAME scanner semantics as parse_bare_tool_calls, applied
  streaming: mode-5 control-char escaping and mode-10 in-string quote
  re-escaping happen inline (a quote holds until the next non-space
  byte decides terminator vs literal — one-byte lookahead, never more).
  Sanitized bytes stream out as argument fragments as they arrive.
  Depth returning to zero ends the object; trailing bytes before the
  wrapper close are discarded (matches the buffered parser).
- **FALLBACK**: nothing was streamed; the raw body goes through
  today's buffered parse_tool_call + recovery chain unchanged.
- **finalize** (wrapper close / flush): DONE = clean; mid-ARGS =
  truncated call — remaining held bytes flush and the call closes
  unbalanced (see trade below); HEAD/FALLBACK = buffered path.

## The pre-registered trade (what is deliberately given up)

Today a wrapped call is validated BEFORE any tool_calls chunk exists;
a malformed body folds into text + end-of-turn recovery, invisibly.
Streaming commits the call at head-parse: a valid-head call whose body
ends broken reaches the client broken (their parse errors, the agent
loop retries with the error in context). That is exactly production
behavior — the model's bytes are the model's bytes — and the two
highest-frequency drift modes (5: literal newlines, present in every
multiline write; 10: verbatim shell quotes) are repaired INLINE by the
streaming sanitizer, so the residual exposure is the rare
structurally-unbalanced body. Bare (wrapper-less) calls keep the full
end-of-turn recovery chain — they never stream. Non-streaming
endpoints unchanged.

## Scope and gates

This round: **/v1/chat/completions streaming** (pi's wire, the burned
customer). Registered residue, same streamer, follow-up rounds:
/v1/messages input_json_delta fragments, /v1/responses
function_call_arguments deltas, CUDA server adoption at next merge.

- Unit (test_tokenizer.cpp, the api_common battery): well-formed call
  streams name + fragments whose concatenation is parse-EQUAL to the
  buffered parse (byte equality is not required — the buffered path
  re-serializes); multiline-content and in-string-quote bodies
  sanitize inline to valid JSON; mode-6 head falls back with raw
  preserved byte-exact; truncated ARGS reports incomplete; adjacent
  calls stream as two independent calls.
- Live (agentic battery): G5s wire shape holds — opener chunk carries
  id/type/name, argument fragments concatenate to valid JSON,
  finish_reason tool_calls; full battery + trace gate green.
- Deploy via the quiet-gap watcher (never yank a live pi turn).

## RESULTS

(pending)
