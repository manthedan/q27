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

Shipped 2026-07-17 (commit 1adb0cd): unit battery + agentic battery +
trace gate ALL PASS on the deployed binary; wire dump shows a
get_weather call streaming as an opener chunk plus six argument
fragments concatenating to valid JSON. G5s gate updated to
production-shape accumulation (the old assertion crashed on the
opener's empty arguments).

## Review round (workflow autoreview, 2026-07-17 night)

17/17 candidates verified, 0 refuted. Four confirmed correctness
defects in this feature's span, all fixed same-night and proven both
ways
(sabotage builds against the pre-fix parser FAIL the new battery
cases):

1. **DONE-state byte drop (packed wrapper).** The streamer discarded
   everything after the first arguments object closed, silently losing
   any second call packed in one wrapper — a regression vs the
   buffered recovery chain. Fix: post-DONE bytes accumulate in
   `trail()` (the call object's own closing `}` swallowed once as
   framing); `close_tool` runs the trail through
   `parse_bare_tool_calls` and emits recovered calls whole; non-call
   trail surfaces as text; pure framing junk is discarded. Battery s7.
2. **Mode-3 not streamed.** The buffered path repaired
   `<content>`-tagged bodies via escape_content_tags; the streamer
   shipped them raw (client-side JSON parse failure, no recovery — an
   UNregistered trade). Fix: inline mode-3 in the streamer — a
   value-position `<content>` opens the string the model forgot
   (shape 1); an in-string `</content>` holds for the same
   one-non-ws-byte lookahead as mode-10 quotes and closes the string
   only before a JSON delimiter (shape 2); literal in-string
   `</content>` mid-value stays literal. Battery s8/s9/s9b at 1-byte
   cuts.
3. **Mode-10 lookahead × truncation repair (bare scan).** On a
   truncated call followed by prose, the lookahead re-escaped the
   real closing quote, swallowed the prose into the open string, and
   the truncation repair then "successfully" parsed one merged garbage
   command. Fix: the lookahead's premise (valid framing follows every
   real close) only holds for BALANCED segments — an unbalanced
   segment with mode-10 re-escapes rescans with the unconditional
   terminator and takes the pre-mode-10 repair path. Battery v16
   (chosen adversarially: a trailing second call re-synchronizes the
   old scanner via colon lookahead and does NOT discriminate; prose
   does).
4. **Keepalive blind spots.** The 5 s keepalive lived only in the
   per-token callback — queue wait (250 ms ticks) and prefill (tens of
   seconds cold at big contexts) stayed wire-silent past the 18 s
   stall window the fix was shipped for. Fix: each streaming handler's
   keepalive hoisted into a named lambda fired from BOTH the token
   callback and the liveness probe; `Runtime::run` now invokes the
   probe with `route_` UNLOCKED in the queue-wait loop (a stalled
   client's TCP backpressure must not block other requests' slot
   acquisition; cancel bookkeeping relocks first).

Registered residues from the review: (a) the in-string sanitizer state
machine now exists in three places (bare scan, scan_namedropped,
streamer) with two idioms — a shared char-level stepper is the
refactor, deferred (risk-heavy, no behavior delta); (b)
/v1/completions streaming carries NO keepalive by design — the replay
bench's TTFT reads first-byte, and a prefill keepalive would fake it;
(c) the MTP/non-chunked prefill path has no live() ticks, so no
keepalive there either (MTP is off on the serving line); (d) cleanups
landed: dead post-close_tool emit_tool deleted, parked
q27_matmul_q4_mm_h PSO now lazy-built (startup never compiles it),
quiet_q4_bench re-armed legs use Q27_METAL_GEMM_HALF_Q4 (a rerun under
the old knob benched float-vs-float), replay bench headline separates
trace-priced from cap-priced (corpus-derived, approximate) turns.
