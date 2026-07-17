# Metal server agentic parity — pre-registration (2026-07-17)

the operator's go 2026-07-17 ("Go for the parity pass") on the external review's
seven agentic blockers, all source-confirmed, plus the eighth found in
verification: the OpenAI chat endpoint (pi's API) also returns no structured
tool_calls — the model's <tool_call> XML arrives as plain content text, so
pi's tool loop cannot execute reads/edits at all. README wording is out of
scope per the operator ("you can ignore the readme changes").

## Scope

All in `src/metal/metal_server.cpp` (+ `stream_format.h` dedup, Makefile
deps, a new live gate script). The CUDA server (`src/server.cu`) is the
reference implementation; the shared machinery already lives header-only in
`src/api_common.h` (anthropic_msgs, anthropic_tools_json, chatml_prompt,
normalize_cc_billing_header, ctx_limit_error_message, parse_tool_call,
parse_bare_tool_calls, StreamSplitter via stream_split.h) — this round is a
wiring port, not a design round. Verified before starting: the tokenizer's
apply_chat_template and encode(chatml_prompt(...)) render identical tokens
(strip_chatml ≡ strip_ctrl; <think>/<|im_start|> matched as added/control
tokens inside encode), so switching the Metal endpoints to the shared
renderer is byte-safe. One deliberate behavior adoption rides along:
chatml_prompt's minimal default system prompt when the client sends none
(the CUDA over-refusal fix, 2026-07-13).

1. **/v1/messages — full CUDA parity** (reference server.cu:1087-1439):
   anthropic_msgs/anthropic_tools_json request mapping (incoming tool_use /
   tool_result / thinking blocks reconstructed to model markers; billing
   header normalized inside anthropic_msgs), StreamSplitter output routing
   (thinking blocks with q27-local signature, tool_use content blocks,
   input_json_delta streaming, wrapper-less bare-call recovery), stop_reason
   "tool_use", context preflight answering 400 + "prompt is too long"
   (ctx_limit_error_message — CC's compact-now signal) before slot claim.
2. **/v1/messages/count_tokens** (reference server.cu:1072-1085): CPU-only,
   counts exactly what /v1/messages prefills for the same body.
3. **/v1/chat/completions — beyond-CUDA** (the CUDA chat endpoint is
   text-only by design; its structured paths are /v1/messages and
   /v1/responses): incoming assistant.tool_calls arrays and role:"tool"
   messages mapped to tool_call_text/tool_response_text (consecutive
   same-role merged, as the CUDA responses handler does); outgoing
   structured message.tool_calls / streaming delta.tool_calls with
   finish_reason "tool_calls"; <think> routed to reasoning_content
   (llama.cpp convention — also closes the registered think-leak residue on
   this endpoint); context preflight with code context_length_exceeded.
4. **/v1/responses**: context preflight only (codex 400 shape). Structured
   function_call items on this endpoint stay a registered residue — no
   Codex traffic on this box; the CUDA reference exists when needed.
5. **/v1/completions**: unchanged (raw continuation endpoint, no template).

## Pre-registered gates (all must pass to ship; any fail → fix or report)

Build/unit: `make build/q27-metal-server` + `make test-metal` +
`build/test_tokenizer` green (test_tokenizer already covers the api_common
mapping on CPU; test_metal_stream keeps covering the SSE shapes after the
Utf8Gate dedup into api_common.h).

Live gates (tools/agentic_parity_gate.sh against the T2 server, ctx 131072,
--suffix 32, port 8213 — greedy decode):
- **G2 count_tokens**: input_tokens > 0 and EXACTLY equals /v1/messages
  usage.input_tokens for the same tooled body.
- **G3 Anthropic tool round**: non-streaming reply has a tool_use block,
  name == get_weather, input parses with a city key; stop_reason
  "tool_use". Streaming twin emits content_block_start(tool_use) and
  input_json_delta events.
- **G4 Anthropic round 2**: history with tool_use + tool_result → 200,
  non-empty text block, stop_reason end_turn (model consumed the result).
- **G5 OpenAI chat tool round**: non-streaming message.tool_calls[0]
  .function.name == get_weather, finish_reason "tool_calls"; streaming
  emits a delta.tool_calls chunk; round 2 (assistant.tool_calls +
  role:"tool" result) → 200 with content text.
- **G6 overflow contract**: >max_prompt prompt → HTTP 400, body contains
  "prompt is too long" on /v1/messages.
- **G7 billing normalization**: two /v1/messages bodies identical except
  the cch= stamp → second response q27_prefix_hit == input_tokens (full
  prefix-cache hit proves the stamp was canonicalized; run as a
  consecutive pair, cache capacity is 1).
- **G8 pi smoke**: non-interactive pi run against provider q27-metal
  executes a real read-tool loop end-to-end (the retry the operator planned,
  scripted).

No timing claims this round; all legs contention-tolerant. Server restart
(same launch line, new binary) is part of the pass per the operator's go.

## RESULTS (2026-07-17, 24 GB M4, T2 @ 131072/--suffix 32 on :8213 — logs/agentic-parity-20260717/)

**SHIPPED. All pre-registered gates PASS on the final binary in one clean
run** (gates-final.out): G2 count_tokens == usage.input_tokens (200) · G3/G3s
tool_use block + input_json_delta + stop_reason tool_use, both modes · G4
tool_result round-trip consumed ("The weather in Paris right now is 14
degrees C, sunny, with light wind.", end_turn) · G5/G5s message.tool_calls
and full-wire-shape delta.tool_calls chunk, finish_reason tool_calls,
round-2 via assistant.tool_calls + role:"tool" · G6 400 + "prompt is too
long: 140020 tokens > 131039 maximum" · G7 cch-varied billing pair full
prefix hit 539/539. **G8: pi executed a real read-tool loop end-to-end**
(pi-smoke.out) — the exact retry that was structurally impossible the night
before.

Unit gates: test-metal green (incl. test_metal_stream after the Utf8Gate
dedup into api_common.h), test_tokenizer 12 PASS / 0 FAIL on the real .tok.

Gate-authoring find (not a server bug): curl -d defaults to
application/x-www-form-urlencoded and httplib caps THAT content type at 8 KB
(CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH) → 413 before our handler.
Real clients send application/json (uncapped). The gate script now sends the
proper header; anyone curl-testing large prompts needs it too.

Codex round: no P1. Adopted all four findings — P2 Anthropic endpoints now
answer errors in Anthropic's envelope (anthropic_guarded; overloaded_error
at 503, invalid_request_error at 400); P2 count_tokens invalid-JSON same;
P3 per-endpoint max_tokens defaults now mirror CUDA (messages 1024,
completions/chat 256, responses 4096 — was a uniform 128); P3 the streaming
OpenAI gate now validates the full tool_calls chunk wire shape.

Post-rebase note: the mini's lane landed mid-round (E2 per-engine GQA
partials, SHADER_ABI 10, its own /v1/responses tools-preamble hardening);
after rebasing onto it the two /v1/responses changes compose (mini's
preamble prepend feeds my context preflight), test-metal, the full gate
battery, and the pi smoke were re-run green on the merged binary
(gates-postmerge-note.out).

Registered residue: /v1/responses structured function_call items (CUDA
reference exists; no codex traffic on this box); error-class split on
/v1/messages (engine failures still map to invalid_request_error 400 rather
than api_error 500 — needs a throw-type split in Runtime::run); the
<think>-leak residue is CLOSED on /v1/messages (thinking blocks) and chat
(reasoning_content) but /v1/responses still flattens; CUDA's chat endpoint
could adopt openai_msgs for the same beyond-parity tool support at the next
CUDA merge.
