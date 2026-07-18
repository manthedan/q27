# DS4-style native agent experiment

This directory tests the highest-risk boundary before importing DS4's full
agent control plane: can a C program own q27's C++/Metal engine directly,
without the HTTP server?

The architecture and proposed port are based on `antirez/ds4`'s native agent,
pinned for this experiment at commit
`80ebbc396aee40eedc1d829222f3362d10fa4c6c`. The upstream `ds4_agent.c` is
10,244 lines and is intentionally **not** copied wholesale yet. Its engine,
DSML protocol, KV payload store, browser, and distributed interfaces are
model-specific. Phase 0 establishes a narrow q27 C ABI first so the eventual
fork has a real target rather than a fake `ds4.h` compatibility layer.

See `THIRD_PARTY_NOTICES.md` for attribution.

## Phase 0: direct generation

`q27_agent.c` is a reduced DS4-style C control loop. The C++ adapter owns:

- q27 tokenizer and ChatML rendering;
- one `MetalEngine` and its shared mapping;
- reset/prefill and serial greedy generation;
- all exception containment at the C ABI.

The C process owns the transcript, terminal loop, streaming output, and
interrupt decision. A dedicated pthread opens, owns, drives, and closes the
engine. Submission deep-copies the complete message array; the UI thread then
drains a bounded owned event queue (`state`, binary `text_delta`, and exactly
one terminal). There is no server, socket, HTTP, SSE, or API-shape translation.

Build on macOS:

```sh
make build/q27-agent
./build/q27-agent MODEL.q27 MODEL.tok --prompt 'Reply with exactly: ready' \
  --no-think --max-tokens 32
```

Interactive mode omits `--prompt`; `:quit` exits. Machine-readable events use:

```sh
./build/q27-agent MODEL.q27 MODEL.tok --prompt 'Reply: ready' \
  --no-think --output-format jsonl
```

Every JSONL row carries monotonic `seq`, `command_id`, event `type`, state,
status, accounting, and exact `data_b64` bytes. Diagnostics remain on stderr.

### Deliberate limitation

Every turn re-renders and re-prefills the complete transcript. This is correct
for the Phase-0 control-path experiment, but it does **not** claim DS4's live
session/KV benefit. The adapter reports `phase0 full-prefill` after each turn
so benchmark output cannot be mistaken for an incremental harness result.

## First smoke result (2026-07-18)

On the M1 mixed tier, with no resident q27 process and no coordination hold:

- noninteractive direct generation returned exactly `ready` (54 prompt tokens,
  1 output token);
- one fresh process retained a two-turn C-side transcript and returned exactly
  `one`, then `two` (52/1 and 68/1 prompt/output tokens);
- both runs used `--no-think`, context 512, and no server process.

Follow-up worker/event smokes returned exactly `worker` and `events` through
the dedicated engine-owner thread. The JSONL smoke produced exactly
`state → text_delta → turn_done`; decoding `data_b64` yielded `events`.
`build/test_q27_agent_worker` uses a fake adapter to gate startup failure,
deep binary message ownership, monotonic lifecycle events, request rejection,
4094-event queue backpressure (5000 deltas), cancellation terminals, and
shutdown ordering without loading a model. Its ASan/UBSan leg is clean.

This establishes direct C → worker → C++ → Metal feasibility. It does not
establish incremental state reuse, tool execution, cancellation recovery, or
parity.

## Graduation gates

Phase 0 graduates only when a coordinated model window demonstrates:

1. a noninteractive answer without a server process;
2. a two-turn interactive transcript;
3. Ctrl-C returns cleanly and a subsequent fresh run succeeds;
4. emitted bytes match the direct Metal CLI/server greedy result for the same
   rendered prompt, modulo the documented API prose trimming convention;
5. ASan/UBSan-clean C-side transcript and output handling in a fake-adapter
   test (to be added before tool execution).

## Next port slice

After Phase 0, port from the pinned DS4 agent in this order:

1. ~~dedicated engine-owner worker and synchronous typed commands~~;
2. ~~queued UI events and a noninteractive JSONL event stream~~;
3. q27 `AgentSession` append/finalize contract;
4. file read/search/edit tools and bounded asynchronous shell jobs;
5. q27 `<tool_call>` parsing and constrained generation (not DSML);
6. transcript/session persistence using `Q27SNAP1` (not DS4 payloads);
7. compaction;
8. optional terminal UI and browser tooling.

The next slice is the q27 `AgentSession` contract: finalize the last emitted
token and append new tokens without reset, with fail-closed reset/re-prefill on
any token-prefix mismatch. Only then may the harness claim incremental prefill
or resident-session performance.
