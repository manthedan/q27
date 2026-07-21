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
- exact token-ledger validation, reset/append prefill, and serial greedy generation;
- final emitted-token ingestion and all exception containment at the C ABI.

The C process owns the transcript, durable manifest, compaction policy,
terminal loop, streaming output, and interrupt decision. A dedicated pthread
opens, owns, drives, snapshots, restores, and closes the engine. Submission deep-copies the complete message array; the UI thread then
drains a bounded owned event queue (`state`, exact `prefill_progress`, binary
`text_delta`, and exactly one terminal). There is no server, socket, HTTP, SSE,
or API-shape translation.

The friendly source-checkout entry point builds and starts B1 with context
32768, adaptive output, automatic tools, and workspace `$PWD`:

```sh
make agent
```

Use `./packaging/bin/q27 agent t2` to select another local/installed pack;
`Q27_AGENT_CONTEXT`, `Q27_AGENT_WORKSPACE`, and `Q27_AGENT_SESSION` override
its defaults. The low-level invocation remains available:

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
status, accounting, and exact `data_b64` bytes. `prefill_progress` reports the
initial cached-prefix jump, fixed 96-new-token completed GPU boundaries, and a
final update before `text_delta`; `cached_tokens + prefill_tokens` is completed
prompt ingestion. Interactive text mode renders the same progress in place on
stderr, leaving stdout clean. Diagnostics remain on stderr.
A watchdog terminal is emitted as `type=generation_stalled`, `status=stalled`,
and exact data `generation stalled`.
Tokenizer counts and Q27SNAP1 save/load use the same internal command/event
boundary. JSONL exposes validated loads and emits a distinct `session_done`
only after durable manifest publication (or a rejected `session_done` on
publication failure); internal counts, compaction work, and the snapshot-only
phase of save are suppressed so no terminal can falsely claim durable
publication early. Successful process exit remains the outer noninteractive
verdict.

## Durable sessions and compaction

`--session FILE` loads an existing session or creates and autosaves one after
each completed top-level turn:

```sh
./build/q27-agent MODEL.q27 MODEL.tok --no-think --session work.q27agent
```

The mode is exact: context size, thinking mode, automatic-tool mode, tokenizer,
and the complete tool preamble/protocol must match on resume. Selection handles
use the `selection-handles-v1` protocol boundary, so older tool-enabled durable
sessions fail closed and must be recreated. A small binary
`Q27AGT2` manifest stores the binary-safe
transcript, owner-pinned tokenizer SHA-1 identity, full snapshot SHA-256
digest, and
the random basename of one immutable `Q27SNAP1` engine blob. Both files are
owner-only regular files (`0600`). Snapshot content is fsynced
and renamed first; the CRC-checked manifest is then published with its own
fsync/rename/directory-fsync transaction. The old snapshot is removed only
after the new manifest is durable, so a crash can leave an unreferenced blob
but cannot publish a manifest pointing at a partial one. If the manifest rename
succeeds but directory durability is uncertain, both old and new snapshots are
retained and the command fails loudly. Snapshot basenames are namespaced to
the manifest filename, so copying or renaming a manifest alone is rejected
rather than creating an untracked shared reference. A stable owner-only lock
file serializes publication, readers retain a shared lock until the engine has
pinned and validated their referenced snapshot, and the manifest's previously
loaded snapshot name is compared under the exclusive writer lock so a stale
concurrent writer fails instead of overwriting a newer turn. The parent directory must be owned by the user and
not group/other writable; symlink file opens are rejected.

Resume pins one snapshot descriptor across SHA-256 verification, metadata
inspection, and restore, then validates tokenizer identity, the model artifact,
KV configuration, complete snapshot layout,
exact token metadata prefix, Metal position, and—when present—the one pending
token against resident logits before restoring the private token ledger. Any
mismatch fails closed and resets uncertain engine state. Save also
validates the resident generated-token ledger against a fresh transcript
render; if decoded bytes retokenize differently, it re-prefills the canonical
closed-message prefix before publication. Snapshot files contain the full
active Metal state and can be large; persistence is
therefore opt-in rather than the default. `:save` forces a save when
`--session` is configured.

Prompt size is counted on the owner thread before every generation. At 75% of
the configured context by default, the harness summarizes older complete root
turns and retains the four most recent root turns. Configure this with
`--compact-at`, `--compact-keep`, and `--compact-tokens`; `:compact` requests it
immediately. An assistant tool call and its following `<tool_response>` are an
indivisible part of one root turn. For write/edit, the intervening harness-authored
raw-payload request and assistant payload are part of that same group, so
compaction never retains only part of a file transaction. Summary generation
cannot call tools. A hidden, bounded
acknowledgement establishes a new exact generated-token prefix before the
recent tail is appended, allowing the compacted transcript to be snapshotted
immediately without pretending old KV state still matches. If no safe cut fits,
the original transcript remains unchanged and generation fails rather than
dropping history.

## Bounded native tools

Interactive users can exercise the control plane directly:

```text
:read relative/path
:search relative/path literal needle
:shell command
```

The C API additionally exposes exact binary atomic write/edit requests.
Generation and tools share one-command admission and the same bounded owned
event queue
(`tool_running → tool_output* → tool_done`). No engine callback launches a
tool.

File tools are rooted at a directory descriptor pinned when `--workspace`
(default `.`) opens; later pathname replacement cannot retarget it. They reject
absolute paths, `..`, empty components, and every symlink component, accept
regular files up to 8 MiB, and cap output at 256 KiB. Write creates only a
nonexistent path, stages content beneath a pinned ACL-free `0700` directory,
and atomically publishes a fixed owner-only, ACL-free, non-executable `0600`
file; it requires an owner-owned destination parent that is not group/other
writable, rejects granting parent ACLs, and never overwrites. Search/edit use
cancellation-aware linear matching. Edit requires exactly one old-byte match,
uses bounded nonblocking lock acquisition, preserves mode, and atomically
exchanges the new file with the destination (Darwin/Linux), validating the
displaced inode and complete bytes before deleting it or atomically rolling
back. Shell is manual by default and becomes model-triggerable only under the
explicit global `--auto-tools` opt-in. It starts in the workspace, combines
exact stdout/stderr bytes, caps output at 256 KiB, and caps
runtime at 60 seconds. To make that bound cover the complete process tree, the
job denies process creation (Darwin sandbox policy / Linux seccomp): builtins
and a final external `exec` work, while pipelines, background jobs, and
forking commands fail. The sole process group is killed/reaped on timeout,
cancellation, or output overflow.

## Opt-in model tool calls

Automatic tools are disabled unless `--auto-tools` is present. **This flag
grants model output local side effects.** File helpers remain workspace-bounded,
but shell uses the workspace only as its current directory and can access any
path allowed to the local account; the no-fork policy is process containment,
not a filesystem or prompt-injection sandbox.

```sh
./build/q27-agent MODEL.q27 MODEL.tok --auto-tools --workspace ./project \
  --prompt 'Read README.md and summarize it'
```

Opt-in mode inserts the fixed
read/search/write/overwrite/edit/edit_selection/shell JSON schemas into the
system message and enables greedy grammar masks after the model emits
`<tool_call>`. The grammar allows only a registered name, a strict JSON object,
and a complete `</tool_call>` closer. For non-body tools generation stops at
that closer. Body tools (write, overwrite, edit, edit_selection) **continue**
free-decoding so a markdown-fenced body can follow in the same turn. Mask
exhaustion, illegal grammar state, truncation, malformed JSON, unknown
arguments, multiple calls in one turn, tool-shaped bodies, or missing fences
fail closed without a filesystem side effect.

Generated file and replacement bytes are deliberately excluded from JSON
(escaping f-strings and multi-line source was a prior death spiral). Instead:

```
<tool_call>{"name":"write","arguments":{"path":"hello.py"}}</tool_call>
```python
print(f"hi {name}")
```
```

`write` creates a new file (path only + fence). `overwrite` creates or replaces
the entire file (path only + fence). Literal `edit` takes path plus a unique
nonempty `old` string capped at 512 bytes, with the replacement in the fence;
use `overwrite` for full-file rewrites. When the response budget permits,
successful automatic `read` and `search` responses also append up to 32 short
selection handles for exact line-aligned content ranges; the final LF or CRLF
terminator stays outside each range so ordinary line replacement preserves it
automatically. The bounded process-local ledger behind each handle owns the
workspace-relative path, selected bytes, byte offset, and SHA-256 of the
complete file. `edit_selection` takes path plus one handle, with replacement
in the fence. Unknown, expired, path-mismatched, or stale handles fail before
mutation; preflight and final publication both revalidate the full-file digest,
exact range, and selected bytes. Handles expire when their 256-entry ledger
slot is reused or the process restarts, so a resumed model must read/search
again rather than trusting transcript-only IDs.

There is **no** second raw-payload chat turn for write/edit: that path re-emitted
tool calls as file content under greedy Bonsai. Bodies that look like
`<tool_call` or `{"name"` are rejected. Empty fenced bodies remain valid for
empty-file create and delete-style edits.

The final mutating tool repeats all validation under its atomic race defenses;
preflight is control-plane preparation, not authority.

The engine callback only publishes binary text deltas. After the generation
terminal, the C control thread strictly parses the completed call and submits a
new owned tool command through `q27_agent_worker_submit_tool`. It captures the
exact output and verifies terminal byte accounting. Before execution it lowers
the tool's output cap to a conservative one-byte-per-token budget derived from
the terminal prompt accounting plus the full generated assistant byte length
(never trusting emitted-token segmentation across re-render), remaining
context, next `max_tokens`, and fixed response-framing reserve; if even the
framing cannot fit, it refuses
before a mutating tool runs. It then appends an explicit `<tool_response>` user
turn with exit/flag metadata and starts the next resident generation.
`--max-tool-rounds` bounds this loop (default 8, maximum 64). JSONL shows
separate monotonic command IDs for generation, tool execution,
and follow-up generation; a post-terminal `selection_handles` event carries
the exact model-visible annotation with the parent tool command ID and its own
monotonic event sequence. `turn_done.tool_call_complete` identifies the
semantic call terminal. The worker's `eos_reached` field distinguishes natural
completion from a generation stopped at its bound and is used internally for
raw payload authority, although internal raw terminals are not emitted to
JSONL. Unless `--max-tokens` is
supplied explicitly, `--auto-tools` raises the per-generation bound from 512
to 4096 when context is at least 8192 tokens, giving a complete raw file
payload practical room to reach EOS. Smaller contexts retain 512 because tool execution also
reserves framing and the next generation. An explicit numeric `--max-tokens`
always wins. `--max-tokens auto` instead derives each generation from the
fresh canonical prompt count, retaining 512 tokens in automatic-tool mode
(256 without tools), and caps any one generation at 16384 tokens. Before a
side effect, tool-response framing and another 512-token continuation margin
are reserved separately. The limit is recomputed after every tool response and any
compaction rewrite. Automatic shell execution
remains subject to the single-process sandbox described above.

### Resident session contract

Every turn still re-renders the complete transcript for validation. Reuse is
allowed only when the prior exact prompt-plus-generated token ledger is a
stable prefix and `MetalEngine::position()` matches its encoded length. The
adapter finalizes each emitted token into resident state when context permits,
then ingests only the newly rendered ChatML suffix. Any token or position
mismatch resets and re-prefills; cancellation, runtime error, or watchdog stall
invalidates the ledger so the next valid request also resets.

Generation has conservative allocation-free no-progress bounds: 64 consecutive
empty decoded tokens, 256 consecutive ASCII whitespace bytes, 128 identical
tokens, or a 2–8-token cycle spanning 256 tokens. The triggering token is not
streamed. Earlier streamed bytes remain an irreversible preview, but the turn
fails with `generation stalled`; incomplete calls and raw payloads are discarded
without side effects.

The no-thinking prefix is retained in the private assistant transcript, even
though it is not duplicated on stdout. This makes the next render token-exact
instead of silently dropping the prefix previously ingested before output.
Terminal accounting reports total `prompt`, already encoded `cached`, newly
encoded `prefill`, and `output` tokens, so reuse cannot be inferred from timing.

## First smoke result (2026-07-18)

On the M1 mixed tier, with no resident q27 process and no coordination hold:

- noninteractive direct generation returned exactly `ready` (54 prompt tokens,
  1 output token);
- one fresh process retained a two-turn C-side transcript and returned exactly
  `one`, then `two` (52/1 and 68/1 prompt/output tokens);
- both runs used `--no-think`, context 512, and no server process.

A later B1 smoke exercised the raw-payload path at context 2048 with adaptive
limits. B1 created `fstring.py` as the exact 37 bytes
`name = "world"\nprint(f"hello {name}")`, despite the nested `print(f"` syntax
that previously broke JSON-embedded content. It then read an existing file,
selected `print(f"hello {name}")` as the unique old text, generated the
24-byte replacement `print(f"goodbye {name}")` in the raw phase, atomically
edited the file, and reread it successfully. Both raw phases used exact-prefix
resident reuse and published mode `0600`; neither source payload passed through
JSON.

Follow-up worker/event smokes returned exactly `worker` and `events` through
the dedicated engine-owner thread. The JSONL smoke produced exactly
`state → text_delta → turn_done`; decoding `data_b64` yielded `events`.
`build/test_q27_agent_worker` uses a fake adapter to gate startup failure,
deep binary message/tool ownership, monotonic generation and tool lifecycle
events, exact prefill-progress ordering/accounting, an explicit
stalled-generation terminal, request rejection, 4094-event queue backpressure
(5000 deltas), cancellation terminals, and shutdown ordering
without loading a model. `build/test_q27_agent_tools` gates binary
read/search/write/edit-selection/shell output, path and symlink rejection,
create-only atomic write publication, exact-one edit publication/mode
preservation, shell workspace, timeout, cancellation, and output bounds.
`build/test_q27_agent_selections` gates annotation budgets, exact handle
resolution, path mismatch, unknown IDs, and 256-slot expiry.
`build/test_q27_agent_stall` gates every watchdog threshold, progress resets,
and a long progressing-output control.
`build/test_q27_agent_persistence` gates binary transcript round trips, CRC
rejection, mode-0600 publication, immutable snapshot replacement, and
tool-boundary-preserving compaction cuts. Their ASan/UBSan legs are clean.

A two-turn resident-session smoke returned exactly `one`, then `two`. The
first turn reported `prompt=51 cached=0 prefill=51 output=1`; the second
reported `prompt=73 cached=52 prefill=21 output=1`, proving exact-prefix reuse
rather than another full reset. The CPU `AgentSession` selftest separately
gates pending-token finalization, append offsets, prefix and engine-position
mismatch fallback, and cancellation/error invalidation.

This establishes direct C → worker → C++ → resident Metal-session feasibility,
including opt-in model-driven tools through a separate control-plane command,
private restart persistence, and bounded boundary-preserving compaction. It
does not establish rich terminal/browser UX or broad DS4 parity.

## Graduation gates

Phase 0 graduates only when a coordinated model window demonstrates:

1. a noninteractive answer without a server process;
2. a two-turn interactive transcript;
3. Ctrl-C returns cleanly and a subsequent fresh run succeeds;
4. emitted bytes match the direct Metal CLI/server greedy result for the same
   rendered prompt, modulo the documented API prose trimming convention;
5. ASan/UBSan-clean C-side transcript, session-ledger, and output handling in
   fake-adapter tests.

## Next port slice

After Phase 0, port from the pinned DS4 agent in this order:

1. ~~dedicated engine-owner worker and synchronous typed commands~~;
2. ~~queued UI events and a noninteractive JSONL event stream~~;
3. ~~q27 `AgentSession` append/finalize contract~~;
4. ~~file read/search/edit tools and bounded asynchronous shell jobs~~;
5. ~~q27 `<tool_call>` parsing and constrained generation (not DSML)~~;
6. ~~transcript/session persistence using `Q27SNAP1` (not DS4 payloads)~~;
7. ~~compaction~~;
8. **terminal UI (in progress on `agent-tui`)** — linenoise editor, status
   footer, slash commands; design note
   `docs/metal/plans/2026-07-21-agent-tui.md`. Browser tooling still deferred.

### Interactive TUI (Phase 1)

On a TTY, interactive mode uses a ds4-style linenoise editor with:

- line editing, history, multiline paste
- sticky status footer (`ctx used/size | idle|prefill|…`)
- slash commands (`/help`, `/quit`, `/save`, `/compact`, `/session`, `/new`,
  `/read`, `/search`, `/shell`) plus legacy colon forms

Non-TTY stdin keeps the plain interruptible line reader. JSONL and
`--prompt` paths are unchanged.
