# Native agent direction — in-process coding agent with KV-as-session

**Status: DESIGN, DEFERRED (2026-07-17, Daniel: "worth trying the native
direction later, once everything is stable"). No code. Do not start
before the stability gate below.** Reference: ds4-agent (ds4_agent.c,
~10.2K lines) — triaged in `2026-07-17-ds4-product-triage.md` (N1).

## The bet, stated honestly

Everything q27 serves today sits *under existing agents* across an API
boundary (Anthropic parity for Claude Code/pi, chat-completions,
responses for codex). That bet is working and stays the product. This
doc is the *other* bet: a native agent where inference is controlled
in-process — no socket, no API impedance — and **the session IS the
on-disk KV cache**. Why it's worth trying here specifically: every hard
mechanism it needs is already shipped and gated in this repo, and the
wins it unlocks are exactly the ones an API boundary makes impossible:

- **Compaction without cache loss.** An API server cannot compact — the
  client owns the transcript. In-process, compaction is a KV-state
  operation (see COMPACT below), so long agent sessions never pay the
  full re-prefill an API client's compact forces.
- **Tool-call byte-exactness by construction.** Tools rendered in the
  model's native syntax, no JSON-schema conversion layer — the
  chatml-rendering parity and exact tool-call byte-replay debt recorded
  against BasicToolConstrainer disappears structurally instead of being
  patched. (ds4 persists the replay map inside the KV files, bounded
  100k IDs — copy that shape.)
- **Session switching at disk speed.** Snapshot restore is 0.12 s where
  re-prefill is ~52 s (2026-07-16-prefix-snapshots.md); /switch between
  live tasks is the feature that number was always for.

## What q27 already owns (the mechanism inventory)

| needed by the agent | shipped mechanism | where |
|---|---|---|
| session persistence | byte-exact save/load-state + disk snapshot store (LRU, budget, admission) | prefix-snapshots Phases 1–2 |
| resume mid-generation | `pending_from_logits` byte-exact resume | snapshot v2 round |
| fast rebuild of stripped/compacted sessions | suffix bursts on agentic traffic (verify width 48) + chunked prefill (Q4 port pending) | suffix-burst-verify, t2-prefill-throughput |
| interactive + background sessions | two-slot scheduler, adaptive quanta, admission 503 | multislot Phase 1 |
| tool-call parsing/constraint | BasicToolConstrainer, bare-call recovery, truncation repair | agentic-parity + responses-residue rounds |
| abandoned work cleanup | cancellation probes + snapshot-on-cancel | abandoned-request-cancellation |

The remainder is **orchestration plus a product surface** — a REPL, a
tool suite, a compaction loop. That is not small, which is why this is
deferred, but none of it is research risk.

## Design (v1 scope)

### D1. Session model = ds4's, verbatim where it fits

Sessions live under `~/.q27/sessions/` (transcript + KV snapshot + tool
replay map, one directory per session). Commands: `/save [name]`,
`/list`, `/switch <name>`, `/del <name>`, `/strip <name>`. `/strip`
drops the KV payload, keeps rendered text; switching to a stripped
session rebuilds by re-prefill. (The store-side strip mechanism ships
earlier and independently as triage item I1 — the agent reuses it.)

### D2. Compaction through the live KV session (the COMPACT port)

Trigger: ≥85% of ctx used or <8192 tokens free. Then: a hidden internal
turn asks the model for a durable task-state summary; roll back the
transcript; rebuild as `[system + summary + verbatim tail aligned to a
user-turn boundary]`; re-sync live KV to the rebuilt transcript
(prefill, suffix-burst-accelerated, then snapshot). The **summary
contract** is the load-bearing part and ports as-is: durable task state
only (goals, files touched, decisions, rejected approaches, next
steps); hard stop at `</think>`/tool-call boundaries; *record the next
step, never execute it*.

### D3. Tools: native syntax, minimal set

v1 tools: bash, read, edit — rendered in the model's native tool syntax
via the existing template path, constraint-decoded, replay map
persisted in the session. **No JSON-schema layer in the native agent.**
Browser tool (ds4_web.c-style CDP against the user's real Chrome, real
cookies) is explicitly v2 — genuinely clever, nothing like it on our
side, but it multiplies surface area and permissions questions; do not
let it into v1.

### D4. What the engine does NOT grow

The agent is a separate binary (`q27-agent`) linking the engine — the
server keeps its narrow no-deps character, same principle as the
Homebrew wrapper decision (D4 there). Shared code moves to the engine
library only when both binaries need it byte-identically (tool
constraint, template rendering — already shared).

## Stability gate (prerequisites, all must hold before Phase 0)

1. Q4 chunk-GEMM ship-bench resolved (shipped or honestly parked) and
   committed; responses-parity residue committed.
2. Suffix-burst gate 6 economics run (the rebuild path's speed story).
3. B1 round 2 + Phase-4 residue gates closed (tier lineup final).
4. Homebrew Phase 2 done (v0.1.0 tagged, LICENSE resolved) — the agent
   ships *into* a distribution, not before one.
5. Serving instance stable under daily pi/Claude Code load with no
   open P1s for two weeks.

## Phases and gates (pre-registered shape; details at un-defer time)

- **Phase 0 — session store + REPL, no tools.** /save /list /switch
  /del /strip over the snapshot store. Gate: switch-restore
  byte-identity vs a never-switched control; stripped-session rebuild
  produces identical generation; 100-switch soak within disk budget.
- **Phase 1 — tools (bash/read/edit) native-syntax loop.** Gate: the
  pi-style read-tool task end-to-end; tool-call byte-replay exactness
  across save/switch (the recorded BasicToolConstrainer debt becomes a
  gate here); no KV mismatch across 50 tool rounds (prefix_hit full).
- **Phase 2 — COMPACT.** Gate: forced-compaction task completion (agent
  finishes a multi-file task through ≥2 compactions); summary-contract
  audit (no execution inside the hidden turn); post-compact KV re-sync
  byte-consistent with the rebuilt transcript; TTFT after compact
  bounded by tail-prefill only.
- **Phase 3 (v2, separate go) — browser tool, background sessions on
  slot 2.**

Kill/park lines: set per-phase at un-defer time with fresh numbers —
pre-registering timings now against a moving engine would be fake
precision. The standing kill condition is product-shaped, not
perf-shaped: if Phase 1 dogfooding loses to `claude` + q27-server on
the same tasks (subjective but recorded), park the direction and keep
Phase 0's session store, which is useful to the server regardless.

## Open questions (answer at un-defer time)

- **Q1:** One binary or two — does `q27-agent` subsume the CLI
  (`q27-metal`) or stand beside it?
- **Q2:** Multi-session concurrency: does /switch preempt slot 1, or do
  background sessions get slot 2 under the existing fair lease?
- **Q3:** Does the native agent speak MCP for external tools, or stay
  closed-world (bash covers most of it)?
- **Q4:** Model-native syntax across tiers: Bonsai distillations
  inherited the template — verify the constrainer's grammar is
  tier-invariant before Phase 1.
