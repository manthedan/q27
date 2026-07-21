# Experimental harness-prefix prewarming

Status: experimental Phase 0 implemented and live-smoked; disabled by default.

## Problem

Pi and Codex send thousands of stable instruction/tool tokens before the live
user message. On the measured Metal path, an 8,355-token Pi prompt took about
five minutes to prefill. Existing automatic disk snapshots make later requests
fast, but the first interactive request still pays the cold prefill.

## Boundary

The shipped token-prefix `DiskSnapshotStore` and `Q27SNAP1` serializer remain
the storage/restore mechanism. Harness capture, initial-request validation, and
prewarm orchestration are experimental:

```text
q27-metal-server --experimental-prefix-cache PRIVATE_DIR
POST /experimental/prefix-cache/prewarm   (loopback + admin token)
tools/experimental_prefix_cache.py
```

Without the flag, the route is not registered and no experimental behavior
runs. With the flag, ordinary auto-save defaults off unless `--snapshot-auto`
is also supplied. MTP is rejected because its warmed lane state is outside the
disk snapshot contract.

## Protocol

The endpoint accepts:

```json
{
  "api": "chat_completions | responses",
  "request": {"the": "ordinary initial harness request"}
}
```

It uses the same Chat Completions or Responses canonicalizer as normal serving,
requires the normalized request to contain only system instructions followed
by one final user message, removes that user message, and renders the closed
static prompt. Before any snapshot side effect, it tokenizes both static and
full prompts and proves the static token vector is a strict exact prefix of the
full vector. It then runs zero-token generation with exact full-prefix save.

The cache key is SHA-1 over the exact token IDs. Deep `Q27SNAP1` validation
still binds model artifact, KV/configuration, blob layout, stored token IDs,
and Metal position. Any later request that differs in tools, instructions,
template output, or tokenizer IDs misses and follows ordinary prefill.

## Local installer

`tools/experimental_prefix_cache.py prewarm` installs a captured JSON request.
Its `proxy` mode listens on loopback, prewarms the first `/v1/chat/completions`
or `/v1/responses` request from the real harness, and then transparently
forwards that original request to the server. The proxy requires a separate
`Q27_PREFIX_CAPTURE_TOKEN` as the temporary harness provider's bearer token
(or `X-Q27-Prefix-Capture-Token`) before it exercises its stored admin
capability. Streaming setup calls receive a one-second SSE comment heartbeat
during the cold install, then the untouched upstream event stream. Subsequent
requests are only forwarded. Target and listener are loopback-only;
credential-bearing admin traffic bypasses all configured HTTP proxies. Both
secrets are environment-only (never argv), consumed locally, and stripped
before forwarding to the model endpoint.

Captured prefixes can contain private project instructions. The experimental
store leaf is owner-owned, non-symlink, stripped of extended ACLs, and forced
to mode `0700`; its canonical ancestor chain rejects granting ACLs, foreign
non-root owners, and non-sticky principal-writable directories while allowing
macOS's common deny-delete home ACL; existing and newly retained snapshots are
ACL-free `0600`.
Nothing is uploaded or bundled.

## Live Phase-0 evidence

M1 model, context 512, fp16 KV, one slot, in-memory prefix cache disabled:

- Chat request: 202 total prompt tokens; 189-token static prefix installed.
- Snapshot: 162 MiB, mode `0600`.
- Fresh server process restored all 189 tokens from the disk tier.
- A changed user message still hit 189 tokens.
- A changed system message missed.
- Unauthorized prewarm returned HTTP 403.
- Responses request: 198 total prompt tokens; 186-token static prefix
  installed and consumed by the ordinary `/v1/responses` path.
- Proxy mode installed a distinct 183-token prefix and forwarded the original
  Chat Completions request successfully.

A real isolated Pi run then exercised the complete setup path using Pi's
installed OpenAI Chat Completions client and ordinary four-tool prompt:

- Pi initial request: 1,331 prompt tokens; exact static prefix: 1,318.
- Prewarm + 233 MiB private snapshot publication + forwarded one-token answer:
  28.28 s wall (`ready`).
- Fresh Metal server process, proxy removed, different user message: disk hit
  1,318/1,331 and complete Pi invocation in 1.64 s wall (`okay`).
- Snapshot mode: `0600`; cache directory: `0700`.

That is about 17x end-user wall improvement for this current minimal Pi
configuration. The older captured Pi configuration was much larger (8,355
prompt tokens), so recipes must remain exact and versioned rather than treating
"Pi" as one stable prompt. Separately, the existing 2,346-token snapshot gate
measured 0.12 s load versus about 52 s prefill (~450x), while the prior live Pi
auto-snapshot measurement reduced a 4:54 cold prompt to 8.1 s on a later
prefix hit.

## Promotion gates

Before this can become supported behavior:

1. Capture real pinned Pi and Codex versions and record cold/install/hit TTFT.
2. Verify tool-loop first and follow-up turns, streaming, cancellation, and
   concurrent-slot behavior through the capture proxy.
3. Add human-readable recipe manifests with harness executable/version digest,
   while retaining exact token IDs as the only matching authority.
4. Add list/verify/remove/prune lifecycle commands.
5. Decide whether selected prefixes should be pinned resident after measuring
   cold and filesystem-warm snapshot load times.
6. Complete adversarial review of prompt privacy and directory replacement
   assumptions before enabling non-loopback administration.

---

## Codex + Claude Code capture (2026-07-20, promotion gate 1 partial)

**Messages API support added (this change).** The prewarm endpoint previously
accepted only `chat_completions` and `responses`, so Claude Code (Anthropic
`/v1/messages`) could not be captured at all. Added a `messages`/`anthropic`
arm that reuses the SAME canonicalizer as ordinary `/v1/messages` serving
(`q27::anthropic_msgs` + `anthropic_tools_json`, `think=true` to match the
serving render), so the prewarmed prefix is byte-for-byte what the live
request prefills. Proxy `ELIGIBLE` gains `/v1/messages -> messages`; direct
`prewarm --api` accepts `messages`/`anthropic`; streaming prewarm failures on
`/v1/messages` now emit the real API's named `event: error` frame instead of
the OpenAI `[DONE]` shape.

**Codex 0.144.6 captured and warm-verified (pinned).** Method: the proxy's
blocking cold-prewarm outlives Codex's HTTP timeout on a large prompt, so a
`--dump-request PATH` proxy mode was added to harvest the exact request JSON
for out-of-band install. One `codex exec -p q27` run dumped
`/tmp/q27-codex-captured.json` (177 KB body, `api=responses`, 20,751-char
`instructions` + 19 tools + 3 input items, `stream=true`). The background
install thread (which keeps running after client disconnect, by design)
completed the snapshot anyway; a direct `prewarm` of the harvested file
confirmed `already_cached: true`.

- Codex initial request: 11,219 prompt tokens; exact static prefix: **10,802
  tokens** (96.3% of the prompt is stable across runs).
- Snapshot: **826 MiB**, mode `0600`, dir `0700`.
- `/stats` after a fresh warm run: `disk_saves: 1`, `disk_hits: 2` — the
  10.8K-token prefix is restored from disk, not re-prefilled.
- Cold first turn (blocking proxy, Codex timeout) never completed
  interactively; warm turn (`codex exec -p q27`, prefix hit) completed
  end-to-end in **29.6 s wall** including Codex startup + model refresh +
  generation. This is the gate-1 Codex number; a clean cold-vs-warm TTFT
  split needs the async-prewarm fix below.

**Persistent Codex config installed:** `~/.codex/q27.config.toml` (named
`q27` profile: provider `q27` -> `http://127.0.0.1:8081/v1`,
`wire_api=responses`, bearer from `Q27_CAPTURE_TOKEN` in `~/.codex/q27.env`,
mode 600). Toggle with `codex -p q27`. Stack launcher `/tmp/q27-stack.sh`
(server ctx 32768 + proxy) with proper PID/port handling.

**Known limitation surfaced (gate 2 blocker):** the proxy's streaming
cold-prewarm blocks the first live request for the full tokenize +
zero-token-gen duration. On a >10K-token Codex/Claude prompt at 32K ctx on
the M1 pack this exceeds the harness HTTP timeout, so the interactive
capture path stalls (the snapshot still lands via the background install
thread, but the first turn is lost). Mitigation used here: `--dump-request`
+ out-of-band direct `prewarm`. Proper fix (async prewarm that returns
immediately + lets the first request ride ordinary prefill while the
snapshot builds, or a longer-heartbeat SSE path) is required before the
blocking proxy is usable interactively for large-prompt harnesses.

**Claude Code 2.1.215: harvested, but its request shape does NOT fit the
prewarm contract — capture blocked on an extractor generalization.**

Capture plumbing (all landed): the proxy accepts Anthropic `x-api-key` as a
capture credential and strips it before forwarding (it is a secret); Claude
Code's `Authorization: Bearer` is driven by `ANTHROPIC_AUTH_TOKEN` (NOT
`ANTHROPIC_API_KEY` — Claude Code prefers its stored OAuth and only honors
`ANTHROPIC_AUTH_TOKEN` for the Bearer header on a custom base URL). Its
request path is `/v1/messages?beta=true` (the proxy's query-stripping path
match already handles the `?beta=true`). With `ANTHROPIC_BASE_URL=:8081` +
`ANTHROPIC_AUTH_TOKEN=<capture>`, one `claude -p` run dumped the exact
request via `--dump-request`: 99.6 KB body, `api=messages`, 3 system blocks
(6,203 chars) + 27 tools + 2 raw messages.

**The blocker (structural, not config).** After the server's OWN
`anthropic_msgs` canonicalizer runs, Claude Code's message list is:

    [system(6203 chars), user(331 chars), system(7672 chars)]

i.e. it ENDS in a system block — the large trailing injected
system-reminder / tool-context — not in a final user message.
`initial_harness_prefix` hard-requires "zero or more system messages
followed by exactly one final user message" and throws `400: prewarm
requires an initial request ending in one user message`. Pi and Codex both
end in a clean user turn, so the assumption held for them; Claude Code
2.1.215 injects its context as a trailing system block, breaking it. The
static prefix is still well-defined (everything before the live user turn,
plus arguably the trailing system block which is itself largely stable), but
the current extractor cannot carve it because the live user message is not
last.

**Fix required (NOT done — deserves its own pre-reg + tests):** generalize
`initial_harness_prefix` to locate the LAST user message anywhere in the
canonical list and treat the tail after it as non-static (or, better, as a
second cacheable-static segment if it is byte-stable across runs). This is a
safety-critical exact-prefix path; do not hack it ad hoc. Interim options:
(a) leave Claude Code on ordinary `--snapshot-auto` (first turn cold, later
identical-prefix turns hit) — zero new code; (b) defer Claude Code until the
extractor is generalized properly. Decision recorded 2026-07-20: write up
and stop; Codex ships as the working capture, Claude Code awaits the
extractor design pass.

**Proxy safety note (autoreview, landed):** because the extractor rejects the
trailing-system shape, the proxy must NOT inline-prewarm `/v1/messages` —
doing so on a first streaming Claude Code request would 400 and fail the live
turn. `/v1/messages` is therefore **dump-only**: the proxy harvests the
request via `--dump-request` for out-of-band `prewarm`, marks itself warmed,
and forwards normally, so live Claude Code turns are never broken by the
capture path. `chat_completions`/`responses` (Pi/Codex, which end in a user
turn) still inline-prewarm. Once the extractor is generalized, `/v1/messages`
can move from dump-only to inline-prewarm.
