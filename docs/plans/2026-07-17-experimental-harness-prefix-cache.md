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
