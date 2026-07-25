# P16 design note: persistent stable-prefix cache (cross-restart, cross-conversation)

Status: **P16a IMPLEMENTED and gated, 2026-07-24** (`src/prefix_cache.h`,
`Engine::pfx_*`, `--prefix-cache`). Opt-in, off by default. Results at the
bottom; two things this note got wrong before the code existed are corrected
in place and flagged.

## The gap

q27's prefix reuse is excellent and entirely volatile. The P8 stable snapshot
and the P9 checkpoint ring live in VRAM and pinned host RAM, and both die with
the process. Two cases pay a full cold prefill that nothing in the current
design can avoid:

1. **Server restart.** Every rebuild-and-restart during development, every
   crash, every machine reboot. The next Claude Code turn re-prefills the whole
   conversation from zero.
2. **A new conversation that shares a system prompt.** CC sends a 20-25K-token
   system + tool-definition block that is byte-identical across sessions
   (`normalize_cc_billing_header` already pins the one part that used to
   mutate). Today the second session prefills it again from scratch, on a cold
   slot, because reuse is per-engine and per-process.

Both are the same miss: we recompute state we have computed before and could
have kept.

## What that miss costs (measured 2026-07-24)

RTX 5090, vanilla `qwen36-27b-mtp` (canonical a2982c51), fp8 KV, `--ctx 65536`,
W12, batching on, `/v1/messages` non-stream, `max_tokens 1`:

| prompt | cold `pf_ms` | cold wall | warm `pf_ms` | warm wall | ratio |
|---|---|---|---|---|---|
| 15,418 tok | 4310 (3,578 t/s) | 4.34 s | 42 (`hit=15411`, `pf=7`) | 0.072 s | 60x |
| 30,733 tok | 9170 (3,351 t/s) | 9.21 s | 42 (`hit=30726`, `pf=7`) | 0.081 s | 114x |

Prefill runs ~3.4K t/s at these depths, so a 25K-token CC system prefix costs
**~7.3 s** cold. That is the per-restart tax, and it is also the tax on the
first turn of every new conversation.

Disk on this box (`/mnt/ai`, CT4000P3PSSD8, 77% full), O_DIRECT so the page
cache is bypassed:

| access | throughput |
|---|---|
| bs=1M, single stream | 714 MB/s |
| bs=16M or 64M, single stream | 1.4 GB/s |
| bs=16M, 4 parallel streams | ~2.4 GB/s aggregate |

## The key already exists: P8's `stable_off`

`chatml_prompt` already computes the char offset where the volatile
assistant-open begins, and `handle`/`/v1/messages` already encode the prompt in
two pieces at that boundary. Everything before `stable_off` re-renders
byte-identically next turn AND next session. That is exactly the right unit to
persist.

**Scope correction (found while implementing).** The pre-implementation draft
claimed this gets the cross-conversation case for free, because the key is the
token prefix itself. It does not, and the reason is the same all-or-nothing
property that motivates the whole design: an entry only helps if its L tokens
are an EXACT prefix of the new prompt, and a recurrent state cannot be
partially restored at some shorter length. Turn 1's stable prefix is
system + tools + the first user message, and the user message is what differs
between conversations -- so a fresh conversation misses it. Getting the shared
win needs an entry whose L lands INSIDE the system block, which means
persisting at a boundary the server picks (a `sys_off` split in
`chatml_prompt`, or the 4096-token ring checkpoints that already fall there).
P16a therefore delivers the RESTART case, which is measured below; the
cross-conversation case is P16b and is not claimed here.

## What has to be persisted

Restoring a prefix of length L needs three things. Only the first is currently
copied off the device.

1. **GDN recurrent state.** `S[il]` (`GDN_HEADS*GDN_DIM*GDN_DIM` floats) plus
   `conv_ring[il]` (`3*GDN_CH` floats) for each of the 47 recurrent layers
   (N_LAYER 64 minus 17 attention). `ckpt_save` already does exactly this D2H
   copy into pinned host memory.
   Size: 47 x (786,432 + 30,720) x 4 B = **153.6 MB**, independent of L.
2. **Attention KV rows [0, L).** 17 attention layers plus the MTP pair, 18
   K/V pairs = 36 buffers. `kv_store_T` writes row-major with stride
   `KVROW = N_KV*HEAD_DIM = 1024`, so a prefix is a **contiguous** chunk of
   `L*KVROW` elements at the start of each buffer. No striding, no gather, and
   the chunk is restorable into an engine with a different `max_ctx`.
   Size per token: **fp8 36.0 KiB, fp16 72.0 KiB, turbo3 14.1 KiB**.
   (CORRECTED from the pre-implementation draft, which said 18.0 KiB at fp8:
   the `pair` term in the auto-ctx sizer is K+V for one layer, 2048 B at fp8,
   so per-token is 18 x 2048, not 18 x 1024. Every derived number below moved.)
3. **The MTP rows run one longer: [0, L+1).** `mtp_warm_T` stores at `base+1`,
   so prefilling through L leaves MTP rows [1, L] populated. Copying only
   [0, L) restores a stale row L, which still decodes CORRECTLY -- bad drafts
   are rejected by verify -- but silently costs acceptance. A bitwise gate
   would never catch it; only a t/s regression would.
4. **The token vector**, for verification (see below).

Entry size for a 25K prefix: **1.08 GB** at fp8, **498 MB** at turbo3.
(Measured at L=26693, fp8: 1.09 GB.)

## Payoff, measured

26,700-token prompt (L=26693 stable), fp8, 5090, vanilla a2982c51,
`--ctx 32768`, greedy, `max_tokens 64`. Three processes: reference with the
cache off, a boot that persists, then a **fresh process** against the same
directory.

| run | wall | `[req]` |
|---|---|---|
| cold prefill, cache off | 8.15 s | `hit=0 pf=26700 pf_ms=7762` |
| warm turn, same process (P8) | 0.43 s | `hit=26693 pf=7 pf_ms=41` |
| **restart, disk restore (page cache cold)** | **1.20 s** | `hit=26693 pf=7 pf_ms=329 pfx=26693` |
| restart, disk restore (page cache warm) | 0.72 s | same, read 246 ms |

The cold-page-cache read of 1.09 GB took **726 ms**, i.e. 1.5 GB/s, matching
the drive's measured single-stream rate. So the honest restart figure is
**6.8x / ~7.0 s saved**, and 11.3x when the entry is still in page cache.
turbo3 halves the blob and should roughly halve the read.

Write cost on the persisting turn: **38 ms** of D2H staging on the critical
path (1.09 GB), then ~1-2 s of file write on a background thread. Total
request time went 8.15 s -> 8.33 s, so the persist costs ~2% of the one turn
that pays it.

## Design

### On-disk layout

```
{cache_dir}/{compat_hash}/{fnv1a64(tokens)}-{L}/
    meta.json      # compat fields, L, token count, mtime, kv_kind, bytes
    toks.bin       # L int32 tokens, the verification payload
    gdn.bin        # 47 x (S | conv_ring), the ckpt_save layout verbatim
    kv.bin         # 18 x contiguous [0,L) chunks, k then v per layer
```

`compat_hash` covers the model file identity (the .q27 tag plus size), the
architecture constants that shape the buffers (`N_LAYER`, `N_KV`, `HEAD_DIM`,
`GDN_HEADS`, `GDN_DIM`, `GDN_CH`), and `kv_kind`. A mismatch means the
directory is simply not consulted -- swapping tiers or KV formats can never
half-load. This mirrors CachyLLama's `set_compat_hash`, which is the one piece
of their design that is unambiguously correct.

### Verification: exact tokens, never a hash

The filename hash only narrows the search. The load path compares `toks.bin`
against the request's prefix **element by element** and refuses on any
mismatch, exactly as `ckpt_best` (`engine.cuh:2902`) already does in RAM with
`std::equal`.

This is not paranoia. CachyLLama shipped SSD checkpoints keyed on a prefix hash
and spent 2026-07-23 fixing it: "server : verify SSD checkpoint token
prefixes", an outside PR `fix/ssd-cache-prefix-verification`, then "score
continuation overlap against stored prefix length". A collision or a stale
entry there silently restores another conversation's state and the model
continues from it. q27's in-RAM design is already immune; the disk tier must
inherit the same rule, not the cheaper one.

### Write path

Persist ONLY when the snapshot boundary is the P8 **stable** one
(`stable_len > base && stable_len < NP`). `generate()` falls back to `NP-1`
when that condition fails, and a blob cut at `NP-1` sits past the
assistant-open and think prefill -- it can never prefix-match a later turn.

This is not hypothetical: the first gate run persisted a second 1.09 GB entry
at `NP-1` on every restore, because a restored request has `base == stable_len`
so the fallback fires. Two entries, one of them permanently unhittable. The
gate is what makes the on-disk set converge to one entry per distinct prefix.

Beyond the boundary rule, persist if all of:

- `L >= --prefix-cache-min` (default 4096; below that, prefill is under ~1.2 s
  and not worth the write), and
- no entry exists for this exact prefix, and
- `L` exceeds the last persisted L for this conversation by
  `--prefix-cache-step` (default 8192)

then stage a copy into a pinned host buffer under the lease that already owns
the GPU, and hand the buffer to a background writer thread. The GPU-side cost
is one D2H of 614 MB (~25 ms) on the critical path; the disk write happens off
it.

The step gate matters: persisting on every turn would write ~600 MB per turn to
a QLC drive. In practice the highest-value entry by far is the FIRST one (the
system + tools block), which is also the one shared across conversations.

### Read path

On a request whose stable prefix misses every in-VRAM snapshot and every ring
checkpoint, consult disk before falling through to cold prefill. On a verified
hit: load GDN state and KV rows, set position to L, then prefill only the
volatile tail. That is the same branch `generate()` already takes on a snapshot
hit (`engine.cuh:3367`), so the integration surface is one new source of
`reuse_len`, not a new code path through the engine.

Warm the cache on boot: prefetch the most recently used entry into the page
cache so the first request after a restart pays no disk latency at all.

### Eviction

LRU by mtime against `--prefix-cache-max-gb` (default 20). Delete whole
directories, never partial entries.

### Flags

| flag | default | meaning |
|---|---|---|
| `--prefix-cache PATH` | off | enable, directory root |
| `--prefix-cache-max-gb N` | 20 | eviction budget |
| `--prefix-cache-min N` | 4096 | do not persist prefixes shorter than this |
| `--prefix-cache-step N` | 8192 | token growth required before re-persisting a conversation |

Off by default. This writes to the user's disk; it should be an explicit opt-in
the way `--api-key` is.

## Correctness gates (pre-registered, before any result counts)

1. **Bitwise restore.** Same prompt, same seed, greedy: a run that restores from
   disk must produce a byte-identical continuation to a run that prefilled cold.
   This is the whole feature in one test, and q27 already has the harness idiom
   for it (the canonical md5 gates).
2. **Verification refuses.** Corrupt one token in `toks.bin`, and a truncated
   `kv.bin`: both must fall through to cold prefill, not load.
3. **Compat refuses.** An entry written by q4s must not load into a q6f server,
   or an fp8 entry into a turbo3 server.
4. **No solo regression.** With `--prefix-cache` unset, the [req] line and
   decode t/s must be unchanged from the current build.
5. **Restart proof.** Boot, run a 25K-token turn, kill the server, reboot it,
   repeat the turn: second-boot TTFT must land within ~2x of the warm number,
   not the cold one.

## What would make this a NO-GO

- **Restore is not bitwise.** If fp8 KV rows or GDN state round-trip through
  disk with any change, the feature is dead as designed; there is no acceptable
  version that silently perturbs a conversation to save 7 s.
- **The D2H staging cost lands on the critical path** for more than ~50 ms.
  Mitigation exists (side stream, or persist only on the first turn), but if it
  cannot be hidden, the write path becomes first-turn-only.
- **Write amplification.** If real CC traffic ends up persisting on most turns
  despite the step gate, the SSD cost outweighs the win. Measure writes per
  session before defaulting anything on.

## Gate results (2026-07-24, all five PASS)

1. **Bitwise restore**: md5 `ae23ddeb8ce9b350` on all five runs -- reference
   cold, reference warm, cache-cold, disk-restore with page cache cold, and
   disk-restore with it warm. A restored continuation is byte-identical to a
   cold-prefilled one.
2. **Verification refuses**: `test_prefix_cache.cpp` forges the exact hazard --
   writes an entry, then rewrites its token payload in place so the filename
   key still matches a prompt it has nothing to do with. The load refuses and
   logs `key hit but TOKENS DIFFER`. Truncated files are not indexed.
3. **Compat refuses**: an entry written under one compat hash is invisible to a
   cache opened with another (different model/tier/KV format), not coerced.
4. **No solo regression**: with `--prefix-cache` unset the `[req]` line is
   byte-identical (the `pfx=` field is emitted only on an actual disk hit) and
   cold/warm timings are unchanged (7762 ms / 41 ms both builds).
5. **Restart proof**: fresh process, 8.15 s -> 1.20 s.

Full slate re-run because `engine.cuh` changed: tri-arch on all 4 binaries, 10
CPU suites + auth_integration, canonicals EXACT (vanilla a2982c51, q4s greedy
f64e7c02 + sampled 900031e9, q5f 683f7f44, q6f 2a4d22ea), ninv ALL PASS,
fused_smoke PASS.

## Phases

- **P16a (DONE)**: stable-prefix entries, background writer, exact-token
  verification, compat hash, LRU eviction, stale-tmp sweep. Opt-in via
  `--prefix-cache`.
- **P16b**: an entry cut inside the system block so a NEW conversation hits
  (see the scope correction above) -- either a `sys_off` split or promoting a
  ring checkpoint; plus boot prefetch of the most recent entry.
- **P16c (DONE, but a measured near-NO-GO -- default OFF)**: host-RAM tier
  between VRAM and disk, `--prefix-cache-ram-gb`. See below.

## P16c result: the page cache was already doing the job

Sizing the lever before building it turned out to be the whole story. A restore
is `alloc + read + import`:

| | 1.09 GB entry | 0.51 GB entry |
|---|---|---|
| import (H2D from pinned) | 38 ms | 18 ms |
| read, page cache warm | 206 ms | 36-39 ms |
| read, page cache cold | 681 ms | (n/a) |

**Import is the floor and no tier can beat it.** That left the read as the only
thing a RAM tier could remove -- and the boot prefetch (below) already removes
most of it by making the first read warm. Measured first-restore-after-restart,
page cache evicted before each boot, n=3 per leg:

| leg | wall | alloc | read |
|---|---|---|---|
| RAM tier OFF | 0.47-0.48 s | 75-84 ms | 36-39 ms |
| RAM tier ON | 0.53-0.54 s | 133-141 ms | 36-39 ms |

The tier makes the first restore **slower**, because its slot (sized for
`max_tokens`) is a bigger pinned allocation than the exact-fit staging buffer,
and there is no read left to save. Where it does win is a repeat restore on a
DIFFERENT slot: 18 ms (import only) vs 52-127 ms from a warm-page-cache read.
That is 40-110 ms, on a path that already costs ~0.5 s against the 8.15 s cold
prefill it replaced.

Verdict: shipped, `--prefix-cache-ram-gb 0` (off) by default. Worth enabling
only when the page cache is under pressure from other work, where the
alternative is a genuinely cold read on every hit. Bitwise verified either way.

**Boot prefetch is the real win here, and it needed a rewrite.** The first
implementation used `posix_fadvise(POSIX_FADV_WILLNEED)`; measured, it did
NOTHING for a 1 GB entry (cold restore still read for 681 ms) -- the call is
advisory and the kernel declines a readahead that size. Replacing it with an
explicit chunked read on a detached thread, which runs under the cover of the
weight upload, takes the first post-restart read from 681 ms to **36-39 ms**.

**One unreproduced outlier, recorded rather than hidden:** two first-reads in a
single gate run took 12.7 s and 17.9 s for a 0.51 GB file that reads at
3.3 GB/s with O_DIRECT moments later. Both were the first read of a file
written seconds earlier and then evicted from page cache. Never reproduced
(3/3 repeats at 36-39 ms). Consistent with QLC garbage collection after a large
write on a 77%-full drive, not with anything in this code path -- but it is the
reason the restore timing log now splits alloc / read / import instead of
reporting one number, since the original single number had folded a
first-touch `cudaMallocHost` into what it called "read".

## Known limits of P16a

- A write is lost if the process dies inside the ~1-2 s background write.
  Nothing partial can be indexed (publish is a rename) and the leftover .tmp is
  swept on the next boot, so the cost is one re-prefill, not corruption.
- The staging buffer is pinned host memory sized by `--prefix-cache-max-tokens`
  (32768 default = ~1.3 GB at fp8), allocated lazily on first use.
- One write in flight per engine; a persist that arrives while the previous is
  still writing is skipped, not queued.

## Prior art

- **Upstream llama.cpp** already ships `--ctx-checkpoints` (in-RAM, 32/slot),
  `--slot-save-path` with `/slots/{id}?action=save|restore` (manual,
  client-driven file persistence), and `--cache-reuse` (KV-shifted partial
  reuse). The persistence primitive is not novel; automatic
  conversation-keyed lookup on top of it is what nobody ships.
- **CachyLLama** (fewtarius, MIT llama.cpp fork, active 2026-07) builds exactly
  that: SSD-backed KV cache with hot/warm/cold tiering, kernel readahead, a
  cross-conversation system-prompt cache, and hybrid-model checkpoint restore.
  Their headline is 23-144x, which is cold-vs-warm on a 7840U/780M where cold
  prefill runs ~110 t/s; the ratio is a statement about how bad iGPU prefill is,
  not about the cache. Their tiering shape and compat hash are worth copying,
  their hash-keyed lookup is not (see Verification above).
