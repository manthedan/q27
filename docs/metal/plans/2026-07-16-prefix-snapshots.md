# Prefix snapshots to disk — ds4 survey item 5 / review-2 residue (P3)

Status: Phases 1 AND 2 LANDED same day, all gates PASS (results below).
Parents: 2026-07-15-ds4-survey.md §5 (disk-KV design, adopt nearly
wholesale), 2026-07-15-expert-review-2-triage.md (stable-boundary
snapshots; the needle runs re-prefilled the same 31K haystack — ~23 min
per question — because prompts differed only in the trailing question).

## Why the economics are lopsided

At 31K context the full engine state is ~2.1 GB fp16-KV (16 attn layers
× 4 KV heads × 256 dims × 2 sides × 2 B × 31K rows) or ~200 MB turbo3,
plus ~1.2 GB GDN state (48 layers × 48 heads × 128×128 f32 recurrent +
conv rings) — call it 1.4–3.3 GB. NVMe moves that in ~1 s against a
: ~23 min re-prefill. Any hit rate above zero pays for the code.

## Phase 1 — engine serializer + CLI gates (this rig, now)

`MetalEngine::save_state(path)` / `load_state(path)`: the
capture_state()/restore_state() composition (per-layer recurrent, conv
ring, active KV rows, MTP caches when present, hidden row, resident
logits, position) streamed through host memory to a file with plain
write/read — ds4's explicit no-mmap lesson adopted (no new VM mappings
on a process already mapping a 7 GB artifact).

Format Q27SNAP1 (LE): magic+version, artifact identity (mapping size +
SHA1 over the WHOLE mapped artifact — the bytes the engine actually
opened, immune to same-size checkpoint swaps and pathname races; lazy,
cached per Shared, ~2 s once and only when snapshots are used — codex
P1+P2 on the landing commit), kv dtype flag, position,
token count + token ids (prefix identity for Phase 2), reserved
byte-prefix SHA1 slot (zeros in Phase 1), then per-layer blobs with
per-blob byte lengths. Fail-loud on any mismatch: magic, identity,
dtype, position > target engine's context, truncated blob.

CLI: `--save-state FILE` (after prompt ingestion, before generation)
and `--load-state FILE` (restore instead of ingesting a prompt, then
generate). Both compose with --prompt/-n; both reject speculative
modes in Phase 1 (MTP interaction gated separately if ever needed —
T2 artifact has no MTP layer).

### Pre-registered gates (all must pass before Phase 2)

1. **Fresh-process byte identity:** process A: prompt → save + generate
   64 greedy tokens (reference). Process B: load → generate 64. The 64
   tokens must be byte-identical. Run for BOTH kv dtypes (fp16, turbo3)
   and both prefill routes if applicable.
2. **Reject matrix:** wrong artifact (identity hash), wrong kv dtype,
   position > context, truncated file — each must throw, none may
   partially mutate engine state (load stages fully to host before any
   GPU write).
3. **Round-trip wall:** save and load MB/s recorded at 2K and the
   longest prompt this rig comfortably holds; the load must be
   dramatically under the equivalent re-prefill (expected ≥100×; if the
   measured ratio is under 10× something is wrong — investigate before
   Phase 2).

## Phase 2 — server keying + LRU (opened same evening)

Design deviation from ds4, recorded with the why: ds4 keys on the SHA1
of the RENDERED BYTE prefix because stateless clients retokenize —
their server never sees tokens, so byte identity plus a
chunk-align/tail-trim heuristic is their only defense against
BPE-boundary drift. q27's server tokenizes every prompt itself, so the
snapshot's stored token ids allow EXACT token-prefix matching: a hit is
"the snapshot's token list is a prefix of this request's token list",
which has no BPE-boundary hazard by construction. The byte-SHA1 header
slot stays reserved for a future stateless-client path.

Mechanics: snapshots saved by request hint (`"snapshot": true` on the
request — app hints + LRU is v1 per the triage) at a stable boundary:
position aligned down to a 96-token prefill-chunk boundary after
trimming a 32-token tail (still adopted — it cuts the question-specific
suffix so the shared haystack prefix is what persists). On admit, scan
Q27_METAL_SNAPSHOT_DIR headers (cheap: header + token ids only), pick
the LONGEST matching token prefix, load, and continue prompt ingestion
for the remaining tokens through the normal quantum path. LRU: touch
files on hit (mtime), evict oldest past Q27_METAL_SNAPSHOT_MAX_MB
(validated env, fail-loud, same class as Q27_METAL_BUDGET_MB).

G7 gate in tools/multislot_gates.py: hit path byte-identical to cold
path; a prompt differing inside the snapshotted prefix must MISS;
eviction under a tiny budget; admission accounting unchanged (snapshot
files are disk, NOT resident — the G6 per-slot term does not grow;
only the in-memory snapshot cache counts).

Non-goals (recorded): tool-call byte replay (ds4's exact-byte map) —
follows the constrain-tools work, not this; cross-machine snapshot
portability (identity hash pins the artifact byte-exactly);
turbo3↔fp16 snapshot conversion (dtype is part of identity).

---

## Phase 1 results (same afternoon, M4 16 GB) — ALL GATES PASS

`tools/snapshot_gate.sh` (fresh-process identity + reject matrix):

1. **Byte identity:** save-run and fresh-process load both byte-identical
   to the uninterrupted reference, 64 greedy tokens, BOTH kv dtypes
   (fp16 and turbo3). Deep-prefix re-check at position 2,346: 32
   continuation tokens byte-identical from a fresh process.
2. **Reject matrix:** truncated header, truncation inside the FINAL
   blob (the fseeko-past-EOF hole codex caught on the landing commit —
   pass 1 now checks the computed end offset against the real file
   size), corrupted artifact identity, wrong kv dtype, position >
   context — all throw with specific messages and generate nothing
   before the first GPU write, so no partial restore is possible.
3. **Round-trip wall (ctx 4096, 2,346-token War-and-Peace prefix):**
   snapshot 311.7 MB; save 0.236 s (~1.3 GB/s), load 0.116 s
   (~2.7 GB/s) vs ~52 s to re-prefill the same prefix — **~450×**,
   comfortably past the ≥100× expectation. At the 31K needle-haystack
   scale this projects to the ~23 min → seconds class the triage
   predicted. After the whole-artifact identity hardening (codex P1),
   the FIRST snapshot operation in a process additionally pays a
   one-time ~3.3 s SHA1 over the 7.15 GB mapping (cached per Shared) —
   amortized to nothing in a long-lived server, and a cold
   single-use CLI load is still ~15× over re-prefill at 2.3K (the
   margin grows linearly with prefix depth).

Phase 2 (server keying, boundary policy, LRU, G7 gate) is now unblocked;
it remains queued behind the KV-codec census on this rig's schedule.

## Phase 2 results (same evening) — G7 PASS on first run

Implementation: `DiskSnapshotStore` in the server (token-prefix
best-match over header peeks, SHA1-of-token-bytes filenames so identical
prefixes overwrite, mtime LRU, all file I/O outside the GPU lease);
hinted saves land mid-prefill at align96(len−32) by capping one chunk's
width, recorded with the stale-logits flag so a same-length request can
never derive a pending token from them; full-length disk hits derive
pending via the ingestion argmax kernel; MTP requests stay cold (lane
warming is outside the snapshot contract). Env: Q27_METAL_SNAPSHOT_DIR
opt-in, Q27_METAL_SNAPSHOT_MAX_MB validated fail-loud (default 8192).
The artifact identity hash is primed at server startup so the first
hinted request never stalls the lease. `/stats` gains
snapshots.{enabled,disk_hits,disk_saves}.

G7 (tools/multislot_gates.py, four fresh server processes): hinted save
persisted exactly one file; a fresh process answered the shared-prefix
question from disk **byte-identically to a cold server** — which also
empirically confirms that resuming at a 96-aligned boundary is
partition-invariant with respect to the cold path's chunk widths; a
prompt perturbed inside the snapshotted prefix missed and still answered
byte-identically to cold; a 1 MB budget evicted the (300 MB-class) file
after the save. Full suite G1–G7 re-run for regressions.
