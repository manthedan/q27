# KV exception snapshot v2: side caches join all three snapshot surfaces

Status: PRE-REGISTERED before implementation (2026-07-17, mini).
the operator funded L7-full as the production config ("fund full l7") and then
"do v2" — remove the v1 exclusion that refuses snapshots under
`Q27_METAL_KV_FP16_CELLS`, so the funded quality config regains the
prefix cache (TTFT) and the disk snapshot tier. Parent:
2026-07-17-kv-except-production.md (v1 exclusion recorded there; codex
rounds 41705bb/88373f7/86b3090 made all three surfaces refuse loudly and
consistently — v2 makes all three WORK consistently instead).

## Why v1 refused (what v2 must actually solve)

The side caches are `allocate_private` (MTLStorageModePrivate) — their
`contents` pointer is nil, so `MetalBackend::read`/`write` (host memcpy
paths used by save_state/load_state) cannot touch them; only `copy()`
(GPU kernel) can. And a snapshot that carries main KV but not side rows
recombines one prompt's main KV with another's side rows on restore
(the 41705bb P2). v2 = side rows ride every snapshot, staged through a
shared bounce buffer for the disk tier.

## Design

Engine (`metal_engine.{h,cpp}`):

1. Keep the parsed per-attn-layer head masks as a member
   (`uint8_t kv_fp16_head_masks_[16]`, accessor for the server) — the
   snapshot config identity.
2. **In-memory (capture_state/restore_state)**: Snapshot gains a flat
   `std::vector<std::shared_ptr<BackendBuffer>> kv_side` in fixed order
   (attn_idx ascending, head ascending, K then V; `position_ *
   HEAD_DIM * 2` bytes each, skipped at position 0 like the main KV).
   `copy()` handles private buffers on both ends; restore copies back.
   Config identity is structural: restore already requires
   `snapshot.owner == this`.
3. **Disk (save_state/load_state)**: header `reserved` gains bit 1 =
   "KV fp16 exception extension present" (bit 0 remains
   !logits_resident). When set, after the standard blobs:
   a length-prefixed 16-byte head-mask blob, then per side entry the K
   and V blobs (length-prefixed, active bytes), same order as (2).
   - save: one shared staging buffer (16 MB, chunked) —
     `copy(side→staging)` then `read(staging)` per chunk.
   - load pass 1 (before any GPU write): extension presence must equal
     the engine's `kv_fp16_except()` (loud reject both ways — an
     env-unset snapshot cannot serve an exception engine and vice
     versa: the masked head's side rows / quantization history would be
     wrong); mask blob CONTENT must equal the engine's masks (same
     lengths ≠ same cells — 8,9 vs 10,11 would otherwise silently
     cross-restore); side blob lengths validated like all blobs.
   - load pass 2: TOCTOU re-checks (lengths + mask content), side
     blobs stream `write(staging)` then `copy(staging→side)`.
   - peek_snapshot unchanged (header + tokens only).
4. Remove all four v1 refusal throws. `snapshot_bytes()` adds the side
   bytes at worst case (2 × max_ctx × 512 per masked head) so G6
   admission and the prefix-cache budget stay honest.

Server (`metal_server.cpp`):

5. Remove the three v1 special-cases: Slot cache capacity back to
   `entries` unconditionally, G6 charges real capacity, snapshot dir no
   longer ignored. Keep one informational startup note that exception
   cells are active.
6. Disk store tag gains a config token when exception cells are active
   (`…t` → `…tx<16 mask nibbles>`), so servers with different cell
   lists sharing a snapshot dir miss cleanly instead of paying a
   request-time loud reject + fallback. Env-unset tags are unchanged —
   existing snapshot files stay valid. load_state's mask check remains
   underneath as defense in depth.

Cost honesty: at ctx 32768 the funded L7-full config adds 4 pairs ×
2 × 32768 × 512 B = 128 MB to every snapshot (in-memory entry and disk
file alike), on top of the ~13 MB the side caches themselves cost.
snapshot_bytes carrying it means multislot admission prices it.

## Pre-registered gates (each a kill line)

1. **Default-off untouched**: `make test-metal` green; env-unset
   `tools/snapshot_gate.sh` ALL PASS; a snapshot file saved by the
   PRE-change binary loads on the POST-change binary byte-identically
   (format compatibility is a contract, not an accident).
2. **Except-arm byte identity (the point)**: snapshot_gate.sh gains a
   turbo3 + L7-full arm — save mid-run, load in a fresh process,
   continuation byte-identical to the uninterrupted reference under the
   same env.
3. **Reject matrix additions, all loud**: env-unset file into exception
   engine; exception file into env-unset engine; mismatched cell lists
   (save 8,9 / load 8..15); truncation inside the final (side) blob.
4. **Server surfaces live under the funded env**: warm repeat request
   byte-identical to cold with `q27_prefix_hit` > 0 (in-memory tier);
   `Q27_METAL_SNAPSHOT_DIR` hinted save then fresh-process restore hit
   (disk tier); startup admission log shows the side-inclusive
   snapshot_bytes.
5. Codex cross-review (gpt-5.6-sol) of the diff, findings triaged,
   before done — standing practice.

## Results (2026-07-17, mini)

1. PASS — `make test-metal` green; env-unset `snapshot_gate.sh` ALL PASS
   (gate hardened en route: `gen()` compared only the first output line,
   which is empty when the decode starts with a newline — now the whole
   generated text; every pre-existing arm passes the stronger check).
   Cross-version: fp16 + turbo3 snapshots saved by the PRE-change binary
   load on the POST-change binary with byte-identical 64-token
   continuations.
2. PASS — l7full arm: save-run and fresh-process load byte-identical to
   the uninterrupted reference under the funded env (side rows at
   position 17 through the staging bounce).
3. PASS — all four new rejects loud: no-side-rows into exception engine,
   side-rows into env-unset engine, equal-size mismatched cell lists
   (0..7 vs 8..15 — the mask-content check specifically), truncation
   inside the final side blob.
4. PASS — live server, funded env: hinted disk save fired
   (tag `…tx0f00000000000000-`, config nibble = L7 mask 0xf), fresh
   process restored 672 tokens from disk (disk_hits 1) with
   byte-identical continuation; in-memory tier warm hit
   (q27_prefix_hit 5) byte-identical to cold. Startup notes read
   "side caches ride prefix/disk snapshots (v2)".
5. Codex cross-review (gpt-5.6-sol, on 4415c53) — three findings, triaged:
   - P1 (pass-2 failure atomicity of load_state): PRE-EXISTING contract,
     not a v2 regression — the Phase-1 design already accepted that
     mid-stream pass-2 failures leave indeterminate state and the server
     resets/falls back (documented at its call site). Disposition:
     contract now stated explicitly in the load_state comment; a fully
     atomic restore needs a double-buffered staging pass and stays
     unfunded.
   - P2 (G6 admission overflow: uint32 --prefix-entries x side-inclusive
     ~4.6 GB snapshot_bytes can wrap uint64 and falsely admit a slot):
     REAL, newly reachable with v2 sizes — fixed, --prefix-entries
     bounded to 4096.
   - P3 (position-0 snapshots demand config match despite carrying no
     history): accepted limitation — nothing saves at position 0 (the
     server snapshots post-prefill, the CLI post-prompt), and relaxing
     presence at 0 would special-case a state that never exists on disk.
   Codex confirmed: staging-bounce synchronization, mask-before-write
   ordering, cross-version format behavior, tag collision-freedom, and
   the side-inclusive snapshot_bytes arithmetic are all correct.
