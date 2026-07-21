# Snapshot artifact-identity digest cache — pre-registration (2026-07-19)

Source: expert review 3 §3 + review 4 (triage:
`2026-07-19-expert-review-3-triage.md` disposition 2). Small, bounded
durability item, NOT bundled with the B1 fusion round (different
subsystem, different risk).

## The defect (verified against source)

`MetalEngine::Shared::snapshot_identity()` computes a SHA1 over the WHOLE
mapped artifact (`mapping_base`/`mapping_size`), lazily on first snapshot
use and cached only for the lifetime of `Shared` (~2 s for 7 GB —
metal_engine.h:39-48, metal_engine.cpp:803-817). Every fresh process that
takes a snapshot re-reads the full weight stream to rediscover a digest
already verified at install time.

## Threat model (what this is and is not)

APFS does not automatically solve artifact identity. macOS's signed and
sealed system volume protects Apple's system content only; user data and
third-party installed content (user directories, /usr/local,
/Applications, the normal q27 model layout) live on the writable Data
volume and do NOT inherit signed-system-volume integrity. The user
immutable file flag can be cleared by the file's owner — it prevents
accidental ordinary writes while set, but is not a cryptographic identity
guarantee against the same account. So an OS-trust shortcut is valid only
under a narrower, explicit opt-in threat model.

(The native agent separately verifies its snapshot FILE by full SHA-256
before restore; that is a different concern from model-artifact identity
and is untouched here.)

## Design: persistent stat-keyed digest cache

Key (obtained via `fstat()` on the SAME open file descriptor backing the
mapping, not by re-statting a pathname):

    device id, inode, file size, mtime_ns, ctime_ns

Value:

    full-artifact SHA-256

Behavior:

1. Cache hit with all key fields identical → reuse the digest.
2. Any mismatch or missing entry → hash the full file, atomically update
   the cache.
3. Explicit audit mode → always rehash.
4. Optional local-development mode → trust the stat identity without ever
   hashing.

Pinning the descriptor and including ctime defeats the ordinary
replacement / in-place modification / file-growth cases without requiring
an installation manifest. inode+mtime+size alone is too weak for the
default.

## Policies (three)

- `cached-hash` (default): persistent stat-keyed digest as above.
- `strict-hash`: full hash every fresh process (current behavior).
- `stat-only`: explicit local/trusted-install opt-in; never hash.

## Gates

- **G1 correctness.** A cache hit returns the identical digest a full
  rehash would produce; any of {replace, in-place modify, truncate/grow,
  chmod/utimes} invalidates the entry. Sabotage arm: modify one weight
  byte without changing size/mtime_ns/ctime_ns is NOT required to be
  caught by the stat key (that is the documented stat-only threat model),
  but MUST be caught by `strict-hash`.
- **G2 startup cost.** Warm `cached-hash` startup identity cost is a small
  metadata lookup (no full weight read); cold cost equals current
  whole-file hash.
- **G3 snapshot semantics unchanged.** A snapshot restores only into an
  engine whose artifact digest matches, exactly as today; the digest value
  recorded in snapshot headers is unchanged in meaning (now SHA-256).

## Ship / kill line

Ship only if fresh-process breakdown confirms the current whole-file
identity scan is material to snapshot/restart latency. If a warm
`cached-hash` lookup saves under ~1 s of user-visible restart time in the
measured configuration, keep the current per-process lazy hash and close
this item as not-worth-it. No broad snapshot-v3 arena rewrite is justified
by the offline eval workloads; this is the smallest change that preserves
current semantics.
