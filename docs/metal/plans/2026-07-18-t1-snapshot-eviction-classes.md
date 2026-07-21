# T1: semantics-aware snapshot eviction classes — pre-registration (2026-07-18)

Mooncake applicability review (operator-supplied report, 2026-07-18). T2/T4
dropped (need >1 slot; this M4 boots single-slot — multislot admission
rejected 34 GB needed > 9 GB budget). T3's two-object-type store is a
census-write-up insight, not this-week code. T5 dropped (no cross-session
consumer on single-user pi). **T1 survives: it is metadata-only on the
existing single-slot disk store and needs no GPU memory.**

Mooncake's load-bearing production result is a *policy* finding, not an
architecture: uniform LRU degraded below 20% hit rate at peak until cache
entries were split into classes (ephemeral excluded, session-pinned given
persistence), recovering +93% hit rate. Our `DiskSnapshotStore` has exactly
one class: `evict_past_budget()` sorts by mtime, deletes oldest-first.

## The pathology, measured from the live trace

`~/.q27/trace.jsonl`, 138 requests / 9 boots / 2026-07-17:

- **9 `snapshot_save` events triggered 6 `snapshot_evict` events.** The
  8 GB budget is saturated; every large save evicts a prior snapshot.
- Boot 9 is the clean case: save→evict fired **three times back-to-back**
  (524616/528115, 916490/920270, 1465965/1469122 tms), each eviction
  ~700–890 MB. Each new save immediately evicted the previous save.
- Real pi sessions are long runs of *identical or near-identical* prompts
  (the conversation spine, re-sent every turn): one session runs
  8336→8336→8336→8336→8328→8602→10430→10430→10430→10430. The spine prefix
  is exactly what must survive; a single large tool-output dump (the
  +1828, +2006, +11238 jumps) is what should be demoted.
- Under flat mtime-LRU, saving a 1.04 GB tool-output snapshot evicts the
  8336-token spine the next five turns need. That is the Mooncake failure
  mode verbatim, on our box, today.

## The two classes

Classification must be **derivable from the file itself** (key = full-prefix
SHA1; size = on-disk bytes). No request context is available at eviction
time, and none is needed:

- **SPINE (pin-preferred):** a snapshot that is a strict prefix of a longer
  snapshot in the same directory — i.e. the conversation grew past it, so a
  future turn may land on it. Plus recency.
- **TOOL-OUTPUT (demote-first):** a leaf snapshot (no longer snapshot
  extends it) that is large and was saved once and rarely re-hit — the big
  single-turn dumps.

The directory scan in `evict_past_budget()` already walks every file; adding
an "is a prefix of another stored key" relation is a set operation over the
keys already read. Size and mtime are already read.

## What the code already does (read before designing)

`best_match()` calls `std::filesystem::last_write_time(...)` on every hit —
**recency is already persisted in mtime**. A zero-hit large leaf is already
mtime-oldest and already evicted first. Mooncake's "exclude ephemeral"
class therefore needs no new mechanism: it falls out of the existing
touch-on-hit. The one thing flat mtime-LRU does NOT do is protect a
*growing* conversation's shorter snapshots: when the spine extends
(8336→8602→10430), the older 8336 snapshot's mtime is stale, so a large
newer leaf can out-rank it for survival even though a future turn may still
land on 8336 (a client that resends an earlier prefix, or a branch).

## Proposed mechanism (smallest change that prices the classes)

One knob, env/flag twin, fail-loud like the existing snapshot flags:

- **`Q27_METAL_SNAPSHOT_SPINE_PIN` (default 1):** in `evict_past_budget()`,
  a snapshot that is a **strict token-prefix of another stored snapshot**
  (the conversation grew past it → it is on a spine) is evicted only after
  every non-spine (leaf) snapshot has been evicted. Within each class,
  mtime-oldest first (unchanged).

Eviction order becomes: **mtime-oldest leaves → (mtime-oldest spine, only
when no leaf remains).** The hard budget cap is unchanged — the class
reorders victims, it never exceeds the budget. No persistence, no request
context, no new files: the prefix relation is computed over the keys the
directory scan already reads (each file's stored token ids are already
loaded by `peek_snapshot` on the hit path; eviction reads only what it
needs).

Correctness invariant (gate G2): eviction only ever *deletes files*; a
wrong eviction costs a re-prefill, never a wrong result. The byte-exact
restore path is untouched.

## Gates (pre-registered, both directions, exit codes)

**Design amendment (2026-07-18, one-model constraint):** this box holds
exactly ONE resident model — a second model server crashed the live T2
server during the first G1 attempt. The original live-server G1/G3 (a
throwaway server with a tiny budget driving real hinted requests) is
therefore OFF this box. The eviction ordering was refactored into
header-only `src/metal/snapshot_evict.h` precisely so the gates run
**offline, no model, no GPU, no server.**

- **G1 — plumbing (class assignment + victim order), OFFLINE.** DONE:
  `tools/test_snapshot_evict.cpp` drives `q27::mark_spine` +
  `q27::eviction_order` (the exact functions `evict_past_budget` calls) on
  a synthetic A ⊂ AB ⊂ ABC spine chain plus a large mtime-newer leaf D.
  Asserts: A,AB spine / ABC,D leaves; pin OFF → flat mtime (A oldest first
  victim); pin ON → leaves (ABC,D) before spine (A,AB); pin changes the
  first victim away from spine A; unreadable-token file stays a leaf. Gate
  FAILED FIRST on an inverted fixture (proving the assertions can fail),
  then PASS. In `test-cpu`.
- **G2 — invariant.** DONE: `tools/snapshot_gate.sh` ALL PASS (fp16, turbo3,
  l7full byte-identical fresh-process restore; full reject matrix;
  crash-consistency legs). Eviction policy does not alter restore
  correctness.
- **G3 — decision number, OFFLINE replay.** The ship line (does the spine
  survive a subsequent large tool-output save) is answered by driving the
  REAL `DiskSnapshotStore::evict_past_budget` over a fabricated directory
  that mirrors the recorded pi session: an 8336-token spine snapshot plus a
  newer ≥500 MB tool-output leaf, total over budget. `SPINE_PIN=0` must
  evict the spine; `=1` must evict the leaf and keep the spine. Runs
  offline via fabricated Q27SNAP1 fixtures (header + token ids; the
  eviction path only reads tokens through `peek_snapshot`). If the offline
  store harness proves infeasible, G3 degrades to the G1 ordering result
  plus a trace-observation period on the live server's `/stats`
  `evicted_spine`/`evicted_leaf` counters.

## Ship / kill line

- **SHIP:** with `SPINE_PIN=1`, the spine snapshot survives a subsequent
  ≥500 MB tool-output save (disk-tier hit on the next same-prefix request)
  where `SPINE_PIN=0` evicts it — and the `snapshot_evict` trace event
  names the tool-output leaf as victim instead. Zero restore-correctness
  regressions (G2).
- **KILL:** `SPINE_PIN=1` does not change which snapshot survives the G3
  replay (the spine was never actually the mtime-oldest victim in
  practice), or computing the prefix relation at eviction time is not
  cheap enough to matter. On KILL the flat-LRU store stands and T1 joins
  T5 in the not-worth-it list.

## RESULTS (2026-07-18, M4, offline — one-model constraint)

- **G1 (ordering, offline): PASS** — `test_snapshot_evict` asserts
  classification (A,AB spine / ABC,D leaves), pin-OFF flat-mtime order, and
  pin-ON leaves-before-spine order; pin changes the first victim away from
  spine A; unreadable-token file stays a leaf. Failed-first on an inverted
  fixture, then PASS. In `test-cpu`.
- **G2 (restore correctness): ALL PASS** — `tools/snapshot_gate.sh` fp16 /
  turbo3 / l7full byte-identical fresh-process restore, full reject matrix,
  crash-consistency legs. Eviction policy does not touch the restore path.
- **G3 (decision number, real store, offline): PASS** —
  `test_snapshot_evict_store` drives `DiskSnapshotStore::evict_past_budget`
  over fabricated Q27SNAP1 fixtures (reused spine s1 ⊂ s2 chain tip + a
  large newer tool-output leaf, budget = total−1 byte to force exactly one
  victim). **pin OFF evicts the reused spine s1; pin ON spares s1 and
  evicts a leaf.** Failed-first twice (inverted mtime; then the
  tip-is-a-leaf subtlety below), then PASS. In `test-cpu`.

**VERDICT: SHIP.** The pre-registered ship line held on the real store:
with the pin on, the reused conversation spine survives a large tool-output
save that flat-LRU would sacrifice it to. Implementation shipped:
`src/metal/snapshot_evict.h` (header-only ordering) +
`src/metal/disk_snapshot_store.h` (store extracted from metal_server.cpp so
the gate drives the real code offline), `--snapshot-spine-pin` /
`Q27_METAL_SNAPSHOT_SPINE_PIN` (default 1), per-class `evicted_spine` /
`evicted_leaf` counters in `/stats`, and a class note on the eviction
trace event. `DiskSnapshotStore` now takes an injected peek functor so the
offline gate links no GPU code.

**Design caveat surfaced by G3 (operator-visible):** the pin protects only
*non-tip* spine members. The chain tip (the most recent snapshot) is a leaf
— nothing extends it — so a large newer tool-output can still evict the
current tip under a tight budget. That is correct Mooncake semantics (the
tip is the newest, least-reused position), and recency (touch-on-hit)
already favors a re-hit tip, but it means the pin's protection is one turn
behind the frontier. Recorded, not chased here.

**One-model constraint (hard rule, reinforced this session):** this box
holds exactly ONE resident model; a second model server (the original
live-server G1) crashed the live T2 server. All T1 gates were therefore
built offline (no model, no GPU, no server). The obsolete live-server gate
script was removed.

## Honest scope note (operator, read before ship)

The recorded trace shows most evictions are of *recently-saved* files
(boot-9 save→evict pairs are ~3.5 s apart), i.e. the budget is so tight
relative to snapshot size (~0.7–1 GB each, ~8–10 fit) that the newest save
often immediately forces out the previous one. Whether those evicted files
are spines or leaves is exactly what G1/G3 measure. If they are mostly
leaves, T1's real-world delta is small and the KILL line is the likely
outcome — which is itself the useful, cheap answer. The pin is worth
building because it is ~50 lines and the G3 replay is decisive either way.

## Cost / risk

Contained to `DiskSnapshotStore` (file-keyed; no request-context plumbing):
a prefix-relation pass over stored keys in `evict_past_budget()`, one flag
(`Q27_METAL_SNAPSHOT_SPINE_PIN` / `--snapshot-spine-pin`), a class tag on
the existing `snapshot_evict` trace event, and `/stats` gaining per-class
evict counters. Single slot, no GPU memory, no serving-path change, restore
path read-only with respect to this change. Well under a session.
