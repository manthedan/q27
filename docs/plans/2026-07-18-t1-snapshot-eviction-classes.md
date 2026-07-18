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

- **G1 — plumbing (class assignment + victim order):** synthetic directory
  with a spine chain (A ⊂ AB ⊂ ABC) plus a large unrelated leaf D, total
  over a small budget. Assert the classifier marks A, AB spine; ABC, D
  leaves; assert eviction removes D (leaf) before any spine member even
  when D is mtime-newer. Sabotage: `SPINE_PIN=0` must revert to flat
  mtime order (D survives, oldest spine evicted). Gate must fail both ways.
- **G2 — invariant:** with the pin active, run `tools/snapshot_gate.sh`:
  64-token continuation byte-identical from a fresh process, both KV
  dtypes, 5-case reject matrix fail-loud. Eviction policy must not alter
  restore correctness.
- **G3 — trace replay A/B (the decision number):** replay the recorded pi
  session (8336-spine, then a ≥500 MB tool-output save, then a request
  that lands back on the spine) with `SPINE_PIN=0` vs `=1`, same budget.
  Ship line below lives on this.

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
