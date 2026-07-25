# Boot prefetch for the disk snapshot store — pre-registered experiment

Audit item A5 ([../PARITY-2026-07-25.md](../PARITY-2026-07-25.md)), Metal
counterpart of upstream P16c (`d465e0e`). **Built, opt-in, default OFF.
Not yet measured on the real restore path — that is what this document
pre-registers, with its kill line, before the run.**

## The claim under test

A server restart leaves the page cache cold, so the first snapshot restore
pays a full disk read. Reading the most recent entries at boot moves that
cost off the first request.

## Host-side lever, measured (2026-07-25, no GPU, no model)

Sequential read throughput on this box's SSD, via `F_NOCACHE` (uncached)
versus a fully page-cached re-read:

| read | measurement |
|---|---|
| uncached, 17 GB artifact | 6072 ms, 6090 ms → **~2.8 GB/s** |
| page-cached, 1 GiB file | 57–64 ms → ~17 GB/s |

So a **1 GiB entry costs ~380 ms cold and ~60 ms warm: a ~320 ms lever**,
once, on the first restore after a restart. Upstream measured 681 → 206 ms
for a 1.09 GB entry on the 5090 box; this SSD is simply faster, and the
lever is correspondingly smaller but the same order.

An earlier attempt to measure this by `dd`-ing a 1 GiB file and reading it
back was **discarded as contaminated** — `dd`'s own writes leave the file in
the page cache, so the "cold" read was partly served from RAM (it reported
201 ms, i.e. optimistic by ~2×). The artifact read above is trustworthy
because that file had not been touched this session.

## Why the Metal design diverges from upstream's

Upstream prefetches **under cover of the weight upload**, arguing it is free
time. That reasoning is discrete-GPU reasoning: their upload is a PCIe H2D
transfer, which leaves the disk idle.

**Metal has no such window.** On unified memory the "upload" is mmap +
page-in of the artifact — the same disk, at the same ~2.8 GB/s measured
above. The artifact is 17 GB, so boot is ~6 s of disk-bound work.
Prefetching 1 GiB concurrently would add ~380 ms of demand to that, i.e.
**contend with boot rather than hide in it** — plausibly spending more than
the 320 ms it saves, and spending it on every boot including the ones where
the snapshot is never used.

So the Metal version runs **after** the engines exist (`Runtime` setup, just
after `evict_past_budget()`), where the disk is genuinely idle and the
server is waiting on its first request. Ordering after eviction also means
it can never warm an entry that is about to be deleted.

Second divergence: the detached thread captures **copies** of the selected
paths and touches no member of the store, so it cannot outlive the store
into freed memory. Upstream's thread reads store state.

Both use explicit chunked reads, not `posix_fadvise(WILLNEED)` — upstream
measured fadvise doing nothing at this size (it is advisory and the kernel
declines a readahead that large).

## Ship line / kill line

Run with `--snapshot-prefetch 2` against a real conversation whose snapshot
is already on disk, page cache evicted between boots, n≥3:

- **SHIP** if first-restore wall time drops by **≥150 ms** (roughly half the
  measured 320 ms/GiB lever, allowing for entries smaller than 1 GiB) **and**
  boot-to-first-token does not regress by more than 50 ms.
- **KILL** if boot regresses by **>100 ms**, or if the restore win is
  **<75 ms**. Either means the idle-disk premise does not hold on unified
  memory and the feature is not worth its complexity. Record it as a NO-GO
  with the mechanism, and delete the flag.

Prefetching more than the working set is the obvious failure mode: with
`--snapshot-prefetch N` the cost scales with N while the benefit is capped
by how many entries the next request actually touches (usually one). Start
at 2.

## Status

- `DiskSnapshotStore::prefetch_recent(n)` — implemented.
- `--snapshot-prefetch 0..64`, default **0 (off)**; surfaced on `/stats` as
  `snapshot_prefetch`.
- Offline gate in `tools/test_snapshot_evict_store.cpp`: selects only
  tag-matching `.q27snap` files (a foreign/mis-tagged file in a shared
  directory must never be read), caps at `max_entries`, no-ops when disabled
  or asked for ≤0, is read-only, and survives the store being destroyed
  immediately after the call. Runs in `make test-cpu`; **G3 PASS**.
- **Unmeasured:** everything above the host-side lever. Needs the
  serialized GPU-exclusive slot.
