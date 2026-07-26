# Boot prefetch for the disk snapshot store — **NO-GO** (measured 2026-07-25)

Audit item A5 ([../PARITY-2026-07-25.md](../PARITY-2026-07-25.md)), Metal
counterpart of upstream P16c (`d465e0e`). Built, pre-registered with a kill
line, **measured, killed, and removed.** The flag and
`DiskSnapshotStore::prefetch_recent` no longer exist; this document is the
record.

## The claim under test

A restart leaves the page cache cold, so the first snapshot restore pays a
full disk read. Reading recent entries at boot moves that cost off the first
request. Upstream measured 681 → 36–39 ms on the 5090 box and shipped it.

## Kill line (pre-registered, before the run)

> **SHIP** if first-restore wall time drops by **≥150 ms** and boot does not
> regress >50 ms. **KILL** if boot regresses >100 ms, or the restore win is
> **<75 ms** — record it as a NO-GO with the mechanism, and delete the flag.

## Result: KILL

A/B, n=3 per arm, real 610 MB snapshot, 7007-token prompt, base M4. Each
boot's own 17 GB artifact read is what evicts the snapshot from the page
cache, so the no-prefetch arm is genuinely cold.

| arm | first-restore request | boot→health |
|---|---|---|
| no prefetch | 17107, 16866, 15721 ms (mean 16565) | mean 35486 ms |
| `--snapshot-prefetch 2` | 18466, 15241, 15267 ms (mean 16325) | mean 34572 ms |

Mean delta **+240 ms**, against a within-arm spread of **1386 ms** and
**3225 ms**. The effect is not resolvable; boot did not regress either.
Restore win is not ≥150 ms → **KILL**.

## Mechanism — why it could never have worked here

The trace breaks the matching request down (`tms` = ms since boot):

| phase | time | share |
|---|---|---|
| disk restore (`request` → `prefix` event) | **230 ms** | **1.6 %** |
| post-restore work (95 residual tokens + 1 output) | **14570 ms** | **98.4 %** |
| total | 14800 ms | |

The 230 ms restore matches the host-side prediction exactly (610 MB at the
measured ~2.8 GB/s uncached = ~218 ms), so the model of the read was right —
**the read was simply never the bottleneck.** Prefetch can remove at most
`230 − 36 = 194 ms`, i.e. **1.3 % of the request it exists to speed up**,
which is an order of magnitude below the run-to-run noise.

This conclusion generalises off this box. Even on a disk 10× slower the read
would be ~2.3 s against ~14.6 s of post-restore work — still not the
dominant term, and prefetch would still be attacking the smaller one.

Upstream's number is not contradicted; their premise differs. Their restore
was ~245 ms end to end, so a 475 ms read *was* the whole cost. On Metal the
restore is 1.6 % of a path dominated by residual prefill.

## What the measurement DID establish (both worth keeping)

**1. The disk snapshot tier is enormously valuable on Metal — far more than
on CUDA.** Same-length prompt, snapshot hit vs no hit:

| | time |
|---|---|
| 6912 of 7007 tokens restored from disk | **14.8 s** |
| non-matching, 5606 tokens, full prefill | **322.2 s** (57.5 ms/token) |

**~21.8× on this shape.** Prefill is brutally memory-wall-bound on a base
M4 (every chunk re-reads a 17 GB model over 120 GB/s), which is exactly why
the tier matters more here than on a 5090. Worth stating plainly in the
serving docs.

**2. New finding — residual prefill after a restore is the real lever.**
Handling the **95** tokens the snapshot did not cover cost **14.57 s**. At
the measured bulk rate of 57.5 ms/token those 95 tokens should cost ~5.5 s,
so roughly 9 s is chunk-granularity overhead: a 95-token tail still pays
close to a full chunk's weight traffic.

That is ~60× the lever boot prefetch was chasing, and it is attackable
without new I/O machinery — e.g. bank snapshots at the exact prompt boundary
rather than a chunk multiple (6912 = 27×256), or make the final partial
chunk cheaper. **Filed as the next A5 successor; not yet investigated.**

## Reproduction

```
./build/q27-metal-server MODEL TOK --port 8231 --ctx 16384 --slots 1 \
    --snapshot-dir DIR --snapshot-auto 512 --trace trace.jsonl
# bank:  POST /v1/completions {"prompt": <~7000 tok>, "max_tokens":4, "snapshot":true}
# restart, then re-POST the same prompt; read `request`/`prefix`/`outcome` tms
# deltas out of trace.jsonl.
```
