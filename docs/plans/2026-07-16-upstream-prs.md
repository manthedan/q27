# Upstream PR queue — approved in principle, HELD until the fork is further along

Status: RECORDED 2026-07-16 night. The maintainer has approved the idea
(merge-back / contributions conversation, Daniel-relayed); Daniel's
explicit call: **do not open PRs yet** — wait until our project is
further along. This doc is the queue so nothing is lost in the meantime.

## Ready-to-send when unheld

1. **`server.cu` CUDA-12.0 compat fix** — lambda-captured structured
   bindings rejected by CUDA 12.0's nvcc; behavior-identical rewrite to
   named tuple references. Fork commit `5d3e981`. Smallest, safest,
   send first.
2. **README width note for 24 GB cards** — the default `Q27_W_MAX=12`
   graph zoo (~2.7 GB) does not fit a 3090 next to the 17.7 GB weights
   (dies in `cudaGraphInstantiate`); `-DQ27_W_MAX=8` fits with ~0.9 GB
   headroom. Evidence: `docs/plans/2026-07-16-3090-graph-oom.md`
   (per-family attribution, both widths measured on yukon). One
   paragraph + maybe a Makefile comment.
3. **`Q27_GRAPH_TRACE=1` instrument** (optional, offer-not-push) —
   per-family graph-memory attribution in `build_spec_graphs`, prints
   the table on instantiate failure so OOM reports self-attribute. Fork
   commit `0a9a65b`-lineage (post-rewrite: resolve by subject
   "instrument: Q27_GRAPH_TRACE=1"). Useful to them now that their
   serving defaults instantiate more graphs (`Q27_BATCH_GRAPH=1` default
   ON since 2026-07-17T00:16Z).

## Send-later candidates (need the merge-back conversation first)

- Backend-split / `src/metal` incremental merge-back series — shape TBD
  with the maintainer (the "agree on a backend split first" plan from
  the issue draft at `~/.claude/jobs/625abbd9/tmp/upstream-issue.md`).
- Format dtypes 6 (`B1_G128`) / 7 (`Q4_1_G32`) registry coordination —
  only matters once merge-back is real.

## Unhold criteria (Daniel decides; suggestions)

The fork reads "further along" when: suffix-burst economics measured
(gate 6), official-tier multislot gates run, Homebrew v0.1.0 tagged and
installable end-to-end. Revisit then.
