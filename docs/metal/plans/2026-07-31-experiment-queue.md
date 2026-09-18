# Shared experiment queue (task spooler) — 2026-07-31

Multiple agents run GPU experiments on this box concurrently. Bare
concurrent runs destroy A/B fidelity (quiescence gating, bandwidth
ceiling). All benchmark/experiment commands MUST go through the shared
task-spooler queue so GPU work executes strictly one job at a time.

## Setup (done 2026-07-31)

- `brew install task-spooler` (formula, 1.0.4 — the viric original; no
  Homebrew cask exists). Binary installs as `ts`.
- `tsp` is a symlink to `ts` (`/opt/homebrew/bin/tsp`) — **use `tsp`**:
  this machine has a pi-shell whose own `ts` (v17.x) can shadow the name.
- One shared queue per box, selected by socket:

```sh
export TS_SOCKET=/tmp/q27-experiments.sock
```

Both agents (upstream-merge and mlx-fast-explore) MUST use the same
`TS_SOCKET`, otherwise they get independent queues and no mutual exclusion.
The queue runs one job at a time (default slots=1) — do not raise it for
measured runs.

## Usage

```sh
# submit (returns job id immediately; jobs run serially)
TS_SOCKET=/tmp/q27-experiments.sock tsp -L mlx-headprune ./build/metal_decode_bench ...

# inspect
TS_SOCKET=/tmp/q27-experiments.sock tsp            # list queue
TS_SOCKET=/tmp/q27-experiments.sock tsp -t <id>    # tail running output
TS_SOCKET=/tmp/q27-experiments.sock tsp -c <id>    # cat finished output
TS_SOCKET=/tmp/q27-experiments.sock tsp -w <id>    # block until done
```

## Discipline

- **Every** measured run goes through the queue: benches, decode/prefill
  A/B pairs, profile captures. No bare runs even "just to check" — a stray
  GPU process invalidates whatever is currently queued.
- Label every submission (`-L <campaign>`) so owners can tell their jobs
  apart in the shared list.
- Job output lands in temp files; copy results into the campaign's logs/
  before clearing (`tsp -C` removes finished jobs).
- Compilation, tests, and other non-GPU work do NOT need the queue.
- The spooler daemon starts on first use; if the socket is stale after a
  reboot, just submit — it respawns. `TS_SOCKET` pointing at a fresh path
  is the only "reset".
