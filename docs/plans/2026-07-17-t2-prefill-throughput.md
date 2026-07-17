# T2 chunked-prefill throughput — measurement pre-registration (2026-07-17 night, STAGED — awaiting Daniel's go)

## The finding (live serving, 24 GB M4)

Daniel's first real pi.dev request ("yo" + pi's ~14 KB system prompt + 6
tools = **8,355 prompt tokens**) appeared hung. Captured the exact body and
timed it against the T2 server: **prefill 8,355 tokens in 4:44 at
--ctx 131072 and 5:23 at --ctx 16384** — ≈ **28 tok/s, ctx-independent**,
so this is NOT context-window scaling. 8,355/96 ≈ 87 chunks → **~3.3 s per
96-token chunk**, against T2 serial decode at ~11.5 tok/s (87 ms/token): the
chunk path is barely 3× decode per token when weight-amortization should
put it well above 10×. Thread samples during the stall: dominant frame
waitUntilCompleted under prefill_chunk with steady chunk progress — a
legitimately slow path, not a wedge. Agentic serving impact is first-class:
every fresh agent session pays a ~5-minute TTFT, requests queue behind it
on the single 128K slot, clients disconnect, and the server grinds on for
nobody (the no-cancel-for-abandoned-requests debt, already registered,
turns one slow request into a pile-up).

Interim mitigation DEPLOYED (2026-07-17 night): the serving instance now
runs with Q27_METAL_SNAPSHOT_DIR=~/.q27/snapshots and a hinted warm request
seeded a disk snapshot at the 96-aligned boundary (~8,256 tokens) *before*
the user turn — pi's system prompt is session-independent (verified: no cwd
in the sniffed body), so every future pi session should restore it and
prefill only the tail. Chronicle entry rides the measurement round.

## Pre-registered measurement (no optimization in this round)

Instrument: metal_prefill_bench (standing) + one served-path timing per
tier via the captured pi body (logs/agentic-parity-20260717/pi-sniff.json
derivative). Quiet machine, one model resident at a time.

- Arm 1 — official tier chunk prefill tok/s (the reference chunk path).
- Arm 2 — T2 chunk prefill tok/s (expect ≈28 if the bench reproduces the
  served number; a large bench-vs-served gap redirects the diagnosis to
  the server loop, not the kernels).
- Arm 3 — B1 chunk prefill tok/s (same bonsai chunk-matmul class; tells
  whether the cost is bonsai-generic or T2-specific).
- Attribution leg: per-op split of one T2 chunk (matmul_quantized vs
  attention_chunk vs GDN vs sync overhead) — Q27_METAL bench flags or a
  one-off timestamped run.

Expectation bands (falsifiable): official ≥ 10× its serial decode rate;
T2 bench within 2× of the served 28 tok/s. Output: a diagnosis naming the
dominant term; any optimization round gets its own pre-registration with
ship/kill lines after the numbers exist.

## RESULTS

(awaiting go)
