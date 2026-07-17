# T2 chunked-prefill throughput — measurement pre-registration (2026-07-17 night; Daniel's go: "yes, please try to fix")

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

## RESULTS (2026-07-17 night, quiet 24 GB M4)

**No regression anywhere — chunked prefill was simply never optimized past
its first shipped schedule, and tonight was the first time real agentic
traffic priced that in.** The arms:

- Official q4q8, 8352-token synthetic prompt: **23.2 tok/s** (4138 ms/chunk,
  12.08 GiB weight stream/chunk at 3.1 GB/s effective). Per-kernel profile:
  **q27_matmul_q4_mm is 91.8%** of the wall at 10.9 ms/dispatch. History:
  that kernel's introducing commit (efb5d5d) recorded "prefill ceiling
  5.7 → 21.3 tok/s" — today's number IS the introduced ceiling.
- T2: **40.2 tok/s at 960 tokens, 27.55 tok/s at 8352** (attention depth
  growth accounts for the fade) vs the **served 28–29 tok/s — the serving
  loop adds ≈ nothing**; the engine runs at its bench ceiling.
- Roofline (metal_mma_roofline, arm C = the T2 chunk GEMM): the standing
  pre-registered verdict holds — **M4 prefill MMA is MATURE at this
  schedule** (~2.3 TFLOP/s with staging; pure-MMA arm A ~3.4; f16-acc,
  function-constant, and stripe levers all previously PARKED). T2 has no
  kernel headroom left on this device.
- The official kernel runs the SAME FLOPs as T2's ~1.9× slower (equal
  effective GB/s over 2× the bytes): q27_matmul_q4_mm still carries the
  original efb5d5d tile schedule and never inherited the T2 GEMM's staging
  maturity. That gap is the one real kernel prize; pre-registered below.

**Fix shipped — auto-snapshots (the serving-tier fix):** with the snapshot
store opted in, prompts ≥ 4096 tokens (Q27_METAL_SNAPSHOT_AUTO; 0 =
hint-only) now behave as hinted, so pi/Claude Code-class clients get the
disk-prefix mitigation without cooperating. Gates (all PASS, live T2
server): cold no-hint pi body auto-saved (disk_saves 1, one-time 4:54);
different user message on the same prefix answered in **8.1 s with
prefix_hit 8256/8368** (disk_hits 1); small prompts save nothing. Codex:
no P1; P2(a) same-path save race REJECTED with evidence (save_state is
lease-serialized; worst case a redundant identical rewrite), P2(b) write
churn on non-repeating large prompts ADOPTED as a documented trade +
env off-switch (this box serves repeated agent prefixes).

## Pre-registered next round — Q4/Q8 chunk-GEMM schedule port (official tier)

Port the mature T2 MM schedule (q27_matmul_t2_mm_h: 32×16 tiles, 64-K walk,
half staging cadence) to q27_matmul_q4_mm/_q8. Ship line: official-tier
prefill bench (8352/96, quiet) ≥ 1.7× current (≥ 39 tok/s); correctness =
CPU-reference gate (1e-3) on all production shapes + ΔNLL within ±0.5% on
the official 8K leg + chunk-width parity legs; < 1.7× after one candidate
ships nothing and records the honest table. Note the summation-order
contract must be checked against the chunk-parity gate's expectations
before promotion (byte-identity vs the old MM kernel is NOT expected).
Beneficiary is the official tier (T2 serving is unaffected); queued behind
Daniel's go alongside B1 select round 2.
