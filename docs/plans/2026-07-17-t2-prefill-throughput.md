# T2 chunked-prefill throughput — measurement pre-registration (2026-07-17 night; the operator's go: "yes, please try to fix")

## The finding (live serving, 24 GB M4)

the operator's first real pi.dev request ("yo" + pi's ~14 KB system prompt + 6
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
the operator's go alongside B1 select round 2.

## Port round RESULTS (2026-07-17 morning, the operator's go: "run official q4 port")

Contract check first: no standing gate expects byte-identity of GEMM output
vs the old kernel — the shape suite is tolerance-gated vs the serial GEMV,
`--chunk-parity` compares widths of the SAME engine build (both on the new
kernel), and the bit-exact contracts (GDN replay) don't touch projections.
Clear to proceed.

**Q4 half kernel (`q27_matmul_q4_mm_h`) — all correctness gates PASS:**

- Written as the mechanical G′ transplant of `q27_matmul_t2_mm_h` (half
  tiles, raw ints, float mixed-MMA accumulators, 64-K staged tiles, two
  accumulator pairs, one flush barrier region) with the Q4 deltas: weight
  stride cols/2, uint2 loads, 256-entry half2 nibble-pair LUT (low nibble
  first, generated not hand-typed), scale group 64 cols = exactly one K-tile
  (index c0/64). Numerics are STRONGER than the float-staged kernel it
  replaces: products bounded by 8×128 = 1024 < 2048 are exact in half
  wherever the MMA rounds, and the old per-value activation staging round
  disappears.
- `test_matmul_shape` suite: PASS both `Q27_METAL_GEMM_HALF` settings
  (true exit codes checked, per the masked-failure lesson).
- `--chunk-parity 384` on the official artifact: widths 17/48/96
  bit-identical to width 12 — every metric exactly zero, exit 0.
- 8K NLL A/B (wikitext-2, `--nll-long 8192 --ctx 8192`): float
  1.5742/1.9217 (0-2k/2k-8k) vs half 1.5742/1.9218 — ΔNLL +0.005%,
  two orders inside the ±0.5% gate.
- Codex autoreview: clean, no P1/P2/P3 (static; suite runs done locally).

**Q8 half kernel — FAILED and PARKED (the empirical answer to the open
product-precision question):** `q27_matmul_q8_mm_h` missed the shape-suite
bound at the high-cancellation 33×5120 repro (5.5e-4 vs 3e-4). Q8 products
(up to 127×127) exceed half's 2048 exact-integer range, and the failure
shows the mixed-precision MMA rounds products at HALF precision, not at
the float accumulator's — Q4/T2/B1 are immune (products ≤ 1024). Q8
stays float-staged (~8% of wall; a split-nibble staging would restore
exactness at 2× the MMA work — not worth it). Kernel kept in the .metal,
never routed, verdict in its comment.

**Ship-line bench: PENDING QUIET MACHINE — commit HELD.** the operator was
actively using the box (WindowServer ~30%, Brave ~25%) and the timing legs
were uninterpretable: the float baseline itself swung 23.44 → 18.26 tok/s
between runs; half read 21.39/18.69/20.76 across three runs (a 960-token
profiled pair suggested ~1.35×, hot 8352 back-to-back ~1.0×). Per the
pre-registered band ("8352/96, quiet"), no promotion on contaminated
numbers: a watcher (verdict file logs/q4port-20260717/quiet_bench.verdict)
waits for ≥10 min input idle, stops the T2 server, runs float/half at 8352
(×2) and 960, restarts the server. ≥1.7× ships default-ON; <1.7× ships
nothing but the parked kernels + this honest table, per the kill line.
