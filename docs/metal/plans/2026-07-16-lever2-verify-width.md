# Lever 2 — verify width past 12 (verify-round-cost plan, second lever)

Status: PRE-REGISTERED before measurement (results appended below the
line). Parent: 2026-07-16-verify-round-cost.md lever 2 (the whole prize
after lever 1 parked at 0.78×). Gates and kill line inherited verbatim:
oracle sweep w ∈ {16, 24, 48}, parity at widths actually dispatched
(vacuous-gate lesson — the width list must include > 12), committed
byte-identity vs width-12 verify, **S(48) < 3× → stop at the width the
curve flattens**.

## What changes (and what explicitly does not)

`VERIFY_CHUNK_MAX = 48` decouples the verify/oracle width ceiling from
`CHUNK_MAX = 12`, exactly as `PREFILL_CHUNK_MAX` decoupled prompt
ingestion. The width-12 surfaces are untouched by construction — the
NLL/KL instruments and the wide-prefill head slicing reference
`CHUNK_MAX` literally, so their contract cannot drift:

- Grows to VERIFY_CHUNK_MAX rows: `cfinal_`, `clogits_` (11.9 → 47.7 MB),
  `cpred_`, and the per-GDN-layer replay parks `park_qkv_/g_/beta_`
  (+~70 MB across 48 layers; ~110 MB/engine total, ×2 slots ≈ 220 MB —
  fine against the 10.7 GiB working set, noted for the future G6
  admission budget).
- Width caps raised to VERIFY_CHUNK_MAX: `chunk_forward(verify=true)`,
  `oracle_round`, CLI `--oracle` (2..48; `-n ≥ width+2` rule unchanged).
- Stays at CHUNK_MAX = 12: `mtp_round` (the MTP lane/mask machinery is
  untestable on this rig — no MTP layer in the T2 artifact; raising it is
  recorded residue for the 24 GB machine), `teacher_force*` NLL/KL
  slicing, `cnll_`/`ctargets_`, `wide_head_stage_`.
- No kernel changes: the chunk kernels already run 96-wide for prefill;
  the verify path's head matmul at 48 rows rides the same
  `matmul_quantized` the chunk-parity gate certified at widths {17,48,96}.

## Gates (all runnable on this rig, short)

1. `test_metal` + `test_metal_ops` (existing suites).
2. Oracle sweep, T2 artifact, `-n 128`, w ∈ {12, 16, 24, 48}: each run
   carries its own built-in gates — serial-reference byte-identity across
   passes 1/3, prompt re-ingest identity, the probe-logits state gate
   (bounded ~1e-3 class, able to fail at 0.25), position identity, and
   the observational agreement counter. w = 12 re-establishes the
   baseline on the post-byte-LUT kernel (the 424 ms round should read
   slightly better than yukon's measurement).
3. Widths actually dispatched: the sweep IS the dispatch evidence
   (Q27_ORACLE_TRACE prints live per round; the w=48 run must show
   live=48 rounds, not silently clamped).
4. Committed byte-identity vs width-12 is structural for the oracle
   (commits ARE the reference lanes, and pass-3 re-measure must
   reproduce them) — the state gate is the non-vacuous check.

## Pre-registered reads

- S(w) curve from the sweep; **S(48) ≥ 3× = the parent plan's bar**
  (break-even ~2.9 tok/round class, inside DSpark's measured 3.8).
- S(48) < 3× → stop at the width where the curve flattens; record the
  flat width as the production VERIFY_CHUNK_MAX and re-scope.
- Round-cost prediction to check against (parent plan arithmetic): if
  the verify chunk stays weight-stream-shaped, round cost should stay
  ~flat in w up to 48 and S should scale ≈ linearly with committed
  tokens per round; if round cost grows with w, the curve flattens where
  the extra head/argmax rows start to bite (head at 48 rows is 4× the
  logits matmul of w=12 — 248320×5120 at x_rows 48).

---

## Results (appended post-measurement, same day, M4 16 GB)

**VERDICT: the S(48) ≥ 3× bar FIRES — S(48) = 3.938× mean (4.138× warm),
state gate PASS at every width, agreement 127/127 throughout.**
VERIFY_CHUNK_MAX = 48 ships.

### The sweep (T2 artifact, -n 128, greedy, ctx 512)

| w | round mean | tok/round | oracle wall | S(w) mean | state gate |
|---|---|---|---|---|---|
| 12 | 356.3 ms | 11.55 | 32.4 tok/s | 2.933× | PASS (0.337) |
| 16 | 369.2 ms | 15.88 | 43.0 tok/s | **3.859×** | PASS (0.337) |
| 24 | 622.6 ms | 21.17 | 34.0 tok/s | 3.041× | PASS (0.337) |
| 48 | 939.4 ms | 42.33 | 45.1 tok/s | **3.938×** | PASS (0.337) |

Serial bracket stable (G ≈ 87–92 ms/tok across all runs). Baseline note
confirmed: the w=12 round reads **356 ms vs yukon's 424 ms** — the
morning's byte-LUT + vector-store round bought the verify batch ~16%
before lever 2 added anything (S(12) 2.21× → 2.93× on kernel work alone).

### The round-cost structure (the sweep's real finding)

Round cost is **flat per 16-token TILE, not flat per width**: the chunk
GEMM tiles tokens 16 per threadgroup, so w ≤ 16 streams the weights once
(356 → 369 ms, +3.6% from w=12 to 16), w = 24 opens a second token tile
and pays a second full weight stream (623 ms), w = 48 pays three
(939 ms). Consequences:

- **Full-tile widths are the sweet spots**: w ∈ {16, 32, 48}. S is
  locally maximal there; mid-tile widths waste a partial weight stream.
- The curve does NOT flatten by 48 (3.86 → 3.94 across full-tile
  points) — per the pre-registered read, 48 stands as the ceiling.
- Per-round break-even (round/serial): ~4.1 tokens at w=16, ~10.7 at
  w=48. Drafter guidance: short-acceptance drafters (DSpark block-4,
  ~3.8 tok/round measured) sit just UNDER w=16's break-even — the
  drafter question stays open per the parent plan's non-goal; long
  exact runs (suffix bursts, agentic traffic) are the w=48 customers.

### Two instrument fixes along the way (both attributed before changing)

1. `argmax_rows` backend cap was a literal 12 — raised to 48 (the kernel
   is one-threadgroup-per-row, width-agnostic). The width-12 NLL cap
   directly below it is untouched (the contract).
2. The oracle state gate's 0.25 probe-logits tolerance **false-failed
   inside the measured numeric class**: this prompt reads max|Δ| 0.3373
   — reproduced bit-for-bit on the pre-lever-2 tree (git-stash A/B), and
   the 384-position chunk-parity serial control puts the class envelope
   at 0.646. Recalibrated to 0.7 (envelope + margin, per the
   margin-aware contract); the gate still fails on real corruption
   (orders-of-magnitude logit movement, probe argmax/position flips).
   yukon's 0.031–0.092 observations were three samples from the shallow
   end of the class.

### Gates

test_metal + test_metal_ops pass; oracle built-in gates (serial
byte-identity across passes 1/3, re-ingest identity, position identity,
probe state) PASS at all four widths; 48-wide rounds actually dispatched
(42.33 tok/round = 48+48+32 over 3 rounds; Q27_ORACLE_TRACE live lines);
multislot suite re-run with the grown buffers (~110 MB/engine:
clogits_ 11.9 → 47.7 MB, gdn parks +70 MB — noted for the G6 admission
budget).
