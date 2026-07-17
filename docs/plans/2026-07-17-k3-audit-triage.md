# k3 audit triage — 2026-07-17

Source: docs/k3-review-2026-07-15.md (full Metal backend audit: kernels, engine,
backend host; static read, tree at 83e4800 — its D5 inventory already cites the
same-day am_pack fix, so the read is current). Every finding was source-verified
before a verdict. Outcome: **nothing rejected** — the cleanest external audit to
date. Three precision corrections, no verdict changes.

## Verdicts

| # | Sev | Verdict | Disposition |
|---|-----|---------|-------------|
| A1 | MED | CONFIRMED | Lane 1: mirror mtp_round's cfinal_→x1_ copy in both teacher-force paths; gate via --chunk-parity hidden-row assert (E3) |
| A2 | LOW | CONFIRMED | Lane 2: drop divergent `j >= 128` clauses (threadgroup-uniform guards stay); host width guards land as E6 (lane 3) |
| A3 | LOW | CONFIRMED | Lane 3 (E6): host guards for 8-simdgroup trees, block+=8 strides, literal-8 reduce_sum, 128-thread turbo + 1024-thread topk dispatches |
| A4 | LOW | CONFIRMED | Lane 2: comment states the argmax invariant as load-bearing |
| A5 | LOW | CONFIRMED | Lane 2: rewrite from the kernel body, not from either stale text |
| A6 | LOW | CONFIRMED | Lane 3: ordering = default per-resource hazard tracking, not encoder serialization; 4 sites |
| A7 | LOW | CONFIRMED* | Lanes 2+3 in lockstep: delete reduce_row + MatvecArgs/MatvecPairArgs.simdgroups. *Correction: MatmulArgs.simdgroups is LIVE (repurposed as tokens_pad, q27_kernels.metal:2818/:2838/:2932) — untouched |
| A8 | LOW | CONFIRMED | Lane 3: "1..96" |
| A9 | LOW | CONFIRMED | Lane 3: route cols%4≠0 through the existing single-matvec fallback |
| B1 | MED | CONFIRMED* | Lane 1: fsync(fileno) before rename + dir fsync; crash failpoint + snapshot_gate.sh legs (E2). *Correction: a torn page over a blob LENGTH prefix fails pass 1 loud; only torn DATA pages load silently — class stands, worst case narrower than stated |
| B2 | MED | CONFIRMED* | Lane 1: advance position_ after successful finish (mirrors mtp_round); failpoint gate (E4). *Correction: exposure is latent, not live — metal_server.cpp:565 makes every request restore-or-reset(), which rewinds position_ and re-encodes over torn rows; mid-request save_state runs only before any throw. Fix is correct-by-construction, not a live-bug patch |
| B3 | MED | CONFIRMED | Lane 1: pass-2 `stored != bytes` throw |
| B4/B5/B9 | — | confirmations accurate | no action (assigned/parked as cited) |
| B6 | LOW | CONFIRMED | Lane 1 (E7): charge mask_pool_ (~2.0 MB worst case), wide_head_stage_ (240 KB), topk staging in fixed_state_bytes |
| B7 | LOW | CONFIRMED | Lane 3: throw on nil attrib_dummy |
| B8 | LOW | CONFIRMED | Lane 1: clamp+log (dtor can't throw); keep the assert for debug builds |
| C1 | LOW | CONFIRMED | parked: stage 3 (on-GPU draw) is the recorded lever; interleaved (value,index) readback noted for the serve path |
| C2–C4 | — | accurate | no action |
| D1 | MED | CONFIRMED | Lane 2 (E1): three Metal sites (:798, :1125, :3154) move to CUDA's reciprocal-multiply form; metal_gemv_bench.cpp:540 official-leg CPU ref moves in lockstep. MSL builds with MTLMathModeSafe, so today's divide is true IEEE division — the one-vs-two-roundings model is exact. Sizing leg pre-registered below |
| D2 | LOW | CONFIRMED | Lane 2: Goldberg log1p (u==1 ? t : log(u)*(t/(u-1))) at both gate-kernel sites — MSL has no log1p; this tracks CUDA log1pf to ~1 ulp. Fixed rather than documented: two lines |
| D3 | LOW | observation | E8 probe → mini lane (tasks/mini-e8-rope-theta.md), queued behind gqa_partials |
| D4 | LOW | CONFIRMED | Lane 1 (E5): leftover NLL row through nll_rows at rows=1 (chunked engines only; the pre-Apple7 all-serial fallback keeps its CPU path — the audit's concern is the mixed regime inside one chunked pass) |
| D5/D7 | — | verified accurate | no action |
| D6 | LOW | CONFIRMED | orchestrator: one METAL_PROGRESS line scoping the bonsai "exact ternary math" claim to serial decode (chunk path consumes int8-quantized activations via matmul_quantized) |

Audit misses (recorded, not held against it): a FOURTH quantize site exists —
q27_b1_x_prep (q27_kernels.metal:598, `round(xv/s)`). Out of D1 parity scope by
decision: B1 is Metal-native (no CUDA twin), its CPU model in metal_gemv_bench
matches the kernel form, and the tier's quality battery was certified two days
ago — churning it buys zero parity. One-line scoping comment added instead.

## Execution shape

Three parallel worktree lanes by file ownership (subagents edit/compile only;
all runs orchestrator-owned):
- **Lane 1** — metal_engine.cpp/.h, metal_cli.cpp, tools/snapshot_gate.sh:
  A1+E3, B1+E2, B3, B2+E4(engine side), D4/E5, B8, E7.
- **Lane 2** — q27_kernels.metal, tools/metal_gemv_bench.cpp: D1/E1, D2, A2,
  A4, A5, A7(MSL side).
- **Lane 3** — metal_backend.mm: A6, A8, A9, B7, E6, A7(host side),
  Q27_METAL_FAIL_FINISH failpoint (E4 backend side).
Cross-lane contracts (frozen here): MatvecArgs/MatvecPairArgs lose their
trailing `uint simdgroups;` on BOTH sides, MatmulArgs untouched; failpoint env
is `Q27_METAL_FAIL_FINISH=<n>` (throw on the nth finish), snapshot failpoint is
`Q27_METAL_SNAP_CRASH=before-fsync|after-rename` (_exit(42) at that point).
Mini: E8 probe task, queued behind mini-e2-gqa-partials.

## Pre-registered verification (bands before runs)

1. **Suites**: test-cpu + test-metal green post-merge. No CUDA changes this
   batch (D1 moves Metal toward CUDA) — yukon untouched.
2. **D1 sizing leg** (official tier, quiet M4): 8K wikitext NLL, same protocol
   as the 0A-q27 battery, pre-change vs post-change binaries. Band: |ΔNLL|
   ≤ 0.5% relative (noise class). Expectation: rare-event class, delta ~0.
   Above band → stop and investigate before commit, per house rule.
3. **Snapshot crash gate** (B1 artifact, 3.79 GB): leg (a) crash before-fsync →
   target path absent or previous snapshot intact + loadable; leg (b) crash
   after-rename → snapshot loads and greedy-16 continuation is byte-identical
   to an uncrashed save. Any silent zero-content load = FAIL.
4. **B2 failpoint gate**: Q27_METAL_FAIL_FINISH mid-generation → throw
   surfaces, position_ NOT advanced for the failed round; reset()+regenerate
   byte-identical to a clean engine.
5. **--chunk-parity hidden-row leg** (T2 artifact): existing parity run must
   now also assert x1_ equality between chunked and serial teacher-forced
   passes (this is what makes A1 a gated fix, not a drive-by).

## RESULTS

(pending)
