# Verify round cost: cut the flat ~430 ms that prices all speculation

**Status: P0 DONE (2026-07-16, 24 GB M4, Daniel-authorized "go quick") — hypothesis
CONFIRMED at 97.5%; levers 1–2 stand, lever 3 formally dead. Kernel work not
started.** Machine: mac-mini (T2 artifact suffices; every gate here runs on it). Follow-on from Gate 0
(`2026-07-15-sibling-drafter-probe.md` §Gate 0 result): oracle rounds cost
~405–445 ms FLAT across verify widths 2..12 — a fixed ~5.1·G overhead that puts
speculation break-even at ~5 committed tokens/round. Every ms off the round
lowers that break-even; at GEMV-class round cost (~100 ms) break-even drops to
~1.2 tokens/round and every drafter shape (DSpark blocks, suffix bursts,
official-tier MTP) flips from loss to win.

## Attribution hypothesis (pre-registered, from existing measurements)

The flat round is NOT mysterious — the arithmetic closes with numbers already
on the books:

- A 12-wide prefill chunk measured ~342 ms pre-widening (35.09 tok/s sweep,
  wide-chunks entry) = the full 7.15 GB T2 weight stream at ~21 GB/s effective
  ("still ~25 GB/s effective" per the standing chunk-GEMM headroom note).
- Serial GEMV streams the same bytes at ~83 GB/s → ~86 ms/token.
- Oracle round = verify chunk (~342 ms class) + batched head + argmax + 2
  CPU syncs + gdn_replay ≈ 430 ms. Width-invariant because the weight stream
  is width-invariant.

**Prediction to confirm in P0: the verify chunk is ≥ 70% of the round; syncs +
replay + head are the remainder.** If P0 refutes this (verify chunk < 70%),
STOP and re-rank — the levers below assume it.

## P0 — round anatomy trace (measurement only, one short run)

`Q27_ORACLE_TRACE=1` (landed with this plan, same convention as
`Q27_MTP_TRACE`) prints per-round: verify-batch wall, cpred read, commit-batch
(gdn_replay + copies) wall. One `--oracle 12 -n 128` run on the T2 artifact.
Deliverable: the 430 ms split. Decides lever ranking below; ~2 minutes of GPU.

**P0 result (2026-07-16, 24 GB M4, `logs/roundcost-p0-20260716/`):** round
424.2 ms mean at w=12 splits **verify batch 412–419 ms (97.5%), prediction
readback 0.03–0.08 ms, commit batch (gdn_replay + copies) 7–15 ms (~2.3%)**.
The ≥70% prediction fires with 27 points to spare. Consequences: lever 3
(sync count) is DEAD — there is ~10 ms total on the table; levers 1–2 hold
the entire prize and their ranking stands. (Same run: S(12) = 2.207×, third
consecutive confirmation; state gate PASS, agreement 127/127.) Cross-tier
note from the same session's official-tier legs: mtp_round on the 17 GiB
official artifact traces verify ≈ 0.50–0.51 s with draft 0.03–0.09 s and
commit ≈ 0.01 s — the same shape (verify-batch-dominated), so the levers
transfer across tiers.

## Levers (ranked by the hypothesis; each gated)

1. **Chunk-GEMM weight-stream efficiency at w ≤ 16 — phase B direct-RHS
   64-row × 32-token half-tile kernel** (already the standing prefill lever,
   ds4-survey item 2 phase B). This is the SAME kernel family the verify chunk
   rides, so it pays twice: prefill wall AND speculation break-even. Target:
   ≥ 2× effective weight stream at width 12 (21 → 40+ GB/s → round ~250 ms →
   break-even ~2.9 tok/round, inside DSpark's measured 3.8). Gates: existing
   chunk shape suite (exit-code-checked), artifact committed tokens
   byte-identical, oracle re-sweep S(12) ≥ 3× as the acceptance measure.
2. **Verify width past 12.** Decouple `VERIFY_CHUNK_MAX` from the NLL/KL
   width-12 contract exactly as `PREFILL_CHUNK_MAX` was decoupled (buffers:
   `clogits_` grows 11.9 → 47.7 MB at 48-wide — fine). At the measured 48-wide
   chunk rate (48.37 tok/s → ~992 ms/chunk) a 48-lane verify is ~20.7 ms/token:
   S(48) ≈ 4× at perfect acceptance. Realistic use is suffix bursts (long exact
   runs on agentic traffic) and multi-block DSpark chains, both of which cap
   out at today's 12 lanes for no kernel reason. Gates: tiled-parity suite at
   the new widths ACTUALLY DISPATCHED (vacuous-gate lesson — the width list
   must include > 12), oracle sweep extended to w ∈ {16, 24, 48}, committed
   byte-identity vs width-12 verify.
3. **Sync count** (2 → 1 per round): fold the commit batch into the verify
   batch is impossible while acceptance is CPU-decided — this lever is
   GPU-side acceptance, already on the roadmap as "GPU-resident
   drafting/acceptance"; only worth scoping after 1–2 land (it saves one
   ~1–5 ms sync + turnaround, not hundreds of ms, unless P0 says otherwise).

## Kill lines

- P0 shows verify chunk < 70% of round → re-rank before any kernel work.
- Lever 1 lands < 1.3× effective stream at width 12 → the direct-RHS shape
  doesn't fit the M4; park and re-scope around lever 2 only.
- Lever 2's oracle S(48) < 3× → width headroom is dispatch-bound too; stop at
  the width the curve flattens.

## Non-goals

Any drafter integration (DSpark port pricing waits for the post-lever oracle
re-sweep); CHUNK_MAX changes to the NLL/KL instruments (contract stays 12);
CUDA-side conductor work (upstream merge 2026-07-16 covers it).
