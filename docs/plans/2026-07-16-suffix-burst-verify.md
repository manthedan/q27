# Suffix-burst batched verification — the speculation lever that ships

Status: PRE-REGISTERED before measurement (results appended below the
line). Machine: 24 GB M4 (T2 for correctness gates; official tier + timing
on a quiet machine, Daniel-authorized). Follow-on from the day's two
closures: DSpark PARKED (2026-07-16-dspark-port-phase0.md — every drafter
shape now measured dead on M4) and lever 2 LANDED (S(48) = 3.94×,
w ∈ {16,32,48} full-tile sweet spots). Suffix bursts are the named w=48
customer: no drafter model, no acceptance loss on exact matches, proposals
cost pennies on CPU.

## What exists and what is missing

- `SuffixDraft` (src/suffixdraft.h): longest-suffix-match proposer over the
  committed stream. Phase-0 traffic prior (2026-07-09, real streams,
  K=16): **cctx (code-context) fires at L ≥ 12 on 35% of positions, average
  burst 11.5**; docs/neutral traffic ≈ 0% — the gate goes silent instead of
  mis-firing. The customer is agentic/code traffic, exactly where q27
  serves.
- `MetalEngine::generate_suffix` (metal_engine.cpp): **serial** — every
  proposal is verified with an individual `step()`, so acceptance saves
  zero wall time. It is a correctness scaffold, not a speedup; width caps
  at 12 for no kernel reason.
- Lever 2's verify surfaces (`VERIFY_CHUNK_MAX = 48`, chunk_forward verify
  path, `oracle_round`'s caller-lane structure) and `mtp_round`'s
  acceptance walk + early-EOS clamp (e765dde) are all in tree.

## Design (engine)

`suffix_round(pending, remaining, eos, lanes[], n_lanes, committed)`:
`oracle_round`'s verify machinery (caller lanes → one verify chunk →
batched head → cpred readback) with `mtp_round`'s REAL acceptance walk and
commit (accepted prefix + bonus/replacement; EOS clamp + encode rules
inherited verbatim from the e765dde fix — commit_n/encoded clamp at first
EOS, pending==eos early return at the driver).

`generate_suffix` becomes the batched driver:
- per round, `SuffixDraft::propose_with(pending, k, out)`;
  `match < minimum_match` → serial `step()` fallback (no verify dispatch,
  the round costs exactly one GEMV — neutral traffic must not regress);
- `match ≥ minimum_match` → `suffix_round` at n_lanes = min(match-capped
  proposal length, width);
- **width policy is full-tile aware**: proposal counts snap DOWN to
  {≤16, 32, 48} boundaries (a 20-lane verify pays the 32-lane weight
  stream; snapping to 16 is strictly better unless the match reaches 32).
  **Recorded alternative (external review, 2026-07-16, decided by
  measurement not argument): round UP instead for long matches** — a
  17–31-lane match verified at 32 pays one extra weight stream
  (~254 ms ≈ 2.8 serial tokens) for up to 15 more evidence-backed lanes,
  positive EV when per-lane acceptance is high (exact suffix runs are).
  v1 ships snap-down (conservative, never regresses); gate 6's burst
  histogram decides whether match lengths 17–31 carry enough mass to
  fund the round-up variant. If round-up ever ships it needs its own
  prefix-invariance gate: committed tokens/state byte-identical with and
  without the padded tail lanes.
- The old serial path stays as `--suffix-serial` — it is the A/B control
  and the byte-level reference.
- Stats grow a burst-length histogram (drafted/accepted per round +
  match-length distribution) — the width-policy evidence.

CLI: `--suffix W` = batched (W = 2..48), `--suffix-serial W` = the old
serial loop (2..12, unchanged semantics). Mutual exclusions as today.

## Gates (each must be able to fail)

1. Unit suites (`make test-cpu test-metal`) green; SuffixDraft unit tests
   unchanged.
2. **Committed-stream A/B, T2, 96 tokens** (prompt with engineered
   repetition so bursts actually fire): batched `--suffix 16/32/48` vs
   `--suffix-serial 12` vs plain serial — byte-identical committed streams
   expected (suffix verify rides the same tolerance-gated chunk GEMM as
   MTP; any divergence must be the documented low-margin class and fails
   the gate otherwise). Negative control: a corrupted-proposal arm must
   produce different acceptance stats (proves the walk is live).
3. **Widths actually dispatched** must include > 12 (vacuous-gate lesson):
   trace must show live lanes ≥ 16 on the repetition prompt.
4. **Stats honesty**: rounds + drafted + accepted + fallback-rounds
   reconcile with the committed token count exactly (mirror the mtp gate).
5. **EOS**: the e765dde clamp is inherited verbatim inside `suffix_round`
   (commit_n/encoded clamp at first EOS in the committed prefix). The v1
   CLI driver runs fixed-count with a never-matching sentinel (clamp
   provably inert, same as the MTP sentinel argument), so the LIVE
   `--eos-gate` extension is DEFERRED to the stream/server suffix
   integration — which only proceeds if gate 6's economics pass. Recorded
   residue, not skipped silently.
6. **Economics (pre-registered ship line)**: on repetition-heavy/agentic
   traffic net ≥ 1.15× vs serial greedy wall (same number as the DSpark
   plan used); on neutral traffic (docs class, drafter silent) regression
   ≤ 2% (fallback overhead must be pennies). Timing legs on a quiet
   machine only.

## Implementation status (2026-07-16 night)

Engine + CLI landed (suffix_round, batched generate_suffix,
`--suffix-serial` control, burst histogram, `Q27_SUFFIX_TRACE` dispatch
trace); unit suites green. Codex round, 3 findings all fixed: P1
remaining<2 underflow guard (mirrors mtp_round); P2 match-capped width
enforced (forward lanes ≤ matched suffix length — lag-copy extrapolation
past the match evidence is not dispatched, so drafted stats measure real
bursts); P2 tool-constraint rejection at driver entry AND in suffix_round
(burst argmax is unmasked; same contract as GPU-resident greedy). Codex
found no state/position/KV divergence vs the serial walk. Correctness
gates 2–4 staged in `tools/suffix_burst_gates_2026-07-16.sh` — **runs on
Daniel's go** (T2, ~7.15 GB, correctness class, desktop-contention
tolerant); gate 6 timing needs the quiet machine.

## Kill lines

- Burst traffic on the real agentic mix fires too rarely to matter
  (net < 1.05× despite per-burst wins) → park the batched path, keep the
  serial scaffold, record the traffic histogram as the evidence.
- Fallback overhead regresses neutral traffic > 2% → the gating policy is
  wrong; fix or park.

## Non-goals

Cross-request suffix indexes (per-request stream only, as today); CUDA-side
changes (upstream has its own speculation); sampled-temperature suffix
verification (greedy-only, as the serial path today); any drafter-model
revival (parked with prejudice this morning).
