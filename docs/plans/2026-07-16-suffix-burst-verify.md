# Suffix-burst batched verification — the speculation lever that ships

Status: PRE-REGISTERED before measurement (results appended below the
line). Machine: 24 GB M4 (T2 for correctness gates; official tier + timing
on a quiet machine, operator-authorized). Follow-on from the day's two
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
the operator's go** (T2, ~7.15 GB, correctness class, desktop-contention
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

---

## RESULTS (2026-07-16 night, quiet 24 GB M4, operator-provided quiet window) — SHIP LINE MET

Gates 2–4 + 6 all PASS (`logs/suffix-gates-20260716/verdicts.txt`):
byte identity across --suffix-serial 12 / --suffix 16/32/48 and both
corrected arms; dispatch-traced live lanes ≥16/≥32; stats reconcile.
Two staged verdicts failed on prompt construction, not engine behavior,
and were re-run corrected: a perfectly-repeating prompt cannot yield
rejection evidence by construction (fixed with a numbered-list prompt —
10 drafts / 3 accepted, walk provably live, bytes identical), and the
canonical prompt's own 96-token continuation self-repeats so its 3
bursts were CORRECT firings (16.4 tok/s, bytes identical); the
docs-class silence claim was re-tested with narrative prose — 0 bursts,
95/95 fallback, wall −0.1%.

**Gate 6 economics: repetition-heavy 2.34× (25.38 vs 10.85 tok/s,
96 tokens in 3 rounds, 0 fallbacks) vs the 1.15× bar; neutral −0.1% vs
the ≤2% bar.** Single-run protocol on a quiet machine; margins are 15×
and 20× the bars respectively, re-measurement not required for the ship
decision. Gate-6 burst histogram note for the round-up-to-tile
alternative: the rep run fired 1×≤16 + 1×32 + 1×48 — matches snapped to
full tiles naturally on this traffic; no evidence yet that 17–31-length
matches carry mass; round-up stays unfunded.

**Next: gate 5 — live EOS through the stream/server suffix integration
(now unlocked), and server plumbing so agentic traffic actually rides
the batched path.**

### Server integration + gate 5 results (2026-07-16 night, 24 GB M4)

Shipped: `--suffix W` on `q27-metal-server` (mutually exclusive with
`--mtp` and `--constrain-tools`, greedy-only like the MTP branch). The
engine loop body is extracted verbatim into `MetalEngine::suffix_step`
(propose → match-cap → snap-down → burst or serial fallback, one round
per call); `generate_suffix` now drives it, and the server's generation
loop gained a fourth branch mirroring the MTP branch's one-round-per-lease
quantum discipline. The drafter is per-request CPU state seeded from the
FULL prompt (including restored prefixes, which never pass through the
step loop). `/stats` speculation grew `suffix_bursts`/`suffix_fallbacks`
so a server whose traffic never rides the batched path is visible.

- **Refactor identity (CLI)**: all six pre-refactor arms from
  `logs/suffix-gates-20260716` byte-identical post-refactor
  (serial/sfx-serial/w16/w32/w48 on the repetition prompt, serial+w48 on
  neutral; `logs/suffix-server-20260716`). Same 3 bursts on the w48 arm.
- **Server A/B**: five greedy arms (repetition, QA, chat-template,
  instruct-repeat, few-shot copy) byte-identical between `--suffix 48`
  and the serial server; repetition arm delta showed bursts fired and
  committed > rounds (the unfakeable signal — 297 committed / 201
  rounds across the suffix-server request set).
- **Gate 5, live EOS**: the QA arm stops with `finish_reason: stop` on
  BOTH servers at the same byte (" Paris"), through the suffix branch's
  new pending==eos driver clamp — the real eos id rides into
  `suffix_round`'s lane clamp exactly as registered. The strong-form
  composition (eos landing mid-burst on the same request as fired
  bursts) was probed twice and is unreachable on this model's chat
  style (T2 CoT-rambles instruct prompts to length); the clamp code
  path is pending-value-dependent only, so the QA evidence covers the
  new code. Recorded honestly, not claimed.
- **Cross-tier**: the same identity battery passes on the fresh B1 pack
  (`logs/b1-firstlight-20260716`), bursts firing — the lever is
  tier-agnostic as designed.

**Gate 5 verdict: PASS.** All six registered gates now closed. Residue:
suffix-mode variant of the multislot gate harness (G1–G3/G5 under
two-slot contention with `--suffix`) — registered, not run; and the
economics on real agentic traffic (tool-call loops via the server) ride
the standing quiet-machine protocol.

## Review-2 amendments (2026-07-18 — dispositions:
2026-07-18-expert-review-2-triage.md)

- **Sampled-path battery pre-registered:** bursts currently engage only
  at temperature 0 (server gates `sfx` on `temperature==0.0f` — coverage
  limitation, not a bug). The sampled acceptance rule: one batched
  forward over the draft block, sample each position, commit the longest
  prefix matching the sampled target tokens, resample at first mismatch
  (distributionally identical to serial sampling). Battery must
  demonstrate it before bursts engage under temperature > 0.
- **Finite-class optimality certificate (from review 1) and the
  closed-condition formulation (review 2):** γ* ≈ 0.89 per-position
  agreement at w=16 for a serial learned drafter on this hardware class
  (drafter floor = F ≈ 14 ms, not bytes); measured DSpark τ = 3.209 is
  far below. The window reopens iff π/β changes (M5/cooperative tensors),
  a block-parallel drafter escapes the F floor with better acceptance, or
  trees. **Cruel corollary recorded: every F reduction raises the
  learned-drafter bar further.**
- **Tree speculation, narrow version only:** branch only where the
  suffix automaton has multiple high-count continuations; external
  hybrid-architecture datapoint ~+15% (draft-ceiling bound) vs +35–42%
  on pure attention. GDN branch state costs recorded as the caution.
- **w=48 widening stands:** review 2's "S peaks at w ≈ 3–8" conflicts
  with our measured tile-flat cost curve (sweet spots {16,32,48}, S(48)
  = 3.94×); measurement wins.
