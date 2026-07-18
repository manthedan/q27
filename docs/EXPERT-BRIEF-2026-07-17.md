# q27 / Quasar — brief for an outside expert (2026-07-17)

Hi —

Follow-up to the 2026-07-15 brief (`EXPERT-BRIEF-2026-07-15.md`). Most of
that letter's questions were answered by measurement in the two days since —
prefill closed MATURE 7% off the vendor's own stack with the lever list
empty, GPU-resident greedy landed, and the speculation question got a
decisive measured answer (below). This letter carries the questions those
answers created.

**Repo:** github.com/manthedan/q27 — **read the `metal` branch.**

## Where things stand (2026-07-17 evening)

- **Hardware under discussion:** base M4 24 GB + M4 mini 16 GB. ~100 GB/s
  practical DRAM stream; ~2.5–2.65 TFLOP effective simdgroup-MMA plateau
  (no Metal-4 cooperative tensors); the physics frames everything below.
- **Decode is at the wall on every tier we serve.** T2 (ternary 2.125 bpw,
  7.15 GB): 8.2–9.4 tok/s observed, 12.4 resident ceiling, GEMV at 93–97
  GB/s — at the stream bound, parity with the vendor's own fork on the
  same pack (8.41 ± 1.36). B1 (binary 1.125 bpw, 3.79 GB): **18.6–18.8
  tok/s warm**, ceiling 19.2 — memory wall 3.36 GiB/token at ~69 GB/s.
- **Prefill closed as MATURE:** 47.2 tok/s on the synthetic T2 mix vs the
  vendor fork's 52.48 — 7% gap, attribution recorded (MMA-plateau-bound;
  direct-RHS, f16-accumulate, barrier-free, K=128 staging, function-
  constant baking all parked by pre-registered measurement).
- **Speculation, resolved by measurement:** learned drafters are dead on
  this hardware — the vendor's own trained DSpark drafter measured
  τ = 3.209 committed/round (712 real rounds, fork harness, our exact
  target) against a break-even of 4.13; the doubly-generous chained upper
  bound reaches S ≤ 1.07 vs our 1.3 ship bar; the vendor's own stack
  corroborates (0.528 at home). The live lever is **suffix bursts** — a
  zero-byte n-gram drafter through our batched verify: 78% mean
  acceptance over 1,093 verified rounds, zero fallbacks, shipped to the
  server with byte-identity gates. Verification is nearly free here:
  round cost is flat per 16-token tile (oracle S(48) = 3.94×).
- **The headline experiment:** a mixed-tier census (checkpoint grafting,
  B1 pack + selected T2 tensor classes). `gdn_qkv` alone recovered
  **114.4%** of the B1→T2 NLL gap at +314.6 MB; `gdn_alphabeta` 46.9% at
  +2.9 MB; nearly every other lone class swap negative. A ~4.1 GB mixed
  pack at ≥T2 quality — decoding ~17–19 tok/s — is the serving candidate
  the combination arm now gates.
- **Serving:** native Anthropic Messages + OpenAI chat/completions +
  `/v1/responses` (full Codex CLI lifecycle, validated end-to-end); disk
  prefix snapshots (8.1 s vs 4:54 cold at 8.3K tokens; cancel-banking
  rescued a real client retry live); 131K context in 24 GB; two-slot fair
  scheduler; whole-session JSONL trace.

## Where I'd most value help

1. **The grafting overshoot — real, or a red flag?** Two QAT sibling
   checkpoints (ternary and binary builds of the same trained family);
   grafting one tensor class (GDN qkv projections, ~315 MB) from the
   ternary donor into the binary pack recovered *more than the entire
   measured quality gap* (114.4%). Is >100% recovery a known signature in
   checkpoint-merging / model-soup literature (interference cancellation,
   regularization effects), or should we treat it as a measurement
   artifact until proven otherwise? What experiment design would you use
   for the combination arm to detect interference between grafted classes
   — and is NLL on held-out text even the right gate, or should we gate
   on task capability directly? Second part: is quantization sensitivity
   concentrated in the recurrent/GDN (linear-attention) pathway a known
   result? It would explain why that one class carries the whole gap.
2. **Why does reading fewer bytes achieve lower bandwidth?** B1 streams
   3.36 GiB/token at ~69 GB/s effective; T2 streams ~6.8 GiB/token at
   ~78.5 GB/s on the same machine, same select-form GEMV shape, same
   4-rows-per-simdgroup structure (isolated kernel bench: B1 is 2.36×
   after restructure — yet the artifact wall didn't move). Registered as
   residue, known NOT to be issue-rate. Smaller working set changing DRAM
   page/bank behavior? Per-group scale-walk stride? We'd like a mechanism
   hypothesis we can falsify with one probe.
3. **The last 30% of serial decode.** ~~T2 resident ceiling 12.4 tok/s vs
   8.2–9.4 observed~~ **CORRECTION (2026-07-17, post-review): the gap is
   already attributed in-repo** (ledger entry 207): launch fault-in inside
   the timed window (fixed same-day via single-MTLBuffer + residency set)
   plus thermal-state mismatch (the ceiling itself re-measured 12.66 →
   10.57 mid-session), GPU busy 96% of wall — decode is kernel-bound at
   the bandwidth limit within a few percent of the same-moment ceiling.
   The question as written was stale; what survives for outside eyes is
   the reviewer's own observation: our synthetic ceiling reuses one
   representative tensor per class, so it does not reproduce the 7 GB
   unique-address footprint (page tables/TLB/file-backing) — whether the
   ceiling number itself is inflated by that compact footprint is
   untested, and a four-arm full-footprint bench is now pre-registered
   (triage doc §3). Phase-4 remainder: per-step readback turnaround and
   the GEMV itself, not residency.
4. **Is n-gram-class drafting the optimum at this bandwidth — and can
   that be made rigorous?** Our measured position: verify passes are
   nearly free (flat per 16-token tile), so speculation economics hinge
   entirely on drafter cost-per-proposal vs acceptance. A trained 0.5 GB
   drafter at τ = 3.2 loses; a zero-byte n-gram drafter at K ≈ 1.5–2
   wins. For ~100 GB/s machines serving 3–8 GB models, is there a drafter
   shape that clears break-even (τ ≥ ~4.2 at block 4, or a scheduling
   trick that moves the bar), or is the trained-drafter window simply
   closed until hardware changes? If the answer is "n-gram is the
   optimum," is that a theorem anyone has stated?
5. **Cross-backend numeric divergence framing (carried, still open).**
   CUDA↔Metal agree byte-exactly for 16 tokens, then diverge at
   low-margin tokens (~128) from accumulation order; teacher-forcing
   confirms numeric path. We treat divergence beyond the canonical gate
   as legitimate. Is there a stronger cross-backend equivalence
   discipline we should adopt?
6. **What are we not asking?** The ledgers record what we tried
   (shipped and parked alike); a fresh eye on what is conspicuously
   absent from this roadmap is worth as much as any answer above.

## Reading order

1. `docs/METAL_PROGRESS.md` — the ledger; the Current-state table at the
   top plus the last ~5 entries are the frontier.
2. `docs/plans/2026-07-17-mixed-tier-census.md` — the grafting census:
   pre-registration, grafting-vs-attribution caveat, ship bands, results
   (readout summary in the ledger's 2026-07-17 evening entry; full table
   publishing from the mini lane).
3. `docs/plans/2026-07-16-dspark-port-phase0.md` — the parked drafter
   port: full contract, measured verdict, and the whitepaper
   reconciliation addendum.
4. `docs/plans/2026-07-17-b1-select-round2.md` — the B1 kernel round and
   the corrected cold-pack-paging diagnosis.
5. `README.md` Speed/Server sections for the calibrated numbers;
   `docs/BENCHMARKING.md` for the CUDA-side methodology.

Practical notes: the repo is self-contained; `make test-cpu test-metal`
runs the full synthetic gate suite without model weights on any Apple
Silicon Mac. Every claim above traces to a dated, gated ledger entry —
we answer follow-ups with measurements, not adjectives.

Thanks — happy to walk through any of it live.

— manthedan
