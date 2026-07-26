# Metal decisions ledger

One line per pre-registered experiment: what was tried, what the measurement
said, and the number that decided it. Every row links to its plan in
[plans/](plans/), which keeps the full pre-registration, kill line, and raw
results.

**The parked list is as load-bearing as the shipped list.** Most of the rows
below are things we did *not* ship, and each one records the mechanism that
killed it — which is the part that stops the idea coming back.

## Shipped

| date | lever | the number |
|---|---|---|
| 07-15 | [Wide prefill chunks + direct-RHS](plans/2026-07-15-wide-chunks.md) | **1.39×** (35.09 → 48.79 tok/s at chunk 96); committed tokens byte-identical to serial |
| 07-16 | [Lever 2 — verify width past 12](plans/2026-07-16-lever2-verify-width.md) | **S(48) = 3.938×** mean vs a ≥3× bar; `VERIFY_CHUNK_MAX = 48` ships |
| 07-16 | [Suffix-burst batched verification](plans/2026-07-16-suffix-burst-verify.md) | gates 2–4 + 6 PASS; byte identity across `--suffix` 16/32/48 |
| 07-16 | [G6 multislot admission + 503](plans/2026-07-16-g6-admission.md) | PASS — and found a real hole first: at 8 workers the 503 path was unreachable dead code |
| 07-16 | [Envelope instrument](plans/2026-07-16-envelope-instrument.md) | zero contradictions, zero NaN, exact-zero repeat at every position (2,048 positions × 4 classes) |
| 07-17 | [Q4 matvec rewrite (official tier)](plans/2026-07-17-q4-rewrite-round.md) | **R = 1.024 / 1.060** — the Q4/Q8 mix now streams at/above the same-run T2 reference (73.99–88.29 GB/s, from 52.44 at funding) |
| 07-17 | [B1 select kernel round 2](plans/2026-07-17-b1-select-round2.md) | r2 (4 rows/simdgroup) promoted; correctness PASS on all 8 shapes — "the artifact prize was not where the funding thought it was" |
| 07-17 | [Metal server agentic parity](plans/2026-07-17-metal-agentic-parity.md) | all pre-registered gates PASS in one clean run |
| 07-17 | [Incremental tool-call streaming](plans/2026-07-17-incremental-tool-call-streaming.md) | shipped `1adb0cd`; wire dump shows six argument fragments concatenating to valid JSON |
| 07-17 | [Abandoned-request cancellation](plans/2026-07-17-abandoned-request-cancellation.md) | 2 s kill of a ~90 s prefill; follow-up answered in **1.15 s** |
| 07-17 | [KV exception snapshot v2](plans/2026-07-17-kv-except-snapshot-v2.md) | side caches ride all three snapshot surfaces; gate hardened en route |
| 07-18 | [Streaming parity: messages + responses](plans/2026-07-18-streaming-parity-messages-responses.md) | 173 insertions / 3 deletions by reusing `ToolCallStreamer` |
| 07-18 | [T1 semantics-aware snapshot eviction](plans/2026-07-18-t1-snapshot-eviction-classes.md) | G1 PASS offline; spine-pin changes the first victim away from the reused spine |

## Parked / killed by measurement

| date | lever | why it died |
|---|---|---|
| 07-15 | [T3 native ternary packing](plans/2026-07-15-t3-packing.md) | **KILLED** — decode-bound at ~50 GB/s vs T2's ~95 on M4 |
| 07-15 | [Cache-block scheduling R1 (head-major)](plans/2026-07-15-cache-block-scheduling.md) | **PARKED** — 1.00× at every depth including 128K |
| 07-16 | [Lever 1 — direct-RHS chunk GEMM](plans/2026-07-16-lever1-direct-rhs.md) | **PARK** — C/D2 = 0.782 at x_rows 12; 22% *slower* than production |
| 07-16 | [f16-accumulate MMA probe](plans/2026-07-16-f16acc-probe.md) | **PARKED** by kill line — C/F = 1.065; a real 6.5% win exists but accumulator width is not the M4 MMA limiter |
| 07-16 | [R3 barrier-free attention](plans/2026-07-16-r3-barrier-free-attention.md) | **PARK entirely** — 0.47–0.53× of production (≈2× slower) at every block size and depth |
| 07-16 | [N=2 slot-batched decode GEMV](plans/2026-07-16-multislot-phase2-probe.md) | **PARK** — s_k = 1.093 vs a 1.31 line; implied e2e ceiling ≈1.08× |
| 07-16 | [Verify round cost](plans/2026-07-16-verify-round-cost.md) | **PARKED** by its own kill line |
| 07-17 | [T2 zero-aware repack](plans/2026-07-17-t2-zero-sim.md) | **KILL** — net saving −0.37%; of 210M groups, 0 are all-zero |
| 07-17 | [e4m3 on the hot KV cells](plans/2026-07-17-kv-e4m3-hot-cells.md) | **KILLED** at the 1,536-position screen — max 2.278 vs a >2.0 kill line, and mean *worse* than control |
| 07-18 | [gdn_pair rescue](plans/2026-07-18-gdn-pair-rescue.md) | **G1 tripped** — the control did not reproduce, so no band arm ran (as designed). Exonerated the pack; the loop was a serving-binary artifact |
| 07-25 | [Boot prefetch for the snapshot store](plans/2026-07-25-boot-prefetch.md) | **NO-GO** — restore read is 230 ms = 1.6% of the path; prefetch could remove ≤194 ms of a 14 s request |
| — | Mixed-tier (M1) thesis | **permanently closed** — A5-on-gdn_pair killed the strongest arm on both corpora; a B1→T2 bridge needs training, not grafting |

## Measurement rounds that set direction

| date | round | finding |
|---|---|---|
| 07-16 | [A/B/C MMA roofline](plans/2026-07-16-mma-roofline.md) | C/Beq = **1.135** → "not mature"; exactly one targeted kernel round permitted, aimed at unpack/conversion ALU |
| 07-16 | [KV-codec steps 1–4](plans/2026-07-16-kv-codec-step1.md) | tail attribution across 8,191 positions/arm; one head owns the extreme tail ([census](plans/2026-07-16-kv-codec-census.md), [step 4](plans/2026-07-17-kv-codec-step4-probe.md)) |
| 07-17 | [E6 Q4/Q8 GEMV efficiency](plans/2026-07-17-e6-q4q8-gemv.md) | R = 0.716 → funded one round (which became the Q4 rewrite above) |
| 07-17 | [FP8-KV control arm](plans/2026-07-17-fp8-kv-control.md) | fp8 mean 0.000626 nats vs turbo3's 0.0115 — the control that priced turbo3 honestly |
| 07-17 | [T2 chunked-prefill throughput](plans/2026-07-17-t2-prefill-throughput.md) | no regression — prefill was simply never optimized past its first shipped schedule |
| 07-17 | [Mixed weight-tier census](plans/2026-07-17-mixed-tier-census.md) | 2 baselines + 25 arms, zero retries, zero aborts |
| 07-18 | [A5 KL(mixed ‖ official)](plans/2026-07-18-kl-pair-a5.md) | cross-model paired-logit gate; plumbing verified exact-zero against self |
| 07-19 | [Task-level suite on mixed packs](plans/2026-07-19-task-level-suite-mixed-packs.md) | four arms published with provenance; a partial run correctly refused itself |

## Open

| item | state |
|---|---|
| [A7 — first-request warm-up](plans/2026-07-25-first-request-warmup.md) | ~9–10 s one-time cost on the first request of a fresh process. Fix attempt #1 (decode-step warm-up) measured as NOT working and reverted |
| A8 — snapshot write holds the GPU lease | 7234 ms for a 610 MB entry, stalling the other slot at the default `--slots 2`. See [PARITY-2026-07-25.md](PARITY-2026-07-25.md) |
| A1 — q4s/q5f/q6f tier validation | loader blocker removed; no canonical/NLL/task ladder run on Metal yet |
| [Sampled MTP](plans/2026-07-21-metal-sampled-mtp.md) | legs 0–1 done; wall-clock and acceptance legs need the GPU slot |
| [Agentic replay bench](plans/2026-07-17-agentic-replay-bench.md), [SWE-bench tier gap](plans/2026-07-20-agentic-swebench-tier-gap.md) | pre-registered, pending first real run |

## Reviews and surveys

Kept in [plans/](plans/) rather than summarized, because their value is the
triage table rather than a verdict: expert reviews
[1](plans/2026-07-15-expert-review-integration.md) ·
[2](plans/2026-07-18-expert-review-2-triage.md) ·
[3](plans/2026-07-19-expert-review-3-triage.md),
[k3 audit triage](plans/2026-07-17-k3-audit-triage.md),
[Metal audit triage](plans/2026-07-17-metal-review-triage.md),
[parity audit](plans/2026-07-17-parity-audit-triage.md),
[ds4 survey](plans/2026-07-15-ds4-survey.md),
[BaseRT survey](plans/2026-07-16-basert-survey-draft.md),
[paper-scan triage](plans/2026-07-16-paper-scan-triage.md).
