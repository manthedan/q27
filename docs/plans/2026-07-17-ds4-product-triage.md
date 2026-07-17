# ds4 product/quant/process triage — the layers the kernel survey never read

**Status: TRIAGED (2026-07-17, Daniel's analysis + review).** The
2026-07-15 ds4 survey covered kernels and infra only (352 lines, nothing
on the agent product, quant tooling, or process files). This doc records
verdicts on the rest of ds4 so each import/skip is traceable. The one
big item — the native in-process agent direction — gets its own design
doc: `2026-07-17-native-agent-direction.md` (**Daniel's decision: worth
trying, DEFERRED until the engine is stable**).

## ADOPT NOW (small, no product-direction decision required)

- **I1 — `/strip`-equivalent for DiskSnapshotStore.** ds4 keeps the
  rendered conversation text but drops the heavy KV payload; switching
  to a stripped session rebuilds KV by re-prefill. Our store has LRU
  eviction only — all-or-nothing. Strip adds a middle disk-budget tier:
  demote cold snapshots to text-only stubs instead of deleting, restore
  = re-prefill (suffix-burst- and, post-Q4-port, faster-chunk-
  accelerated). Cheap, purely server-side, composes with the existing
  budget accounting. Gate: stub restore produces byte-identical
  generation vs never-evicted (the prefix-cache tag already encodes the
  prompt; NLL/argmax equivalence over one canonical body).
- **I2 — `--trace` whole-session logging.** One stream: rendered
  prompts, snapshot/cache decisions, tool-parser events. The responses/
  codex parity rounds this week reconstructed exactly this by hand from
  four logs. Diagnostic switch, no semantic variant — allowed even
  under ds4's own flag rule.
- **I3 — `QA_BEFORE_RELEASES.md`.** Single release checklist collecting
  the gates now scattered across chronicle + make targets. This is also
  the first concrete Homebrew Phase-2 task that isn't the shader embed
  (B4 needs a release-cut habit; a habit needs a checklist).
- **I4 — frontier benchmarking shape.** Report instantaneous tok/s at
  context frontiers (2k/4k/8k/…) from one load, instead of whole-run
  averages — the 32K 430→180 tok/min decay was measured by hand. Fold
  into the existing bench binaries on next touch; not its own round.

## UPGRADES to already-queued items (method changes, same queue slots)

- **U1 — mixed weight-tier pack (upgrades 2026-07-17-mixed-tier-census).
  [2026-07-17 pm: census is LIVE on the mini — q27_mix.py + 19-arm
  driver, commits dc3392d..d6f90ff. U1(a) positional v1 is superseded by
  the running census; U1(b) imatrix selector stands as the v2 method.]**
  ds4 ships the precedent: base 2-bit GGUF with last-six-layer experts
  spliced from Q4 → "statistically closer to full Q4 than plain 2-bit"
  on output-agreement. Two method changes: (a) **v1 needs no census** —
  a positional heuristic (last-K layers at T2, rest B1) is runnable now,
  zero engine work (per-tensor dtype already routes), gated on
  KL/`--envelope` vs T2 + the agentic probes; (b) **v2 selector = an
  imatrix-style importance collector** — an envelope-arm instrument
  recording per-tensor activation importance over a domain-matched
  rendered corpus (including tool-call prompts), so the boosted set is
  chosen by data, not position. Instrument-first and production-exact,
  same pattern the fp8-KV control arm validated. Skip ds4's IQ2 tables
  (see S1). The census doc becomes the v2 selection round.
- **U2 — capability-eval harness (new gate infrastructure, needed by
  B1 residue + U1).** We have NLL/KL/needle only; PPL ratios don't
  answer "does the agent still finish tasks", and mixed packs need that
  gate. Scope minimal, in order: output-agreement vs the strong tier
  (ds4's own sufficient first gate — our KL instrument is strictly
  better than their ad-hoc check); then a small extractor+grader with
  golden self-tests — **the grader must prove it can fail** (our
  masked-failure lesson applied to quality gates). Capability sets
  (GPQA-class) only if a decision ever needs them. ds4's glm5.2
  continuation-fixture corpus is the regression-vector model.

## DEFERRED (design doc written, gated on stability)

- **N1 — native in-process agent** (KV-as-session, /save /switch
  /strip, COMPACT-style compaction, model-native tool syntax, CDP
  browser tool). See `2026-07-17-native-agent-direction.md` for the
  full design, prerequisites, and gates. Not on the finishable list; it
  is a second product, deliberately.

## SKIP (with mechanisms)

- **S1 — IQ2 lookup tables.** Dependent-gather decode — the exact
  mechanism that killed T3 on M4 (2026-07-15-t3-packing.md). Their
  sequential 2-bit code is the fast path on our hardware too; B1's
  select-form already is that answer at 1.125 bpw.
- **S2 — DSA sparse-attention indexer.** DeepSeek V4-specific (22 GB at
  1M ctx); q27 is dense hybrid, no transfer.
- **S3 — SSD expert streaming / mixed streaming experts / MoE
  machinery.** No MoE in this model family.
- **S4 — ROCm / Strix Halo.** Out of scope; backends are CUDA + Metal.
- **S5 — distributed pipelined prefill.** Already in the someday pile
  with honest sizing (2026-07-15-ds4-survey.md item 7); their code
  doesn't change the decode-gets-slower arithmetic.
- **S6 — steering vectors.** Orthogonal research tooling.

## Notes for the record

- **ds4 AGENT.md flag rule** ("no permanent semantic variants behind
  flags; diagnostic switches only") vs our env-knob culture: ours are
  probe infrastructure with measured park/ship verdicts — defended. The
  real lesson: knobs that *shipped semantics* (snapshot auto, KV fp16
  cells, max-tokens default) should be promoted to documented flags at
  Homebrew time; probe knobs stay probes.
- **glm5.2 branch (26K-line second-arch patch)** — watch item: the
  reference for how a single-model engine survives going multi-arch, if
  qwen variants ever land on the roadmap. No action.
