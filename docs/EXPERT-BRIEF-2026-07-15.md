# q27 / Quasar — brief for an outside expert (2026-07-15)

Hi —

I'm working on **q27 ("Quasar")**, a narrow, from-scratch inference engine for
**Qwen3.6-27B-MTP** (hybrid GDN + attention, trained-in MTP heads) — one model
family, as fast as possible, in the spirit of antirez's ds4. It started as a
CUDA engine for a single RTX 5090/3090 and now has a full **Metal port** for
Apple Silicon that has become the most active front. I'd value an outside
expert pass over our approach and a few specific open problems.

**Repo:** github.com/manthedan/q27 — **read the `metal` branch** (`master` is
frozen at the last CUDA-only milestone; everything current, including all
CUDA history, is on `metal`).

## Where things stand

- **CUDA (production):** 224–246 t/s aggregate on a 5090 via self-speculation
  (trained-in MTP ladder + free suffix drafter through one shared-KV MMA
  verify; 5.3–5.8 accepted tokens per weight read on live Claude-Code
  traffic). +47% decode over tuned llama.cpp same-day/same-GPU. Native
  Anthropic/OpenAI serving endpoints. History in `BUILDLOG.md`.
- **Metal (most active):** byte-exact 16-token parity with the CUDA oracle;
  layer-major chunked prefill; batched MTP verification; online-softmax
  attention (flat in context); GQA KV-reuse kernels; **turbo3 3-bit KV**
  (1.56 bits/value, depth-flat quality at 32K); packed-dot GEMV at 84–93
  GB/s (~70–85% of the M4's practical stream bandwidth).
- **Ternary tier (new, landed):** PrismML released "Bonsai 27B" — ternary
  ({−1,0,+1}, 2.125 bpw) and binary ({−1,+1}, 1.125 bpw) QAT builds of our
  exact base model. We repacked the ternary GGUF losslessly into our format
  and serve it: first fully-resident 27B decode on a 24 GB M4 at ~12.4 tok/s
  resident ceiling, byte-identical to their reference stack, quality cost
  measured depth-uniform (~2.2× PPL vs our official tier at 32K). Binary
  tier and a denser ternary packing (1.75 bpw) are planned next.
- **Hardware:** M4 24 GB + M4 mini 16 GB (Metal), RTX 5090 + 3090 (CUDA).

The engineering culture is measurement-first: every optimization lands behind
byte-exact or tolerance-gated A/B gates, negative results are recorded, and
the progress ledgers double as lab notebooks.

## Reading order (key docs)

1. `README.md` — project overview and headline results.
2. `docs/METAL_PROGRESS.md` — **the Metal ledger**; the last ~10 entries are
   the current frontier. Best single document for state.
3. `docs/plans/2026-07-14-ternary-tier.md` — the ternary tier end-to-end
   (motivation, format, kernels, gates, whitepaper follow-ups).
4. `docs/plans/2026-07-15-ds4-survey.md` — our analysis of ds4's Metal
   engine vs ours, leverage-ranked; doubles as our near-term Metal roadmap.
5. Active plan docs (all `docs/plans/2026-07-15-*`): `binary-tier`,
   `t3-packing`, `cache-block-scheduling`, `metal-multislot`,
   `sibling-drafter-probe`, `gemm-half-staging`, `kl-kv-gate`, and
   `resident-greedy` (in flight — may land a few days after this letter).
6. CUDA background as needed: `BUILDLOG.md`,
   `docs/plans/2026-07-13-mtp-draft-head.md` (drafter economics),
   `docs/dflash-block-verify-design.md` (block-drafter analysis),
   `docs/SPEC.md`, `docs/FORMAT.md`.

## Where I'd most value help

1. **Metal prefill GEMM utilization.** Decode is bandwidth-bound and healthy;
   prefill is compute-bound at ~25 GB/s effective weight stream through a
   32-row × 16-token `simdgroup_matrix` GEMM (half-staging just landed,
   +1.22×). Open questions: how far can M4-class matrix units realistically
   be pushed on quantized-weight GEMMs; direct-device-RHS vs staged;
   whether our 12-token prefill chunk (vs ds4's 4096) is leaving the
   dominant factor on the table. (`ds4-survey` items 2–3.)
2. **Per-token sync elimination on Metal.** The ternary tier's remaining gap
   (~9 vs ~12.5 tok/s) is argmax-readback sync + command-buffer turnaround.
   Planned: GPU-resident token feedback (argmax writes device-side, embedding
   reads it, K steps per command buffer). Sanity-check the design;
   MTLSharedEvent / event-driven alternatives welcome. (`resident-greedy`.)
3. **Speculative decoding on low-bandwidth Apple Silicon.** Batch-1
   verification doesn't amortize (our measurement, PrismML's, and llama.cpp
   community evidence all agree). Is this wall structural or an overhead
   artifact? Our drafter options and probe plan are in
   `sibling-drafter-probe` — a critique of the decision gates would be
   valuable.
4. **Cross-backend numeric divergence framing.** CUDA and Metal agree
   byte-exactly for 16 tokens, then diverge at low-margin tokens (~token 128)
   from accumulation-order differences; teacher-forcing confirms
   numeric-path, not architecture. We treat committed-token divergence
   beyond the canonical gate as legitimate. Is our tolerance framing sound,
   or is there a better cross-backend equivalence discipline?
5. **Long-context attention scheduling.** turbo3 KV is depth-flat on quality;
   wall time still decays with depth. The cache-block-scheduling plan
   (R1 head-major layout, token-tiled causal GQA) is designed but unreviewed.
6. **Anything we're not asking.** The ledgers record what we tried; a fresh
   eye on what's conspicuously absent from the roadmap is worth as much as
   answers to the questions above.

Practical notes: the repo is self-contained (no external ML frameworks);
`make test-cpu test-metal` runs the full synthetic gate suite without model
weights on any Apple Silicon Mac. Model artifacts are large (7–17 GB) and
downloadable via scripts/docs in-tree.

Thanks — happy to walk through any of it live.

Daniel
