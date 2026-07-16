# Sibling-drafter probe: Ternary-Bonsai-1.7B drafting for the 27B tiers

**Status: proposed** (2026-07-15). This is the P1-style probe gating option 2
of the binary-tier doc's drafter follow-up (`2026-07-15-binary-tier.md`).
Drafting remains parked per the ternary doc's follow-up 6 resolution; this
probe costs ~1 hour, zero engine work, and either kills the idea cheaply or
produces the numbers that would justify an engine project.

## Why probe the dominated option

DSpark dominates the 1.7B at equal byte budget on paper (trained against the
target, block-parallel, confidence-scheduled — whitepaper §6). Three reasons
the 1.7B is still worth one hour:

1. **Integration simplicity.** DSpark needs hidden-state taps from five
   target layers + shared embeddings threaded through the engine. The 1.7B
   is standard two-model speculation; our batched chunk verify already
   exists, and the drafter side is just a second, tiny engine.
2. **Target-agnostic.** DSpark is trained against this Bonsai 27B's hidden
   states; its acceptance on the incoming agentic-coding variant (§9,
   "shortly") is unknown. A generic sibling drafter survives artifact swaps.
3. **Dual-purpose measurement.** Run in *their* stack, the probe is also a
   direct same-hardware test of the batch-1 Apple Silicon verification-
   amortization wall (our ledger follow-up 6; their §9 limitation) — the
   wall that parks ALL drafting. Whatever the acceptance number, we learn
   whether two-model spec-decode nets positive on an M4 at all.

## Phase 0 protocol (their stack, zero engine work)

Machine: either M4 (memory ~8 GB weights total — fine on both). Caffeinated;
matched-thermal protocol; standard machine rules (24 GB machine: ask first;
runs serialized with any other GPU work).

1. Download `prism-ml/Ternary-Bonsai-1.7B-gguf` (~0.6 GB); md5 into
   `CHECKSUMS.md5`.
2. **Vocab/tokenizer identity gate:** the 1.7B pack's tokenizer and vocab
   size must match the 27B pack (draft tokens must be target-valid).
   Mismatch → probe dead immediately, record and stop.
3. Draft-cost baseline: `llama-bench` on the 1.7B alone under the fork
   binaries (`prism-b9591-62061f9`, already certified on both machines).
   Expect ~80–100 tok/s class; record actual.
4. Speculative run: the fork's two-model path (`llama-speculative` /
   `llama-server --model-draft`), greedy (`top_k: 1` — their server's
   `"samplers": []` is nondeterministic, known from the ternary spike),
   draft depth sweep (4/8), on two prompt sets: the ternary spike's agentic
   mix (our real workload; their weakest category) and a prose prompt.
   Record: per-step acceptance, accepted length/round, end-to-end tok/s vs
   the 8.41 ± 1.36 tok/s 27B-alone baseline (re-measure baseline
   back-to-back, same session, matched thermal).
5. Optional, free while loaded: same sweep against the **binary** 27B pack
   if Phase 0 of the binary tier has landed it (drafter economics improve
   as the target gets faster only if acceptance holds — record both).
6. **DSpark counterpart probe (same session if feasible):** their fork
   implements the DSpark path and default-disables it on Apple Silicon
   (§6.2 — "not enabled by default there"). Locate the enable flag +
   drafter pack (whitepaper ref [7]; possibly among the org's unlisted
   repos), force-enable, and run the same prompt sets: accepted length and
   net tok/s on our M4 in their stack. This prices any future DSpark port
   with same-hardware numbers next to the sibling's — the port-vs-sibling
   decision should never be made with their H100 table alone. If the flag
   or pack cannot be found in an hour, record that and move on.

## Decision gates (write results here; honor them)

- **Acceptance ≥ ~0.75 per step AND net ≥ +25% tok/s in their stack** →
  the wall is disproven for this shape on this hardware; fund the engine
  project (see below).
- **Acceptance ≥ ~0.75 but net loss in their stack** → the wall is
  confirmed as overhead, not acceptance; park with the number recorded and
  revisit after the GPU-resident sync work lands (`resident-greedy` doc) —
  our per-round overhead will be lower than their stack's, so their net
  loss does not bind our engine forever. This is the expected outcome.
- **Acceptance < ~0.6 on the agentic mix** → kill the 1.7B option
  permanently (drafter quality, not overhead, is the binding constraint);
  DSpark or nothing for these tiers.

## If funded: the engine project (own plan doc, not this one)

Cross-model speculation is a real project and would get its own plan:
multi-arch `validate_architecture` (Qwen3.6-1.7B dims beside 27B/bonsai
variants), a second small engine instance, draft loop feeding the existing
batched chunk verify, and a deliberate one-model-load policy amendment with
a measured pressure test (~7.7 GB combined weights + KV — far from the
17+17 crash scenario, but the rule is amended only explicitly, never
implicitly). Sequencing: after `resident-greedy` (which it depends on for
round overhead) and behind the binary tier's Phase 0 in priority; it
displaces nothing on the current critical path.

## Follow-up ladder if funded (in order; each gates the next)

1. **Scheduling policy first, free:** adaptive draft width + confidence-
   gated early exit on the drafter's own top-token margin — the CUDA
   ladder's `dexit`/`pmin` policy ported to the two-model loop. Recovers
   most of what fixed-width drafting loses; measure before any training.
2. **Distillation finetune, last resort, eyes open:** KL-distill an FP16
   Qwen3.6-1.7B toward 27B outputs over our real traffic (5090 generates
   teacher data; LoRA fits), quantize to Q4 (~1 GB drafter — no QAT
   latents are published for any 1.7B: `Bonsai-1.7B-unpacked` (checked
   2026-07-15) is the *binary* model dequantized to FP16 as an
   HF-compatibility artifact, not training masters, and no ternary-1.7B
   unpacked repo exists; if distilling, that unpacked binary FP16 is a
   better starting point than the raw Qwen3.6-1.7B base — Bonsai-family
   behavior out of the gate). Caveats recorded up front: this forfeits
   both advantages
   that justified the sibling over DSpark (zero-training, target-agnostic
   across artifact swaps like the incoming agentic variant); at the moment
   drafter training is funded, a DSpark port must be re-compared as the
   alternative for the same effort.

## Non-goals

DSpark port (separate evaluation if ever; contract analysis already exists
in `dflash-block-verify-design.md`); 4B/8B siblings as drafters (strictly
worse economics than 1.7B at 3–8× the stream cost; revisit only if 1.7B
acceptance is the binding failure); any engine work before the gates above;
any drafter training before the follow-up ladder's step 1 is measured.
