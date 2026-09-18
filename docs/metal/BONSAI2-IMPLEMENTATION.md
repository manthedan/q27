# Bonsai 2 implementation — 2026-09-18

Branch: `revival/bonsai2`, upstream base `0f1f1d496476929560fdcb93d736b796aabb2f69`.
Worktree: `/Users/macthedan/projects/q27-revival`. Original checkout untouched.

## Frozen inputs

- HF: `prism-ml/Ternary-Bonsai-2-27B-gguf`, revision
  `6ed5e12bf84b7a63069882c91dd9e9218647d17b`.
- `Ternary-Bonsai-2-27B-PQ2_0.gguf`: 7,206,168,928 bytes;
  SHA-256 `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`.
- Prism llama.cpp: `1a07bfa5f4144274c8f1c9963821dd9d9a51854b` (`prism`).

Header inventory: 851 tensors; 402 PQ2_0, 353 F32, 96 BF16 (GDN alpha/beta).
401 input-rotated matrices, inverse embedding, grouped GDN value ordering,
block 1024, explicit signs at widths 5120/6144/17408. No MTP layer.
`general.name` is `Hf`, so substring-based Qwen3.8 dialect detection is NOT
sufficient. Conversion must retain the source name and explicitly identify the
model's chat profile. BF16 gates must widen exactly, not be requantized.

## Gates registered before the first model run

1. CPU format/metadata fixtures: byte-exact PQ2_0 codes/scales; BF16 -> F32
   exact; reject absent/unknown/inconsistent rotation and quantization metadata.
   Existing repack/loader tests stay green.
2. GPU transform: independent dense CPU Hadamard reference at production
   widths, forward/inverse order and grouped-GDN permutation. Absolute error
   <= 2e-5 for bounded test inputs; inverse roundtrip <= 2e-5. Invalid extents,
   aliasing, unsupported sizes fail before dispatch.
3. Pinned real-artifact teacher-forcing vs Prism, identical token IDs, one
   model resident at a time, fp16 KV. At least 32 positions initially, followed
   by a multi-prompt gate: relative logit RMSE <= 0.02, mean KL(reference ||
   q27) <= 0.01 nats, max KL <= 0.05 nats. Argmax must agree for reference
   margins > 0.25. Record greedy agreement too, but do not mistake one plausible
   continuation for a correctness proof. Investigate failures; do not silently
   relax limits.
4. Legacy T2/B1/q4s model validation and synthetic Metal gates; correct no-MTP
   and serial-prefill rejection behavior. No claim of full long-context parity
   or task quality from the short reference gate.
5. Restore native-agent/TUI and packaging only against the new shared API;
   test CPU control-plane behavior before model-driven tools. Do not copy old
   `api_common.h`, tokenizer, or server snapshots over current upstream.

## Scope

Text-only PQ2_0 first. PTQ1_0, batched float-activation prefill, speculative
acceleration and release publication remain separate milestones. Never enable
old activation-quantized Bonsai chunking merely to recover an old benchmark.
Model files and raw logs stay ignored. Small reproducible evidence summaries
will be committed. GPU/model runs are strictly serialized.

## Progress

- Clean sibling worktree created; all current upstream changes inherited.
- Pinned GGUF downloaded and SHA-256 verified. Streaming repack, tokenizer,
  strict rotation policy, exact BF16 widening, and Metal transforms implemented.
- Initial 41-position reference probe passed, followed by **340/340 argmax
  agreement** on the three-input registered gate. Worst KL < 8.4e-8 nats.
- CPU/API/loader/repack suites and GPU transform/backend/ops suites passed;
  legacy T2/B1/q4s artifact validation passed.
- Native agent and TUI recovered; all 9 native CPU suites and 8 Rust tests
  pass. Native read-tool round trip and saved-session restart (41 cached
  tokens) pass. One earlier cold snapshot attempt timed out at 180 seconds;
  root cause/soak remains open rather than declaring release readiness.
- Bonsai-specific Chat/Messages/Responses/SSE/count_tokens smoke passed.
  Review exposed and fixed XML mask-cache aliasing across tools, native
  device-pool exhaustion across generations, and HTTP's stale JSON-only
  constraint wiring. CPU tests compare 2,148 XML prefix/schema states with
  direct token simulation and exercise 128 full 64-slot device-pool epochs.
  All six streaming/non-streaming AUTO XML tool routes now engage the matching
  schema-aware grammar; live tests pass with `--constrain-tools` enabled.
  The q4s recovery script's explicit snapshot-publication failpoint did not
  fire: upstream intentionally ignores hints on serial-only Bonsai. That gate
  is NOT claimed as passed or weakened.
- Kernel-agent harness recovered and baseline-only mechanics tested. Its
  new production-width fixture initially exposed cancellation roundoff in the
  old unbounded activation ramp (~800 at width 5120). Existing narrow fixtures
  and the 3e-4 tolerance are unchanged; added wide inputs are scaled by 1/128
  to stay at normalized activation magnitudes. No optimization candidate ran.
- Source launcher and user guide added. Homebrew/prebuilt publication, PTQ1_0,
  accelerated Bonsai prefill, full task/long-context quality and 16 GiB mini
  validation are intentionally not part of this source-checkout milestone.

## Mini migration and review continuation

Continued in `$HOME/projects/q27-revival` on mac-mini (M4 / 16 GiB), leaving
its separate `q27` checkout untouched. All 963 migrated tracked files and
executable bits matched the laptop manifest before further edits. Yukon's
hash-locked Linux preparation exited 0; only the packed model, tokenizer and
notices were delivered. Both model hashes verified on the mini.

Additional accepted review findings and fixes:

- The native agent still used the old generic tokenizer template. Qwen3.8/
  Bonsai 2 now delegates to the current profile-aware renderer for generation,
  counting, save and restore, preserving reasoning and `stable_off`. The
  prefilled opener seeds both inline output and the thinking-budget tracker;
  thinking toggles cannot double-prefix a no-think transcript. Legacy-family
  rendering remains unchanged.
- The HTTP gate's aggregate engagement count could mask a missing route.
  Each request now has its own log boundary and exact call/name/argument checks.
- Empty XML parameter allowlists were conflated with absent schemas. Presence
  survives alignment, engagement resets and cache keys. The separate
  `additionalProperties` policy preserves open objects (including empty and
  required-only schemas) while enforcing required keys; contradictory closed
  schemas fail rather than silently weakening them. Legacy list-only empty
  schemas retain their permissive fallback. This is name/required-key constraint support, not
  full JSON Schema value validation.
- HTTP could decode unmasked after device-pool exhaustion. Native and HTTP
  now share the checked serial transition. At completed serial-step boundaries,
  a full pool is recycled along with **all** device-index maps, retaining CPU
  masks. Valid long names complete at 8/64-slot test capacities without growing
  device memory; unrecoverable upload/grammar failures still abort before the
  next engine step. CUDA's speculative caller policy is unchanged.

Mini gates pass: ten native CPU suites, eight Rust tests, all Metal transform/
backend/ops tests, widened production-row harness baseline, all six HTTP XML
routes plus empty/required-only schema cases, native read/search, and default-
thinking budget/save/restart with 83 cached tokens. Repack fixtures also pass
on Yukon under normal and optimized Python. Model/GPU work remained serialized.
These are new context-2048 mini results, not copied laptop parity/timings; the
340-position Prism comparison was not rerun. Larger-context/default-8192 mini
validation and cold persistence soak remain open.

**Review closeout:** structured Codex run 9 exited **0**, with **no accepted/
actionable findings**. Seven findings from this continuation were accepted and
resolved (including the evidence-file omission); four earlier handoff fixes
were retained. No findings were rejected. The final command was:

```sh
~/.pi/agent/skills/autoreview/scripts/autoreview --mode local --engine codex \
  --prompt-file logs/revival/review9-notes.md \
  --dataset docs/metal/evidence/bonsai2-mini-2026-09-18.json \
  --output logs/revival/review9.txt --json-output logs/revival/review9.json \
  --stream-engine-output
```

The [validated review result](evidence/bonsai2-review-2026-09-18.json) is retained.
Only closeout documentation/evidence changed after the clean review; no extra
review was run. See [mini evidence](evidence/bonsai2-mini-2026-09-18.json) for
exact test commands, fresh measurements and local log digests. This checkpoint
is local only: no push, tag, package, Homebrew change or publication.

Current instructions: [BONSAI2.md](BONSAI2.md). Compact evidence:
[evidence/bonsai2-2026-09-18.json](evidence/bonsai2-2026-09-18.json).
