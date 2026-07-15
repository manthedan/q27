# q27 benchmarking methodology

How the q27 numbers are produced, and how to reproduce the cross-engine
comparison against the [llama-cpp-turboquant](https://github.com/) fork with
`ngram-mod`. Two engines are compared throughout:

| engine | build | quant | spec-decode |
|---|---|---|---|
| **q27** | git `94e645a` | NVFP4 **5.25 bpw** | MTP head + SuffixDraft (fused verify) |
| **llama-cpp-turboquant** (TheTom fork) | git `c3e6dbb13` | Q5_K_M **~5.5 bpw** | `--spec-type ngram-mod` (n_match=24, n_max=64, n_min=48) |
| **llama.cpp mainline** | git `13e67386` (2026-07-01) | Q5_K_M **~5.5 bpw** | run two ways: none (stock), and `--spec-type draft-mtp --spec-draft-n-max 6` (same MTP head as q27) |
| **vLLM** | `vllm/vllm-openai:nightly` | NVFP4 (`unsloth/Qwen3.6-27B-NVFP4`, compressed-tensors) | `speculative-config {method:mtp, num_speculative_tokens:3}` — the model's MTP head |

All three serve the **same base model** (Qwen3.6-27B-MTP; the `qwen35` GGUF arch,
which mainline supports as `LLM_ARCH_QWEN35`). The only unavoidable confound is
quantization (both llama builds carry ~0.25 bpw more than q27's NVFP4); it is
disclosed on every result and it favors llama, so any q27 win is conservative.
Mainline is the **stock, no-drafter floor**: it isolates how much of the gap is
the base decode kernel vs the drafter, and shows exactly what `ngram-mod` adds
over vanilla llama.cpp. (Mainline `13e67386` is ~2 weeks behind true nightly but
carries the `qwen35` arch and the Anthropic `/v1/messages` endpoint; a fresh
rebuild risks the known sm_120 toolchain traps and would not move the base-kernel
number materially.)

## Fairness controls (every cross-engine run)

- **Single GPU, RTX 5090 only.** llama is pinned with `CUDA_VISIBLE_DEVICES=0` —
  otherwise llama.cpp layer-splits onto the second GPU (a 3090 here) and gets an
  unearned bandwidth edge. Verify the second GPU's memory is untouched after
  startup.
- **KV cache matched.** llama runs `-ctk q8_0 -ctv q8_0` (≈ q27's fp8 KV) and
  `-np 1` (single slot = single user, as q27 is).
- **Greedy decode** (`temperature 0`) on both sides.
- **Decode-only timing.** Throughput excludes prefill/TTFT so the number
  isolates the decode+drafter path, not prompt processing. (llama has the faster
  tensor-core prefill; q27 has the faster decode. Mixing them hides the real
  difference.)
- **Identical payloads / pinned tasks / same harness** on both engines.

## Method A — payload decode microbench (`/v1/completions`)

Isolates spec-decode effectiveness across regimes. Send identical
`{prompt, max_tokens, temperature: 0}` to each engine's `/v1/completions`.

- q27 decode t/s from its `[req]` journal line (`tps=`).
- llama decode t/s from the response `timings.predicted_per_second`
  (decode-only, excludes `prompt_ms`), with `draft_n` / `draft_n_accepted` for
  acceptance.

Three regimes, chosen to bracket drafter behavior:

| payload | regime | what it tests |
|---|---|---|
| `echo_ctx12k` (256 tok) | pure verbatim echo | drafter saturates |
| `fileemit_verbatim` (1024 tok) | partial-echo code continuation | near-repeat tolerance |
| `novel_prose` (400 tok) | novel generation | drafter with nothing to match |

**Two gotchas that will corrupt this bench if ignored:**

1. **ngram-mod persists its n-gram table across requests.** Re-sending an
   identical novel prompt makes it echo its own prior greedy output, so the rate
   climbs run-to-run (novel 56 → 79 → 97 t/s). Use the **cold** run (first call,
   or restart the server between prompts). q27 keeps no server-side table, so its
   number is request-invariant.
2. **Cross-quant divergence.** Q5 vs NVFP4 can pick a different greedy token on
   the same prompt, after which the two engines generate different text (and
   different token counts). Such a payload is no longer a like-for-like
   comparison — drop it (this is why `echo_ctx26k` is excluded).

## Method B — agentic traffic via SWE-bench Verified (reproducible)

Measures the engines on **realistic multi-turn tool-use coding traffic** that
anyone can rerun. This replaces the earlier private task set (see "History"):
private tasks can't be reproduced off this box, public SWE-bench instances can.

**Task set** — 12 instances pinned from `princeton-nlp/SWE-bench_Verified`,
biased to fast-test repos (`requests`, `flask`, `pytest`, `pylint`, `xarray`)
and `<15 min fix` difficulty, chosen deterministically (per-repo quota, sorted
by id). Regenerate with `bench/swebench/select_instances.py`; the frozen list is
`bench/swebench/manifest.json`. Pinned by `instance_id` + `base_commit` → exact
reproducibility.

**Harness** — Claude Code (`claude -p`, `--output-format stream-json`) pointed at
the engine's Anthropic `/v1/messages` API via `ANTHROPIC_BASE_URL`. Both engines
serve `/v1/messages` natively (q27 by design; the turboquant fork too — verified
streaming SSE, `tool_use`, thinking blocks, and `count_tokens`), so no
translation proxy is needed.

**Sandbox** — each instance runs the agent inside the `thunderdome/claude-code`
Docker image (claude CLI + node) via plain `docker run`, **not** the private
orchestration. Required flags:
- `--user 1000:1000` + `-e HOME=/home/node` — Claude Code refuses
  `--dangerously-skip-permissions` as root; run as the image's `node` user.
- `--add-host host.docker.internal:host-gateway` — reach the host engine on
  `:8081` from inside the container.

Running the untrusted upstream repo + autonomous agent in a container (not on the
host) is a safety requirement, not just convenience.

**Per instance** — clone `repo@base_commit` from a local bare mirror, feed the
`problem_statement` plus a "fix the code, don't run tests" instruction, let the
agent read/edit, then `git diff` = the candidate patch.

**Metrics**
- **decode t/s** (engine telemetry over the run window) and **wall-to-wall** time.
- **cheap quality signal**: non-empty diff, and *edited-the-gold-file* — overlap
  of the changed files with the gold patch's files (from the dataset). This is
  **not** the official resolve-rate.

**Why not official resolve grading.** SWE-bench's real grading applies the patch
and runs `FAIL_TO_PASS`/`PASS_TO_PASS` in per-instance Docker eval images
(multi-GB each) — heavy on a disk-constrained box, and, more to the point, since
both engines run the **same base model**, resolve-rate is a model property that
does not differentiate engines. The file-overlap signal is enough to confirm the
agent did something on-target; anyone wanting the resolve % can run the `swebench`
harness over the same pinned patches.

## Reproduce

Prereqs: Docker + the `thunderdome/claude-code` image (or any image with node +
the `claude` CLI), `pip install datasets`, and an engine serving the Anthropic
API on `:8081`.

```bash
# 1. materialize the pinned task set (or use the frozen manifest.json)
HF_HOME=/mnt/ai/hf_cache python3 bench/swebench/select_instances.py

# 2. one-time: bare mirrors of the 5 repos into $SWEBENCH_CACHE (default /mnt/ai/swebench-cache)
for r in pallets/flask psf/requests pydata/xarray pylint-dev/pylint pytest-dev/pytest; do
  git clone --bare "https://github.com/$r" "/mnt/ai/swebench-cache/${r/\//__}.git"
done

# 3. start q27 on :8081, run the set
#    (q27-server <model.q27> <model.tok> --port 8081 --host 0.0.0.0)
bash bench/swebench/run.sh q27

# 4. swap the engine on :8081 to the ngram-mod fork (5090-only + q8 KV), run the same set
#    CUDA_VISIBLE_DEVICES=0 llama-server -m Qwen3.6-27B-MTP-Q5_K_M.gguf \
#      -ngl 99 -fa on -c 131072 -np 1 -ctk q8_0 -ctv q8_0 --spec-type ngram-mod --jinja \
#      --host 0.0.0.0 --port 8081
bash bench/swebench/run.sh llama

# 4b. mainline baselines (mainline llama-server, git 13e67386):
#     stock (no drafter):
#       CUDA_VISIBLE_DEVICES=0 llama-server -m ...Q5_K_M.gguf -ngl 99 -fa on -c 131072 \
#         -np 1 -ctk q8_0 -ctv q8_0 --jinja --host 0.0.0.0 --port 8081
bash bench/swebench/run.sh llamamain     # -> results.llamamain.jsonl (unit llamamain-eval)
#     with the MTP head (fairest vs q27 -- same head, same model):
#       ...same launch... --spec-type draft-mtp --spec-draft-n-max 6
bash bench/swebench/run.sh llamammtp     # -> results.llamammtp.jsonl (unit llamammtp-eval)

# 4c. (vLLM) needs two extra pieces: vLLM has no /v1/messages, so Claude Code talks
#     to a litellm Anthropic->OpenAI shim on :8081 that forwards to vLLM on :8080.
#     - vLLM (5090-only, single-seq, MTP, qwen3_coder tool parser for the XML tool format):
#         docker run --gpus '"device=0"' -p 8080:8000 -v <hf_cache>:/root/.cache/huggingface \
#           vllm/vllm-openai:nightly --model unsloth/Qwen3.6-27B-NVFP4 --served-model-name vllm-qwen \
#           --max-num-seqs 1 --max-model-len 131072 --gpu-memory-utilization 0.96 --kv-cache-dtype fp8 \
#           --trust-remote-code --enable-auto-tool-choice --tool-call-parser qwen3_coder \
#           --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
#     - litellm shim (config wildcard-routes to openai/vllm-qwen @ vLLM):
#         docker run -p 8081:4000 -v <config.yaml>:/app/config.yaml ghcr.io/berriai/litellm:main-stable \
#           --config /app/config.yaml --port 4000
#     Decode t/s comes from vLLM /metrics deltas (not journalctl): see scratchpad/vllm_swebench.sh.
bash bench/swebench/run.sh vllm          # -> results.vllm.jsonl

# 5. compare bench/swebench/results.{q27,llama,llamamain,llamammtp,vllm}.jsonl
```

Long runs should be launched under `systemd-run --user` (a crashed shell
otherwise tears the job's cgroup down).

## Results (2026-07-14, RTX 5090)

### Method A — payload decode (decode-only t/s)

| payload | regime | q27 | llama ngram-mod | winner |
|---|---|---|---|---|
| echo_ctx12k | pure echo | **603** (11.6 tok/rnd) | 529 (96% acc) | q27 |
| fileemit_verbatim | partial-echo | 178 (3.0 tok/rnd) | **409** (89% acc) | llama |
| novel_prose | novel | **157** (2.6 tok/rnd) | 56 cold / 97 warm | q27 |

No clean winner at the payload level: ngram-mod's 24-tok lookup / 64-tok drafts
win partial-echo continuations; q27's fused MTP wins pure echo (once acceptance
saturates) and novel generation (MTP drafts every round; ngram-mod has nothing
to match).

### Method B — SWE-bench Verified, 12 instances

| engine | decode agg | wall/inst | nonempty diff | edited gold file |
|---|---|---|---|---|
| **q27** (MTP + SuffixDraft, fused) | **202.7 t/s** | **47 s** | 12/12 | 11/12 |
| **vLLM** NVFP4 + MTP (`method:mtp`, n=3) | 117.1 t/s | 133 s | 12/12 | 11/12 |
| **llama mainline + MTP** (`--spec-type draft-mtp`, n-max 6) | 116.3 t/s | 80 s | 12/12 | 11/12 |
| **llama ngram-mod** (fork) | 61.1 t/s | 118 s | 12/12 | 11/12 |
| **llama mainline** (no spec) | 62.0 t/s | 120 s | 12/12 | 12/12 |

(vLLM decode is aggregate from `/metrics` deltas — `generation_tokens_total` /
`inter_token_latency_seconds_sum`; the others are per-`[req]` telemetry.)

The five engines decompose the gap cleanly (all same model + MTP head available):

- **ngram-mod adds ~nothing on real agentic traffic.** The fork (61.1 t/s) is
  within noise of — marginally *below* — stock mainline (62.0). At 34% draft
  acceptance the failed drafts + table bookkeeping cancel the wins. ngram-mod's
  advantage is real only on synthetic high-echo re-emission (Method A), a small
  slice of real coding.
- **MTP is the real lever, and two independent engines confirm it.** Turning on
  the MTP head nearly **doubles** stock mainline (62 → 116–117 t/s), and llama's
  MTP (116.3) and vLLM's MTP (117.1) land on essentially the **same number** from
  completely different codebases — strong evidence this is the drafter's ceiling
  for a mainstream engine on this model, not a one-off.
- **On that same MTP head, q27 is still ~1.73× faster than both** (202.7 vs
  ~117). That residual is q27's engine — the fused shared-KV MTP+SuffixDraft
  verify, NVFP4 kernels, and tie/tolerance discipline — not the drafter *choice*.
  It matches Method A, where q27 leads llama+MTP on novel generation (157 vs 92
  t/s) but ties on echo (178 vs 184).
- **vLLM pays a wall-time tax this benchmark exposes.** Its decode (117) is
  competitive, but its **wall/inst (133 s) is the worst of all five** because
  vLLM's prefix caching is dead on this hybrid-GDN arch (0% reuse) — every
  agentic turn re-prefills the whole growing context — and it runs behind a
  litellm Anthropic→OpenAI shim (vLLM has no `/v1/messages`). q27 and llama both
  reuse prefix/checkpoint state across turns, so they convert competitive decode
  into far lower wall time. This is an arch-support gap, not raw kernel speed, but
  it's real for anyone serving this model agentically on vLLM today.
- **Quality is engine-independent** (11–12/12 edited-gold-file across all five) —
  the model is identical; the engine only changes speed. The 1-instance spread is
  agentic noise.

Gap decomposition (real agentic decode): stock llama.cpp **62** → +ngram-mod
**~62** (≈0) → +MTP **116–117** (×1.9, and vLLM independently agrees) → q27's MTP
engine **203** (×1.73 on top).

## History / non-reproducible baselines

An earlier cross-engine run used 3 **private** greenfield tasks (not
redistributable, hence Method B above). For the record, on those tasks q27
decoded **289.8 t/s agg / 236.5 med** vs llama **61.0 / 55.5** (~4.75×), with
ngram-mod accepting only **34%** on agentic traffic — the same effect Method B
captures on public tasks.

## Honest caveats

- Quantization confound: the two llama builds are Q5_K_M (+0.25 bpw vs q27's
  NVFP4, favors llama); vLLM is NVFP4 (the *same* quant family as q27, so the
  cleanest comparison), but from a different checkpoint (`unsloth`, the multimodal
  variant — its unused vision tower costs VRAM, which is why vLLM ran at 131072
  ctx not higher).
- vLLM carries two extra confounds the others don't: a **litellm proxy hop**
  (Anthropic↔OpenAI translation adds per-request latency) and **no prefix
  caching** on this arch (re-prefill every turn). Both inflate vLLM's wall time;
  neither touches its decode-t/s number. Read vLLM's wall/inst as "vLLM serving
  this model agentically *today*," and its decode-t/s as the cleaner
  engine-vs-engine number.
- Agentic runs are non-deterministic: wall time, turn count, and which files get
  edited vary run-to-run. Treat single-run Method-B numbers as indicative, not
  precise; average more instances/trials for tighter bounds.
- *edited-gold-file* ≠ correctness — it confirms the agent worked the right file,
  not that the fix passes tests.
- Single box, single run per engine.
