# Quasar (q27)

**Quasar** is a small native inference engine optimized first for
**Qwen3.6-27B-MTP** (a hybrid Gated-DeltaNet + attention model with
trained-in MTP heads), with support for the **PrismML Bonsai-27B** ternary
distillation of the same architecture. It is deliberately narrow and
completely self-contained, neither a generic GGUF runner nor a wrapper
around another runtime. It serves one model family on two backends, CUDA
for RTX 3090/5090-class cards and Metal for Apple Silicon, as fast as
possible, and it includes the loading, prompt rendering, tool-call
handling, KV state management, and server APIs needed to sit directly
under a coding agent.

This project is built in the spirit of
[antirez/ds4](https://github.com/antirez/ds4), for a different model and a
different architecture. Where ds4 targets DeepSeek V4, q27 targets the
Qwen3.6 hybrid. That model interleaves 48 linear-attention (Gated
DeltaNet) layers with 16 full-attention layers and one MTP draft layer,
an arrangement that makes almost every standard inference trick (paged
KV, prefix caching, speculative verification) behave differently than it
does on a pure-attention model.

## Motivations

Batch-1 agentic serving is its own discipline. An engine that sits under a
coding agent lives at batch size 1, decode-bound, with long mostly-stable
prompts, heavy tool traffic, and a user watching the tokens arrive.
General-purpose engines leave large factors on the table in exactly that
regime. q27's answer is to specialize. The weight layout is designed for
the GPU it runs on, speculation is built around the model's own trained
MTP head and a zero-cost suffix drafter, KV state snapshots and restores
byte-exactly, and the measurement discipline described below kills
attractive ideas before they ship.

## Acknowledgements

- [antirez/ds4](https://github.com/antirez/ds4): the model for what this
  project is trying to be.
- [signalnine/q27](https://github.com/signalnine/q27): the upstream CUDA
  engine this repository is forked from. The CUDA backend, the `.q27`
  format, and the original benchmarks are theirs. This fork adds the
  Metal backend, the Bonsai ternary tier, and the Apple Silicon serving
  work, and merges upstream regularly.
- [PrismML](https://huggingface.co/prism-ml): the Bonsai-27B
  ternary/binary QAT distillations and their llama.cpp fork, which
  serves as the reference implementation for those packs.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): the semantic
  reference for the architecture's forward pass.

## Status

q27 is beta quality, because inference is a complicated matter, but it is
measured beta. Every performance claim below traces to a dated entry in a
committed log. Correctness changes ship byte-identity or envelope gates
against the serial path, experiments are pre-registered with kill lines
before they run, and negative results are recorded with their mechanisms.
The parked list is as load-bearing as the shipped list.

- **Metal** is the serving stack and the active development backend. It is
  mature for single-request use on base M4 and serves two slots: layer-major
  prefill to width 96, decode at 98–99% of its own resident-weight ceiling,
  factor-2 tiled GQA attention, batched MTP and suffix-burst speculation,
  batched verification to width 48, disk prefix snapshots, and a fair
  two-slot scheduler with full memory admission accounting. The ledger
  records an official-tier two-slot MTP pass (813 committed tokens over 281
  rounds), but its raw multislot harness output is not committed, so release
  status remains provisional pending an evidence-publishing rerun. The
  numerical-envelope constants are also provisional.
- **CUDA** remains in-tree and is the fastest engine we have measured on
  this model (5090; see Speed), but **CUDA serving adoption is dropped
  (2026-07-18), not deferred — do not touch the CUDA code.** Yukon (the
  RTX 3090) stays as-is and is now a behavioral/numeric oracle only: the
  16-token canonical gate is byte-exact across CUDA and Metal. On the
  24 GB 3090 the merged width-8 server is validated at 16K context (boots
  at 23,230–23,232 MiB, passes the canonical gate); width 12 exceeds the
  card by ~980 MiB of prebuilt graph families, a build-width limit.
- **The mixed-tier (M1) thesis is permanently closed.** The gdn_pair rescue
  exonerated the pack itself (the repetition loop was a serving-binary
  artifact), and A5-on-gdn_pair then KILLED the strongest arm on both
  corpora. A B1→T2 bridge requires training/distillation, not byte-level
  grafting — now a measured result, not a conjecture.
- **No live traffic yet.** The native-agent direction (opt-in model-driven
  tools, durable sessions, compaction) has exact restart persistence and
  boundary-preserving compaction, but broad product validation is open and
  is gated behind the task-suite/native-agent graduation work.

## More documentation

The lab records are the real documentation of *why* things are the way
they are.

- [docs/MODELS.md](docs/MODELS.md): **which pack (quant) and context window
  for your Mac** — the M-chip speed table, the quant×context memory-fit
  table, and the recommendation guide behind `q27 recommend`. Start here as
  a user.
- [docs/metal/METAL_PROGRESS.md](docs/metal/METAL_PROGRESS.md): the Metal ledger,
  with a maintained **Current state** table on top of an append-only
  dated chronicle. Start here for the Apple Silicon side.
- [docs/BUILDLOG.md](docs/BUILDLOG.md): the CUDA build log (P0–P9).
- [docs/BENCHMARKING.md](docs/BENCHMARKING.md): cross-engine
  methodology, fairness controls, and reproduce steps for the SWE-bench
  numbers.
- [docs/FORMAT.md](docs/FORMAT.md): the `.q27` container and every
  dtype (Q4/Q8/T2/B1/Q4_1), with encodings verified against their
  sources.
- [docs/SPEC.md](docs/SPEC.md): per-layer forward-pass semantics.
- [docs/plans/](docs/plans/): pre-registered experiment plans, each with
  gates and kill lines and with results appended below the line.
- [docs/metal/NOTEBOOK.md](docs/metal/NOTEBOOK.md): the scratch pad this README used
  to be, preserving upstream's working notes, performance model, risk
  register, and progress logs verbatim. No longer maintained.

## Model weights

**Easy way (Homebrew):** `brew install manthedan/tap/q27`, then `q27 pull`
auto-picks the right pack for your RAM and `q27 serve` boots it. See
[docs/MODELS.md](docs/MODELS.md) for the pack/context/speed tables and the
experimental packs. The manual flow below is what `q27 pull` runs under the
hood.

Everything is Apache-2.0. The three official-model tiers live in one repo
([signalnine/Qwen3.6-27B-MTP-q27](https://huggingface.co/signalnine/Qwen3.6-27B-MTP-q27)),
and the ternary tier is repacked from PrismML's QAT pack:

| tier | file | size | runs on | pick it when |
|---|---|---|---|---|
| **default** (5.25 bpw) | `qwen36-27b-mtp.q27` | ~17 GiB | 24 GB+ GPU, 24 GB Mac | max speed with the trained MTP head; the only official tier that fits 24 GB |
| q6 (6.0 bpw) | `qwen36-27b-mtp-q6.q27` | larger | 32 GB | +0.35% PPL margin over Q5_K_M |
| q6k (6.8 bpw) | `qwen36-27b-mtp-q6k.q27` | larger | 32 GB | GGUF-matching quality, ~10% slower |
| **T2 ternary** (2.25 bpw) | repack of `Ternary-Bonsai-27B-Q2_0.gguf` | 7.15 GB | 16 GB Mac | fully-resident 27B on small machines; ~2.1–2.3× PPL vs official, task quality holds |
| ~~M1 candidate~~ (mixed; thesis CLOSED) | `bonsai-27b-m1.q27` | 3.83 GB | 16 GB Mac | mixed-graft thesis permanently closed: gdn_pair pack exonerated, A5-on-gdn_pair KILLED on both corpora; a B1→T2 bridge needs training, not grafting |
| B1 binary (1.125 bpw) | `bonsai-27b-b1.q27` | 3.79 GB | 16 GB Mac | smallest fully-resident tier; full quality/probe battery passed |

```bash
# official tier + tokenizer (swap --include for the tier you picked)
huggingface-cli download signalnine/Qwen3.6-27B-MTP-q27 \
  --include qwen36-27b-mtp.q27 qwen36-27b-mtp.tok CHECKSUMS.md5 \
  --local-dir models/qwen36-27b-mtp
```

Ternary/binary packs come from PrismML on Hugging Face and repack
losslessly (bit-exact round-trip gated) with:

```bash
python3 tools/repack.py Ternary-Bonsai-27B-Q2_0.gguf ternary-bonsai-27b-t2.q27
```

The M1 candidate was a byte-level graft of the companion B1/T2 checkpoints
(B1 bulk plus T2 GDN alpha/beta and attention-Q in blocks 21–42). The
mixed-graft thesis is now **permanently closed**: the gdn_pair rescue
exonerated the pack (the repetition loop was a serving-binary artifact),
and A5-on-gdn_pair KILLED the strongest arm on both corpora. History and
the negative result are recorded in
[`docs/metal/plans/2026-07-17-mixed-tier-census.md`](docs/metal/plans/2026-07-17-mixed-tier-census.md).

Verify every download against its `CHECKSUMS.md5`. The fine-tune variant
of the official model is `signalnine/Qwopus3.6-27B-v2-MTP-q27`.

## Speed

Every number here is measured, dated, and reproducible from the logs.
CUDA numbers are upstream's, re-validated on this fork byte-exactly.

| machine | tier | decode | notes |
|---|---|---|---|
| RTX 5090 | default | **202.7 t/s** agentic (231–246 t/s aggregate) | fused MTP + SuffixDraft verify; 12 pinned SWE-bench instances under Claude Code |
| RTX 3090 (24 GB) | default + turbo3 KV | **102.2 t/s** median | pre-merge w8 server, 131K context; merged w8 fit validated at 16K |
| M4 (24 GB) | default | ~12.5 t/s | batched MTP, 58.5% acceptance |
| M4 (16 GB mini) | T2 | ~11.7 t/s greedy | 99% of the machine's own resident-weight ceiling |
| M4 (24 GB) | B1 | **~18.7 t/s** greedy warm | memory wall: 3.36 GiB/token @ ~69 GB/s; select round 2 shipped (kernel 2.36×, wall noise — strictly dominates) |
| M4 | T2 prefill | **47.2 t/s** | width-96 layer-major chunks, prefill closed as MATURE on this hardware |

For calibration, the same model on the same 5090 runs at 117.1 t/s
through vLLM NVFP4+MTP, 116.3 t/s through mainline llama.cpp+MTP, and
62.0 t/s through llama.cpp without speculation, under the same
methodology ([docs/BENCHMARKING.md](docs/BENCHMARKING.md)).

The single-stream performance model is simple and holds on both backends.
Decode is weight-read-bound, so
`t/s ≈ bandwidth × efficiency ÷ bytes-per-step × accepted-tokens-per-round`.
Everything q27 does either reads fewer bytes (tiers, turbo3 KV), reads
them at higher efficiency (measured 81–90% of DRAM speed-of-light on the
3090 and 98–99% of ceiling on M4), or commits more tokens per read (MTP,
suffix bursts, batched verification).

## Speculative decoding

The model ships a trained MTP draft layer, and q27 treats speculation as
a first-class subsystem.

- **CUDA:** fused shared-KV MTP + SuffixDraft verification is the
  default serving stack, and it accounts for most of the 1.7× lead over
  other engines running the *same* MTP head.
- **Metal:** MTP verification is batched at widths 2–12 on the official
  tier, and batched verification is measured viable to width 48 (oracle
  S(48) = 3.94×, with cost flat per 16-token tile, so 16, 32, and 48
  are the efficient widths). Suffix-burst verification feeds exact
  repetition matches from a zero-cost CPU drafter through that wide
  verify path (`--suffix`, with `--suffix-serial` as the A/B control).
- **Measured and killed, so you don't have to:** a separately trained
  1.7B sibling drafter (vocabulary lineage mismatch), and the DSpark
  block-diffusion drafter, fully contract-extracted and losslessly
  repacked, then parked when its measured acceptance (τ = 3.2
  committed/round against a 4.1 break-even) put even a generous chained
  upper bound at 1.07× against a pre-registered 1.3× gate. The fixtures
  and a one-command re-gate are banked for future hardware.

## Long context

`Q27_KV=turbo3`, a 50-byte transformed KV codec, keeps the full
262,144-token native window practical. On CUDA it is validated flat to
361K tokens (needle 6/6 at depths beyond native, NLL buckets flat to
256K). The pre-merge w8 server used it to promote a 24 GB 3090 from a
32K box to a 131K box; the merged w8 server's 131K fit leg was not run
before CUDA serving adoption was dropped, and is no longer planned. On
Metal, 32K NLL is depth-flat (PPL 5.318 single-pass) with 6/6 needle
retrieval, and tail-focused codec allocation is the active KV-quality
frontier. Disk prefix snapshots restore a 2,346-token prefix byte-exactly
in 0.12 s where re-prefill takes ~52 s, which turns minutes into seconds
per question on repeated long documents.

## CLI

CUDA (`build/q27`) and Metal (`build/q27-metal`) share conventions, and
the CLI keeps conservative reference defaults so canonical gates stay
bitwise.

```bash
# CUDA: canonical smoke (md5 of the output line is the bitwise canonical)
./build/q27 model.q27 --tokens "760,6511,314,9338,369" -n 128 --ctx 2048 --spec

# Metal: greedy, MTP, suffix-burst, and long-context modes
./build/q27-metal model.q27 model.tok --prompt "..." -n 128
./build/q27-metal model.q27 model.tok --prompt "..." -n 128 --mtp 8
./build/q27-metal model.q27 model.tok --prompt "..." -n 128 --suffix 48
./build/q27-metal model.q27 model.tok --tokens IDS --ctx 131072 --kv turbo3
```

Quality and correctness instruments ride the same binary: `--nll`,
`--nll-long`, `--kl-kv`, `--envelope`, `--chunk-parity`, `--oracle`,
`--eos-gate`, `--save-state`/`--load-state`.

## Native coding agent

The friendly source-checkout command builds and starts the B1 agent in the
current directory:

```bash
make agent
```

The equivalent current-source supervisor command is
`./packaging/bin/q27 agent`. It defaults to B1, context 32768,
`--max-tokens auto`, automatic tools, and the physical path of the current
workspace. Packs and ordinary native-agent flags remain available:

```bash
make agent                                      # B1 in the current project
Q27_AGENT_PACK=t2 make agent                    # choose another local pack
Q27_AGENT_SESSION=work.q27agent make agent
Q27_AGENT_CONTEXT=16384 make agent
Q27_AGENT_WORKSPACE=/path/to/project make agent
```

The published v0.3 Homebrew archive predates this supervisor subcommand;
`q27 agent` will become the installed spelling in the next tagged release.
Friends testing it now should use a source checkout and `make agent`.

`Q27_AGENT_PACK`, `Q27_AGENT_MAX_TOKENS`, and `Q27_AGENT_SESSION` provide
script-friendly defaults. The wrapper holds a private kernel lifetime lock
shared with `q27 serve`, preventing concurrent multi-GB model launches. The
lock is inherited across `exec` and released automatically on every process
exit, including signals and crashes. Read/search results expose short,
digest-bound line-selection handles so B1 can edit one exact occurrence without
regenerating a unique `old` string; stale selections fail before mutation.
Conservative no-progress bounds terminate pathological output explicitly as
`generation stalled`, invalidate resident reuse, and discard incomplete calls
or raw payloads without side effects. File helpers stay workspace-bounded, but
automatic shell retains the local account's
filesystem access; use a disposable checkout when testing an untrusted prompt.

## Server

Both backends serve the native Anthropic Messages API plus OpenAI
`/v1/chat/completions`, `/v1/completions`, and `/v1/responses`, with true
token streaming, stop handling, disconnect cancellation, tool-call
constraint decoding, and `count_tokens`, all shaped so Claude Code
compacts correctly and Codex CLI gets its full item lifecycle
(function_call items, reasoning items, custom_tool_call bridging).

```bash
./build/q27-server model.q27 model.tok --port 8080          # CUDA
./build/q27-metal-server model.q27 model.tok --port 8080    # Metal (--slots 2)
```

Point a coding agent at it:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8080"
export ANTHROPIC_API_KEY="placeholder"
export ANTHROPIC_DEFAULT_OPUS_MODEL="q27"   # server is single-model
claude

# Codex CLI (Responses API), one-shot provider override:
codex -c model_provider=q27 -c model=q27-metal \
  -c 'model_providers.q27.base_url="http://localhost:8080/v1"' \
  -c 'model_providers.q27.wire_api="responses"'
```

The CUDA server's zero-config defaults are the measured Claude-Code
stack (fp8 KV on sm_89+, suffix drafter, fast-head, auto-sized `--ctx`),
and `Q27_PROFILE=ref` restores conservative reference behavior. The
Metal server runs two slots on one model mapping with a fair FIFO GPU
lease, adaptive prefill quanta (96/48/12), full per-slot memory admission
accounting, and a dedicated 503 on overload. `/health` reports the
resident model artifact by name.

**Metal serving knobs (shipped semantics; CLI flags since Homebrew
Phase-2 pre-tag — each flag falls back to its env twin when absent, and
an explicit flag wins over the env):**

```bash
./build/q27-metal-server model.q27 model.tok --port 8080 \
  --snapshot-dir ~/.q27/snapshots --max-tokens-default 16384 \
  --trace ~/.q27/trace.jsonl
```

| knob | env fallback | default | effect |
|---|---|---|---|
| `--trace <path>` | — | off | whole-session JSONL: rendered prompts, prefix/snapshot decisions, tool recoveries, cancels, errors |
| `--snapshot-dir <path>` | `Q27_METAL_SNAPSHOT_DIR` | off | disk prefix snapshots (load-bearing for agent TTFT: measured 8.1 s vs 4:54 cold at 8,256 tokens) |
| `--snapshot-max-mb N` | `Q27_METAL_SNAPSHOT_MAX_MB` | 8192 | snapshot directory budget, LRU demote |
| `--snapshot-auto N` | `Q27_METAL_SNAPSHOT_AUTO` | 4096 | prompts ≥ N tokens auto-save at a 96-aligned boundary; 0 = hint-only |
| `--max-tokens-default N` | `Q27_METAL_MAX_TOKENS_DEFAULT` | per-endpoint | generation cap when the client sends none/null (pi sends null; 16384 recommended for agents) |
| `--budget-mb N` | `Q27_METAL_BUDGET_MB` | half working set | multislot admission budget (test/override hook) |
| `Q27_METAL_KV_FP16_CELLS` | env-only (engine ctor plumbing pending) | off | turbo3 KV fp16 exception cells (production: `8,9,10,11,12,13,14,15`) |

**The server has no auth and binds 127.0.0.1 by default.** Reaching it
from other machines or containers requires an explicit `--host 0.0.0.0`.

## Backends

- **CUDA:** one dual-arch binary (sm_86 + sm_120). fp8-KV and e4m3 MMA
  need sm_89+, while Ampere runs fp16-MMA verify with fp16 or turbo3
  KV. 24 GB cards use the `q27-server-w8` build; width 12 exceeds the
  measured graph-memory envelope.
- **Metal:** developed and gated on base M4, at 24 GB and 16 GB.
  Everything targets plain `simdgroup_matrix`, with no Metal-4
  cooperative-tensor dependence. The official tier wants a 24 GB
  machine, T2 is fully resident on 16 GB, and one model load at a time
  is a hard memory-policy rule on 24 GB machines.

## Test vectors and gates

The canonical prompt is token IDs `760,6511,314,9338,369` ("The capital
of France is"). Its 16-token continuation is byte-exact across CUDA and
Metal (SHA `6c1d4328…`), and the 128-token CUDA output line has a pinned
md5. `make test-cpu test-metal` is the default gate (synthetic,
memory-safe, no artifact), while artifact-scale gate scripts live in
`tools/` and write their verdicts to `logs/`.

A few house rules shaped the codebase, in case you wonder why the logs
look the way they do. Experiments are pre-registered with kill lines
before they run. Gates must be demonstrated *able to fail* before a pass
counts, because we have been burned by gates that could not. Binaries are
rebuilt before every gate run (the stale-binary rule). Committed-token
changes require byte-identity or a measured, documented tolerance class,
and every code change gets an adversarial review round before it counts
as done.

## Debugging notes

Each subsystem has a trace env. `Q27_METAL_PROFILE=1` attributes GPU time
per kernel, `Q27_MTP_TRACE`/`Q27_ORACLE_TRACE`/`Q27_SUFFIX_TRACE` print
per-round speculation anatomy, and `Q27_BATCH=1` turns on the
experimental CUDA continuous batching. Synthetic benches
(`metal_decode_bench`, `metal_prefill_bench`, `metal_gemv_bench`,
`metal_attn_bench`) reproduce every kernel-level number without loading
the artifact. If a run misbehaves, the first three suspects are a stale
binary, a page-thrashed mmap from two model loads, and desktop contention
on a timing leg, in that order, from experience.

## Development process

This engine is developed with strong AI assistance. Implementation and
review are largely done by LLM agents, with independent adversarial
review on every change, while humans lead the ideas, the priorities, the
measurement discipline, and every authorization to touch a GPU. We say
this openly because it shaped how the project was built. The
pre-registration and kill-line culture exists precisely so that no one,
human or model, gets to believe their own optimism.
