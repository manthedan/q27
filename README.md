# Quasar (q27)

**Quasar** is a small native inference engine optimized first for
**Qwen3.6-27B-MTP** (a hybrid Gated-DeltaNet + attention model with
trained-in MTP heads), with support for the **PrismML Bonsai-27B** ternary
distillation of the same architecture. It is intentionally narrow: not a
generic GGUF runner, not a wrapper around another runtime — it is
completely self-contained. One model family, two backends (CUDA for
RTX 3090/5090-class cards, Metal for Apple Silicon), as fast as possible,
with the loading, prompt rendering, tool-call handling, KV state
management, and server APIs needed to sit directly under a coding agent.

This project is built in the spirit of
[antirez/ds4](https://github.com/antirez/ds4), for a different model and a
different architecture. Where ds4 targets DeepSeek V4, q27 targets the
Qwen3.6 hybrid — 48 linear-attention (Gated DeltaNet) layers interleaved
with 16 full-attention layers and one MTP draft layer, which makes almost
every standard inference trick (paged KV, prefix caching, speculative
verification) behave differently than it does on a pure-attention model.

## Motivations

Batch-1 agentic serving is its own discipline. An engine that sits under a
coding agent lives at batch size 1, decode-bound, with long mostly-stable
prompts, heavy tool traffic, and a user watching the tokens arrive.
General-purpose engines leave large factors on the table in exactly that
regime. q27's answer is to specialize: one weight layout designed for the
GPU it runs on, speculation built around the model's own trained MTP head
and a zero-cost suffix drafter, KV state that snapshots and restores
byte-exactly, and a measurement discipline (below) that kills attractive
ideas before they ship.

## Acknowledgements

- [antirez/ds4](https://github.com/antirez/ds4) — the model for what this
  project is trying to be.
- [signalnine/q27](https://github.com/signalnine/q27) — the upstream CUDA
  engine this repository is forked from. The CUDA backend, the `.q27`
  format, and the original benchmarks are theirs; this fork adds the Metal
  backend, the Bonsai ternary tier, and the Apple Silicon serving work,
  and merges upstream regularly.
- [PrismML](https://huggingface.co/prism-ml) — the Bonsai-27B ternary/
  binary QAT distillations and their llama.cpp fork, which serves as the
  reference implementation for those packs.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the semantic
  reference for the architecture's forward pass.

## Status

Beta quality, because inference is a complicated matter — but measured
beta: every performance claim below traces to a dated entry in a committed
log, correctness changes ship byte-identity or envelope gates against the
serial path, and experiments are pre-registered with kill lines before
they run. Negative results are recorded with their mechanisms; the parked
list is as load-bearing as the shipped list.

- **CUDA** is the production backend. On a 5090 it is the fastest engine
  we have measured on this model by a wide margin (see Speed). On 24 GB
  3090-class cards the width-8 server build serves 131K context with
  turbo3 KV — **but note a live regression:** the post-conductor-merge
  server currently OOMs in `cudaGraphInstantiate` on 24 GB even at w8; a
  pre-merge w8 build still serves, the merged CLI passes the canonical
  gate, and the fix direction (lazy graph families) is recorded. Until it
  lands, treat merged-*server* 3090 support as broken.
- **Metal** is mature for single-request use on base M4 and serves two
  slots: layer-major prefill to width 96, decode at 98–99% of its own
  resident-weight ceiling, factor-2 tiled GQA attention, batched MTP and
  suffix-burst speculation, batched verification to width 48, disk
  prefix snapshots, and a fair two-slot scheduler with full memory
  admission accounting. Multi-slot MTP on the official tier and the
  numerical-envelope constants are still provisional.
- **Continuous batching** (CUDA) passed its bars (1.31–1.35× two-slot
  aggregate) but ships default-off pending live validation.

## More documentation

The lab records are the real documentation of *why* things are the way
they are:

- [docs/METAL_PROGRESS.md](docs/METAL_PROGRESS.md) — the Metal ledger: a
  maintained **Current state** table on top, an append-only dated
  chronicle underneath. Start here for the Apple Silicon side.
- [docs/BUILDLOG.md](docs/BUILDLOG.md) — the CUDA build log (P0–P9).
- [docs/BENCHMARKING.md](docs/BENCHMARKING.md) — cross-engine methodology,
  fairness controls, and reproduce steps for the SWE-bench numbers.
- [docs/FORMAT.md](docs/FORMAT.md) — the `.q27` container and every dtype
  (Q4/Q8/T2/B1/Q4_1), with encodings verified against their sources.
- [docs/SPEC.md](docs/SPEC.md) — per-layer forward-pass semantics.
- [docs/plans/](docs/plans/) — pre-registered experiment plans, each with
  gates and kill lines, results appended below the line.
- [docs/NOTEBOOK.md](docs/NOTEBOOK.md) — the scratch pad this README used
  to be: upstream's working notes, performance model, risk register, and
  progress logs, preserved verbatim and no longer maintained.

## Model weights

Everything is Apache-2.0. Three official-model tiers, one repo
([signalnine/Qwen3.6-27B-MTP-q27](https://huggingface.co/signalnine/Qwen3.6-27B-MTP-q27)),
plus the ternary tier repacked from PrismML's QAT pack:

| tier | file | size | runs on | pick it when |
|---|---|---|---|---|
| **default** (5.25 bpw) | `qwen36-27b-mtp.q27` | ~17 GiB | 24 GB+ GPU, 24 GB Mac | max speed with the trained MTP head; the only official tier that fits 24 GB |
| q6 (6.0 bpw) | `qwen36-27b-mtp-q6.q27` | larger | 32 GB | +0.35% PPL margin over Q5_K_M |
| q6k (6.8 bpw) | `qwen36-27b-mtp-q6k.q27` | larger | 32 GB | GGUF-matching quality, ~10% slower |
| **T2 ternary** (2.25 bpw) | repack of `Ternary-Bonsai-27B-Q2_0.gguf` | 7.15 GB | 16 GB Mac | fully-resident 27B on small machines; ~2.1–2.3× PPL vs official, task quality holds |
| B1 binary (1.125 bpw) | not yet produced | 3.9 GB | — | quality gate passed; native kernel decision (Phase 0B) pending |

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

Verify every download against its `CHECKSUMS.md5`. Fine-tune variant of
the official model: `signalnine/Qwopus3.6-27B-v2-MTP-q27`.

## Speed

All numbers are measured, dated, and reproducible from the logs; nothing
here is a projection. CUDA numbers are upstream's, re-validated on this
fork byte-exactly.

| machine | tier | decode | notes |
|---|---|---|---|
| RTX 5090 | default | **202.7 t/s** agentic (231–246 t/s aggregate) | fused MTP + SuffixDraft verify; 12 pinned SWE-bench instances under Claude Code |
| RTX 3090 (24 GB) | default + turbo3 KV | **102.2 t/s** median | w8 server, 131K context |
| M4 (24 GB) | default | ~12.5 t/s | batched MTP, 58.5% acceptance |
| M4 (16 GB mini) | T2 | ~11.7 t/s greedy | 99% of the machine's own resident-weight ceiling |
| M4 | T2 prefill | **47.2 t/s** | width-96 layer-major chunks; prefill is closed as MATURE on this hardware |

For calibration, the same model on the same 5090 through other engines
(same methodology, [docs/BENCHMARKING.md](docs/BENCHMARKING.md)): vLLM
NVFP4+MTP 117.1 t/s, mainline llama.cpp+MTP 116.3 t/s, llama.cpp without
speculation 62.0 t/s.

The single-stream performance model is simple and it holds on both
backends: decode is weight-read-bound, so
`t/s ≈ bandwidth × efficiency ÷ bytes-per-step × accepted-tokens-per-round`.
Everything the project does is one of: read fewer bytes (tiers, turbo3
KV), read them at higher efficiency (measured 81–90% of DRAM
speed-of-light on the 3090; 98–99% of ceiling on M4), or commit more
tokens per read (MTP, suffix bursts, batched verification).

## Speculative decoding

The model ships a trained MTP draft layer, and q27 treats speculation as
a first-class subsystem rather than an add-on:

- **CUDA:** fused shared-KV MTP + SuffixDraft verification is the default
  serving stack and is most of the 1.7× lead over other engines running
  the *same* MTP head.
- **Metal:** batched MTP verification (widths 2–12) on the official tier;
  batched verification is measured viable to **width 48** (oracle
  S(48) = 3.94×, cost flat per 16-token tile, so 16/32/48 are the
  efficient widths); **suffix-burst verification** feeds exact
  repetition matches from a zero-cost CPU drafter through that wide
  verify path (`--suffix`, with `--suffix-serial` as the A/B control).
- **What we measured and killed, so you don't have to:** a separately
  trained 1.7B sibling drafter (vocabulary lineage mismatch), and the
  DSpark block-diffusion drafter — fully contract-extracted, losslessly
  repacked, then parked when its measured acceptance (τ = 3.2
  committed/round vs a 4.1 break-even) put even a generous chained upper
  bound at 1.07× against a pre-registered 1.3× gate. The fixtures and a
  one-command re-gate are banked for future hardware.

## Long context

`Q27_KV=turbo3` (a 50-byte transformed KV codec) keeps the full 262,144
native window practical: validated flat to 361K tokens on CUDA (needle
6/6 at depths beyond native, NLL buckets flat to 256K), and promotes a
24 GB 3090 from a 32K box to a 131K box. On Metal, 32K NLL is depth-flat
(PPL 5.318 single-pass) with 6/6 needle retrieval; KV-quality research
(tail-focused codec allocation) is the active frontier. Disk prefix
snapshots restore a 2,346-token prefix byte-exactly in 0.12 s vs ~52 s of
re-prefill — the difference between minutes and seconds per question on
repeated long documents.

## CLI

CUDA (`build/q27`) and Metal (`build/q27-metal`) share conventions; the
CLI keeps conservative reference defaults so canonical gates stay bitwise.

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

## Server

Both backends serve the native Anthropic Messages API plus OpenAI
`/v1/chat/completions` and `/v1/completions`, with true token streaming,
stop handling, disconnect cancellation, tool-call constraint decoding,
and `count_tokens` — shaped so Claude Code compacts correctly.

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
```

The CUDA server's zero-config defaults are the measured Claude-Code
stack (fp8 KV on sm_89+, suffix drafter, fast-head, auto-sized `--ctx`);
`Q27_PROFILE=ref` restores conservative reference behavior. The Metal
server runs two slots on one model mapping with a fair FIFO GPU lease,
adaptive prefill quanta (96/48/12), full per-slot memory admission
accounting, and a dedicated 503 on overload.

**The server has no auth and binds 127.0.0.1 by default.** Reaching it
from other machines or containers requires an explicit `--host 0.0.0.0`.

## Backends

- **CUDA:** one dual-arch binary (sm_86 + sm_120); fp8-KV and e4m3 MMA
  need sm_89+, Ampere runs fp16-MMA verify and fp16/turbo3 KV. 24 GB
  cards use the `q27-server-w8` build (and see the Status regression
  note).
- **Metal:** developed and gated on base M4 (24 GB and 16 GB); everything
  targets plain `simdgroup_matrix`, with no Metal-4 cooperative-tensor
  dependence. The official tier
  wants a 24 GB machine; T2 is fully resident on 16 GB. One model load
  at a time is a hard memory-policy rule on 24 GB machines.

## Test vectors and gates

The canonical prompt is token IDs `760,6511,314,9338,369` ("The capital
of France is"); its 16-token continuation is byte-exact across CUDA and
Metal (SHA `6c1d4328…`), and the 128-token CUDA output line has a pinned
md5. `make test-cpu test-metal` is the default gate (synthetic,
memory-safe, no artifact); artifact-scale gate scripts live in `tools/`
and write their verdicts to `logs/`.

House rules that shaped the codebase, in case you wonder why the logs
look the way they do: experiments are pre-registered with kill lines
before they run; gates must be demonstrated *able to fail* before a pass
counts (we have been burned); binaries are rebuilt before every gate run
(stale-binary rule); committed-token changes require byte-identity or a
measured, documented tolerance class; and every code change gets an
adversarial review round before it counts as done.

## Debugging notes

Per-subsystem trace envs: `Q27_METAL_PROFILE=1` (per-kernel GPU time
attribution), `Q27_MTP_TRACE`, `Q27_ORACLE_TRACE`, `Q27_SUFFIX_TRACE`
(per-round speculation anatomy), `Q27_BATCH=1` (CUDA continuous batching,
experimental). Synthetic benches (`metal_decode_bench`,
`metal_prefill_bench`, `metal_gemv_bench`, `metal_attn_bench`) reproduce
every kernel-level number without loading the artifact. If a run
misbehaves, the first three suspects are a stale binary, a page-thrashed
mmap (two model loads), and desktop contention on a timing leg — in that
order, from experience.

## About the development process

This engine is developed with strong AI assistance — implementation and
review are largely done by LLM agents (with independent adversarial
review on every change), with humans leading the ideas, the priorities,
the measurement discipline, and every authorization to touch a GPU. We
say this openly because it shaped how the project was built: the
pre-registration/kill-line culture exists precisely so that no one, human
or model, gets to believe their own optimism.
