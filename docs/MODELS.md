# Choosing a q27 pack: models, quants, and context

This guide answers three questions for an Apple-silicon user:

1. **Which pack (quant) should I install?**
2. **What context window fits on my machine?**
3. **What speed should I expect?**

The `q27` wrapper automates the first two (`q27 recommend` reads your RAM
and prints this table for your machine). This document is the reference
behind it, plus the experimental packs the wrapper will not auto-recommend.

A note on honesty: the **B1 / T2 / M1 packs are a different model**
(Bonsai-27B, a QAT distillation of Qwen3.6-27B), not a quantization of the
official checkpoint. "Smaller" there trades model, not just bits. The
**default / q4s / q6 / q6k / q8** packs are the official Qwen3.6-27B-MTP
with its trained MTP head.

---

## 1. The packs

| pack | model | bpw | size | min RAM | route | status |
|---|---|---|---|---|---|---|
| **b1** | Bonsai-27B binary | 1.125 | 3.5 GiB | 8 GB | float GEMV | supported |
| **t2** | Bonsai-27B ternary | 2.25 | 6.7 GiB | 16 GB | float GEMV | supported |
| **m1** | Bonsai-27B mixed B1/T2 graft | ~1.13 | 3.6 GiB | 8 GB | float GEMV | **experimental** |
| **default** | Qwen3.6-27B-MTP | 5.25 | 16.5 GiB | 24 GB | int8 GEMV + MTP | supported |
| **q4s** | Qwen3.6-27B-MTP | ~4.6 | 14.4 GiB | 20 GB | int8 GEMV + MTP | supported |
| **q6** | Qwen3.6-27B-MTP | 6.0 | 19.1 GiB | 28 GB | int8 GEMV + MTP | supported |
| **q6k** | Qwen3.6-27B-MTP | 6.8 | 21.7 GiB | 32 GB | int8 GEMV + MTP | supported |
| **q8** | Qwen3.6-27B-MTP | 8.1 | 26.5 GiB | 36 GB | int8 GEMV + MTP | **experimental / UNVALIDATED** |

- **b1** is the smallest fully-resident 27B and the fastest decode on a
  16 GB machine. It passed the full quality/probe battery.
- **t2** is the quality/speed sweet spot on 16 GB. Task quality holds;
  ~2.1–2.3× official PPL.
- **m1** is a byte-level graft of the B1/T2 checkpoints. **The mixed-graft
  thesis is permanently closed** (the gdn_pair pack was exonerated, then
  A5-on-gdn_pair KILLED the strongest arm on both corpora — a B1→T2 bridge
  needs training, not grafting). We keep the pack for users who want to
  experiment; it is not a supported tier. Install with `q27 pull m1` after
  building it locally (`tools/q27_mix.py`).
- **default** is the max-speed official tier with the trained MTP head and
  the only official tier that fits 24 GB (borderline — wants an idle
  machine). **q4s** is the more comfortable 20 GB choice.
- **q8** is the high-fidelity tier. **It is UNVALIDATED upstream pending
  48GB-class hardware** — which is exactly why we need your report. If you
  have a 36–48 GB+ machine, `q27 pull q8 && q27 report --full` and send us
  the bundle; that is how this tier gets validated.

## 2. Expected decode speed by chip (planning numbers)

Decode is weight-read-bound, so tokens/sec scales with memory bandwidth and
shrinks with bytes-per-token. These are **greedy** single-stream estimates
from our memory-wall model, calibrated on measured base-M4 results (B1
measures ~19 tok/s, T2 ~11.7 tok/s). Treat as ±15%.

| pack | bytes/token | M4 (120 GB/s) | M4 Pro (273 GB/s) | M4 Max (546 GB/s) |
|---|---|---|---|---|
| **b1** | 3.4 GiB | ~20 | ~47 | ~93 |
| **t2** | 6.7 GiB | ~12 | ~26 | ~53 |
| **default** | 16.5 GiB | ~5.5* | ~12* | ~25* |
| **q8** | 26.5 GiB | ~3.5* | ~8* | ~16* |

\* Official tiers carry the trained MTP head, so **real** decode is higher
than the greedy figure (MTP commits multiple tokens per weight read; ~12.5
tok/s measured for default on a base M4 with MTP). The greedy column is the
floor.

Other M-chips, by memory bandwidth (decode scales ~linearly):

| chip | bandwidth | relative to M4 |
|---|---|---|
| M4 | 120 GB/s | 1.0× |
| M4 Pro | 273 GB/s | 2.3× |
| M4 Max | 546 GB/s | 4.6× |
| M3 Ultra | ~800 GB/s | ~6.7× |

(M3 Ultra figure is a published-spec estimate; we have not measured one.
If you have one, `q27 report --full` tells us the real number.)

Prefill is compute-bound, not bandwidth-bound, and does not scale with
bandwidth the same way; it is roughly flat across packs at a given chip
(official tiers measure ~47 tok/s synthetic width-96 on a base M4).

## 3. Context window vs memory

KV cache grows with context. fp16 KV is the default; `--kv turbo3` (a
50-byte transformed codec) cuts it ~5×. This table is the **attention KV
only** (the GDN recurrent state is small and fixed); add it to the pack's
weight size, then keep the total under ~60% of your RAM so macOS, your
browser, and the page cache have room.

| context | fp16 KV (no MTP) | fp16 KV (official, +MTP) | turbo3 KV |
|---|---|---|---|
| 8 K | 0.5 GiB | 0.6 GiB | 0.1 GiB |
| 16 K | 1.1 GiB | 1.1 GiB | 0.2 GiB |
| 32 K | 2.1 GiB | 2.3 GiB | 0.4 GiB |
| 65 K | 4.3 GiB | 4.6 GiB | 0.8 GiB |
| 131 K | 8.6 GiB | 9.1 GiB | 1.7 GiB |
| 262 K | 17.2 GiB | 18.3 GiB | 3.4 GiB |

**Rule of thumb:** `weights + KV(ctx) ≤ 0.6 × RAM`.

Worked examples:

- **16 GB Mac, t2 (6.7 GiB):** budget ≈ 9.6 GB. 6.7 + fp16 KV to 32 K
  (2.1) = 8.8 GB ✓. With turbo3 you can reach 131 K (1.7) = 8.4 GB ✓.
- **16 GB Mac, b1 (3.5 GiB):** budget ≈ 9.6 GB. fp16 KV to 65 K (4.3) =
  7.8 GB ✓; turbo3 to 262 K (3.4) = 6.9 GB ✓.
- **24 GB Mac, default (16.5 GiB):** budget ≈ 14.4 GB. fp16 KV only fits
  ~8 K (0.6) = 17.1 GB — too tight with the OS running. **Use turbo3**:
  131 K (1.7) = 18.2 GB, workable on an idle machine. This is why default
  on 24 GB is "borderline."
- **48 GB Mac, q8 (26.5 GiB):** budget ≈ 28.8 GB. fp16 KV to 16 K (1.1) =
  27.6 GB ✓; turbo3 to 131 K (1.7) = 28.2 GB ✓.

## 4. The recommendation (what `q27 recommend` does)

| your RAM | recommended pack | context | why |
|---|---|---|---|
| 16 GB | **t2** | 32 K fp16 / 131 K turbo3 | best quality that fits comfortably |
| 16 GB (fastest) | **b1** | 65 K fp16 / 262 K turbo3 | smallest + fastest, still a real 27B |
| 20–24 GB | **q4s** or **default** | turbo3 to 131 K | official model + MTP; default only if idle |
| 28–32 GB | **q6** / **q6k** | fp16 to 32 K, turbo3 far | higher official fidelity |
| 36–48 GB+ | **q8** | fp16 to 16 K, turbo3 to 131 K | max fidelity — **please send a report** |

Install and serve:

```bash
brew install manthedan/tap/q27
q27 pull            # auto-picks the recommended pack for your RAM
q27 serve           # boots the server on it (refuses if one is running)
export ANTHROPIC_BASE_URL=http://localhost:8080 && claude
```

Switch packs any time: `q27 pull <name>`, stop the server, `q27 serve
<name>`. Weights are mmap'd, so a warm-cache switch is seconds.

**Faster downloads.** `q27 pull` downloads with `curl` (single stream, ~10
MB/s on most links) and resumes if interrupted. To go faster: set
[`HF_TOKEN`](https://huggingface.co/settings/tokens) to move off the
anonymous per-IP quota (required for gated repos); and, only if your
connection is much faster than ~10 MB/s, install the `hf` CLI with a parallel
backend (`pip install "huggingface_hub[hf_xet]"`) and pull with
`Q27_USE_HF_CLI=1 q27 pull <name>`. The CLI opt-in is off by default because
on plain-LFS repos (our current packs) `curl` is already as fast.

## 5. Help us validate the big tiers

The maintainer's laptop tops out at 24 GB, so **q6 / q6k / q8 and the
long-context legs are developed against user reports.** If you have a
beefier machine:

```bash
q27 pull q8            # or q6 / q6k
q27 report --full      # benches every installed pack + captures machine,
                       # thermal, per-tier tok/s, and a generation sample
```

This writes `q27-report-<timestamp>.tgz` in your current directory. It
contains **no model weights and no private prompts** — just machine specs,
thermal anchors, per-tier tok/s, artifact md5s, and one fixed-prompt
generation sample. Email it back and we can diagnose and fix the tiers we
cannot run locally.

## 6. Try the native agent (experimental)

The formula also installs **`q27-agent`**, a native C/C++ agent that links the
engine directly — no server, no HTTP, no Anthropic/OpenAI shim. It owns the
transcript, durable sessions, auto-compaction, and a set of local tools
(read / search / write / edit / shell). **It is a Phase-0 experiment, not a
finished product**: the terminal UX is a plain read-eval loop (a real TUI is on
the roadmap), and it is the thing we most want feedback on.

It uses whatever pack you already pulled — the `q27 agent` wrapper resolves
the model and tokenizer for you (default pack **b1**, or pass another):

```bash
q27 agent          # b1
q27 agent t2       # or default / any installed pack
```

**Read this before you run it.** `q27 agent` turns the model's local tools on
automatically and uses your **current directory** as the agent's workspace.
The file tools stay inside that directory, but the shell tool runs with your
local account's normal filesystem access — so **`cd` into a scratch directory
first, not a real project:**

```bash
mkdir -p /tmp/q27-play && cd /tmp/q27-play
q27 agent
```

Type a prompt, Enter; `:quit` exits. Try `Create hello.py that prints hello,
then run it`.

Two things to know. First, the default is **b1** specifically (not "whatever
fits your RAM" like `q27 serve`), so if you pulled only a bigger pack, pass
its name — `q27 agent t2` — or you'll get `b1 is not installed`. Second,
**the agent and the server can't run at the same time**: each loads the full
model, and the wrapper refuses to start one while the other is up
(`... already running`). Stop the first before starting the other; that guard
is deliberate memory safety, not a bug.

Sessions are opt-in and durable (autosave + resume). Point the wrapper at a
session file and it reuses it across runs:

```bash
cd /tmp/q27-play
Q27_AGENT_SESSION=work.q27agent q27 agent b1
```

A one-shot answer with no tools and no session (bypass the wrapper, straight
to the binary):

```bash
q27-agent ~/.q27/models/b1/*.q27 ~/.q27/models/b1/*.tok \
  --prompt 'Reply with exactly: ready' --no-think
```

Useful env knobs for the wrapper: `Q27_AGENT_PACK` (default pack, instead of
the `[name]` argument), `Q27_AGENT_WORKSPACE` (override the workspace without
`cd`), `Q27_AGENT_CONTEXT` (default 32768), `Q27_AGENT_MAX_TOKENS` (default
`auto`). Inside the agent,
`--max-tool-rounds N` (default 8) bounds how many tool calls one turn can
chain, and `--compact-at`/`--compact-keep` tune auto-compaction.

Recommended tiers: **b1** or **t2** on 16 GB (fast, and the raw-payload file
path is most exercised there); any official tier on bigger machines.

**What to report:** crashes or hangs, a session that fails to resume, the
model writing outside the workspace (the current dir, or `Q27_AGENT_WORKSPACE`),
a tool call looping past
`--max-tool-rounds`, or anything where the agent's answer diverges from the
same prompt through `q27 serve`. The exact command line and the session file
are usually enough to reproduce it. Note the agent is CLI-only and not part of
`q27 report` yet — for now just describe what you saw and include the command.
