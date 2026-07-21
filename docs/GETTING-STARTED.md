# Getting started with q27

The short path from zero to a running 27B model on Apple silicon. Everything
here goes through the `q27` command the Homebrew install gives you.

## 1. Install

```bash
brew install manthedan/tap/q27
```

This installs the Metal engine, the OpenAI/Anthropic-compatible server, the
`q27` wrapper, and the experimental `q27-agent`. **Model weights are not
included** — you pull a pack next.

Requires: macOS on Apple silicon (M1 or newer). Memory depends on the pack:
8 GB for **b1**, 16 GB for **t2**, 24 GB+ for **default**. `q27 recommend`
tells you what fits your machine.

## 2. Pull a model pack

`q27` picks the right pack for your RAM, or you name one:

```bash
q27 pull            # auto-pick by RAM
q27 pull b1         # or a specific pack
```

This downloads the pre-repacked artifact and its tokenizer, checksum-verified.
No Python or build step needed. See [MODELS.md](MODELS.md) for what each pack
is and the speed/context tables — but as a rule of thumb: **16 GB → t2**
(best quality that fits) or **b1** (smallest + fastest); **24 GB+ → default**

**Speed up downloads (optional).** Pulls use `curl` by default — a single
stream, ~10 MB/s on most connections. Two ways to do better:

- **Set `HF_TOKEN`** ([create one](https://huggingface.co/settings/tokens)).
  Anonymous downloads share a per-IP quota; a token lifts you to the
  authenticated tier and is required for gated repos. `q27 pull` picks it up
  automatically.
- **On a fast connection, opt into the `hf` CLI** for parallel downloads.
  Install it plus a parallel backend (`pip install "huggingface_hub[hf_xet]"`),
  then `Q27_USE_HF_CLI=1 q27 pull b1`. This only helps if your link is faster
  than ~10 MB/s and the repo is Xet-backed; on our current packs plain `curl`
  is already as fast, so leave it off unless you've measured a win.
(the official model with its MTP head).

## 3. Run it

**Serve it** (OpenAI/Anthropic-compatible endpoint on port 8080):

```bash
q27 serve
export ANTHROPIC_BASE_URL=http://localhost:8080 && claude
```

**Or try the native agent** (no server; links the engine directly). It turns
on local file/shell tools and uses your current directory as its workspace,
so `cd` into a scratch directory first:

```bash
mkdir -p /tmp/q27-play && cd /tmp/q27-play
q27 agent
```

Type a prompt, Enter; `:quit` exits. Try `Create hello.py that prints hello,
then run it`. See [MODELS.md](MODELS.md) §6 for sessions, tool limits, and the
safety model.

**Which pack does the agent use?** Unlike `q27 serve` (which auto-picks for
your RAM), `q27 agent` defaults to **b1** — so if you pulled a different pack,
name it: `q27 agent t2` (or `default`, `q8`, …). It errors with
`b1 is not installed` if you only pulled a bigger pack; that's your cue to
pass the pack you have. `q27 ls` shows what you've pulled.

**One model at a time.** The server and the agent each load the full model,
so only one can run at once — starting either while the other is up gives
`q27: a low-level q27 server or native agent is already running.` Stop the
first (`Ctrl-C` / `:quit`) before starting the other. This is a deliberate
memory-safety guard, not a bug.

## 3b. Sampling + MTP (optional)

Default decode is still **greedy** (temperature 0). For Qwen-card-style
sampling:

```bash
# Native agent: set temperature; top_p=0.95 and top_k=20 fill in if omitted
Q27_AGENT_TEMPERATURE=0.6 q27 agent default
# Agent MTP free-decode is opt-in (official packs only; engages only where
# the tool grammar cannot — fenced bodies / tools-off):
Q27_AGENT_TEMPERATURE=0.6 Q27_AGENT_MTP=4 q27 agent default

# Metal CLI (source checkout): greedy MTP vs sampled MTP
./build/q27-metal MODEL.q27 MODEL.tok --mtp 4 -n 64 --prompt "..."
./build/q27-metal MODEL.q27 MODEL.tok --mtp 4 \
  --temperature 0.7 --top-p 0.95 --top-k 20 --seed 1 -n 64 --prompt "..."
```

On an **official MTP pack** (`default` / `qwen36-27b-mtp`), `temp > 0` +
`--mtp` uses **sampled MTP** (greedy drafts, rejection-sample accept). Bonsai
packs have no MTP layer and fall back to plain serial sampling. Force plain
sample for A/B: `Q27_SAMPLE_PLAIN=1`.

Quiet-machine overnight suite (units + gate + t/s A/B + temp curve):

```bash
MODEL=models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok \
  tools/overnight_sampled_mtp.sh
```

Details: [metal/plans/2026-07-21-metal-sampled-mtp.md](metal/plans/2026-07-21-metal-sampled-mtp.md).

## 4. Useful next steps

```bash
q27 recommend       # the pack + context table for THIS machine
q27 ls              # installed packs + which is servable
q27 bench --fast    # quick benchmark of an installed pack
q27 report --full   # diagnostic bundle (helps validate the big tiers)
```

Switch packs any time: `q27 pull <name>`, then `q27 serve <name>`. Weights
are mmap'd, so a warm switch is seconds.

## Where to read more

- [MODELS.md](MODELS.md) — packs, quants, context windows, expected speeds.
- [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md) — the quality gates behind each tier.
- [SECURITY-MODEL.md](SECURITY-MODEL.md) — the agent's tool/workspace safety model.
- [metal/](metal/) — the Metal development records (if you want the *why*).
