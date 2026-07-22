# Packaging

## Homebrew (macOS, Apple silicon)

The one-command install:

    brew install manthedan/tap/q27

This installs the engine binaries (`q27-metal`, `q27-metal-server`, the
synthetic benches) **and** the `q27` supervisor wrapper, which owns the
multi-GB weight lifecycle:

    q27 recommend      # table of packs that fit THIS machine + the pick
    q27 pull           # download + verify the recommended pack (~/.q27/models)
    q27 serve          # boot the server (refuses if one is already running)
    q27 ls             # what's installed
    q27 bench --fast   # quick synthetic ceiling + a real decode
    q27 report --full  # build a send-back diagnostic bundle (see below)

The current source checkout adds the native-agent supervisor planned for the
next tagged release. Friends testing it before that release should build from
source, place the B1 artifact/tokenizer under `models/` (or `q27 pull b1`),
and run:

    make agent
    # or one-time PATH shim (uses ~/.grok/bin when first on PATH):
    make install-dev-q27
    q27 agent b1

On a TTY, `q27 agent` prefers the Rust Ratatui client (`q27-tui` / FP1);
scripts and pipes fall back to the classic linenoise agent. Force either UI
with `Q27_AGENT_UI=tui` or `Q27_AGENT_UI=classic`.

In `q27-tui`, thinking is **collapsed by default** (one-line summary after the
turn). Press **`t`** on an empty prompt to expand/collapse the full `<think>`
trace. Live streaming still shows the open think span in full while it runs.
**Esc** / **Ctrl-C** cancel the active turn; **`/cancel`** is the same.

The underlying `./packaging/bin/q27 agent [pack]` defaults to context 32768,
adaptive generation limits, automatic tools, and the physical current
workspace. **Decode is pack-split:** Bonsai packs (`b1`/`t2`/…) soft-default
to sampling (T=0.6, top_p=0.95, top_k=20) for loop-break; official MTP packs
stay greedy unless you set temperature. Override with:

| Env | Default | Meaning |
|-----|---------|---------|
| `Q27_AGENT_PACK` | `b1` | Model pack name |
| `Q27_AGENT_CONTEXT` | `32768` | Context window |
| `Q27_AGENT_WORKSPACE` | cwd | Tool sandbox root |
| `Q27_AGENT_MAX_TOKENS` | `auto` | Whole-turn gen bound (thinking **+** answer); `auto` ≤ 16384 |
| `Q27_AGENT_MAX_THINK_TOKENS` | *unset* (**off**) | After N tokens in open `<think>`, force `</think>` and **continue the answer**; **not** a model feature |
| `Q27_AGENT_NO_THINK` | *unset* | Set non-zero to pass `--no-think` (empty think prefill; answer directly) |
| `Q27_AGENT_SESSION` | *unset* | Session snapshot path |
| `Q27_AGENT_UI` | `auto` | `tui` \| `classic` \| `auto` |
| `Q27_AGENT_TEMPERATURE` / `TOP_P` / `TOP_K` / `SEED` | pack-split | Sampling; `Q27_AGENT_TEMPERATURE=0` forces greedy on bonsai |
| `Q27_AGENT_MTP` | *unset* (**off**) | Opt-in MTP draft width (fenced-body bursts on official packs) |

When temperature > 0, the wrapper fills Qwen-card companions `top_p=0.95` and
`top_k=20` unless overridden — and `top_k=20` engages Metal's GPU top-k path
for sampled MTP on official packs when `Q27_AGENT_MTP` is set.

**Thinking budget note.** Qwen-style packs do not expose a native “thinking
budget.” Without `Q27_AGENT_MAX_THINK_TOKENS`, long reasoning is only bounded
by `Q27_AGENT_MAX_TOKENS` (or adaptive `auto`). To cap runaway think under
sampled decode:

```sh
Q27_AGENT_MAX_THINK_TOKENS=2048 Q27_AGENT_MAX_TOKENS=4096 q27 agent b1
# or disable thinking entirely:
Q27_AGENT_NO_THINK=1 q27 agent b1
```

When the think cap hits, the agent injects `</think>\n\n` (stream + KV) and
keeps decoding the answer within remaining `max_tokens`. It only hard-stops
mid-think if there is no room left for that close sequence.

The supervisor holds one private lifetime lock across `agent` and `serve` so
concurrent launches cannot double-load multi-GB weights. Automatic file tools
are workspace-bounded, but shell retains the local account's filesystem
authority.

To use the currently published package, point another coding-agent harness at
the server:

    export ANTHROPIC_BASE_URL=http://localhost:8080 && claude

Model artifacts (`.q27`) and tokenizers (`.tok`) are NOT in the formula —
they are multi-GB and carry their own licenses. `q27 pull` fetches and
checksum-verifies them per `packaging/models.tsv`. See
[docs/MODELS.md](../docs/MODELS.md) for the pack/context/speed tables.

### Developing the big tiers remotely

The maintainer's laptop tops out at 24 GB, so **q6 / q6k / q8 and the
long-context legs are built against user reports.** On a 28 GB+ machine:

    q27 pull q8
    q27 report --full     # writes q27-report-<ts>.tgz; email it back

The bundle has no weights and no private prompts — machine specs, thermal
anchors, per-tier tok/s, artifact md5s, and one fixed-prompt generation
sample. That is enough to diagnose and fix tiers we cannot run locally.

### Direct install (no tap)

    brew install --formula ./packaging/homebrew/q27.rb

Or from the published tap (Formula/q27.rb mirrors
packaging/homebrew/q27.rb; update both on release):

    brew tap manthedan/tap && brew install manthedan/tap/q27

Installs `q27-metal` (CLI), `q27-metal-server` (OpenAI/Anthropic/Responses
API server), and `q27-tokenize` (corpus tokenizer). The Metal shader is
installed to the formula's `pkgshare` and its path baked into the binaries
(`-DQ27_SHADER_PATH=...`), so they run from any directory; inside a source
checkout the checkout's shader still wins, and `Q27_METAL_SOURCE` overrides
both. Every candidate passes the same shader ABI-tag check.

Model artifacts (`.q27`) and tokenizers (`.tok`) are deliberately not
packaged — they are multi-GB and carry their own model licenses. See the
top-level README for repack instructions.

Release flow: tag (`vX.Y.Z`), push the tag, then update `url`/`sha256` in
`packaging/homebrew/q27.rb`:

    curl -sL https://github.com/manthedan/q27/archive/refs/tags/vX.Y.Z.tar.gz | shasum -a 256

## Weights (not packaged)

The T2 artifact is a bit-exact repack of PrismML's Ternary-Bonsai-27B
Q2_0 GGUF (QAT ternary of the Qwen3.6-27B base; Apache-licensed per the
whitepaper — VERIFY the HF model-card license field before publishing).
Candidate release artifact:

    ternary-bonsai-27b-t2.q27  7,154,196,992 bytes
    sha256 25392b471d2e5c55798c2ea77d8b53fbbd1f720318e5ff23bd8c49e5d378e549

**Decision (2026-07-16): q27 does not redistribute weights.** The repack
is a lossless, deterministic container transform of PrismML's published
pack, so users fetch from the source and repack locally:

    tools/fetch_weights.sh

The script pins the upstream revision (upstream verified live:
huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf, license apache-2.0,
ungated; commit 20e435f5), verifies the source GGUF against HF's LFS
sha256, repacks (CPU-only, ~1-3 min on Apple silicon, memory-bounded —
the byte-copy + ternary scan + bit-exact round-trip gate; the 7.2 GB
download dominates), verifies the artifact against the published q27
digest, and exports the tokenizer from the same GGUF. Deps: python3
with numpy + gguf (checked, with the pip command printed).

Own HF uploads are warranted only for artifacts that do NOT exist
upstream (e.g. a future binary-tier or KV-codec pack we produce) or if
upstream hosting disappears. If that day comes: Apache-2.0 + NOTICE
preserving PrismML/Qwen attribution, modification statement, source
checksum, repro command, measured quality numbers — and IMMUTABILITY:
prefix snapshots (Q27SNAP1) pin the artifact byte-exactly, so a
re-upload with different bytes silently invalidates user snapshot
directories; version a changed artifact, never replace it.
