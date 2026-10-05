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
    q27 bench --fast   # a real greedy decode on the first installed pack
    q27 report --full  # build a send-back diagnostic bundle (see below)

`q27 agent` runs the native coding agent on the default Bonsai 2 pack:

    q27 pull b2        # 6.7 GB, SHA-256 verified
    q27 agent          # interactive; tools are bounded to the current directory

On a TTY, `q27 agent` prefers the Rust Ratatui client (`q27-tui` / FP1);
scripts and pipes fall back to the classic linenoise agent. Force either UI
with `Q27_AGENT_UI=tui` or `Q27_AGENT_UI=classic`.

In `q27-tui`, thinking is **collapsed by default** (one-line summary after the
turn). **Ctrl-T** expands/collapses the full `<think>` trace; **Ctrl-Y** switches
the theme and **Ctrl-O** toggles markdown rendering. Live streaming still shows the open think span
in full while it runs. **Esc** / **Ctrl-C** cancel the active turn; **`/cancel`**
is the same.

`q27 agent [pack]` uses adaptive generation limits, automatic tools, and the
physical current workspace. **Decode is profile-driven:** `packaging/models.tsv`
assigns a stable `model_profile` independent of pack name and quant label.
Bonsai 2 (`bonsai2-qwen38-v1`) runs thinking-on with T=1.0, top_p=0.95,
top_k=20, a 16K context (the measured fp16-KV ceiling on a 16 GiB Mac), and
suffix-burst speculation (`--suffix 16`; exact for greedy and sampled decode).
Bonsai 1 profiles soft-default to T=0.6; the Qwen3.8 thinking profile uses
T=1.0/0.95/20; Qwen3.6 stays greedy unless you set temperature. Override with:

| Env | Default | Meaning |
|-----|---------|---------|
| `Q27_AGENT_PACK` | `b2` | Model pack name |
| `Q27_AGENT_CONTEXT` | `16384` (Bonsai 2), else `32768` | Context window |
| `Q27_AGENT_WORKSPACE` | cwd | Tool sandbox root |
| `Q27_AGENT_MAX_TOKENS` | `auto` | Whole-turn gen bound (thinking **+** answer); `auto` ≤ 16384 |
| `Q27_AGENT_MAX_THINK_TOKENS` | *unset* (**off**) | After N tokens in open `<think>`, force `</think>` and **continue the answer**; **not** a model feature |
| `Q27_AGENT_NO_THINK` | *unset* | Set non-zero to pass `--no-think` (empty think prefill; answer directly) |
| `Q27_AGENT_SESSION` | *unset* | Session snapshot path |
| `Q27_AGENT_UI` | `auto` | `tui` \| `classic` \| `auto` |
| `Q27_AGENT_TEMPERATURE` / `TOP_P` / `TOP_K` / `SEED` | model profile | Sampling; explicit values win and temperature 0 forces greedy |
| `Q27_AGENT_MTP` | *unset* (**off**) | Opt-in MTP draft width (fenced-body bursts on official packs); replaces suffix bursts when set |
| `Q27_SERVE_CONTEXT` / `Q27_SERVE_SLOTS` | `16384` / `1` (Bonsai 2) | Server window and slot count for Bonsai 2 packs; other profiles use the server's `--ctx auto` and its slot default |
| `Q27_SUFFIX` | `16` (Bonsai 2) | Suffix-burst speculation width for `q27 serve`/`q27 agent`; `0` disables; skipped when you pass `--suffix`, `--mtp`, or `--constrain-tools` |
| `Q27_SERVE_THINK` | profile default | `0` disables or `1` enables the server thinking default |
| `Q27_SERVE_TEMPERATURE` / `TOP_P` / `TOP_K` | model profile | Server defaults for requests that omit sampling fields; explicit request fields still win |

When temperature > 0, the wrapper fills Qwen-card companions `top_p=0.95` and
`top_k=20` unless overridden — and `top_k=20` engages Metal's GPU top-k path
for sampled MTP on official packs when `Q27_AGENT_MTP` is set.

**Thinking budget note.** Qwen-style packs do not expose a native “thinking
budget.” Without `Q27_AGENT_MAX_THINK_TOKENS`, long reasoning is only bounded
by `Q27_AGENT_MAX_TOKENS` (or adaptive `auto`). To cap runaway think under
sampled decode:

```sh
Q27_AGENT_MAX_THINK_TOKENS=2048 Q27_AGENT_MAX_TOKENS=4096 q27 agent
# or disable thinking entirely:
Q27_AGENT_NO_THINK=1 q27 agent
```

When the think cap hits, the agent injects `</think>\n\n` (stream + KV) and
keeps decoding the answer within remaining `max_tokens`. It only hard-stops
mid-think if there is no room left for that close sequence.

The supervisor holds one private lifetime lock across `agent` and `serve` so
concurrent launches cannot double-load multi-GB weights. Packaged releases use
the native `q27-lock-exec` helper (owner/mode/identity checks plus `openat` and
nonblocking `flock`); launching a model has no Python dependency. Automatic
file tools are workspace-bounded, but shell retains the local account's
filesystem authority.

To use the currently published package, point another coding-agent harness at
the server:

    export ANTHROPIC_BASE_URL=http://localhost:8080 && claude

Model artifacts (`.q27`) and tokenizers (`.tok`) are NOT in the formula —
they are multi-GB and carry their own licenses. `q27 pull` fetches and
checksum-verifies them per `packaging/models.tsv`; `q27 recommend` lists every
pack. metal-v0.7.0 validates the Bonsai 2 packs (`b2`, `b2-t3`); the other rows
are marked experimental and were last validated in metal-v0.6.1. See
[BONSAI2.md](../docs/metal/BONSAI2.md) for gates, speeds and context limits.

The Qwen3.8 pack (not re-validated in metal-v0.7.0) is `q38` (c-small): `q27 pull q38` resolves the
artifact and exact tokenizer at immutable HF commit
`fc7656476a9a7e83d58151f99620fe19b22b3688`, verifies both SHA-256 values,
and publishes neither file until the staged pair passes. Every serve/agent
launch re-verifies both products, and a stale neighboring `*.tok` cannot
shadow the manifest-selected tokenizer. `q38-q4s` remains the local-only differential control.
The shared `qwen38-thinking-v1` profile enables thinking, drives native-agent
sampling, and is reported by `/health`; startup fails if that profile is
paired with a non-Qwen3.8 artifact.

### Developing the big tiers remotely

The maintainer's laptop tops out at 24 GB, so **q6 / q6k / q8 and the
long-context legs are built against user reports.** On a 28 GB+ machine:

    q27 pull q8
    q27 report --full     # writes q27-report-<ts>.tgz; email it back

The bundle has no weights and no private prompts — machine specs, thermal
anchors, per-tier tok/s, artifact md5s, and one fixed-prompt generation
sample. That is enough to diagnose and fix tiers we cannot run locally.

### Direct install (no tap)

The prebuilt release archive is directly runnable after extraction; add its
`bin/` directory to `PATH` or invoke `bin/q27`. The launcher resolves the
archive's sibling binaries and packaged supervisor without absolute paths.

    ./q27-VERSION-macos-arm64/bin/q27 help

For a Homebrew-managed install:

    brew install --formula ./packaging/homebrew/q27.rb

Or from the published tap (Formula/q27.rb mirrors
packaging/homebrew/q27.rb; update both on release):

    brew tap manthedan/tap && brew install manthedan/tap/q27

Installs `q27-metal` (CLI), `q27-metal-server` (OpenAI/Anthropic/Responses
API server), and `q27-tokenize` (corpus tokenizer). The Metal shader is
installed to the formula's `pkgshare`; binaries resolve it relative to their
real executable path in both release-tarball (`bin/../share`) and Homebrew
(`bin/../share/q27`) layouts, so no build-machine path is retained. Source
checkouts keep the cwd fallback, `Q27_METAL_SOURCE` remains the first explicit
override, and every candidate passes the same shader ABI-tag check.

Model artifacts (`.q27`) and tokenizers (`.tok`) are deliberately not
packaged — they are multi-GB and carry their own model licenses. See the
top-level README for repack instructions.

Release flow: assemble the prebuilt archive
(`tools/assemble_release_tarball.sh metal-vX.Y.Z DIR`, binaries built with
`MACOSX_DEPLOYMENT_TARGET=13.0`), stage a candidate formula pinned to that exact file
(`tools/stage_homebrew_formula.sh packaging/homebrew/q27.rb ARCHIVE OUT X.Y.Z`
writes a local `file://` url, the version and the archive's sha256), smoke it
with `tools/homebrew_release_smoke.sh`, then tag, attach the same archive to the
GitHub release, and publish the formula with the release `url` and that same
sha256. The source-archive sha256 is not the formula's sha256.

## Weights (separate from the formula)

### Bonsai 2 (`b2`, `b2-t3`)

`b2` is a lossless repack of Prism ML's `Ternary-Bonsai-2-27B-PQ2_0.gguf`
(Apache 2.0; "Created using Bonsai by Prism ML") into the slim T2 container
Metal requires, hosted with its LICENSE, NOTICE and repro command at
`manthedan/q27-bonsai-packs` (commit `6d59ea13`):

    bonsai2-27b-t2-slim.q27  7,195,341,824 bytes
    sha256 b85094a0f53c68c25fc73c28e70e05462620417698c7b08a8977d475391f0aeb

`b2-t3` (`bonsai2-27b-t3-slim.q27`) and the shared tokenizer come from
upstream's `signalnine/Bonsai-2-27B-q27`. Both reproduce byte for byte with
`tools/repack.py --bonsai2-container t2|t3 --slim` (docs/metal/BONSAI2.md).

### Bonsai 1 T2

The T2 artifact is a bit-exact repack of PrismML's Ternary-Bonsai-27B
Q2_0 GGUF (QAT ternary of the Qwen3.6-27B base; Apache-licensed per the
whitepaper — VERIFY the HF model-card license field before publishing).
Candidate release artifact:

    ternary-bonsai-27b-t2.q27  7,154,196,992 bytes
    sha256 25392b471d2e5c55798c2ea77d8b53fbbd1f720318e5ff23bd8c49e5d378e549

The 2026-07-16 blanket “q27 does not redistribute weights” note is superseded
for explicitly registered direct packs. A direct derivative may be hosted
only after license/attribution review, pinned source and publication commits,
reproducible digests, and immutable artifact naming. The formula itself still
contains no weights. T2 can also be reproduced directly from its source with:

    tools/fetch_weights.sh

The script pins the upstream revision (upstream verified live:
huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf, license apache-2.0,
ungated; commit 20e435f5), verifies the source GGUF against HF's LFS
sha256, repacks (CPU-only, ~1-3 min on Apple silicon, memory-bounded —
the byte-copy + ternary scan + bit-exact round-trip gate; the 7.2 GB
download dominates), verifies the artifact against the published q27
digest, and exports the tokenizer from the same GGUF. Deps: python3
with numpy + gguf (checked, with the pip command printed).

Hosted derivatives require their upstream license and NOTICE, a modification
statement, source checksums, repro command, and measured quality evidence.
Immutability is load-bearing: prefix snapshots pin artifact bytes, while the
registry pins the HF commit and both product digests. Version changed bytes
under a new filename/commit; never replace a published artifact in place.
