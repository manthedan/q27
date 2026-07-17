# Packaging

## Homebrew (macOS, Apple silicon)

Direct install from a checkout or raw file:

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
