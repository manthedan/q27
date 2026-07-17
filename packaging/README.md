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

Distribution channel should be Hugging Face Hub (a 7.15 GB single file
exceeds GitHub Releases' 2 GB/file cap). Publish with: Apache-2.0
license + NOTICE preserving PrismML/Qwen attribution and stating the
modification ("repacked GGUF Q2_0 -> q27 T2_G128, codes byte-identical,
scales separated; no MTP head"), the source GGUF's checksum, the repack
command (tools/repack.py, quant_policy bonsai-t2-v1) and its bit-match
gate, measured quality numbers (NLL/PPL and the kl-kv calibration), and
the tokenizer (.tok) alongside. Treat published artifacts as IMMUTABLE:
prefix snapshots (Q27SNAP1) pin the artifact byte-exactly, so any
re-upload with different bytes silently invalidates user snapshot
directories — version a changed artifact instead of replacing it.
