# Packaging

## Homebrew (macOS, Apple silicon)

Direct install from a checkout or raw file:

    brew install --formula ./packaging/homebrew/q27.rb

Or from a tap carrying this formula as `Formula/q27.rb`:

    brew tap manthedan/q27 && brew install q27

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
