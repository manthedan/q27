#!/usr/bin/env bash
# Assemble the prebuilt macOS arm64 release tarball for a metal-vX.Y.Z tag.
# Binaries must already be built with MACOSX_DEPLOYMENT_TARGET pinned.
# Usage: tools/assemble_release_tarball.sh <version> <staging-dir>
set -eu
cd "$(dirname "$0")/.."

VERSION=${1:?usage: assemble_release_tarball.sh <version> <staging-dir>}
STAGE=${2:?usage: assemble_release_tarball.sh <version> <staging-dir>}
ROOT="$STAGE/q27-$VERSION-macos-arm64"

rm -rf "$ROOT"
mkdir -p "$ROOT/bin" "$ROOT/share" "$ROOT/packaging/bin" "$ROOT/packaging/lib" \
         "$ROOT/share/q27-tools" "$ROOT/doc"

# Prebuilt binaries (C++ + Rust TUI). All must be present — a partial
# tarball is worse than none.
for b in q27-metal q27-metal-server q27-agent q27-tui tokenize_to_bin \
         metal_decode_bench metal_prefill_bench; do
    [ -x "build/$b" ] || { echo "missing build/$b" >&2; exit 1; }
    # No Homebrew/store rpaths allowed in a relocatable tarball.
    if otool -L "build/$b" | grep -qE "homebrew|/usr/local/opt"; then
        echo "build/$b links a Homebrew cellar path — rebuild clean" >&2
        exit 1
    fi
    cp "build/$b" "$ROOT/bin/$b"
done

# Runtime shader source (compiled at runtime; path baked via Q27_SHADER_PATH
# with a bin/../share relative fallback).
cp src/metal/q27_kernels.metal "$ROOT/share/q27_kernels.metal"

# Supervisor wrapper tree (resolves lib/models.tsv relative to itself).
cp packaging/bin/* "$ROOT/packaging/bin/"
cp packaging/lib/q27_bench_lib.sh "$ROOT/packaging/lib/"
cp packaging/models.tsv "$ROOT/packaging/models.tsv"
cp tools/repack.py tools/export_tokenizer.py "$ROOT/share/q27-tools/"

# User-facing docs.
cp README.md docs/GETTING-STARTED.md docs/MODELS.md docs/QA_BEFORE_RELEASES.md \
   docs/SECURITY-MODEL.md "$ROOT/doc/"
cp LICENSE "$ROOT/" 2>/dev/null || true

tar -C "$STAGE" -czf "$STAGE/q27-$VERSION-macos-arm64.tar.gz" \
    "q27-$VERSION-macos-arm64"
shasum -a 256 "$STAGE/q27-$VERSION-macos-arm64.tar.gz"
