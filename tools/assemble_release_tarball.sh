#!/usr/bin/env bash
# Assemble the prebuilt macOS arm64 release tarball for a metal-vX.Y.Z tag.
# Binaries must already be built with MACOSX_DEPLOYMENT_TARGET pinned.
# Usage: tools/assemble_release_tarball.sh <version> <staging-dir>
set -eu
cd "$(dirname "$0")/.."

VERSION=${1:?usage: assemble_release_tarball.sh <version> <staging-dir>}
STAGE=${2:?usage: assemble_release_tarball.sh <version> <staging-dir>}
ROOT="$STAGE/q27-$VERSION-macos-arm64"
# Must match the formula's `depends_on macos:` floor (ventura = 13.0).
MIN_MACOS=${Q27_MIN_MACOS:-13.0}

# A binary built without MACOSX_DEPLOYMENT_TARGET inherits the build
# machine's OS as its floor and would refuse to launch on older Macs.
check_minos() {
    local got
    got=$(otool -l "$1" | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')
    [ "$got" = "$MIN_MACOS" ] || {
        echo "$1 targets macOS ${got:-?}, want $MIN_MACOS — rebuild with MACOSX_DEPLOYMENT_TARGET=$MIN_MACOS" >&2
        exit 1
    }
}

rm -rf "$ROOT"
mkdir -p "$ROOT/bin" "$ROOT/share" "$ROOT/packaging/bin" "$ROOT/packaging/lib" \
         "$ROOT/share/q27-tools" "$ROOT/doc"

# Prebuilt binaries (C++ + Rust TUI). All must be present — a partial
# tarball is worse than none.
for b in q27-metal q27-metal-server q27-agent q27-tui tokenize_to_bin \
         metal_prefill_bench; do
    [ -x "build/$b" ] || { echo "missing build/$b" >&2; exit 1; }
    # No Homebrew/store rpaths allowed in a relocatable tarball.
    if otool -L "build/$b" | grep -qE "homebrew|/usr/local/opt"; then
        echo "build/$b links a Homebrew cellar path — rebuild clean" >&2
        exit 1
    fi
    check_minos "build/$b"
    cp "build/$b" "$ROOT/bin/$b"
done
[ -x build/q27-lock-exec ] || {
    echo "missing build/q27-lock-exec (the seventh required executable)" >&2
    exit 1
}
if otool -L build/q27-lock-exec | grep -qE "homebrew|/usr/local/opt"; then
    echo "build/q27-lock-exec links a Homebrew cellar path — rebuild clean" >&2
    exit 1
fi
check_minos build/q27-lock-exec
[ -x packaging/q27-release-launcher ] || {
    echo "missing packaging/q27-release-launcher" >&2
    exit 1
}
cp packaging/q27-release-launcher "$ROOT/bin/q27"
chmod 755 "$ROOT/bin/q27"

# Runtime shader source (compiled at runtime and discovered from
# bin/../share; release binaries must not embed a build-machine path).
cp src/metal/q27_kernels.metal "$ROOT/share/q27_kernels.metal"

# Supervisor wrapper tree (resolves lib/models.tsv relative to itself). The
# source checkout carries a build shim at packaging/bin/q27-lock-exec; replace
# it with the native executable so the packaged product has no Python runtime.
cp packaging/bin/* "$ROOT/packaging/bin/"
cp build/q27-lock-exec "$ROOT/packaging/bin/q27-lock-exec"
cp packaging/lib/q27_bench_lib.sh "$ROOT/packaging/lib/"
cp packaging/models.tsv "$ROOT/packaging/models.tsv"
# export_tokenizer.py imports its GGUF reader from prism_gguf.py.
cp tools/repack.py tools/export_tokenizer.py tools/prism_gguf.py "$ROOT/share/q27-tools/"

# User-facing docs.
cp README.md docs/QA_BEFORE_RELEASES.md docs/SECURITY-MODEL.md \
   docs/metal/BONSAI2.md experiments/ds4-agent/THIRD_PARTY_NOTICES.md "$ROOT/doc/"
cp packaging/README.md "$ROOT/doc/PACKAGING.md"
cp LICENSE "$ROOT/" 2>/dev/null || true

tar -C "$STAGE" -czf "$STAGE/q27-$VERSION-macos-arm64.tar.gz" \
    "q27-$VERSION-macos-arm64"
shasum -a 256 "$STAGE/q27-$VERSION-macos-arm64.tar.gz"
