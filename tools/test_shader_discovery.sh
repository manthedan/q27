#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${Q27_SHADER_DISCOVERY_BIN:-$ROOT/build/test_metal}"
SHADER="$ROOT/src/metal/q27_kernels.metal"

if [[ "$(uname -s)" != Darwin ]]; then
  echo "shader discovery test requires macOS" >&2
  exit 1
fi
[[ -x "$BIN" ]] || { echo "missing $BIN; run make build/test_metal" >&2; exit 2; }
[[ -f "$SHADER" ]] || { echo "missing $SHADER" >&2; exit 2; }

tmp="$(mktemp -d "${TMPDIR:-/tmp}/q27-shader-discovery.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/cwd" "$tmp/tar/bin" "$tmp/tar/share" \
         "$tmp/cellar/bin" "$tmp/cellar/share/q27" "$tmp/prefix/bin"
cp "$BIN" "$tmp/tar/bin/test_metal"
cp "$BIN" "$tmp/cellar/bin/test_metal"
cp "$SHADER" "$tmp/tar/share/q27_kernels.metal"
cp "$SHADER" "$tmp/cellar/share/q27/q27_kernels.metal"
ln -s ../../cellar/bin/test_metal "$tmp/prefix/bin/test_metal"

run_staged() {
  local label="$1" binary="$2" output="$tmp/$1.out"
  (
    cd "$tmp/cwd"
    unset Q27_METAL_SOURCE
    "$binary"
  ) >"$output" 2>&1
  grep -q 'Metal matvec: OK' "$output" || {
    echo "$label layout did not execute the Metal probe" >&2
    cat "$output" >&2
    exit 1
  }
}

run_staged tarball "$tmp/tar/bin/test_metal"
run_staged cellar "$tmp/cellar/bin/test_metal"
run_staged cellar-symlink "$tmp/prefix/bin/test_metal"

# The explicit override remains first even when a packaged relative candidate
# exists. Corrupt the staged copy; the valid override must still launch.
printf '%s\n' '// Q27_SHADER_ABI 0' >"$tmp/tar/share/q27_kernels.metal"
(
  cd "$tmp/cwd"
  Q27_METAL_SOURCE="$SHADER" "$tmp/tar/bin/test_metal"
) >"$tmp/override.out" 2>&1
grep -q 'Metal matvec: OK' "$tmp/override.out"

# Without the override, an existing wrong-ABI relative shader fails closed
# rather than falling through to a source checkout in the caller's cwd.
if (
  cd "$tmp/cwd"
  unset Q27_METAL_SOURCE
  "$tmp/tar/bin/test_metal"
) >"$tmp/abi.out" 2>&1; then
  echo "wrong-ABI executable-relative shader was accepted" >&2
  exit 1
fi
grep -q 'shader ABI mismatch' "$tmp/abi.out"

echo "executable-relative Metal shader discovery: PASS"
