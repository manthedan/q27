#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HELPER="$ROOT/tools/stage_homebrew_formula.sh"
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk 'NR==1 {print $1}'
  else
    shasum -a 256 "$1" | awk 'NR==1 {print $1}'
  fi
}
tmp="$(mktemp -d "${TMPDIR:-/tmp}/q27-stage-formula.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/asset/root"
printf 'fixture\n' >"$tmp/asset/root/file"
tar -C "$tmp/asset" -czf "$tmp/q27-test.tar.gz" root
source_sha="$(sha256_file "$ROOT/packaging/homebrew/q27.rb")"
asset_sha="$(sha256_file "$tmp/q27-test.tar.gz")"
asset_path="$(cd "$tmp" && pwd -P)/q27-test.tar.gz"

"$HELPER" "$ROOT/packaging/homebrew/q27.rb" "$tmp/q27-test.tar.gz" \
  "$tmp/q27.rb" 9.8.7 >"$tmp/helper.out"
grep -Fqx "  url \"file://$asset_path\"" "$tmp/q27.rb"
grep -Fqx '  version "9.8.7"' "$tmp/q27.rb"
grep -Fqx "  sha256 \"$asset_sha\"" "$tmp/q27.rb"
[[ "$(sha256_file "$ROOT/packaging/homebrew/q27.rb")" = "$source_sha" ]]
[[ "$(stat -f '%Lp' "$tmp/q27.rb" 2>/dev/null || stat -c '%a' "$tmp/q27.rb")" = 600 ]]

if "$HELPER" "$ROOT/packaging/homebrew/q27.rb" "$tmp/q27-test.tar.gz" \
     "$tmp/q27.rb" 9.8.7 >"$tmp/overwrite.out" 2>&1; then
  echo "staging helper overwrote an existing formula" >&2
  exit 1
fi
grep -q 'refusing to overwrite' "$tmp/overwrite.out"

awk '/^[[:space:]]*sha256 / { print; print; next } { print }' \
  "$ROOT/packaging/homebrew/q27.rb" >"$tmp/duplicate.rb"
if "$HELPER" "$tmp/duplicate.rb" "$tmp/q27-test.tar.gz" \
     "$tmp/rejected.rb" 9.8.7 >"$tmp/rejected.out" 2>&1; then
  echo "staging helper accepted duplicate formula assignments" >&2
  exit 1
fi
[[ ! -e "$tmp/rejected.rb" ]]
grep -q 'exactly one url, version, and sha256' "$tmp/rejected.out"

ln -s "$tmp/q27-test.tar.gz" "$tmp/asset-link.tar.gz"
if "$HELPER" "$ROOT/packaging/homebrew/q27.rb" "$tmp/asset-link.tar.gz" \
     "$tmp/link.rb" 9.8.7 >"$tmp/link.out" 2>&1; then
  echo "staging helper accepted a symlink asset" >&2
  exit 1
fi
[[ ! -e "$tmp/link.rb" ]]

interpolation="$tmp/q27-#{system('false')}.tar.gz"
backslash="$tmp/q27-back\\slash.tar.gz"
space="$tmp/q27 space.tar.gz"
fragment="$tmp/q27#fragment.tar.gz"
percent="$tmp/q27%20escape.tar.gz"
for unsafe in "$interpolation" "$backslash" "$space" "$fragment" "$percent"; do
  cp "$tmp/q27-test.tar.gz" "$unsafe"
  if "$HELPER" "$ROOT/packaging/homebrew/q27.rb" "$unsafe" \
       "$tmp/unsafe.rb" 9.8.7 >"$tmp/unsafe.out" 2>&1; then
    echo "staging helper embedded an unsafe Ruby asset path" >&2
    exit 1
  fi
  [[ ! -e "$tmp/unsafe.rb" ]]
  grep -q 'unsupported local asset path' "$tmp/unsafe.out"
done

echo "staging Homebrew formula helper: PASS"
