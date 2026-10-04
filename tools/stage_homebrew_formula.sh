#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 INPUT_FORMULA LOCAL_TARBALL OUTPUT_FORMULA VERSION" >&2
  exit 2
fi
input="$1"
asset="$2"
output="$3"
version="$4"

[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]] || {
  echo "invalid formula version: $version" >&2
  exit 2
}
for path in "$input" "$asset"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "required regular non-symlink file missing: $path" >&2
    exit 1
  }
done
[[ ! -e "$output" && ! -L "$output" ]] || {
  echo "refusing to overwrite staging formula: $output" >&2
  exit 1
}
parent="$(cd "$(dirname "$output")" && pwd -P)"
[[ -d "$parent" && ! -L "$parent" ]] || {
  echo "staging formula parent must be a real directory: $parent" >&2
  exit 1
}
asset_dir="$(cd "$(dirname "$asset")" && pwd -P)"
asset="$asset_dir/$(basename "$asset")"
# Keep the file URI byte-for-byte valid without depending on an URL encoder.
# The release staging location is under /private/tmp and needs only RFC 3986
# unreserved path bytes plus '/'. Fail closed for spaces, fragments, percent
# escapes, Ruby interpolation, and every other ambiguous character.
case "$asset" in
  *[!A-Za-z0-9._~/-]*)
    echo "unsupported local asset path: $asset" >&2; exit 1 ;;
esac
tar -tzf "$asset" >/dev/null
if command -v sha256sum >/dev/null 2>&1; then
  sha="$(sha256sum "$asset" | awk 'NR==1 {print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  sha="$(shasum -a 256 "$asset" | awk 'NR==1 {print $1}')"
else
  echo "sha256sum or shasum is required" >&2
  exit 1
fi
[[ "$sha" =~ ^[0-9a-f]{64}$ ]] || { echo "could not hash $asset" >&2; exit 1; }
url="file://$asset"

tmp="$(mktemp "$parent/.q27-formula.XXXXXX")"
cleanup() { rm -f "$tmp"; }
trap cleanup EXIT
chmod 600 "$tmp"
if ! awk -v url="$url" -v version="$version" -v sha="$sha" '
  BEGIN { urls=versions=shas=0 }
  /^[[:space:]]*url[[:space:]]+"/ {
    urls++; indent=$0; sub(/[^[:space:]].*$/, "", indent)
    print indent "url \"" url "\""; next
  }
  /^[[:space:]]*version[[:space:]]+"/ {
    versions++; indent=$0; sub(/[^[:space:]].*$/, "", indent)
    print indent "version \"" version "\""; next
  }
  /^[[:space:]]*sha256[[:space:]]+"/ {
    shas++; indent=$0; sub(/[^[:space:]].*$/, "", indent)
    print indent "sha256 \"" sha "\""; next
  }
  { print }
  END { if (urls != 1 || versions != 1 || shas != 1) exit 42 }
' "$input" >"$tmp"; then
  echo "formula must contain exactly one url, version, and sha256 assignment" >&2
  exit 1
fi

grep -Fqx "  url \"$url\"" "$tmp"
grep -Fqx "  version \"$version\"" "$tmp"
grep -Fqx "  sha256 \"$sha\"" "$tmp"
if command -v ruby >/dev/null 2>&1; then ruby -c "$tmp" >/dev/null; fi
mv "$tmp" "$output"
trap - EXIT
printf '%s  %s\n' "$sha" "$output"
