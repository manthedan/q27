#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/q27-fetch-manifest.XXXXXX")"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT
mkdir -p "$tmp/pkg/bin" "$tmp/pkg/lib" "$tmp/fake-bin" "$tmp/remote" "$tmp/home"
cp "$root/packaging/bin/q27-fetch" "$tmp/pkg/bin/"
cp "$root/packaging/lib/q27_bench_lib.sh" "$tmp/pkg/lib/"
chmod +x "$tmp/pkg/bin/q27-fetch"

printf 'artifact-A\n' >"$tmp/remote/artifact-a.q27"
printf 'Q27Ttokenizer-A\n' >"$tmp/remote/token-a.tok"
printf 'artifact-B\n' >"$tmp/remote/artifact-b.q27"
printf 'Q27Ttokenizer-B\n' >"$tmp/remote/token-b.tok"
sha() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'
  fi
}
md5f() {
  if command -v md5 >/dev/null 2>&1; then md5 -q "$1"
  else md5sum "$1" | awk '{print $1}'
  fi
}
rev_a=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
rev_b=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
art_a_sha="$(sha "$tmp/remote/artifact-a.q27")"
tok_a_sha="$(sha "$tmp/remote/token-a.tok")"
art_b_sha="$(sha "$tmp/remote/artifact-b.q27")"
tok_b_sha="$(sha "$tmp/remote/token-b.tok")"
art_a_md5="$(md5f "$tmp/remote/artifact-a.q27")"
art_b_md5="$(md5f "$tmp/remote/artifact-b.q27")"
zeros=0000000000000000000000000000000000000000000000000000000000000000

cat >"$tmp/pkg/models.tsv" <<EOF
name	display	quant_policy	artifact_file	artifact_md5	weights_gb	min_ram_gb	hf_repo	source_kind	tokenizer	experimental	notes	model_profile	artifact_sha256	hf_revision	tokenizer_file	tokenizer_sha256
good	Good	fixture	artifact-a.q27	$art_a_md5	0	1	example/repo	direct	direct	no	fixture	qwen36-v1	$art_a_sha	$rev_a	token-a.tok	$tok_a_sha
complete	Complete staged pair	fixture	artifact-a.q27	$art_a_md5	0	1	example/repo	direct	direct	no	fixture	qwen36-v1	$art_a_sha	$rev_a	token-a.tok	$tok_a_sha
moved	Moved	fixture	artifact-b.q27	$art_b_md5	0	1	example/repo	direct	direct	no	fixture	qwen36-v1	$art_b_sha	$rev_b	token-b.tok	$tok_b_sha
bad-art	Bad artifact	fixture	artifact-a.q27	$art_a_md5	0	1	example/repo	direct	direct	no	fixture	qwen36-v1	$zeros	$rev_a	token-a.tok	$tok_a_sha
bad-tok	Bad tokenizer	fixture	artifact-a.q27	$art_a_md5	0	1	example/repo	direct	direct	no	fixture	qwen36-v1	$art_a_sha	$rev_a	token-a.tok	$zeros
legacy	Legacy fixture	fixture	artifact-a.q27	$art_a_md5	0	1	local-only	local	direct	no	fixture	qwen36-v1
partial	Partial manifest	fixture	artifact-a.q27	$art_a_md5	0	1	local-only	local	direct	no	fixture	qwen36-v1			token-a.tok
EOF

cat >"$tmp/fake-bin/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${MOCK_REMOTE:?}" "${MOCK_CURL_LOG:?}"
dest= url= resume=0
while (($#)); do
  case "$1" in
    -o) dest="$2"; shift 2 ;;
    --continue-at) resume=1; shift 2 ;;
    http://*|https://*) url="$1"; shift ;;
    *) shift ;;
  esac
done
[[ -n "$dest" && -n "$url" ]] || exit 2
printf '%s\n' "$url" >>"$MOCK_CURL_LOG"
case "$url" in
  */resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/artifact-a.q27) src=artifact-a.q27 ;;
  */resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/token-a.tok) src=token-a.tok ;;
  */resolve/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/artifact-b.q27) src=artifact-b.q27 ;;
  */resolve/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/token-b.tok) src=token-b.tok ;;
  *) exit 22 ;;
esac
offset=0
if (( resume )) && [[ -f "$dest" ]]; then offset=$(wc -c <"$dest" | tr -d ' '); fi
size=$(wc -c <"$MOCK_REMOTE/$src" | tr -d ' ')
printf 'offset=%s\n' "$offset" >>"$MOCK_CURL_LOG"
(( offset > size )) && exit 22   # range not satisfiable
if [[ -n "${MOCK_DROP_ONCE:-}" && -e "$MOCK_DROP_ONCE" ]]; then
  rm -f "$MOCK_DROP_ONCE"          # transport drop halfway: keep what arrived
  head -c $(( size / 2 )) "$MOCK_REMOTE/$src" >"$dest"
  exit 18
fi
tail -c +$(( offset + 1 )) "$MOCK_REMOTE/$src" >>"$dest"
EOF
chmod +x "$tmp/fake-bin/curl"
export PATH="$tmp/fake-bin:/usr/bin:/bin:/usr/sbin:/sbin"
export MOCK_REMOTE="$tmp/remote" MOCK_CURL_LOG="$tmp/curl.log" Q27_HOME="$tmp/home"

"$tmp/pkg/bin/q27-fetch" good >/dev/null
grep -Fq "/resolve/$rev_a/artifact-a.q27" "$tmp/curl.log"
grep -Fq "/resolve/$rev_a/token-a.tok" "$tmp/curl.log"
[[ "$(sha "$tmp/home/good/artifact-a.q27")" = "$art_a_sha" ]]
[[ "$(sha "$tmp/home/good/token-a.tok")" = "$tok_a_sha" ]]

# A prior process may die after completing both downloads but before digest
# validation/publication. Valid complete .part files publish without a range
# request (and therefore cannot strand on an HTTP 416 at EOF).
mkdir -p "$tmp/home/complete"
cp "$tmp/remote/artifact-a.q27" "$tmp/home/complete/artifact-a.q27.part"
cp "$tmp/remote/token-a.tok" "$tmp/home/complete/token-a.tok.part"
: >"$tmp/curl.log"
"$tmp/pkg/bin/q27-fetch" complete >/dev/null
[[ ! -s "$tmp/curl.log" ]]
[[ "$(sha "$tmp/home/complete/artifact-a.q27")" = "$art_a_sha" ]]
[[ "$(sha "$tmp/home/complete/token-a.tok")" = "$tok_a_sha" ]]

resolved_art="$(bash -c '. "$1"; q27_artifact_path good' _ "$tmp/pkg/lib/q27_bench_lib.sh")"
[[ "$resolved_art" = "$tmp/home/good/artifact-a.q27" ]]
printf 'corrupt-artifact\n' >"$tmp/home/good/artifact-a.q27"
if bash -c '. "$1"; q27_artifact_path good' _ "$tmp/pkg/lib/q27_bench_lib.sh" >/dev/null 2>&1; then
  echo "runtime resolver accepted a corrupt pinned artifact" >&2; exit 1
fi
cp "$tmp/remote/artifact-a.q27" "$tmp/home/good/artifact-a.q27"

# Legacy rows fall back to MD5 at runtime and still require tokenizer magic.
mkdir -p "$tmp/home/legacy"
cp "$tmp/remote/artifact-a.q27" "$tmp/home/legacy/artifact-a.q27"
printf 'bad-tokenizer\n' >"$tmp/home/legacy/legacy.tok"
bash -c '. "$1"; q27_artifact_path legacy' _ "$tmp/pkg/lib/q27_bench_lib.sh" >/dev/null
if bash -c '. "$1"; q27_tokenizer_path legacy' _ "$tmp/pkg/lib/q27_bench_lib.sh" >/dev/null 2>&1; then
  echo "legacy runtime resolver accepted bad tokenizer magic" >&2; exit 1
fi
printf 'Q27Tlegacy-tokenizer\n' >"$tmp/home/legacy/legacy.tok"
bash -c '. "$1"; q27_tokenizer_path legacy' _ "$tmp/pkg/lib/q27_bench_lib.sh" >/dev/null
mkdir -p "$tmp/home/partial"
printf 'Q27Tlooks-valid\n' >"$tmp/home/partial/token-a.tok"
if bash -c '. "$1"; q27_tokenizer_path partial' _ "$tmp/pkg/lib/q27_bench_lib.sh" >/dev/null 2>&1; then
  echo "exact tokenizer filename without digest was accepted" >&2; exit 1
fi
printf 'corrupt-legacy-artifact\n' >"$tmp/home/legacy/artifact-a.q27"
if bash -c '. "$1"; q27_artifact_path legacy' _ "$tmp/pkg/lib/q27_bench_lib.sh" >/dev/null 2>&1; then
  echo "legacy runtime resolver ignored registered artifact MD5" >&2; exit 1
fi

# A stale valid-looking tokenizer must never shadow the manifest filename, and
# corruption of the exact file must fail closed even while that stale file exists.
printf 'Q27Tstale-but-valid-looking\n' >"$tmp/home/good/stale.tok"
resolved="$(bash -c '. "$1"; q27_tokenizer_path good' _ "$tmp/pkg/lib/q27_bench_lib.sh")"
[[ "$resolved" = "$tmp/home/good/token-a.tok" ]]
printf 'Q27Tcorrupt\n' >"$tmp/home/good/token-a.tok"
printf 'stale-part-must-not-publish\n' >"$tmp/home/good/artifact-a.q27.part"
if bash -c '. "$1"; q27_tokenizer_path good' _ "$tmp/pkg/lib/q27_bench_lib.sh" >/dev/null 2>&1; then
  echo "corrupt exact tokenizer was accepted because a stale extra existed" >&2
  exit 1
fi
: >"$tmp/curl.log"
"$tmp/pkg/bin/q27-fetch" good >/dev/null
grep -Fq "/resolve/$rev_a/token-a.tok" "$tmp/curl.log"
! grep -Fq "/resolve/$rev_a/artifact-a.q27" "$tmp/curl.log"
[[ "$(sha "$tmp/home/good/artifact-a.q27")" = "$art_a_sha" ]]
[[ ! -e "$tmp/home/good/artifact-a.q27.part" ]]

# Revision is registry data, not hard-coded main: a second immutable revision
# resolves different bytes and filenames successfully.
: >"$tmp/curl.log"
"$tmp/pkg/bin/q27-fetch" moved >/dev/null
grep -Fq "/resolve/$rev_b/artifact-b.q27" "$tmp/curl.log"
grep -Fq "/resolve/$rev_b/token-b.tok" "$tmp/curl.log"
! grep -Fq '/resolve/main/' "$tmp/curl.log"

# A transport drop mid-download keeps the received bytes; the retry resumes
# from that offset instead of starting over.
rm -rf "$tmp/home/moved"
: >"$tmp/curl.log"
: >"$tmp/drop-once"
MOCK_DROP_ONCE="$tmp/drop-once" "$tmp/pkg/bin/q27-fetch" moved >/dev/null
grep -A1 -F "/resolve/$rev_b/artifact-b.q27" "$tmp/curl.log" | grep -q '^offset=[1-9]'
[[ "$(sha "$tmp/home/moved/artifact-b.q27")" = "$art_b_sha" ]]

rm -rf "$tmp/home/bad-art" "$tmp/home/bad-tok"
if "$tmp/pkg/bin/q27-fetch" bad-art >/dev/null 2>&1; then
  echo "bad artifact SHA-256 was accepted" >&2; exit 1
fi
[[ ! -e "$tmp/home/bad-art/artifact-a.q27" ]]
if "$tmp/pkg/bin/q27-fetch" bad-tok >/dev/null 2>&1; then
  echo "bad tokenizer SHA-256 was accepted" >&2; exit 1
fi
[[ ! -e "$tmp/home/bad-tok/artifact-a.q27" ]]
[[ ! -e "$tmp/home/bad-tok/token-a.tok" ]]

echo "pinned direct artifact/tokenizer manifest: PASS"
