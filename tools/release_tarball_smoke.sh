#!/usr/bin/env bash
# Fresh-HOME smoke of an assembled release tarball, without Homebrew:
# extract to a temp dir, then pull/recommend/serve/agent through bin/q27 under
# a scrubbed environment (no source tree on PATH, no Q27_BIN_DIR).
# Usage: tools/release_tarball_smoke.sh TARBALL PACK [SEED_DIR]
# SEED_DIR (optional) holds an already-downloaded PACK's files; they are
# hard-linked in and must still pass `q27 pull` verification.
set -euo pipefail

[[ $# -eq 2 || $# -eq 3 ]] || {
  echo "usage: $0 TARBALL PACK [SEED_DIR]" >&2; exit 2; }
tarball="$1"
pack="$2"
seed="${3:-}"
[[ -f "$tarball" ]] || { echo "tarball not found: $tarball" >&2; exit 1; }
root="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"
# The smoke later runs from its temp dir: pin a relative interpreter path now.
case "$PYTHON" in */*) ;; *) PYTHON="$(command -v "$PYTHON")" ;; esac
case "$PYTHON" in
  /*) ;;
  *) PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")" ;;
esac
port=8080
if ! "$PYTHON" - "$port" <<'PY'
import socket, sys
with socket.socket() as probe:
    probe.bind(("127.0.0.1", int(sys.argv[1])))
PY
then
  echo "refusing to use occupied smoke port $port" >&2; exit 1
fi

smoke="$(mktemp -d "${TMPDIR:-/tmp}/q27-tarball-smoke.XXXXXX")"
server_pid=""
cleanup() {
  set +e
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null
  fi
  rm -rf "$smoke"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

tar -C "$smoke" -xzf "$tarball"
dist="$(find "$smoke" -mindepth 1 -maxdepth 1 -type d -name 'q27-*-macos-arm64')"
[[ -n "$dist" && -x "$dist/bin/q27" ]] || {
  echo "tarball has no q27-*-macos-arm64/bin/q27" >&2; exit 1; }
# Reach the launcher through a symlink in a private bin dir, as users do.
mkdir -p "$smoke/userbin"
ln -s "$dist/bin/q27" "$smoke/userbin/q27"
q27="$smoke/userbin/q27"

qhome="$smoke/home/models"
mkdir -p "$qhome" "$smoke/home/run" "$smoke/home/snapshots"
chmod 700 "$smoke/home" "$qhome" "$smoke/home/run" "$smoke/home/snapshots"
if [[ -n "$seed" ]]; then
  mkdir -p "$qhome/$pack"
  for f in "$seed"/*; do
    [[ -f "$f" ]] && { ln "$f" "$qhome/$pack/" 2>/dev/null || cp "$f" "$qhome/$pack/"; }
  done
fi
clean_path="$smoke/userbin:/usr/bin:/bin:/usr/sbin:/sbin"
# Run from outside any source tree. No subshell: the backgrounded server's $!
# must be the env process that execs down to q27-metal-server.
cd "$smoke"
run_q27() {
  env -i HOME="$HOME" PATH="$clean_path" TMPDIR="${TMPDIR:-/tmp}" \
    Q27_HOME="$qhome" Q27_RUN_DIR="$smoke/home/run" \
    Q27_SNAPSHOT_DIR="$smoke/home/snapshots" "$q27" "$@"
}

run_q27 help >/dev/null
run_q27 pull "$pack" >"$smoke/pull.out" 2>&1 || { cat "$smoke/pull.out" >&2; exit 1; }
run_q27 recommend >"$smoke/recommend.out"
# recommend names one pack per machine; any pack under test must at least fit.
grep -Eq "^$pack[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+yes" "$smoke/recommend.out" || {
  echo "recommend does not list $pack as fitting:" >&2; cat "$smoke/recommend.out" >&2; exit 1; }
model_profile="$(awk -F'\t' -v n="$pack" '$1==n { print $13 }' "$dist/packaging/models.tsv")"

env -i HOME="$HOME" PATH="$clean_path" TMPDIR="${TMPDIR:-/tmp}" \
  Q27_HOME="$qhome" Q27_RUN_DIR="$smoke/home/run" \
  Q27_SNAPSHOT_DIR="$smoke/home/snapshots" \
  "$q27" serve "$pack" >"$smoke/server.out" 2>"$smoke/server.err" &
server_pid=$!
for _ in {1..600}; do
  curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
  kill -0 "$server_pid" 2>/dev/null || {
    echo "packaged server exited before readiness" >&2; cat "$smoke/server.err" >&2; exit 1; }
  sleep 0.5
done
curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null
env -i HOME="$HOME" PATH="$(dirname "$PYTHON"):/usr/bin:/bin" TMPDIR="${TMPDIR:-/tmp}" \
  "$PYTHON" "$root/tools/packaged_api_driver.py" "http://127.0.0.1:$port" "$model_profile"
kill "$server_pid"; wait "$server_pid" || true
server_pid=""

# The native agent is a separate model consumer: only after the server is gone.
env -i HOME="$HOME" PATH="$clean_path" TMPDIR="${TMPDIR:-/tmp}" PYTHON="$PYTHON" \
  Q27_HOME="$qhome" "$root/tools/packaged_agent_driver.sh" "$q27" "$pack"
echo "release tarball smoke: PASS ($pack, $(basename "$tarball"))"
