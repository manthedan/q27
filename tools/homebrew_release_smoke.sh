#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 && $# -ne 4 ]]; then
  echo "usage: $0 STAGING_FORMULA QWEN38_PACK [LOCAL_MODEL LOCAL_TOKENIZER]" >&2
  exit 2
fi
formula="$1"
pack="$2"
local_model="${3:-}"
local_tokenizer="${4:-}"
if [[ -n "$local_model" ]]; then
  for input in "$local_model" "$local_tokenizer"; do
    [[ -f "$input" && ! -L "$input" ]] || {
      echo "local validation inputs must be regular non-symlink files: $input" >&2
      exit 1
    }
  done
fi
[[ -f "$formula" && ! -L "$formula" ]] || {
  echo "staging formula must be a regular non-symlink file: $formula" >&2
  exit 1
}
[[ -n "$pack" ]] || { echo "Qwen3.8 pack name is empty" >&2; exit 2; }
command -v brew >/dev/null 2>&1 || { echo "Homebrew is required" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }
PYTHON="${PYTHON:-$(command -v python3)}"
[[ -x "$PYTHON" ]] || { echo "python3 is required for the release gate" >&2; exit 1; }

expected_ram="${Q27_HOMEBREW_EXPECT_RAM_GB:-24}"
ram_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
[[ "$ram_gb" -eq "$expected_ram" ]] || {
  echo "Homebrew release smoke requires exactly ${expected_ram} GB RAM; got $ram_gb" >&2
  exit 1
}
if [[ -n "$(brew list --versions q27 2>/dev/null || true)" ]]; then
  echo "refusing to disturb an existing q27 Homebrew installation" >&2
  exit 1
fi
if ! "$PYTHON" - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    probe.bind(("127.0.0.1", 8080))
PY
then
  echo "refusing to use occupied q27 smoke port 8080" >&2
  exit 1
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
smoke="$(mktemp -d "${TMPDIR:-/tmp}/q27-homebrew-smoke.XXXXXX")"
installed_attempted=0
server_pid=""
cleanup() {
  set +e
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null
    wait "$server_pid" 2>/dev/null
  fi
  if [[ "$installed_attempted" -eq 1 ]] &&
     [[ -n "$(brew list --versions q27 2>/dev/null || true)" ]]; then
    brew uninstall --formula q27 >/dev/null 2>&1 || true
  fi
  rm -rf "$smoke"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

installed_attempted=1
brew install --formula "$formula"
[[ -n "$(brew list --versions q27 2>/dev/null || true)" ]] || {
  echo "Homebrew did not register the candidate q27 formula" >&2
  exit 1
}
prefix="$(brew --prefix q27)"
q27="$prefix/bin/q27"
lock_exec="$prefix/libexec/q27/bin/q27-lock-exec"
[[ -x "$q27" && -x "$lock_exec" ]] || {
  echo "installed package is missing q27 or native q27-lock-exec" >&2
  exit 1
}
file "$lock_exec" | grep -q 'Mach-O.*arm64'
if head -c 64 "$lock_exec" | grep -qi python; then
  echo "installed q27-lock-exec is a script, not the staged native helper" >&2
  exit 1
fi
registry="$prefix/libexec/q27/models.tsv"
pack_policy="$(awk -F'\t' -v n="$pack" '
  $1==n { found++; kind=$9; experimental=$11; artifact=$4; digest=$5 }
  END { if(found!=1) exit 1; print kind "\t" experimental "\t" artifact "\t" digest }
' "$registry")" || {
  echo "selected Qwen3.8 pack is absent or duplicated in the installed registry" >&2
  exit 1
}
IFS=$'\t' read -r source_kind experimental artifact_file artifact_md5 <<<"$pack_policy"
case "$source_kind/$experimental" in
  direct/no)
    [[ -z "$local_model" ]] || {
      echo "public direct-pack smoke does not accept local model inputs" >&2; exit 1; }
    pack_mode=public
    ;;
  local/yes)
    [[ -n "$local_model" ]] || {
      echo "experimental local-pack validation requires LOCAL_MODEL and LOCAL_TOKENIZER" >&2
      exit 1
    }
    [[ "$(md5 -q "$local_model")" = "$artifact_md5" ]] || {
      echo "local validation model does not match the registry MD5" >&2; exit 1; }
    pack_mode=local
    ;;
  *)
    echo "pack must be either public direct/non-experimental or local/experimental" >&2
    exit 1
    ;;
esac
brew test q27

qhome="$smoke/home/models"
run_dir="$smoke/home/run"
snapshots="$smoke/home/snapshots"
mkdir -p "$qhome" "$run_dir" "$snapshots"
chmod 700 "$smoke/home" "$qhome" "$run_dir" "$snapshots"
clean_path="$prefix/bin:$(dirname "$(command -v brew)"):/usr/bin:/bin:/usr/sbin:/sbin"
run_q27() {
  env -i HOME="$HOME" PATH="$clean_path" TMPDIR="${TMPDIR:-/tmp}" \
    Q27_HOME="$qhome" Q27_RUN_DIR="$run_dir" Q27_SNAPSHOT_DIR="$snapshots" \
    "$q27" "$@"
}

if [[ "$pack_mode" = public ]]; then
  run_q27 pull "$pack"
else
  mkdir -p "$qhome/$pack"
  cp "$local_model" "$qhome/$pack/$artifact_file"
  cp "$local_tokenizer" "$qhome/$pack/$(basename "$local_tokenizer")"
  chmod 600 "$qhome/$pack/$artifact_file" "$qhome/$pack/$(basename "$local_tokenizer")"
fi
[[ -f "$qhome/$pack/$artifact_file" ]]
run_q27 recommend >"$smoke/recommend.out"
if [[ "$pack_mode" = public ]]; then
  grep -Fq "Recommended pack for this machine: $pack" "$smoke/recommend.out"
fi

env -i HOME="$HOME" PATH="$clean_path" TMPDIR="${TMPDIR:-/tmp}" \
  Q27_HOME="$qhome" Q27_RUN_DIR="$run_dir" Q27_SNAPSHOT_DIR="$snapshots" \
  "$q27" serve "$pack" >"$smoke/server.out" 2>"$smoke/server.err" &
server_pid=$!
for _ in {1..600}; do
  if curl -fsS --max-time 1 http://127.0.0.1:8080/health >"$smoke/health.json" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "packaged q27 server exited before readiness" >&2
    cat "$smoke/server.err" >&2
    exit 1
  fi
  sleep 0.5
done
curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null

env -i HOME="$HOME" PATH="$(dirname "$PYTHON"):/usr/bin:/bin:/usr/sbin:/sbin" \
  TMPDIR="${TMPDIR:-/tmp}" "$PYTHON" "$root/tools/packaged_api_driver.py" \
  http://127.0.0.1:8080 qwen38-thinking-v1
kill "$server_pid"
wait "$server_pid" || true
server_pid=""

# The native agent is a separate model consumer and must run only after the
# packaged server and its inherited flock are gone.
env -i HOME="$HOME" PATH="$clean_path" TMPDIR="${TMPDIR:-/tmp}" \
  PYTHON="$PYTHON" Q27_HOME="$qhome" \
  "$root/tools/packaged_agent_driver.sh" "$q27" "$pack"

brew uninstall --formula q27
installed_attempted=0
trap - EXIT INT TERM HUP
rm -rf "$smoke"
echo "Homebrew local-asset Qwen3.8 release smoke: PASS"
