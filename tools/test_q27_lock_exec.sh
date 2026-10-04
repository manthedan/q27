#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${Q27_LOCK_EXEC_TEST_BIN:-$ROOT/build/q27-lock-exec}"
[[ -x "$BIN" ]] || { echo "missing $BIN; run make build/q27-lock-exec" >&2; exit 2; }

tmp="$(mktemp -d "${TMPDIR:-/tmp}/q27-lock-exec.XXXXXX")"
holder_pid=""
cleanup() {
  set +e
  [[ -z "$holder_pid" ]] || kill "$holder_pid" 2>/dev/null
  [[ -z "$holder_pid" ]] || wait "$holder_pid" 2>/dev/null
  rm -rf "$tmp"
}
trap cleanup EXIT
lock="$tmp/run/consumer.lock"

stat_mode() {
  if [[ "$(uname -s)" == Darwin ]]; then stat -f '%Lp' "$1"
  else stat -c '%a' "$1"
  fi
}
stat_uid() {
  if [[ "$(uname -s)" == Darwin ]]; then stat -f '%u' "$1"
  else stat -c '%u' "$1"
  fi
}
stat_links() {
  if [[ "$(uname -s)" == Darwin ]]; then stat -f '%l' "$1"
  else stat -c '%h' "$1"
  fi
}

set +e
"$BIN" >"$tmp/usage.out" 2>&1
usage_rc=$?
set -e
[[ "$usage_rc" -eq 2 ]]
grep -q 'usage: q27-lock-exec LOCK COMMAND' "$tmp/usage.out"

cat >"$tmp/holder.sh" <<'SH'
#!/bin/sh
set -eu
fd=${Q27_SUPERVISOR_LOCK_FD:?}
[ -e "/dev/fd/$fd" ]
printf '%s\n' "$$" >"$Q27_HOLDER_PID_FILE"
printf '%s\n' "$fd" >"$Q27_HOLDER_FD_FILE"
: >"$Q27_HOLDER_READY"
trap 'exit 0' TERM INT HUP
while :; do sleep 1; done
SH
chmod 755 "$tmp/holder.sh"
Q27_HOLDER_PID_FILE="$tmp/holder.pid" Q27_HOLDER_FD_FILE="$tmp/holder.fd" \
  Q27_HOLDER_READY="$tmp/holder.ready" \
  "$BIN" "$lock" "$tmp/holder.sh" &
holder_pid=$!
for _ in {1..50}; do [[ -e "$tmp/holder.ready" ]] && break; sleep 0.05; done
[[ -e "$tmp/holder.ready" ]]
[[ "$(cat "$tmp/holder.pid")" = "$holder_pid" ]]
kill -0 "$holder_pid"
[[ -f "$lock" && ! -L "$lock" ]]
[[ "$(stat_mode "$tmp/run")" = 700 ]]
[[ "$(stat_mode "$lock")" = 600 ]]
[[ "$(stat_uid "$lock")" = "$(id -u)" ]]
[[ "$(stat_links "$lock")" = 1 ]]

# Every simultaneous contender must observe the held flock and fail with the
# retryable helper code; none may execute its command.
pids=()
for i in {1..12}; do
  (
    set +e
    "$BIN" "$lock" /usr/bin/touch "$tmp/contender-ran-$i" \
      >"$tmp/contender-$i.out" 2>&1
    printf '%s\n' "$?" >"$tmp/contender-$i.rc"
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
for i in {1..12}; do
  [[ "$(cat "$tmp/contender-$i.rc")" = 75 ]]
  [[ ! -e "$tmp/contender-ran-$i" ]]
  grep -q 'already running' "$tmp/contender-$i.out"
done

kill "$holder_pid"
wait "$holder_pid" || true
holder_pid=""

# The lock descriptor and PID survive helper exec, then the kernel releases the
# flock when the consumer exits.
"$BIN" "$lock" /bin/sh -c \
  'test -n "$Q27_SUPERVISOR_LOCK_FD" && test -e "/dev/fd/$Q27_SUPERVISOR_LOCK_FD"'
"$BIN" "$lock" /usr/bin/touch "$tmp/reacquired"
[[ -e "$tmp/reacquired" ]]

# Aliased/non-regular lock leaves and missing commands fail closed.
rm -f "$lock"
mkdir "$lock"
if "$BIN" "$lock" /usr/bin/true >"$tmp/directory.out" 2>&1; then
  echo "lock directory was accepted" >&2
  exit 1
fi
grep -q 'cannot launch locked consumer' "$tmp/directory.out"
rmdir "$lock"
touch "$tmp/target"
ln -s "$tmp/target" "$lock"
if "$BIN" "$lock" /usr/bin/true >"$tmp/symlink.out" 2>&1; then
  echo "lock symlink was accepted" >&2
  exit 1
fi
grep -q 'cannot launch locked consumer' "$tmp/symlink.out"
rm -f "$lock" "$tmp/target"
if "$BIN" "$lock" q27-command-that-does-not-exist >"$tmp/exec.out" 2>&1; then
  echo "missing command was accepted" >&2
  exit 1
fi
grep -q 'cannot launch locked consumer' "$tmp/exec.out"

echo "native q27-lock-exec contracts: PASS"
