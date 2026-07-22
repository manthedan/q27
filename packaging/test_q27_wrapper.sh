#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/q27-wrapper.XXXXXX")"
blocked_pid=""
fd_consumer_pid=""
fd_child_pid=""
cleanup() {
    local pid
    set +e
    if [ -s "$TMP/fd-child-pid" ]; then
        pid="$(cat "$TMP/fd-child-pid" 2>/dev/null)"
        case "$pid" in ""|*[!0-9]*) ;; *) [ "$pid" = "$$" ] || kill "$pid" 2>/dev/null ;; esac
    fi
    for pid in "$blocked_pid" "$fd_consumer_pid" "$fd_child_pid"; do
        case "$pid" in ""|*[!0-9]*) continue ;; esac
        [ "$pid" = "$$" ] || kill "$pid" 2>/dev/null
    done
    [ -z "$blocked_pid" ] || wait "$blocked_pid" 2>/dev/null
    [ -z "$fd_consumer_pid" ] || wait "$fd_consumer_pid" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT
mkdir -p "$TMP/home/b1" "$TMP/bin" "$TMP/work space"
touch "$TMP/home/b1/bonsai-27b-b1.q27" "$TMP/home/b1/model.tok"

cat >"$TMP/bin/pgrep" <<'EOF'
#!/bin/sh
if [ "${Q27_TEST_RESIDENT:-0}" = 1 ]; then
    case "$*" in *q27-agent*) exit 0 ;; esac
fi
exit 1
EOF
cat >"$TMP/bin/q27-agent" <<'EOF'
#!/bin/sh
printf '%s\n' "$@" >"$Q27_CAPTURE"
[ -n "${Q27_CAPTURE_PID:-}" ] && printf '%s\n' "$$" >"$Q27_CAPTURE_PID"
if [ "${Q27_TEST_BLOCK:-0}" = 1 ]; then
    : >"$Q27_TEST_STARTED"
    trap 'exit 0' TERM INT HUP
    # Stay in this process so the PID/kill test does not depend on a shell
    # child inheriting the synthetic consumer's descriptors.
    while :; do :; done
fi
exit 0
EOF
cat >"$TMP/bin/q27-metal-server" <<'EOF'
#!/bin/sh
printf '%s\n' "$@" >"$Q27_SERVER_CAPTURE"
EOF
chmod +x "$TMP/bin/pgrep" "$TMP/bin/q27-agent" "$TMP/bin/q27-metal-server"

export Q27_HOME="$TMP/home"
export Q27_BIN_DIR="$TMP/bin"
export Q27_CAPTURE="$TMP/args"
export Q27_AGENT_CONTEXT=1234
export Q27_AGENT_WORKSPACE="$TMP/work space"
export Q27_AGENT_MAX_TOKENS=77
export Q27_AGENT_SESSION="$TMP/session file.q27agent"
export Q27_RUN_DIR="$TMP/run"
lock_available() {
    "$PYTHON" - "$Q27_RUN_DIR/consumer.lock" <<'PY'
import fcntl, os, sys
try:
    fd = os.open(sys.argv[1], os.O_RDWR)
except FileNotFoundError:
    raise SystemExit(0)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(1)
os.close(fd)
PY
}
wait_lock_clear() {
    local i
    for i in {1..30}; do
        lock_available && return 0
        sleep 0.1
    done
    return 1
}
assert_lock_private() {
    "$PYTHON" - "$Q27_RUN_DIR/consumer.lock" <<'PY'
import os, stat, sys
st = os.lstat(sys.argv[1])
assert stat.S_ISREG(st.st_mode)
assert stat.S_IMODE(st.st_mode) == 0o600
assert st.st_uid == os.geteuid()
assert st.st_nlink == 1
PY
}
Q27_CAPTURE_PID="$TMP/consumer-pid" PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" agent b1 --no-think \
    >"$TMP/stdout" 2>"$TMP/stderr" &
wrapper_pid=$!
wait "$wrapper_pid"
[ "$(cat "$TMP/consumer-pid")" = "$wrapper_pid" ] || {
    echo "FAIL: wrapper PID was not preserved across consumer exec" >&2; exit 1; }

cat >"$TMP/expected" <<EOF
$TMP/home/b1/bonsai-27b-b1.q27
$TMP/home/b1/model.tok
--context
1234
--max-tokens
77
--auto-tools
--workspace
$TMP/work space
--session
$TMP/session file.q27agent
--no-think
EOF
diff -u "$TMP/expected" "$TMP/args"
[ ! -s "$TMP/stdout" ] || {
    echo "FAIL: supervisor banner polluted agent stdout" >&2; exit 1; }
grep -q '^q27 agent: pack=b1 context=1234 workspace=' "$TMP/stderr"
grep -q 'ui=classic' "$TMP/stderr" || {
    echo "FAIL: non-TTY wrapper test should use classic agent UI" >&2
    cat "$TMP/stderr" >&2
    exit 1
}
grep -q 'automatic tools enabled' "$TMP/stderr"

# When q27-tui is on PATH and Q27_AGENT_UI=tui, the wrapper must launch the
# TUI (and pass agent path via Q27_AGENT) rather than q27-agent directly.
cat >"$TMP/bin/q27-tui" <<'EOF'
#!/bin/sh
printf 'Q27_AGENT=%s\n' "${Q27_AGENT:-}" >"$Q27_TUI_CAPTURE"
printf '%s\n' "$@" >>"$Q27_TUI_CAPTURE"
exit 0
EOF
chmod +x "$TMP/bin/q27-tui"
export Q27_TUI_CAPTURE="$TMP/tui-args"
export Q27_AGENT_UI=tui
PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" agent b1 --no-think \
    >"$TMP/tui-stdout" 2>"$TMP/tui-stderr"
unset Q27_AGENT_UI
grep -q 'ui=tui' "$TMP/tui-stderr" || {
    echo "FAIL: expected ui=tui in stderr" >&2; cat "$TMP/tui-stderr" >&2; exit 1; }
grep -q "^Q27_AGENT=$TMP/bin/q27-agent\$" "$TMP/tui-args" || {
    echo "FAIL: TUI did not receive Q27_AGENT pointing at q27-agent" >&2
    cat "$TMP/tui-args" >&2; exit 1; }
cat >"$TMP/tui-expected" <<EOF
Q27_AGENT=$TMP/bin/q27-agent
--
$TMP/home/b1/bonsai-27b-b1.q27
$TMP/home/b1/model.tok
--context
1234
--max-tokens
77
--auto-tools
--workspace
$TMP/work space
--session
$TMP/session file.q27agent
--no-think
EOF
diff -u "$TMP/tui-expected" "$TMP/tui-args"
# Ensure classic fake agent was not also invoked for the TUI path.
[ ! -f "$TMP/args-from-tui-path" ] || true
wait_lock_clear || {
    echo "FAIL: consumer flock remained held after normal child exit" >&2; exit 1; }

# The wrapper PID is the consumer PID, so kill $! terminates the model process;
# the kernel then releases the inherited flock automatically.
Q27_CAPTURE="$TMP/args-blocked" Q27_CAPTURE_PID="$TMP/blocked-pid" \
    Q27_TEST_BLOCK=1 Q27_TEST_STARTED="$TMP/blocked-started" \
    PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" agent b1 >/dev/null 2>"$TMP/blocked.err" &
blocked_pid=$!
for i in {1..30}; do
    [ -e "$TMP/blocked-started" ] && break
    sleep 0.1
done
[ -e "$TMP/blocked-started" ] &&
    [ "$(cat "$TMP/blocked-pid")" = "$blocked_pid" ] &&
    assert_lock_private || {
        echo "FAIL: blocking consumer did not hold a private lock" >&2; exit 1; }
if PATH="$TMP/bin:/usr/bin:/bin" "$ROOT/packaging/bin/q27" agent b1 \
       >"$TMP/locked.out" 2>"$TMP/locked.err"; then
    echo "FAIL: active consumer lock was ignored" >&2
    exit 1
fi
grep -q 'already running' "$TMP/locked.err"
kill "$blocked_pid"
wait "$blocked_pid" || true
if kill -0 "$blocked_pid" 2>/dev/null; then
    echo "FAIL: kill of wrapper PID left consumer alive" >&2
    exit 1
fi
blocked_pid=""
wait_lock_clear || {
    echo "FAIL: consumer flock remained held after PID-based termination" >&2; exit 1; }

# Production consumers mark the inherited lock close-on-exec immediately, so
# a surviving tool subprocess cannot retain authority after the model exits.
cat >"$TMP/fd-consumer.py" <<'PY'
import fcntl, os, subprocess, sys
fd = int(os.environ.pop("Q27_SUPERVISOR_LOCK_FD"))
fcntl.fcntl(fd, fcntl.F_SETFD, fcntl.fcntl(fd, fcntl.F_GETFD) | fcntl.FD_CLOEXEC)
child = subprocess.Popen(["/bin/sleep", "30"], close_fds=False)
with open(sys.argv[1], "w", encoding="ascii") as out:
    out.write(str(child.pid))
while True:
    pass
PY
"$ROOT/packaging/bin/q27-lock-exec" "$Q27_RUN_DIR/consumer.lock" \
    "$PYTHON" "$TMP/fd-consumer.py" "$TMP/fd-child-pid" &
fd_consumer_pid=$!
for i in {1..30}; do
    [ -s "$TMP/fd-child-pid" ] && break
    sleep 0.1
done
[ -s "$TMP/fd-child-pid" ] || {
    echo "FAIL: descriptor consumer did not start" >&2; exit 1; }
fd_child_pid="$(cat "$TMP/fd-child-pid")"
kill "$fd_consumer_pid"
wait "$fd_consumer_pid" 2>/dev/null || true
fd_consumer_pid=""
kill -0 "$fd_child_pid" 2>/dev/null || {
    echo "FAIL: descriptor test child did not survive consumer" >&2; exit 1; }
wait_lock_clear || {
    echo "FAIL: tool child inherited the consumer flock" >&2; exit 1; }
kill "$fd_child_pid" 2>/dev/null || true
for i in {1..30}; do
    kill -0 "$fd_child_pid" 2>/dev/null || break
    sleep 0.1
done
if kill -0 "$fd_child_pid" 2>/dev/null; then
    echo "FAIL: descriptor test child did not terminate" >&2
    exit 1
fi
rm -f "$TMP/fd-child-pid"
fd_child_pid=""

# Non-regular and aliased lock paths fail closed; the helper never treats a
# directory or symlink destination as successful lock acquisition.
rm -f "$Q27_RUN_DIR/consumer.lock"
mkdir "$Q27_RUN_DIR/consumer.lock"
if PATH="$TMP/bin:/usr/bin:/bin" "$ROOT/packaging/bin/q27" agent b1 \
       >"$TMP/lock-directory.out" 2>"$TMP/lock-directory.err"; then
    echo "FAIL: lock directory was accepted" >&2
    exit 1
fi
grep -q 'cannot launch locked consumer' "$TMP/lock-directory.err"
rmdir "$Q27_RUN_DIR/consumer.lock"
touch "$TMP/lock-target"
ln -s "$TMP/lock-target" "$Q27_RUN_DIR/consumer.lock"
if PATH="$TMP/bin:/usr/bin:/bin" "$ROOT/packaging/bin/q27" agent b1 \
       >"$TMP/lock-symlink.out" 2>"$TMP/lock-symlink.err"; then
    echo "FAIL: lock symlink was accepted" >&2
    exit 1
fi
grep -q 'cannot launch locked consumer' "$TMP/lock-symlink.err"
rm -f "$Q27_RUN_DIR/consumer.lock" "$TMP/lock-target"
Q27_CAPTURE="$TMP/args-after-lock-errors" PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" agent b1 --no-think \
    >/dev/null 2>"$TMP/after-lock-errors.err"
wait_lock_clear || {
    echo "FAIL: flock remained held after lock-error recovery launch" >&2; exit 1; }

# Server and agent launches share the same lifetime lock.
Q27_SERVER_CAPTURE="$TMP/server-args" PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" serve b1 >/dev/null
head -2 "$TMP/server-args" >"$TMP/server-model-paths"
printf '%s\n%s\n' "$TMP/home/b1/bonsai-27b-b1.q27" \
    "$TMP/home/b1/model.tok" >"$TMP/server-model-expected"
diff -u "$TMP/server-model-expected" "$TMP/server-model-paths"
wait_lock_clear || {
    echo "FAIL: server consumer flock remained held after child exit" >&2; exit 1; }

# A source checkout resolves its gitignored model tree without requiring the
# installed ~/.q27 layout.
mkdir -p "$TMP/source/models/bonsai-27b-b1" \
         "$TMP/source/models/qwen36-27b-mtp"
touch "$TMP/source/models/bonsai-27b-b1/bonsai-27b-b1.q27" \
      "$TMP/source/models/qwen36-27b-mtp/qwen36-27b-mtp.tok"
Q27_HOME="$TMP/not-installed" Q27_SOURCE_ROOT="$TMP/source" \
    Q27_CAPTURE="$TMP/args-source" PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" agent b1 --no-think \
    >/dev/null 2>"$TMP/source.err"
head -2 "$TMP/args-source" >"$TMP/source-paths"
printf '%s\n%s\n' \
    "$TMP/source/models/bonsai-27b-b1/bonsai-27b-b1.q27" \
    "$TMP/source/models/qwen36-27b-mtp/qwen36-27b-mtp.tok" \
    >"$TMP/source-paths-expected"
diff -u "$TMP/source-paths-expected" "$TMP/source-paths"
wait_lock_clear || {
    echo "FAIL: source-fallback consumer flock remained held" >&2; exit 1; }

# The default workspace uses the physical cwd so the worker's final-component
# O_NOFOLLOW check does not reject a project entered through a symlink.
mkdir -p "$TMP/physical-project"
ln -s "$TMP/physical-project" "$TMP/project-link"
unset Q27_AGENT_WORKSPACE Q27_AGENT_SESSION
(
    cd "$TMP/project-link"
    Q27_CAPTURE="$TMP/args-physical" \
        PATH="$TMP/bin:/usr/bin:/bin" \
        "$ROOT/packaging/bin/q27" agent b1 \
        >/dev/null 2>"$TMP/physical.err"
)
awk 'seen { print; exit } $0 == "--workspace" { seen=1 }' \
    "$TMP/args-physical" >"$TMP/workspace-arg"
(cd "$TMP/physical-project" && pwd -P) >"$TMP/workspace-expected"
diff -u "$TMP/workspace-expected" "$TMP/workspace-arg"
wait_lock_clear || {
    echo "FAIL: physical-workspace consumer flock remained held" >&2; exit 1; }

# Help works after an optional pack without requiring weights or admission.
Q27_HOME="$TMP/not-installed" Q27_CAPTURE="$TMP/help-args" \
    PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" agent b1 --help >/dev/null
grep -qx -- '--help' "$TMP/help-args"

# Resident detection keys off anchored executable argv, not a .q27 argument
# suffix or an unrelated process that merely mentions the binary path.
agent_pattern='^([^[:space:]]*/)?q27-agent([[:space:]]|$)'
printf '%s\n' '/tmp/q27-agent extensionless-model tok' | grep -Eq "$agent_pattern"
if printf '%s\n' 'tail -f /tmp/q27-agent' | grep -Eq "$agent_pattern"; then
    echo "FAIL: resident pattern matched a non-executable argument" >&2
    exit 1
fi
if Q27_TEST_RESIDENT=1 PATH="$TMP/bin:/usr/bin:/bin" \
       "$ROOT/packaging/bin/q27" agent b1 \
       >"$TMP/resident.out" 2>"$TMP/resident.err"; then
    echo "FAIL: resident native agent was not detected" >&2
    exit 1
fi
grep -q 'already running' "$TMP/resident.err"

if PATH="$TMP/bin:/usr/bin:/bin" "$ROOT/packaging/bin/q27" agent missing \
       >"$TMP/missing.out" 2>"$TMP/missing.err"; then
    echo "FAIL: unknown agent pack succeeded" >&2
    exit 1
fi
grep -q 'unknown pack missing' "$TMP/missing.err"

PATH="$TMP/bin:/usr/bin:/bin" "$ROOT/packaging/bin/q27" help >"$TMP/help"
grep -q 'q27 agent \[name\]' "$TMP/help"

echo "q27 wrapper selftest: PASS"
