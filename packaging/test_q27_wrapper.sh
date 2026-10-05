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
printf 'fixture-b1-artifact\n' >"$TMP/home/b1/bonsai-27b-b1.q27"
printf 'Q27Tfixture-b1-tokenizer\n' >"$TMP/home/b1/model.tok"
printf 'fixture-q38-artifact\n' >"$TMP/q38-good.q27"
mkdir -p "$TMP/home/b2"
printf 'fixture-b2-artifact\n' >"$TMP/home/b2/bonsai2-27b-t2-slim.q27"
printf 'Q27Tfixture-b2-tokenizer\n' >"$TMP/home/b2/qwen38-27b-mtp.tok"
md5_fixture() {
    if command -v md5 >/dev/null 2>&1; then md5 -q "$1"
    else md5sum "$1" | awk '{print $1}'
    fi
}
b1_md5="$(md5_fixture "$TMP/home/b1/bonsai-27b-b1.q27")"
q38_md5="$(md5_fixture "$TMP/q38-good.q27")"
sha256_fixture() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
    else sha256sum "$1" | awk '{print $1}'
    fi
}
b2_sha="$(sha256_fixture "$TMP/home/b2/bonsai2-27b-t2-slim.q27")"
b2_tok_sha="$(sha256_fixture "$TMP/home/b2/qwen38-27b-mtp.tok")"
awk -F'\t' -v OFS='\t' -v b1="$b1_md5" -v q38="$q38_md5" \
    -v b2="$b2_sha" -v b2tok="$b2_tok_sha" '
    $1 == "b1" { $5=b1 }
    $1 == "q38-q4s" { $5=q38 }
    $1 == "b2" { $5=""; $14=b2; $17=b2tok }
    { print }
' "$ROOT/packaging/models.tsv" >"$TMP/models.tsv"
export Q27_REGISTRY="$TMP/models.tsv"

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
[ -n "${Q27_PROFILE_CAPTURE:-}" ] && printf '%s\n' "${Q27_MODEL_PROFILE:-}" >"$Q27_PROFILE_CAPTURE"
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
[ -n "${Q27_SERVER_PROFILE_CAPTURE:-}" ] && printf '%s\n' "${Q27_MODEL_PROFILE:-}" >"$Q27_SERVER_PROFILE_CAPTURE"
exit 0
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
# Deterministic sampling baseline: the operator's ambient Q27_AGENT_* sampling
# vars must not leak into these scenarios (the bonsai soft-default keys off
# whether Q27_AGENT_TEMPERATURE is SET, so a stray export inverts the policy).
unset Q27_AGENT_TEMPERATURE Q27_AGENT_TOP_P Q27_AGENT_TOP_K Q27_AGENT_SEED \
      Q27_AGENT_MTP Q27_AGENT_MAX_THINK_TOKENS Q27_AGENT_NO_THINK Q27_AGENT_UI \
      Q27_SERVE_THINK Q27_SERVE_TEMPERATURE Q27_SERVE_TOP_P Q27_SERVE_TOP_K \
      Q27_METAL_TEMPERATURE_DEFAULT Q27_METAL_TOP_P_DEFAULT Q27_METAL_TOP_K_DEFAULT
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
--temperature
0.6
--top-p
0.95
--top-k
20
--no-think
EOF
diff -u "$TMP/expected" "$TMP/args"
# b1 is bonsai-b1-v1: the pack-split soft-default must fire when
# Q27_AGENT_TEMPERATURE is unset, and the stderr hint must name the escape.
grep -q 'bonsai default; Q27_AGENT_TEMPERATURE=0 for greedy' "$TMP/stderr" || {
    echo "FAIL: bonsai soft-default hint missing from stderr" >&2; exit 1; }

# Greedy escape: explicit TEMPERATURE=0 passes --temperature 0 with NO
# top-p/top-k companions (stays on the binary's pure-argmax path).
Q27_CAPTURE="$TMP/args-greedy" Q27_CAPTURE_PID="$TMP/consumer-pid-greedy" \
    Q27_AGENT_TEMPERATURE=0 PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" agent b1 --no-think \
    >"$TMP/stdout-greedy" 2>"$TMP/stderr-greedy"
grep -A1 '^--temperature$' "$TMP/args-greedy" | grep -qx '0' || {
    echo "FAIL: TEMPERATURE=0 escape did not pass --temperature 0" >&2; exit 1; }
if grep -q '^--top-p$' "$TMP/args-greedy"; then
    echo "FAIL: TEMPERATURE=0 escape added top-p companions" >&2; exit 1
fi
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
--temperature
0.6
--top-p
0.95
--top-k
20
--no-think
EOF
diff -u "$TMP/tui-expected" "$TMP/tui-args"
# Ensure classic fake agent was not also invoked for the TUI path.
[ ! -f "$TMP/args-from-tui-path" ] || true

# One-shot flags bypass the TUI even when q27-tui is on PATH (r22 codex P2):
# --prompt is a non-interactive invocation and must reach the classic agent.
rm -f "$TMP/tui-args"
Q27_TUI_CAPTURE="$TMP/tui-args" \
    PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" agent b1 --prompt "hello there" \
    >"$TMP/oneshot-stdout" 2>"$TMP/oneshot-stderr"
[ ! -f "$TMP/tui-args" ] || {
    echo "FAIL: one-shot --prompt was routed to the TUI" >&2
    cat "$TMP/tui-args" >&2; exit 1; }
grep -q -- "--prompt" "$TMP/args" || {
    echo "FAIL: one-shot --prompt did not reach the classic agent" >&2
    cat "$TMP/args" >&2; exit 1; }

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
Q27_SERVER_CAPTURE="$TMP/server-args" \
    Q27_SERVER_PROFILE_CAPTURE="$TMP/server-profile" \
    PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" serve b1 >/dev/null
head -2 "$TMP/server-args" >"$TMP/server-model-paths"
printf '%s\n%s\n' "$TMP/home/b1/bonsai-27b-b1.q27" \
    "$TMP/home/b1/model.tok" >"$TMP/server-model-expected"
diff -u "$TMP/server-model-expected" "$TMP/server-model-paths"
grep -Fxq 'bonsai-agent-v1' "$TMP/server-profile"
wait_lock_clear || {
    echo "FAIL: server consumer flock remained held after child exit" >&2; exit 1; }

# The reproduced Qwen3.8 q4s control remains local/experimental after c-small
# won the release comparison. Its runtime behavior is keyed by the registry
# profile, not the q38-* handle.
Q27_HOME="$TMP/q38-home" PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" pull q38-q4s \
    >"$TMP/q38-pull.out" 2>"$TMP/q38-pull.err"
grep -q 'q38-q4s is EXPERIMENTAL' "$TMP/q38-pull.out"
grep -q 'q38-q4s is a LOCAL artifact, not downloadable' "$TMP/q38-pull.out"
grep -Fq "$TMP/q38-home/q38-q4s/qwen38-27b-mtp-q4s.q27" "$TMP/q38-pull.out"
grep -q "Expected artifact MD5: $q38_md5" "$TMP/q38-pull.out"
touch "$TMP/q38-home/q38-q4s/qwen38-27b-mtp-q4s.q27" \
      "$TMP/q38-home/q38-q4s/qwen38-27b-mtp.tok"
if Q27_HOME="$TMP/q38-home" PATH="$TMP/bin:/usr/bin:/bin" \
       "$ROOT/packaging/bin/q27" pull q38-q4s \
       >"$TMP/q38-bad.out" 2>"$TMP/q38-bad.err"; then
    echo "FAIL: mismatched local q38 artifact was accepted" >&2
    exit 1
fi
grep -q 'md5 MISMATCH' "$TMP/q38-bad.err"
if Q27_HOME="$TMP/q38-home" PATH="$TMP/bin:/usr/bin:/bin" \
       "$ROOT/packaging/bin/q27" serve q38-q4s \
       >"$TMP/q38-invalid-serve.out" 2>"$TMP/q38-invalid-serve.err"; then
    echo "FAIL: q38 wrapper launched a checksum-rejected artifact" >&2
    exit 1
fi
grep -q 'q38-q4s not installed' "$TMP/q38-invalid-serve.err"
# Continue with checksum-valid tiny fixtures to unit-test profile-to-argument
# dispatch. Real Qwen3.8 model execution lives in test-metal-qwen38-{serving,agent}.
cp "$TMP/q38-good.q27" "$TMP/q38-home/q38-q4s/qwen38-27b-mtp-q4s.q27"
printf 'Q27Tfixture-q38-tokenizer\n' >"$TMP/q38-home/q38-q4s/qwen38-27b-mtp.tok"

Q27_HOME="$TMP/q38-home" Q27_SERVER_CAPTURE="$TMP/q38-server-args" \
    Q27_SERVER_PROFILE_CAPTURE="$TMP/q38-server-profile" \
    PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" serve q38-q4s >/dev/null
grep -Fxq -- '--think' "$TMP/q38-server-args"
grep -A1 '^--temperature-default$' "$TMP/q38-server-args" | grep -qx '1.0'
grep -A1 '^--top-p-default$' "$TMP/q38-server-args" | grep -qx '0.95'
grep -A1 '^--top-k-default$' "$TMP/q38-server-args" | grep -qx '20'
grep -Fxq 'qwen38-thinking-v1' "$TMP/q38-server-profile"
wait_lock_clear || {
    echo "FAIL: q38 serve consumer flock remained held" >&2; exit 1; }
Q27_HOME="$TMP/q38-home" Q27_SERVER_CAPTURE="$TMP/q38-server-no-think-args" \
    Q27_SERVE_THINK=0 Q27_SERVE_TEMPERATURE=0.7 Q27_SERVE_TOP_P=0.8 \
    Q27_SERVE_TOP_K=11 PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" serve q38-q4s >/dev/null
if grep -Fxq -- '--think' "$TMP/q38-server-no-think-args"; then
    echo "FAIL: Q27_SERVE_THINK=0 did not override the qwen38 profile" >&2
    exit 1
fi
grep -A1 '^--temperature-default$' "$TMP/q38-server-no-think-args" | grep -qx '0.7'
grep -A1 '^--top-p-default$' "$TMP/q38-server-no-think-args" | grep -qx '0.8'
grep -A1 '^--top-k-default$' "$TMP/q38-server-no-think-args" | grep -qx '11'

Q27_HOME="$TMP/q38-home" Q27_CAPTURE="$TMP/q38-agent-args" \
    Q27_PROFILE_CAPTURE="$TMP/q38-agent-profile" \
    PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" agent q38-q4s >/dev/null 2>"$TMP/q38-agent.err"
grep -A1 '^--temperature$' "$TMP/q38-agent-args" | grep -qx '1.0'
grep -A1 '^--top-p$' "$TMP/q38-agent-args" | grep -qx '0.95'
grep -A1 '^--top-k$' "$TMP/q38-agent-args" | grep -qx '20'
grep -Fxq 'qwen38-thinking-v1' "$TMP/q38-agent-profile"
grep -q 'qwen38 profile; Q27_AGENT_TEMPERATURE=0 for greedy' "$TMP/q38-agent.err"
if grep -Fxq -- '--no-think' "$TMP/q38-agent-args"; then
    echo "FAIL: qwen38 profile disabled thinking" >&2
    exit 1
fi
wait_lock_clear || {
    echo "FAIL: q38 consumer flock remained held after child exit" >&2; exit 1; }

# Bonsai 2 (bonsai2-qwen38-v1) is the default agent pack: 16K context,
# thinking left on, trained sampling, and suffix-16 speculation unless the
# operator picked a width, disabled it, or chose MTP.
(
    unset Q27_AGENT_CONTEXT
    Q27_CAPTURE="$TMP/b2-agent-args" Q27_PROFILE_CAPTURE="$TMP/b2-agent-profile" \
        PATH="$TMP/bin:/usr/bin:/bin" \
        "$ROOT/packaging/bin/q27" agent >/dev/null 2>"$TMP/b2-agent.err" || { cat "$TMP/b2-agent.err" >&2; exit 1; }
)
grep -Fxq "$TMP/home/b2/bonsai2-27b-t2-slim.q27" "$TMP/b2-agent-args"
grep -Fxq "$TMP/home/b2/qwen38-27b-mtp.tok" "$TMP/b2-agent-args"
grep -Fxq 'bonsai2-qwen38-v1' "$TMP/b2-agent-profile"
grep -A1 '^--context$' "$TMP/b2-agent-args" | grep -qx '16384'
grep -A1 '^--temperature$' "$TMP/b2-agent-args" | grep -qx '1.0'
grep -A1 '^--top-p$' "$TMP/b2-agent-args" | grep -qx '0.95'
grep -A1 '^--top-k$' "$TMP/b2-agent-args" | grep -qx '20'
grep -A1 '^--suffix$' "$TMP/b2-agent-args" | grep -qx '16'
if grep -Fxq -- '--no-think' "$TMP/b2-agent-args"; then
    echo "FAIL: bonsai2 profile disabled thinking" >&2; exit 1
fi
wait_lock_clear || {
    echo "FAIL: b2 agent consumer flock remained held" >&2; exit 1; }
b2_suffix_count() { grep -cx -- '--suffix' "$1" || true; }
Q27_CAPTURE="$TMP/b2-agent-nosuffix" Q27_SUFFIX=0 PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" agent b2 >/dev/null 2>&1
[ "$(b2_suffix_count "$TMP/b2-agent-nosuffix")" = 0 ] || {
    echo "FAIL: Q27_SUFFIX=0 did not disable suffix bursts" >&2; exit 1; }
Q27_CAPTURE="$TMP/b2-agent-mtp" Q27_AGENT_MTP=2 PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" agent b2 >/dev/null 2>&1
[ "$(b2_suffix_count "$TMP/b2-agent-mtp")" = 0 ] || {
    echo "FAIL: Q27_AGENT_MTP still added --suffix" >&2; exit 1; }
grep -A1 '^--mtp$' "$TMP/b2-agent-mtp" | grep -qx '2'
Q27_CAPTURE="$TMP/b2-agent-width" PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" agent b2 --suffix 4 >/dev/null 2>&1
[ "$(b2_suffix_count "$TMP/b2-agent-width")" = 1 ] || {
    echo "FAIL: explicit --suffix was doubled by the wrapper default" >&2; exit 1; }
grep -A1 '^--suffix$' "$TMP/b2-agent-width" | grep -qx '4'
wait_lock_clear || {
    echo "FAIL: b2 agent consumer flock remained held" >&2; exit 1; }

# serve: profile args first, then the caller's args forwarded verbatim.
Q27_SERVER_CAPTURE="$TMP/b2-server-args" \
    Q27_SERVER_PROFILE_CAPTURE="$TMP/b2-server-profile" \
    PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" serve b2 --ctx 8192 >/dev/null
grep -Fxq 'bonsai2-qwen38-v1' "$TMP/b2-server-profile"
grep -Fxq -- '--think' "$TMP/b2-server-args"
grep -A1 '^--think-budget$' "$TMP/b2-server-args" | grep -qx '0'
grep -A1 '^--temperature-default$' "$TMP/b2-server-args" | grep -qx '1.0'
grep -A1 '^--top-p-default$' "$TMP/b2-server-args" | grep -qx '0.95'
grep -A1 '^--top-k-default$' "$TMP/b2-server-args" | grep -qx '20'
grep -A1 '^--suffix$' "$TMP/b2-server-args" | grep -qx '16'
tail -2 "$TMP/b2-server-args" | tr '\n' ' ' | grep -qx -- '--ctx 8192 '
wait_lock_clear || {
    echo "FAIL: b2 serve consumer flock remained held" >&2; exit 1; }
Q27_SERVER_CAPTURE="$TMP/b2-server-nothink" Q27_SERVE_THINK=0 \
    PATH="$TMP/bin:/usr/bin:/bin" \
    "$ROOT/packaging/bin/q27" serve b2 --suffix 8 >/dev/null
if grep -Eqx -- '--think|--think-budget' "$TMP/b2-server-nothink"; then
    echo "FAIL: Q27_SERVE_THINK=0 did not override the bonsai2 profile" >&2; exit 1
fi
[ "$(b2_suffix_count "$TMP/b2-server-nothink")" = 1 ] || {
    echo "FAIL: serve doubled an explicit --suffix" >&2; exit 1; }
grep -A1 '^--suffix$' "$TMP/b2-server-nothink" | grep -qx '8'
wait_lock_clear || {
    echo "FAIL: b2 serve consumer flock remained held" >&2; exit 1; }

# metal-v0.7.0 validates only Bonsai 2: the inherited Qwen rows stay listed
# (they fit 24 GB) but are flagged experimental/not re-validated, so a 24 GB
# machine is recommended the validated b2 pack, never an unvalidated tier.
cat >"$TMP/bin/sysctl" <<'EOF'
#!/bin/sh
[ "$*" = "-n hw.memsize" ] && { echo 25769803776; exit 0; }
exec /usr/sbin/sysctl "$@"
EOF
chmod +x "$TMP/bin/sysctl"
PATH="$TMP/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
    "$ROOT/packaging/bin/q27" recommend >"$TMP/recommend-24gb"
grep -Eq '^q38[[:space:]]+14\.63G[[:space:]]+24G[[:space:]]+yes\(exp\)' "$TMP/recommend-24gb"
grep -Fq 'Recommended pack for this machine: b2' "$TMP/recommend-24gb"
rm -f "$TMP/bin/sysctl"

# Pinned product mismatches may not fall through to legacy source-tree globs.
# The installed artifact and same-named source fallback are both wrong here;
# neither may reach the native agent (tokenizer resolution is likewise exact).
mkdir -p "$TMP/pinned-home/q38" "$TMP/pinned-source/models/qwen36-27b-mtp" \
         "$TMP/pinned-source/models/qwen38-27b-mtp"
touch "$TMP/pinned-home/q38/qwen38-27b-mtp-c-small.q27" \
      "$TMP/pinned-source/models/qwen38-27b-mtp/qwen38-27b-mtp-c-small.q27"
printf 'Q27Tcorrupt-installed\n' >"$TMP/pinned-home/q38/qwen38-27b-mtp.tok"
printf 'Q27Tstale-qwen36\n' >"$TMP/pinned-source/models/qwen36-27b-mtp/qwen36-27b-mtp.tok"
printf 'Q27Twrong-qwen38\n' >"$TMP/pinned-source/models/qwen38-27b-mtp/qwen38-27b-mtp.tok"
if Q27_HOME="$TMP/pinned-home" Q27_SOURCE_ROOT="$TMP/pinned-source" \
       Q27_CAPTURE="$TMP/pinned-agent-args" PATH="$TMP/bin:/usr/bin:/bin" \
       "$ROOT/packaging/bin/q27" agent q38 \
       >"$TMP/pinned-agent.out" 2>"$TMP/pinned-agent.err"; then
    echo "FAIL: pinned q38 product mismatch fell through to a source glob" >&2
    exit 1
fi
grep -q 'q38 is not installed' "$TMP/pinned-agent.err"
[ ! -e "$TMP/pinned-agent-args" ]

# An extracted release exposes bin/q27 and resolves its sibling bin directory
# without Q27_BIN_DIR or a source-tree build fallback. Exercise through a
# second-level symlink as users commonly add the launcher to a private PATH.
archive="$TMP/archive/q27-test-macos-arm64"
mkdir -p "$archive/bin" "$archive/packaging/bin" "$archive/packaging/lib" \
         "$TMP/archive-home/b1" "$TMP/archive-prefix/bin"
cp "$ROOT/packaging/q27-release-launcher" "$archive/bin/q27"
cp "$TMP/bin/q27-metal-server" "$archive/bin/q27-metal-server"
cp "$ROOT/packaging/bin/q27" "$archive/packaging/bin/q27"
cp "$ROOT/build/q27-lock-exec" "$archive/packaging/bin/q27-lock-exec"
cp "$ROOT/packaging/lib/q27_bench_lib.sh" "$archive/packaging/lib/"
cp "$ROOT/packaging/models.tsv" "$archive/packaging/models.tsv"
chmod 755 "$archive/bin/q27" "$archive/bin/q27-metal-server" \
          "$archive/packaging/bin/q27" "$archive/packaging/bin/q27-lock-exec"
printf 'fixture-b1-artifact\n' >"$TMP/archive-home/b1/bonsai-27b-b1.q27"
printf 'Q27Tfixture-b1-tokenizer\n' >"$TMP/archive-home/b1/model.tok"
ln -s ../../archive/q27-test-macos-arm64/bin/q27 "$TMP/archive-prefix/bin/q27"
Q27_HOME="$TMP/archive-home" Q27_RUN_DIR="$TMP/archive-run" \
    Q27_SERVER_CAPTURE="$TMP/archive-server-args" \
    Q27_SERVER_PROFILE_CAPTURE="$TMP/archive-server-profile" \
    PATH="/usr/bin:/bin" "$TMP/archive-prefix/bin/q27" serve b1 >/dev/null
head -2 "$TMP/archive-server-args" >"$TMP/archive-server-paths"
printf '%s\n%s\n' "$TMP/archive-home/b1/bonsai-27b-b1.q27" \
    "$TMP/archive-home/b1/model.tok" >"$TMP/archive-server-expected"
diff -u "$TMP/archive-server-expected" "$TMP/archive-server-paths"
grep -Fxq 'bonsai-agent-v1' "$TMP/archive-server-profile"

# A source checkout resolves its gitignored model tree without requiring the
# installed ~/.q27 layout.
mkdir -p "$TMP/source/models/bonsai-27b-b1" \
         "$TMP/source/models/qwen36-27b-mtp"
printf 'fixture-b1-artifact\n' >"$TMP/source/models/bonsai-27b-b1/bonsai-27b-b1.q27"
printf 'Q27Tfixture-b1-tokenizer\n' >"$TMP/source/models/qwen36-27b-mtp/qwen36-27b-mtp.tok"
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
