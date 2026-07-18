# COORDINATION — read before ANY heavy action on this box (2026-07-17)

This 24 GB M4 crashed TWICE today (~11:44, ~12:00), both times because TWO
17 GB `q27-metal-server` instances were resident at once. Two agents are
working this repo concurrently (pi session + claude/cmux sessions). The
second crash was one agent starting a server while the other agent's server
was already healthy on :8213 and serving gates.

## Protocol for every agent on this box

1. **NEVER launch `q27-metal-server` without BOTH checks first:**
   the argv[0] pgrep in item 2 and
   `curl -s -m 2 http://127.0.0.1:8213/health`.
   If either shows a live server: DO NOT LAUNCH. Use the running one.
2. **NEVER kill a server by the launch line's `$!`** — `caffeinate` forks,
   so `$!` is the wrapper and the 17 GB server survives as an orphan (this
   caused crash #1). Note `pgrep -f "q27-metal-server.*--port"` ALSO
   matches the wrapper (its command line contains the pattern): match the
   server by argv[0] —
   `ps -Ao pid,args | awk '$2=="./build/q27-metal-server" && /--port 8213/ {print $1}'`
   — kill that, then verify the list is EMPTY (not just the wrapper gone)
   before any relaunch.
3. **ONE 17 GB resident max.** No benches, no second server, no model loads
   while the T2 serving instance is up. The quiet-bench watcher
   (`tools/quiet_q4_bench.sh`) stops the server first and is the only
   sanctioned exception.
4. **Gates run against an already-running server only, serialized** (one
   suite at a time, nothing else heavy concurrently).
5. **After any reboot: let the box settle before loading 17 GB** — load < 8
   and no post-crash Spotlight/mdworker storm. Right after a crash boot the
   system does housekeeping; a 17 GB load on top has already killed us.

## State of play (2026-07-17 12:05, pi session)

- Serving instance is DOWN (it is the operator's pi backend; bring it back per
  the protocol above once the box is calm — exact env in
  `tools/quiet_q4_bench.sh:start_server`).
- The residue-round binary (build 11:48, round-3 codex fix in
  metal_server.cpp) has BOTH gate suites ALL PASS on disk:
  `responses-gates-p2fix3.log`, `agentic-gates-p2fix3.log`. No re-run
  needed unless the binary changes again.
- Codex round 3 (per-segment recovery verification) was killed mid-review
  by the 12:00 crash; rerun pending. Rounds 1-2 logs: `codex-resid.log`,
  `codex-resid-round2.log`.
- Q4-port quiet ship bench: watcher re-armed via
  `tools/quiet_q4_bench.sh` + hold file `/tmp/q27-quiet-bench.hold`
  (relaunch the watcher after any reboot — it does not survive one).
