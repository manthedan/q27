#!/usr/bin/env python3
"""Multislot Phase 1 gates G1/G2/G3/G5 (docs/plans/2026-07-15-multislot-phase1.md).

Starts the Metal server with --slots 2 and checks:
  G2  determinism: the same greedy request returns byte-identical text when
      the server is idle (96-wide prefill quanta) and when the other slot is
      busy (12/48-wide quanta) — the width policy must be quality-neutral
      end to end, and quantum-split decode must equal whole-run decode.
  G1  isolation: two different requests running concurrently each return
      exactly the text they return solo.
  G3  cancel: a client that disconnects mid-stream leaves no state behind —
      the same request re-run afterwards returns the solo text.
  G5  wait honesty: /stats reports nonzero busy-arrival gate waits after the
      concurrency phase, and every request was admitted.

Run once with the greedy server and once with --mtp 4 (arg: mtp).
"""
import http.client
import json
import subprocess
import sys
import threading
import time

HOST, PORT = "127.0.0.1", 8123
MODEL = "models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27"
TOK = "models/qwen36-27b-mtp/qwen36-27b-mtp.tok"

PROMPT_A = ("The measurement discipline that keeps a kernel project honest is simple to state "
            "and hard to keep: pre-register the kill criterion, pin every environment knob, "
            "rebuild the artifact binaries, and only then run the A/B. The first corollary is")
PROMPT_B = ("A scheduling quantum is the unit of GPU work between opportunities to yield. "
            "If the quantum is a whole prefill, a concurrent decode stream waits seconds; "
            "if it is one chunk, the wait is bounded by")


def request(prompt, n=64, timeout=600, **extra):
    c = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    body = json.dumps({"prompt": prompt, "max_tokens": n, **extra})
    c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    data = json.loads(r.read())
    c.close()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {data}")
    return data["choices"][0]["text"]


def stream_then_cancel(prompt, pieces=3):
    """Read a few SSE chunks then drop the connection mid-generation."""
    c = http.client.HTTPConnection(HOST, PORT, timeout=600)
    body = json.dumps({"prompt": prompt, "max_tokens": 64, "stream": True})
    c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    got = 0
    while got < pieces:
        if not r.read1(256):
            break
        got += 1
    c.sock.close()  # hard disconnect mid-stream


def stats():
    c = http.client.HTTPConnection(HOST, PORT, timeout=30)
    c.request("GET", "/stats")
    data = json.loads(c.getresponse().read())
    c.close()
    return data


def wait_ready(proc, deadline=180):
    start = time.time()
    while time.time() - start < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited early")
        try:
            c = http.client.HTTPConnection(HOST, PORT, timeout=2)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                c.close()
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError("server did not become ready")


def main():
    mtp = len(sys.argv) > 1 and sys.argv[1] == "mtp"
    cmd = ["build/q27-metal-server", MODEL, TOK, "--port", str(PORT),
           "--ctx", "2048", "--slots", "2"]
    if mtp:
        cmd += ["--mtp", "4"]
    label = "mtp4" if mtp else "greedy"
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    failures = []
    try:
        wait_ready(proc)

        # Solo references (idle server, widest prefill quanta).
        solo_a = request(PROMPT_A)
        solo_b = request(PROMPT_B)
        # Repeat solo A: exercises the prefix-cache restore path.
        if request(PROMPT_A) != solo_a:
            failures.append("G2: solo repeat (prefix-cache restore) diverged")

        # G1: concurrent A+B, three rounds; every result must equal its solo text.
        for round_no in range(3):
            results = {}
            def run(name, prompt):
                try:
                    results[name] = request(prompt)
                except Exception as e:  # noqa: BLE001
                    results[name] = f"ERROR: {e}"
            ta = threading.Thread(target=run, args=("a", PROMPT_A))
            tb = threading.Thread(target=run, args=("b", PROMPT_B))
            ta.start(); tb.start(); ta.join(); tb.join()
            if results["a"] != solo_a:
                failures.append(f"G1 round {round_no}: A diverged under concurrency")
            if results["b"] != solo_b:
                failures.append(f"G1 round {round_no}: B diverged under concurrency")

        # G1s: the sampled quantum loop (fixed seed => deterministic) must
        # also be concurrency-invariant.
        sampled_kw = {"temperature": 0.8, "top_k": 40, "seed": 12345}
        solo_s = request(PROMPT_A, **sampled_kw)
        results = {}
        def run_s(name, prompt, kw):
            try:
                results[name] = request(prompt, **kw)
            except Exception as e:  # noqa: BLE001
                results[name] = f"ERROR: {e}"
        ts = threading.Thread(target=run_s, args=("s", PROMPT_A, sampled_kw))
        tb2 = threading.Thread(target=run_s, args=("b", PROMPT_B, {}))
        ts.start(); tb2.start(); ts.join(); tb2.join()
        if results["s"] != solo_s:
            failures.append("G1s: sampled request diverged under concurrency")
        if results["b"] != solo_b:
            failures.append("G1s: greedy companion diverged under concurrency")

        # Gq: queue pressure — five concurrent requests over two slots force
        # ticketed waiters; all must complete (a wrong-waiter wakeup wedges
        # this with an idle slot) and repeats must match solo.
        results = {}
        def run_q(i):
            prompt = PROMPT_A if i % 2 == 0 else PROMPT_B
            try:
                results[i] = (prompt, request(prompt, n=16))
            except Exception as e:  # noqa: BLE001
                results[i] = (prompt, f"ERROR: {e}")
        solo_a16 = request(PROMPT_A, n=16)
        solo_b16 = request(PROMPT_B, n=16)
        threads = [threading.Thread(target=run_q, args=(i,), daemon=True) for i in range(5)]
        for t in threads: t.start()
        # Shared deadline: on the wedge this gate targets, sequential
        # per-thread joins would burn 120 s each and later gates would then
        # stall on their own HTTP timeouts — fail fast and skip them.
        deadline = time.time() + 120
        for t in threads:
            t.join(max(0.0, deadline - time.time()))
        wedged = any(t.is_alive() for t in threads)
        if wedged:
            failures.append("Gq: queued request wedged (still blocked at the shared 120 s deadline); "
                            "skipping remaining gates")
        for i, (prompt, text) in sorted(results.items()):
            want = solo_a16 if prompt is PROMPT_A else solo_b16
            if text != want:
                failures.append(f"Gq: request {i} diverged or errored under queue pressure")
        if wedged:
            print(f"[{label}] FAIL:")
            for f in failures:
                print(f"  - {f}")
            return 1

        # G3: cancel mid-stream, then the same request must match solo.
        stream_then_cancel(PROMPT_B)
        time.sleep(1)
        if request(PROMPT_B) != solo_b:
            failures.append("G3: post-cancel rerun diverged")

        # G5: the concurrency phase must have produced busy-arrival waits.
        s = stats()
        busy = {k: v for k, v in s.get("gate_wait_by_arrival", {}).items() if k != "idle"}
        if s.get("slots") != 2:
            failures.append(f"G5: expected 2 slots, got {s.get('slots')}")
        if not busy:
            failures.append("G5: no busy-arrival bucket recorded despite concurrent load")
        # The Phase 1 guarantee: at most one active scheduling quantum of
        # GATE wait (slot admission -> first lease). The widest quantum is a
        # 96-token prefill chunk (~2 s); 3 s allows thermal margin. The
        # pre-ticket-lock lease measured 5491 ms here (starved behind a
        # whole generation), so this bound is proven able to fail. QUEUE
        # wait (arrival -> admission) is a different quantity — bounded by
        # QUEUE_MAX generations, reported but not bounded here.
        for phase, ws in busy.items():
            if ws["max_ms"] > 3000:
                failures.append(f"G5: {phase}-arrival max gate wait "
                                f"{ws['max_ms']:.0f} ms exceeds one quantum bound")
        print(f"[{label}] /stats: {json.dumps(s)}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)

    if failures:
        print(f"[{label}] FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"[{label}] G1/G2/G3/G5 PASS (solo texts {len(solo_a)}/{len(solo_b)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
