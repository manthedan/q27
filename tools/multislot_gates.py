#!/usr/bin/env python3
"""Multislot Phase 1 gates G1/G2/G3/G5 (docs/metal/plans/2026-07-15-multislot-phase1.md).

Starts the Metal server with --slots 2 and checks:
  G2  determinism: the same greedy request returns byte-identical text when
      the server is idle (96-wide prefill quanta) and when the other slot is
      busy (12/48-wide quanta) — the width policy must be quality-neutral
      end to end, and quantum-split decode must equal whole-run decode.
  G1  isolation: two different requests running concurrently each return
      exactly the text they return solo.
  G3  cancel: a client that disconnects mid-stream leaves no state behind —
      the same request re-run afterwards returns the solo text.
  G4  constraint isolation (greedy config only): a grammar-constrained tool
      request on one slot runs concurrently with a plain request on the
      other; the plain text must equal solo (a leaked mask zeroes almost the
      whole vocabulary — divergence would be dramatic), the constrained text
      must equal ITS solo run, and constraint action is proven, not assumed:
      the declared tool name is one the model never produces naturally, so
      it can only appear in the output if the mask steered decoding
      (vacuous-gate lesson). Server stderr must show the engage/close pairs.
  G5  wait honesty: /stats reports nonzero busy-arrival gate waits after the
      concurrency phase, and every request was admitted.
  G6  admission (greedy config only; own server launch): with
      Q27_METAL_BUDGET_MB sized to fit exactly one slot, /stats must report
      slots == 1 (the degradation actually fired — the default suite asserts
      slots == 2, so both directions of the admission decision are
      exercised), and saturating the single slot's ticket queue must produce
      at least one HTTP 503 with the documented overloaded_error body while
      every other request completes identically (no wedge, shared deadline).

Run once with the greedy server and once with --mtp 4 (arg: mtp).
"""
import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

HOST, PORT = "127.0.0.1", 8123
# Official-tier runs override via env (the MTP gates' real target artifact);
# defaults stay T2 so the mini's rig is unchanged.
MODEL = os.environ.get("Q27_GATE_MODEL", "models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27")
TOK = os.environ.get("Q27_GATE_TOK", "models/qwen36-27b-mtp/qwen36-27b-mtp.tok")

PROMPT_A = ("The measurement discipline that keeps a kernel project honest is simple to state "
            "and hard to keep: pre-register the kill criterion, pin every environment knob, "
            "rebuild the artifact binaries, and only then run the A/B. The first corollary is")
PROMPT_B = ("A scheduling quantum is the unit of GPU work between opportunities to yield. "
            "If the quantum is a whole prefill, a concurrent decode stream waits seconds; "
            "if it is one chunk, the wait is bounded by")

# G4: the few-shot demonstrates get_weather, but the DECLARED tool is a name
# the model would never produce on its own — it can only appear via the mask.
PROMPT_TOOL = ("You are an assistant with one tool.\n"
               "User: What is the weather in Berlin?\n"
               'Assistant: <tool_call>{"name": "get_weather", "arguments": '
               '{"city": "Berlin"}}</tool_call>\n'
               "User: What is the weather in Paris?\n"
               "Assistant:")
TOOL_NAME = "zz_paris_weather_probe"
TOOLS = [{"type": "function", "function": {"name": TOOL_NAME}}]


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


def request_raw(prompt, n=64, timeout=600, port=PORT, **extra):
    """Like request() but returns (status, parsed_body) without raising."""
    c = http.client.HTTPConnection(HOST, port, timeout=timeout)
    body = json.dumps({"prompt": prompt, "max_tokens": n, **extra})
    c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    data = json.loads(r.read())
    c.close()
    return r.status, data


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


def wait_ready(proc, deadline=180, port=PORT):
    start = time.time()
    while time.time() - start < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited early")
        try:
            c = http.client.HTTPConnection(HOST, port, timeout=2)
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
    else:
        # G4 needs the constrainer; it only affects requests that pass tools,
        # so the other gates run unchanged. (--constrain-tools rejects --mtp.)
        cmd += ["--constrain-tools"]
    label = "mtp4" if mtp else "greedy"
    stderr_file = tempfile.NamedTemporaryFile(prefix="multislot_gates_", suffix=".log")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_file)
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
        deadline = time.monotonic() + 120
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
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

        # G4: constrained tool decode on one slot, plain decode on the other.
        if not mtp:
            solo_t = request(PROMPT_TOOL, n=48, tools=TOOLS)
            if f'"{TOOL_NAME}"' not in solo_t:
                failures.append("G4: declared tool name absent from solo constrained "
                                "output — mask never steered decoding (vacuous)")
            for round_no in range(2):
                results = {}
                def run_t(name, prompt, kw):
                    try:
                        results[name] = request(prompt, **kw)
                    except Exception as e:  # noqa: BLE001
                        results[name] = f"ERROR: {e}"
                tt = threading.Thread(target=run_t, args=("t", PROMPT_TOOL,
                                                          {"n": 48, "tools": TOOLS}))
                tp = threading.Thread(target=run_t, args=("b", PROMPT_B, {}))
                tt.start(); tp.start(); tt.join(); tp.join()
                if results["t"] != solo_t:
                    failures.append(f"G4 round {round_no}: constrained request diverged "
                                    "under concurrency")
                if results["b"] != solo_b:
                    failures.append(f"G4 round {round_no}: plain request diverged beside "
                                    "a constrained slot (mask leak class)")
            stderr_file.flush()
            srv_log = open(stderr_file.name, errors="replace").read()
            engaged = srv_log.count("[toolgram] engaged")
            closed = srv_log.count("[toolgram] call closed")
            disengaged = srv_log.count("[toolgram] disengaged")
            if engaged < 3 or closed < 3:
                failures.append(f"G4: expected >=3 engage/close pairs in server stderr, "
                                f"saw engaged={engaged} closed={closed}")
            if disengaged:
                failures.append(f"G4: {disengaged} grammar disengage(s) — constraint "
                                "broke mid-call")

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
        # The quantum bound is tier- and mode-dependent: greedy chunked
        # prefill's widest quantum is a 96-token chunk (~2 s on T2; 3 s with
        # margin). MTP prompt warming is token-serial and deliberately ONE
        # quantum for the whole suffix (documented Phase 1 limitation), so on
        # the official tier (~3.6 tok/s serial ingest, ~150-token prompts) a
        # legitimate quantum reaches ~40 s; 45 s bounds it while still
        # failing the starvation class (waiting behind a whole generation:
        # ingestion + 96 decode tokens ≈ 70 s+). Official-tier MTP measured
        # 11.9 s max here (2026-07-16).
        # Default stays the strict T2 chunk bound in BOTH modes (codex P2:
        # a mode-keyed 45 s would mask T2 starvation on the default rig);
        # official-tier MTP invocations pass Q27_GATE_QUANTUM_MS=45000.
        quantum_bound_ms = int(os.environ.get("Q27_GATE_QUANTUM_MS", "3000"))
        for phase, ws in busy.items():
            if ws["max_ms"] > quantum_bound_ms:
                failures.append(f"G5: {phase}-arrival max gate wait "
                                f"{ws['max_ms']:.0f} ms exceeds one quantum bound "
                                f"({quantum_bound_ms} ms)")
        # Nonzero-speculation assert (external-review adoption, 2026-07-16):
        # byte-identity under --mtp means nothing if speculation never
        # engaged. committed > rounds is unfakeable — a serial fallback
        # commits exactly one token per round.
        if mtp:
            spec = s.get("speculation", {})
            if not spec.get("rounds"):
                failures.append("MTP: zero speculation rounds — the gate ran vacuously")
            elif spec.get("committed", 0) <= spec["rounds"]:
                failures.append(f"MTP: {spec['committed']} committed over {spec['rounds']} "
                                "rounds — no drafts ever accepted, gate vacuous")
        print(f"[{label}] /stats: {json.dumps(s)}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        stderr_file.close()

    if failures:
        print(f"[{label}] FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    if not mtp and run_g6() != 0:
        return 1
    if not mtp and run_g7() != 0:
        return 1
    gates = "G1/G2/G3/G5" if mtp else "G1/G2/G3/G4/G5/G6/G7"
    print(f"[{label}] {gates} PASS (solo texts {len(solo_a)}/{len(solo_b)} chars)")
    return 0


def run_g6():
    """Admission gate: budget fits exactly one slot; queue overflow 503s."""
    env = dict(os.environ, Q27_METAL_BUDGET_MB="1000")
    cmd = ["build/q27-metal-server", MODEL, TOK, "--port", str(PORT + 1),
           "--ctx", "2048", "--slots", "2"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env=env)
    failures = []
    try:
        wait_ready(proc, port=PORT + 1)
        c = http.client.HTTPConnection(HOST, PORT + 1, timeout=30)
        c.request("GET", "/stats")
        slots = json.loads(c.getresponse().read()).get("slots")
        c.close()
        if slots != 1:
            failures.append(f"G6: budget for one slot admitted {slots} slots "
                            "(degradation never fired — vacuous)")
        # Saturate: 1 running + QUEUE_MAX=8 tickets; the overflow must 503.
        results = {}
        def run_q(i):
            try:
                results[i] = request_raw(PROMPT_A, n=32, port=PORT + 1)
            except Exception as e:  # noqa: BLE001
                results[i] = (0, {"error": {"message": str(e), "type": "transport"}})
        threads = [threading.Thread(target=run_q, args=(i,), daemon=True)
                   for i in range(12)]
        for t in threads: t.start()
        deadline = time.monotonic() + 180
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
        if any(t.is_alive() for t in threads):
            failures.append("G6: request wedged at the shared deadline")
        ok = [d for st, d in results.values() if st == 200]
        overloaded = [d for st, d in results.values()
                      if st == 503 and d.get("error", {}).get("type") == "overloaded_error"]
        other = [(st, d) for st, d in results.values()
                 if st not in (200, 503)]
        if not overloaded:
            failures.append("G6: no 503 overloaded_error despite 12-over-(1 slot + 8 tickets)")
        if not ok:
            failures.append("G6: no request completed normally under overload")
        if other:
            failures.append(f"G6: unexpected statuses {[st for st, _ in other]}")
        texts = {d["choices"][0]["text"] for d in ok}
        if len(texts) > 1:
            failures.append("G6: successful responses diverged under overload (greedy determinism)")
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    if failures:
        print("[greedy] G6 FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"[greedy] G6 PASS (slots=1 under 1000 MB budget, "
          f"{len(overloaded)} x 503 overloaded_error, {len(ok)} completed)")
    return 0


def run_g7():
    """Disk prefix snapshots (Phase 2): a hinted request persists a boundary
    snapshot; a fresh server process serves a shared-prefix request from
    disk byte-identically to a cold server; a prompt perturbed inside the
    prefix misses; a tiny budget evicts. One long haystack, two questions."""
    import shutil
    import tempfile

    haystack = " ".join(
        f"Fact {i}: the {w} subsystem reports nominal telemetry on channel {i * 7 % 31}."
        for i, w in enumerate(
            ("attention gdn ffn embed logits sampler tokenizer scheduler cache "
             "paging residency lease quantum ticket snapshot census roofline "
             "envelope oracle verify draft commit replay straddle boundary "
             "prefill decode stream cancel budget admission eviction").split()))
    q1 = haystack + " QUESTION: which channel does the attention subsystem use?"
    q2 = haystack + " QUESTION: which subsystem reports on channel zero?"
    q_miss = "Fact 0X:" + q2[8:]   # perturbed INSIDE the snapshotted prefix

    port = PORT + 2
    failures = []
    snapdir = tempfile.mkdtemp(prefix="q27_g7_snap.")
    colddir = tempfile.mkdtemp(prefix="q27_g7_cold.")

    def launch(env_extra):
        env = dict(os.environ, **env_extra)
        proc = subprocess.Popen(
            ["build/q27-metal-server", MODEL, TOK, "--port", str(port), "--ctx", "2048"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        wait_ready(proc, port=port)
        return proc

    def snap_stats():
        c = http.client.HTTPConnection(HOST, port, timeout=30)
        c.request("GET", "/stats")
        data = json.loads(c.getresponse().read())
        c.close()
        return data.get("snapshots", {})

    def ask(prompt, **extra):
        st, data = request_raw(prompt, n=24, port=port, **extra)
        if st != 200:
            raise RuntimeError(f"G7 request failed: HTTP {st}: {data}")
        return data["choices"][0]["text"]

    try:
        # Server 1: hinted request saves a boundary snapshot.
        proc = launch({"Q27_METAL_SNAPSHOT_DIR": snapdir})
        try:
            ask(q1, snapshot=True)
            s = snap_stats()
            files = [f for f in os.listdir(snapdir) if f.endswith(".q27snap")]
            if s.get("disk_saves") != 1 or len(files) != 1:
                failures.append(f"G7: hinted save did not persist (saves={s.get('disk_saves')}, files={files})")
        finally:
            proc.terminate(); proc.wait(timeout=30)
        # Server 2: FRESH process (empty in-memory cache), same dir — the
        # shared-prefix question must hit disk; the perturbed prompt must miss.
        hit_text = miss_text = None
        proc = launch({"Q27_METAL_SNAPSHOT_DIR": snapdir})
        try:
            hit_text = ask(q2)
            s = snap_stats()
            if s.get("disk_hits") != 1:
                failures.append(f"G7: shared-prefix request did not hit disk (hits={s.get('disk_hits')})")
            miss_text = ask(q_miss)
            s = snap_stats()
            if s.get("disk_hits") != 1:
                failures.append("G7: perturbed-prefix request falsely HIT the snapshot")
        finally:
            proc.terminate(); proc.wait(timeout=30)
        # Server 3: cold reference for the same question (separate empty dir).
        proc = launch({"Q27_METAL_SNAPSHOT_DIR": colddir})
        try:
            cold_text = ask(q2)
            cold_miss = ask(q_miss)
        finally:
            proc.terminate(); proc.wait(timeout=30)
        if hit_text != cold_text:
            failures.append("G7: disk-hit continuation diverged from the cold path "
                            f"(hit {hit_text!r} vs cold {cold_text!r})")
        if miss_text != cold_miss:
            failures.append("G7: perturbed-prefix (miss) continuation diverged from cold")
        # Server 4: eviction under a 1 MB budget — the save happens, then the
        # directory is brought back under budget (the file is far larger).
        evdir = tempfile.mkdtemp(prefix="q27_g7_evict.")
        proc = launch({"Q27_METAL_SNAPSHOT_DIR": evdir, "Q27_METAL_SNAPSHOT_MAX_MB": "1"})
        try:
            ask(q1, snapshot=True)
            total = sum(os.path.getsize(os.path.join(evdir, f)) for f in os.listdir(evdir)
                        if f.endswith(".q27snap"))
            if total > 1 * 1024 * 1024:
                failures.append(f"G7: eviction left {total} bytes under a 1 MB budget")
        finally:
            proc.terminate(); proc.wait(timeout=30)
            shutil.rmtree(evdir, ignore_errors=True)
    finally:
        shutil.rmtree(snapdir, ignore_errors=True)
        shutil.rmtree(colddir, ignore_errors=True)
    if failures:
        print("[greedy] G7 FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[greedy] G7 PASS (hinted save persisted, fresh-process disk hit byte-identical "
          "to cold, perturbed prefix missed, 1 MB budget evicted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
