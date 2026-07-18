#!/usr/bin/env python3
"""Self-test for the agentic replay bench pair. Stdlib only, no model,
no network beyond a loopback mock. Proves failure both ways (masked-
failure lesson): the extractor must DROP what it claims to drop, the
bench must price only what it can account, and dead-server / empty-
corpus directions must exit nonzero (real exit codes checked, never
grep for PASS).
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT = os.path.join(HERE, "agentic_replay_extract.py")
BENCH = os.path.join(HERE, "agentic_replay_bench.py")

checks = []


def check(name, ok):
    checks.append((name, ok))
    print("%-58s %s" % (name, "PASS" if ok else "FAIL"))


def run(argv, **kw):
    return subprocess.run([sys.executable] + argv, capture_output=True,
                          text=True, **kw)


# --- synthetic source trace for the extractor --------------------------------

def source_trace(path):
    ev = []
    ev.append({"kind": "boot"})
    # replayable chat turn
    ev.append({"kind": "request", "api": "chat", "id": "c-1",
               "prompt_tokens": 900, "max_tokens": 4096,
               "rendered": "P1 " * 10})
    ev.append({"kind": "outcome", "api": "chat", "id": "c-1",
               "finish": "stop", "output_tokens": 120, "prefix_hit": 800})
    # cancelled turn: dropped
    ev.append({"kind": "request", "api": "chat", "id": "c-2",
               "prompt_tokens": 900, "rendered": "P2"})
    ev.append({"kind": "cancel", "id": "c-2", "phase": "generate"})
    ev.append({"kind": "outcome", "api": "chat", "id": "c-2",
               "finish": "tool_calls", "output_tokens": 50, "prefix_hit": 0})
    # truncated rendered: dropped
    ev.append({"kind": "request", "api": "chat", "id": "c-3",
               "prompt_tokens": 2000,
               "rendered": {"truncated": True, "bytes": 99999, "head": "x"}})
    ev.append({"kind": "outcome", "api": "chat", "id": "c-3",
               "finish": "stop", "output_tokens": 10, "prefix_hit": 0})
    # sub-floor gate fixture: dropped
    ev.append({"kind": "request", "api": "messages", "id": "m-1",
               "prompt_tokens": 200, "rendered": "small"})
    ev.append({"kind": "outcome", "api": "messages", "id": "m-1",
               "finish": "end_turn", "output_tokens": 30, "prefix_hit": 0})
    # boot boundary: same id namespace restarts; dangling request dropped
    ev.append({"kind": "request", "api": "chat", "id": "c-9",
               "prompt_tokens": 900, "rendered": "dangling"})
    ev.append({"kind": "boot"})
    # second replayable turn after reboot, reusing id c-1
    ev.append({"kind": "request", "api": "messages", "id": "c-1",
               "prompt_tokens": 1500, "rendered": "P3"})
    ev.append({"kind": "outcome", "api": "messages", "id": "c-1",
               "finish": "end_turn", "output_tokens": 60, "prefix_hit": 1400})
    # non-replay api: dropped
    ev.append({"kind": "request", "api": "count_tokens", "id": "",
               "prompt_tokens": 5000})
    with open(path, "w") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")


# --- mock SSE completions server ---------------------------------------------

class Mock(BaseHTTPRequestHandler):
    n = 0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Mock.n += 1
        rid = "cmpl-mock-%d" % Mock.n
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.05)  # measurable TTFT
        for i in range(3):
            chunk = {"id": rid, "choices": [{"text": "tok%d " % i,
                                             "finish_reason": None}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.02)
        fin = {"id": rid, "choices": [{"text": "", "finish_reason": "length"}]}
        self.wfile.write(b"data: " + json.dumps(fin).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


def main():
    d = tempfile.mkdtemp(prefix="replay-selftest-")
    src = os.path.join(d, "source-trace.jsonl")
    corpus = os.path.join(d, "corpus.jsonl")
    source_trace(src)

    # extractor: exactly the two replayable turns survive, in order
    r = run([EXTRACT, "--trace", src, "--out", corpus])
    check("extract exits 0", r.returncode == 0)
    items = [json.loads(l) for l in open(corpus)]
    check("extract keeps exactly the 2 replayable turns", len(items) == 2)
    check("extract preserves trace order + fields",
          len(items) == 2 and items[0]["source_id"] == "c-1"
          and items[0]["output_tokens"] == 120
          and items[1]["prompt_tokens"] == 1500)
    check("extract names every drop class",
          all(k in r.stdout for k in
              ("truncated=1", "cancelled=1", "small=1", "no_outcome=1",
               "other_api=1")))

    # extractor must-fail: all turns filtered -> exit 2
    r = run([EXTRACT, "--trace", src, "--out", corpus + ".x",
             "--min-prompt-tokens", "999999"])
    check("extract exits 2 when nothing is replayable", r.returncode == 2)

    # bench against the mock server, exact accounting via a fake
    # server-side trace joined on the mock's response ids
    httpd = HTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d" % httpd.server_address[1]
    strace = os.path.join(d, "server-trace.jsonl")
    with open(strace, "w") as f:
        for i, out in ((1, 120), (2, 60)):
            f.write(json.dumps({"kind": "outcome", "api": "completions",
                                "id": "cmpl-mock-%d" % i,
                                "output_tokens": out,
                                "prefix_hit": 100 * i}) + "\n")
    summary = os.path.join(d, "summary.json")
    r = run([BENCH, corpus, "--base-url", url, "--server-trace", strace,
             "--out", summary])
    check("bench exits 0", r.returncode == 0)
    s = json.load(open(summary))["summary"] if os.path.exists(summary) else {}
    check("bench prices both turns exactly from the trace join",
          s.get("priced_turns") == 2 and s.get("total_output_tokens") == 180)
    check("bench measures a real TTFT (mock delays 50ms)",
          0.01 < s.get("ttft_median_s", 0) < 5.0)
    check("bench headline is tokens/wall",
          s and abs(s["agentic_effective_output_tok_s"]
                    - s["total_output_tokens"] / s["total_wall_s"]) < 1e-9)
    httpd.shutdown()

    # bench must-fail: dead server is run-invalidating, nonzero exit
    r = run([BENCH, corpus, "--base-url", "http://127.0.0.1:9",
             "--timeout", "2"])
    check("bench exits nonzero on a dead server", r.returncode != 0)

    bad = os.path.join(d, "empty.jsonl")
    open(bad, "w").close()
    r = run([BENCH, bad, "--base-url", url])
    check("bench exits nonzero on an empty corpus", r.returncode != 0)

    failed = [n for n, ok in checks if not ok]
    print("\nagentic replay selftest: %s (%d checks)"
          % ("ALL PASS" if not failed else "FAIL: " + ", ".join(failed),
             len(checks)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
