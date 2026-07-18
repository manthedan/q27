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
import stat
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT = os.path.join(HERE, "agentic_replay_extract.py")
BENCH = os.path.join(HERE, "agentic_replay_bench.py")
PREFIX = os.path.join(HERE, "prefix_recurrence.py")

checks = []


def check(name, ok):
    checks.append((name, ok))
    print("%-58s %s" % (name, "PASS" if ok else "FAIL"))


def run(argv, **kw):
    return subprocess.run([sys.executable] + argv, capture_output=True,
                          text=True, **kw)


def mock_runtime():
    return {"identity_schema":2,"server_sha1":"f"*40,"shader_sha1":"a"*40,
        "shader_abi":"// Q27_SHADER_ABI 13",
        "platform":{"sysname":"Darwin","release":"test","machine":"arm64","metal_device":"mock"},
        "protocol":{"context":8192,"kv":"fp16","mtp":0,"suffix":0,"slots":1,
            "prefix_entries":1,"constrain_tools":False,"snapshots":False,
            "snapshot_auto_min":0,"snapshot_max_bytes":0,"max_tokens_default":0,
            "kv_fp16_except":False,"kv_fp16_cell_masks":"0"*32,
            "kv_side_codec":"none","gemm_half":True,"gemm_half_q4":False,
            "gqa_tile":2,"gqa_block":1024,"gqa_threshold":2048,"gpu_sample":True,
            "resident":True,"bare_system":False,"tool_strict":False,
            "test_failpoints":False,"tokenizer":"source.tok","tokenizer_sha1":"b"*40}}


# --- synthetic source trace for the extractor --------------------------------

def source_trace(path):
    ev = []
    source_identity = {"model":"source.q27","artifact_sha1":"e"*40,
                       "runtime":mock_runtime()}
    ev.append({"kind": "boot", "boot_id":"source-boot-1", **source_identity})
    # replayable chat turn
    ev.append({"kind": "request", "api": "chat", "id": "c-1",
               "prompt_tokens": 900, "max_tokens": 4096,
               "sampling": {"temperature": 0.0, "top_p": 0.9, "top_k": 20, "seed": 7},
               "stops": ["<END>"], "tool_names": ["bash"], "snapshot": True,
               "rendered": "P1 " * 10})
    ev.append({"kind": "outcome", "api": "chat", "id": "c-1",
               "finish": "stop", "terminal": "eos",
               "output_tokens": 120, "prefix_hit": 800})
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
    ev.append({"kind": "boot", "boot_id":"source-boot-2", **source_identity})
    # second replayable turn after reboot, reusing id c-1
    ev.append({"kind": "request", "api": "responses", "id": "c-1",
               "prompt_tokens": 1500, "rendered": "P3"})
    ev.append({"kind": "outcome", "api": "responses", "id": "c-1",
               "finish": "end_turn", "terminal": "eos",
               "output_tokens": 60, "prefix_hit": 1400})
    # Capped turn whose API finish is overridden by a parsed tool call. The
    # underlying terminal remains length and must drive replay classification.
    ev.append({"kind": "request", "api": "chat", "id": "c-cap",
               "prompt_tokens": 1600, "max_tokens": 40, "rendered": "CAP TOOL"})
    ev.append({"kind": "outcome", "api": "chat", "id": "c-cap",
               "finish": "tool_calls", "terminal": "length",
               "output_tokens": 40, "prefix_hit": 0})
    # non-replay api: dropped
    ev.append({"kind": "request", "api": "count_tokens", "id": "",
               "prompt_tokens": 5000})
    with open(path, "w") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")


# --- mock SSE completions server ---------------------------------------------

class Mock(BaseHTTPRequestHandler):
    n = 0
    finish_reason = None

    def do_GET(self):
        body = json.dumps({"model":"mock.q27","artifact_sha1":"a"*40,
            "boot_id":"mock-boot","runtime":mock_runtime()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
            # A buffered read(4096) would not return until the turn ends;
            # keep a large post-first-event gap so the TTFT gate detects it.
            time.sleep(0.25 if i == 0 else 0.02)
        finish = Mock.finish_reason or ("length" if body.get("prompt") == "CAP TOOL" else "stop")
        fin = {"id": rid, "choices": [{"text": "", "finish_reason": finish}]}
        self.wfile.write(b"data: " + json.dumps(fin).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


def main():
    d = tempfile.mkdtemp(prefix="replay-selftest-")
    src = os.path.join(d, "source-trace.jsonl")
    corpus = os.path.join(d, "corpus.jsonl")
    source_trace(src)

    # extractor: exactly the three replayable turns survive, in order
    r = run([EXTRACT, "--trace", src, "--out", corpus])
    check("extract exits 0", r.returncode == 0)
    check("extract writes corpus mode 0600",
          os.path.exists(corpus) and stat.S_IMODE(os.stat(corpus).st_mode) == 0o600)
    items = [json.loads(l) for l in open(corpus)]
    check("extract keeps exactly the 3 replayable turns", len(items) == 3)
    check("extract preserves trace order + fields",
          len(items) == 3 and items[0]["source_id"] == "c-1"
          and items[0]["output_tokens"] == 120
          and items[0]["sampling"]["top_k"] == 20
          and items[0]["stops"] == ["<END>"]
          and items[0]["tool_names"] == ["bash"]
          and items[0]["snapshot"] is True
          and items[1]["prompt_tokens"] == 1500
          and items[1]["source_api"] == "responses"
          and items[2]["finish"] == "tool_calls"
          and items[2]["terminal"] == "length")
    check("extract names every drop class",
          all(k in r.stdout for k in
              ("truncated=1", "cancelled=1", "small=1", "no_outcome=1",
               "other_api=1")))
    # Explicit zero is a valid immediate-EOS turn; a missing accounting field
    # is malformed evidence and must not be inferred as zero.
    ztrace,zout=os.path.join(d,"zero-trace.jsonl"),os.path.join(d,"zero.jsonl")
    zid={"model":"source.q27","artifact_sha1":"e"*40,"runtime":mock_runtime()}
    zevents=[{"kind":"boot","boot_id":"zboot",**zid},
        {"kind":"request","api":"chat","id":"z0","prompt_tokens":900,"rendered":"zero"},
        {"kind":"outcome","api":"chat","id":"z0","terminal":"eos","output_tokens":0},
        {"kind":"request","api":"chat","id":"missing","prompt_tokens":900,"rendered":"missing"},
        {"kind":"outcome","api":"chat","id":"missing","terminal":"eos"}]
    with open(ztrace,"w") as f:
        for e in zevents: f.write(json.dumps(e)+"\n")
    zr=run([EXTRACT,"--trace",ztrace,"--out",zout])
    zitems=[json.loads(l) for l in open(zout)] if os.path.exists(zout) else []
    check("extract keeps explicit zero output and rejects missing accounting",
          zr.returncode==0 and len(zitems)==1 and zitems[0]["output_tokens"]==0)
    mixed = os.path.join(d, "mixed-boot-trace.jsonl")
    mixed_events = [json.loads(l) for l in open(src)]
    boot_n = 0
    for e in mixed_events:
        if e.get("kind") == "boot":
            boot_n += 1
            if boot_n == 2:
                e["artifact_sha1"] = "9" * 40
    with open(mixed, "w") as f:
        for e in mixed_events: f.write(json.dumps(e) + "\n")
    r = run([EXTRACT, "--trace", mixed, "--out", corpus + ".mixed"])
    check("extract rejects incompatible source boots", r.returncode != 0)

    # Prefix recurrence must retain same IDs across boots and exclude the
    # empty-id count_tokens request from the generation population.
    r = run([PREFIX, src, "--since", "999999"])
    check("prefix recurrence scopes IDs and excludes cancelled/dangling/count_tokens",
          r.returncode == 0 and "5 requests across" in r.stdout)
    # Partial hits pay only their uncovered suffix; a recurrence is savable
    # only when its previous occurrence paid prefill work.
    recur = os.path.join(d, "recurrence-trace.jsonl")
    rev = [
        {"kind":"boot"},
        {"kind":"request","api":"chat","id":"r1","ts":10,"prompt_tokens":1000,
         "token_head":list(range(1000)),"rendered":"same"},
        {"kind":"prefix","id":"r1","ts":10,"tier":"cold","hit":0},
        {"kind":"outcome","api":"chat","id":"r1","ts":11,"output_tokens":1},
        {"kind":"request","api":"chat","id":"r2","ts":20,"prompt_tokens":1200,
         "token_head":list(range(1000))+list(range(5000,5200)),"rendered":"same"},
        {"kind":"prefix","id":"r2","ts":20,"tier":"memory","hit":800},
        {"kind":"outcome","api":"chat","id":"r2","ts":21,"output_tokens":1},
        {"kind":"request","api":"chat","id":"r3","ts":30,"prompt_tokens":1300,
         "token_head":list(range(1000))+list(range(7000,7300)),"rendered":"same"},
        {"kind":"prefix","id":"r3","ts":30,"tier":"memory","hit":1300},
        {"kind":"outcome","api":"chat","id":"r3","ts":31,"output_tokens":1},
    ]
    with open(recur, "w") as f:
        for e in rev:
            f.write(json.dumps(e) + "\n")
    r = run([PREFIX, recur, "--since", "999999", "--gap", "60"])
    check("prefix recurrence prices partial misses and prior paid state",
          r.returncode == 0 and
          "prefill paid (uncovered suffixes): 1,400 tokens; of that, 200" in r.stdout)

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
        for i, out, terminal in ((1, 120, "eos"), (2, 60, "eos"),
                                 (3, 40, "length")):
            f.write(json.dumps({"kind": "outcome", "api": "completions",
                                "id": "cmpl-mock-%d" % i,
                                "output_tokens": out,
                                "prefix_hit": 100 * i,
                                "terminal": terminal}) + "\n")
    summary = os.path.join(d, "summary.json")
    r = run([BENCH, corpus, "--base-url", url, "--server-trace", strace,
             "--out", summary])
    check("bench exits 0", r.returncode == 0)
    s = json.load(open(summary))["summary"] if os.path.exists(summary) else {}
    check("bench prices all turns exactly from the trace join",
          s.get("priced_turns") == 3 and s.get("total_output_tokens") == 220)
    check("bench measures first content event, not completion wall",
          0.01 < s.get("ttft_median_s", 0) < 0.20)
    check("bench headline is tokens/wall",
          s and abs(s["agentic_effective_output_tok_s"]
                    - s["total_output_tokens"] / s["total_wall_s"]) < 1e-9)
    Mock.finish_reason = "length"
    r = run([BENCH, corpus, "--base-url", url, "--server-trace", strace,
             "--limit", "1"])
    check("bench rejects replay/source finish mismatch", r.returncode != 0)
    Mock.finish_reason = None

    # bench must-fail: dead server is run-invalidating, nonzero exit
    r = run([BENCH, corpus, "--base-url", "http://127.0.0.1:9",
             "--server-trace", strace, "--timeout", "2"])
    check("bench exits nonzero on a dead server", r.returncode != 0)

    bad = os.path.join(d, "empty.jsonl")
    open(bad, "w").close()
    r = run([BENCH, bad, "--base-url", url, "--server-trace", strace])
    check("bench exits nonzero on an empty corpus", r.returncode != 0)
    r = run([BENCH, corpus, "--base-url", "http://example.com:80",
             "--server-trace", strace])
    check("bench rejects non-loopback replay destinations", r.returncode != 0)
    httpd.shutdown()

    failed = [n for n, ok in checks if not ok]
    print("\nagentic replay selftest: %s (%d checks)"
          % ("ALL PASS" if not failed else "FAIL: " + ", ".join(failed),
             len(checks)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
