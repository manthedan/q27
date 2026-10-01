#!/usr/bin/env python3
"""Bonsai 2 prefix-snapshot equivalence (needs chunked prefill, T2 packs):
greedy output must not depend on whether the prefix came from a snapshot --
fresh server (no snapshots) vs save-then-hit in one process vs hit after a
restart (disk load). Runs one server at a time, context 4096.

Usage: test_bonsai2_snapshot_reuse.py SERVER MODEL TOKENIZER"""
import json, os, socket, subprocess, sys, tempfile, threading, time, urllib.request

SERVER, MODEL, TOK = sys.argv[1:4]
README = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "README.md")).read()[:9000]
PAYLOAD = {"model": "q27-metal", "system": "You are a concise assistant.\n\n" + README,
           "messages": [{"role": "user", "content": "In two sentences, what is this project?"}],
           "max_tokens": 48, "temperature": 0}


def start(snapdir):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    args = [SERVER, MODEL, TOK, "--host", "127.0.0.1", "--port", str(port), "--ctx", "4096",
            "--slots", "1", "--think-budget", "0"]
    if snapdir:
        args += ["--snapshot-dir", snapdir, "--snapshot-max-mb", "4096"]
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    ready = threading.Event(); lines = []
    def drain():
        for line in p.stderr:
            lines.append(line)
            if "listening on http://" in line: ready.set()
    threading.Thread(target=drain, daemon=True).start()
    if not ready.wait(300): p.kill(); sys.exit("server not ready:\n" + "".join(lines[-20:]))
    return p, port, lines


def ask(port, snapshot):
    body = dict(PAYLOAD, snapshot=True) if snapshot else PAYLOAD
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/messages", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    text = "".join(b.get("text", "") + b.get("thinking", "") for b in d["content"])
    return text, d.get("q27_prefix_hit", 0), round(time.time() - t, 1)


def stop(p):
    p.terminate(); p.wait(30)


p, port, _ = start(None)
fresh = ask(port, False); stop(p)
print("fresh      hit=%d %5.1fs" % (fresh[1], fresh[2]))
with tempfile.TemporaryDirectory() as d:
    os.chmod(d, 0o700)
    p, port, lines = start(d)
    saved = ask(port, True)
    hit = ask(port, False); stop(p)
    print("save       hit=%d %5.1fs" % (saved[1], saved[2]))
    print("same-proc  hit=%d %5.1fs" % (hit[1], hit[2]))
    print("snapshot files:", len(os.listdir(d)))
    p, port, _ = start(d)
    disk = ask(port, False); stop(p)
    print("restart    hit=%d %5.1fs" % (disk[1], disk[2]))
ok = fresh[0] == saved[0] == hit[0] == disk[0] and hit[1] > 0 and disk[1] > 0
print("prompt-tail text:", repr(fresh[0][:160]))
print("RESULT", "PASS" if ok else "FAIL")
if not ok:
    for name, r in (("fresh", fresh), ("saved", saved), ("hit", hit), ("disk", disk)):
        print(name, repr(r[0]))
    sys.exit(1)
