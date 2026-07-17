#!/usr/bin/env python3
"""Self-test for gen_runner.py against a stdlib mock server. No model, no GPU.

Checks real exit codes and output contents, including the must-fail
direction (dead server -> nonzero exit, error rows)."""
import json, os, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "gen_runner.py")


class Mock(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        if self.path == "/v1/messages":
            payload = {"content": [{"type": "text", "text": f"echo:{prompt}"}]}
        elif self.path == "/v1/chat/completions":
            payload = {"choices": [{"message": {"content": f"echo:{prompt}"}}]}
        else:
            self.send_error(404)
            return
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def run(prompts_path, out_path, base_url, api):
    return subprocess.run(
        [sys.executable, RUNNER, prompts_path, "--out", out_path,
         "--base-url", base_url, "--api", api, "--timeout", "5"],
        capture_output=True, text=True)


def main():
    fails = []
    srv = HTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    with tempfile.TemporaryDirectory() as td:
        prompts = os.path.join(td, "p.jsonl")
        with open(prompts, "w") as f:
            f.write(json.dumps({"prompt_id": "p0", "prompt": "alpha"}) + "\n")
            f.write(json.dumps({"prompt_id": "p1", "prompt": "beta"}) + "\n")

        for api in ("anthropic", "openai"):
            out = os.path.join(td, f"out_{api}.jsonl")
            r = run(prompts, out, base, api)
            if r.returncode != 0:
                fails.append(f"{api}: exit {r.returncode}, stderr: {r.stderr}")
                continue
            rows = [json.loads(l) for l in open(out)]
            if [x["text"] for x in rows] != ["echo:alpha", "echo:beta"]:
                fails.append(f"{api}: wrong texts {rows}")
            if [x["prompt_id"] for x in rows] != ["p0", "p1"]:
                fails.append(f"{api}: wrong prompt_ids")

        # must-fail: dead server -> exit 1, rows carry error + empty text
        out = os.path.join(td, "out_dead.jsonl")
        r = run(prompts, out, "http://127.0.0.1:9", "anthropic")
        if r.returncode == 0:
            fails.append("dead server: expected nonzero exit, got 0")
        else:
            rows = [json.loads(l) for l in open(out)]
            if len(rows) != 2 or any("error" not in x or x["text"] != ""
                                     for x in rows):
                fails.append(f"dead server: bad rows {rows}")

        # must-fail: duplicate prompt_id -> refused
        dup = os.path.join(td, "dup.jsonl")
        with open(dup, "w") as f:
            f.write(json.dumps({"prompt_id": "p0", "prompt": "x"}) + "\n")
            f.write(json.dumps({"prompt_id": "p0", "prompt": "y"}) + "\n")
        r = run(dup, os.path.join(td, "out_dup.jsonl"), base, "anthropic")
        if r.returncode == 0:
            fails.append("duplicate prompt_id: expected refusal, got exit 0")

    srv.shutdown()
    if fails:
        print("selftest_gen FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("selftest_gen: all checks passed (both APIs, dead-server and "
          "duplicate-id must-fail directions)")


if __name__ == "__main__":
    main()
