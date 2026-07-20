#!/usr/bin/env python3
"""Self-test for agent_runner.py against a stdlib mock /v1/messages server.
No model, no GPU. Exercises the tool loop end-to-end (the mock drives real
tool calls through the sandbox) plus the failure-mode classifications and
sandbox path-escape refusal."""
import json, os, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "agent_runner.py")

ISSUE = "add the missing constant"


class Mock(BaseHTTPRequestHandler):
    """Scripted two-step conversation: turn 1 reads the buggy file, turn 2
    edits it, turn 3 replies with no tool calls (done). The 'gold' file is
    src/fixme.py; the edit targets it, so gold_hit must come out true."""
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = len([m for m in body["messages"] if m["role"] == "assistant"])
        if self.path != "/v1/messages":
            self.send_error(404); return
        if n == 0:
            content = [{"type": "tool_use", "id": "t1", "name": "read_file",
                        "input": {"path": "src/fixme.py"}}]
            sr = "tool_use"
        elif n == 1:
            content = [{"type": "tool_use", "id": "t2", "name": "edit_file",
                        "input": {"path": "src/fixme.py", "old": "X = 1",
                                  "new": "X = 2"}}]
            sr = "tool_use"
        else:
            content = [{"type": "text", "text": "Fixed the constant."}]
            sr = "end_turn"
        payload = {"stop_reason": sr, "content": content,
                   "usage": {"input_tokens": 100, "output_tokens": 25}}
        data = json.dumps(payload).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)
    def log_message(self, *a):
        pass


class TruncMock(BaseHTTPRequestHandler):
    """Always truncates (max_tokens) and never calls a tool -> the instance
    must classify failure_mode='truncated' and gold_hit=False."""
    def do_POST(self):
        payload = {"stop_reason": "max_tokens",
                   "content": [{"type": "thinking", "thinking": "ran out"}],
                   "usage": {"input_tokens": 50, "output_tokens": 8192}}
        data = json.dumps(payload).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)
    def log_message(self, *a):
        pass


def make_repo(root, iid="inst1"):
    repo = os.path.join(root, iid, "repo")
    os.makedirs(os.path.join(repo, "src"))
    with open(os.path.join(repo, "src", "fixme.py"), "w") as f:
        f.write("X = 1\n")
    subprocess.run(["git", "-C", repo, "init", "-q"], check=True)
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True)
    subprocess.run(["git", "-C", repo, "-c", "user.email=t@t", "-c",
                    "user.name=t", "commit", "-qm", "init"], check=True)
    return repo


def manifest(path, gold, iid="inst1"):
    json.dump([{"instance_id": iid, "repo": "x/y", "base_commit": "z",
                "difficulty": "<15 min fix", "problem_statement": ISSUE,
                "gold_files": gold}], open(path, "w"))


def main():
    fails = []
    with tempfile.TemporaryDirectory() as td:
        make_repo(td)
        man = os.path.join(td, "m.json"); manifest(man, ["src/fixme.py"])

        # happy path: loop runs, edit lands, gold_hit true, end_turn
        srv = HTTPServer(("127.0.0.1", 0), Mock)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        out = os.path.join(td, "ok.jsonl")
        r = subprocess.run([sys.executable, RUNNER, "--manifest", man, "--work", td,
                            "--out", out, "--base-url",
                            f"http://127.0.0.1:{srv.server_address[1]}",
                            "--max-turns", "20", "--timeout", "5"],
                           capture_output=True, text=True)
        srv.shutdown()
        if r.returncode != 0:
            fails.append(f"happy path exit {r.returncode}: {r.stderr}")
        else:
            row = json.loads(open(out).read())
            if not row["gold_hit"] or row["failure_mode"] != "hit":
                fails.append(f"happy path: expected hit, got {row}")
            if row["files_changed"] != ["src/fixme.py"]:
                fails.append(f"happy path: wrong files {row['files_changed']}")
            if row["turns"] != 3 or row["stop_reasons"][-1] != "end_turn":
                fails.append(f"happy path: wrong turns/stops {row}")
            # the edit must actually be on disk in the repo
            if open(os.path.join(td, "inst1", "repo", "src", "fixme.py")).read() != "X = 2\n":
                fails.append("happy path: edit did not land in repo")

        # truncated: max_tokens with no tool call -> failure_mode truncated.
        # Separate instance id + fresh repo so leftover edits from the happy
        # path cannot make gold_hit spuriously true.
        make_repo(td, "instT")
        manT = os.path.join(td, "mT.json"); manifest(manT, ["src/fixme.py"], "instT")
        srv = HTTPServer(("127.0.0.1", 0), TruncMock)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        out = os.path.join(td, "tr.jsonl")
        r = subprocess.run([sys.executable, RUNNER, "--manifest", manT, "--work", td,
                            "--out", out, "--base-url",
                            f"http://127.0.0.1:{srv.server_address[1]}",
                            "--max-turns", "3", "--timeout", "5"],
                           capture_output=True, text=True)
        srv.shutdown()
        row = json.loads(open(out).read())
        if row["failure_mode"] != "truncated" or row["gold_hit"]:
            fails.append(f"truncated: wrong classification {row}")
        if row["excl_turns"] < 1:
            fails.append(f"truncated: excl not counted {row}")

        # must-fail: dead server -> nonzero exit, runner_error row
        out = os.path.join(td, "dead.jsonl")
        r = subprocess.run([sys.executable, RUNNER, "--manifest", man, "--work", td,
                            "--out", out, "--base-url", "http://127.0.0.1:9",
                            "--timeout", "2"], capture_output=True, text=True)
        if r.returncode == 0:
            fails.append("dead server: expected nonzero exit")
        else:
            row = json.loads(open(out).read())
            if row.get("failure_mode") != "runner_error":
                fails.append(f"dead server: wrong row {row}")

        # must-fail: missing repo -> nonzero exit
        man2 = os.path.join(td, "m2.json")
        json.dump([{"instance_id": "ghost", "repo": "x/y", "base_commit": "z",
                    "problem_statement": "p", "gold_files": []}], open(man2, "w"))
        r = subprocess.run([sys.executable, RUNNER, "--manifest", man2, "--work", td,
                            "--out", os.path.join(td, "g.jsonl"),
                            "--base-url", "http://127.0.0.1:9"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            fails.append("missing repo: expected nonzero exit")

    # sandbox path escape must be refused (no server needed)
    sys.path.insert(0, HERE)
    import agent_runner
    with tempfile.TemporaryDirectory() as td:
        sb = agent_runner.Sandbox(os.path.join(td, "repo"))
        os.makedirs(sb.root, exist_ok=True)
        try:
            sb.resolve("../../etc/passwd")
            fails.append("sandbox: path escape not refused")
        except ValueError:
            pass

    if fails:
        print("selftest_agent FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("selftest_agent: all checks passed (tool loop, gold_hit, "
          "truncated/dead-server/missing-repo must-fail, sandbox escape refused)")


if __name__ == "__main__":
    main()
