#!/usr/bin/env python3
"""q27-native SWE-bench agent runner (tier-gap probe, 2026-07-20).

Drives one SWE-bench instance as a multi-turn tool-use conversation against
our own Anthropic endpoint (/v1/messages) -- NOT Claude Code, NOT Docker.
The model gets a small tool set (read/write/edit/grep/glob/bash) sandboxed
to the checked-out instance repo, works up to the turn cap, and stops when
it emits no tool calls. The final repo diff is scored by gold_file overlap
(cheap signal; no test execution for v1).

Stop-reason discipline mirrors the fixed capability suite: every turn's
stop_reason is recorded. A turn that stops 'max_tokens'/'length' is a
truncated think/call and is counted in 'excl'; the instance is scored on
the diff it actually produced (gold_file_hit), and truncated turns are a
recorded failure mode, not silently zero.

Requests are strictly serial (house rule). Run only inside a coordinated
GPU slot against a server whose owner expects the traffic. Stdlib only.
"""
import argparse, fnmatch, json, os, subprocess, sys, time, urllib.request, urllib.error


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code,
                                    "redirect refused: " + newurl, headers, fp)


DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

BASH_TIMEOUT_S = 20
READ_CAP = 24000
GREP_CAP = 12000
BASH_CAP = 12000


def tool_schema():
    P = {"type": "object", "properties": {}, "required": []}
    def props(**kw):
        d = dict(P); d["properties"] = kw; return d
    return [
        {"name": "read_file", "description": "Read a UTF-8 text file in the repo (optionally a 1-based line range). Returns its contents, truncated if huge.",
         "input_schema": props(path={"type": "string"}, start={"type": "integer"}, end={"type": "integer"}) | {"required": ["path"]}},
        {"name": "write_file", "description": "Overwrite (or create) a file in the repo with the given full contents.",
         "input_schema": props(path={"type": "string"}, content={"type": "string"}) | {"required": ["path", "content"]}},
        {"name": "edit_file", "description": "Replace the FIRST occurrence of `old` with `new` in a file. old must match exactly, including indentation.",
         "input_schema": props(path={"type": "string"}, old={"type": "string"}, new={"type": "string"}) | {"required": ["path", "old", "new"]}},
        {"name": "grep", "description": "Search file contents under the repo for a regex pattern (ripgrep-style substring). Returns matching file:line:content lines.",
         "input_schema": props(pattern={"type": "string"}, path={"type": "string"}) | {"required": ["pattern"]}},
        {"name": "glob", "description": "List repo files matching a glob pattern (e.g. '**/*.py').",
         "input_schema": props(pattern={"type": "string"}) | {"required": ["pattern"]}},
        {"name": "run_bash", "description": "Run a short shell command in the repo (no network). Use for listing, sed, quick python checks. Do NOT run the full test suite.",
         "input_schema": props(cmd={"type": "string"}) | {"required": ["cmd"]}},
    ]


class Sandbox:
    """Confine all tool effects to the instance repo root."""
    def __init__(self, root):
        self.root = os.path.realpath(root)

    def resolve(self, path):
        p = os.path.realpath(os.path.join(self.root, path.lstrip("/")))
        if p != self.root and not p.startswith(self.root + os.sep):
            raise ValueError(f"path escapes repo: {path!r}")
        return p

    def read_file(self, a):
        p = self.resolve(a["path"])
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        s, e = a.get("start"), a.get("end")
        if s is not None or e is not None:
            s = max(1, int(s or 1)); e = int(e or len(lines))
            lines = lines[s - 1:e]
        return "\n".join(lines)[:READ_CAP]

    def write_file(self, a):
        p = self.resolve(a["path"])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(a["content"])
        return f"wrote {len(a['content'])} bytes to {a['path']}"

    def edit_file(self, a):
        p = self.resolve(a["path"])
        with open(p, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        if a["old"] not in txt:
            return "ERROR: `old` not found in file (exact match required)"
        txt = txt.replace(a["old"], a["new"], 1)
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)
        return f"edited {a['path']}"

    def grep(self, a):
        base = self.resolve(a.get("path") or ".")
        out = []
        for dp, dn, fn in os.walk(base):
            dn[:] = [d for d in dn if d not in (".git", "__pycache__", "node_modules")]
            for f in fn:
                fp = os.path.join(dp, f)
                try:
                    with open(fp, encoding="utf-8", errors="replace") as fh:
                        for ln, line in enumerate(fh, 1):
                            if a["pattern"] in line:
                                out.append(f"{os.path.relpath(fp, self.root)}:{ln}:{line.rstrip()[:200]}")
                except (OSError, UnicodeDecodeError):
                    continue
                if len("\n".join(out)) > GREP_CAP:
                    return "\n".join(out)[:GREP_CAP] + "\n... (truncated)"
        return "\n".join(out)[:GREP_CAP] or "(no matches)"

    def glob(self, a):
        out = []
        for dp, dn, fn in os.walk(self.root):
            dn[:] = [d for d in dn if d not in (".git", "__pycache__", "node_modules")]
            for f in fn:
                rel = os.path.relpath(os.path.join(dp, f), self.root)
                if fnmatch.fnmatch(rel, a["pattern"]) or fnmatch.fnmatch(f, a["pattern"]):
                    out.append(rel)
        return "\n".join(sorted(out))[:GREP_CAP] or "(no matches)"

    def run_bash(self, a):
        try:
            p = subprocess.run(a["cmd"], shell=True, cwd=self.root,
                               capture_output=True, text=True, timeout=BASH_TIMEOUT_S,
                               env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": self.root})
            return (p.stdout + p.stderr)[:BASH_CAP] or f"(exit {p.returncode}, no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {BASH_TIMEOUT_S}s"

    def dispatch(self, name, a):
        fn = {"read_file": self.read_file, "write_file": self.write_file,
              "edit_file": self.edit_file, "grep": self.grep,
              "glob": self.glob, "run_bash": self.run_bash}.get(name)
        if fn is None:
            return f"ERROR: unknown tool {name!r}"
        try:
            return fn(a)
        except (OSError, ValueError, KeyError) as e:
            return f"ERROR: {e}"


def post_messages(base_url, body, timeout):
    url = base_url.rstrip("/") + "/v1/messages"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with DIRECT_OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def run_instance(inst, repo_root, base_url, max_turns, max_tokens, timeout):
    """Run one instance. Returns a result dict (never raises on model oddities)."""
    sb = Sandbox(repo_root)
    sysp = ("You are a careful software engineer fixing a real GitHub issue in the "
            "repository at the workspace root. Investigate with the tools, then make "
            "the minimal source edit that fixes it. Do NOT add new test files and do "
            "NOT run the full test suite. When the fix is complete, reply with a short "
            "summary and no tool calls.")
    user = ("ISSUE:\n" + inst["problem_statement"] +
            "\n\nThe repository is checked out at the workspace root. Fix the issue by "
            "editing source files, then stop.")
    messages = [{"role": "user", "content": user}]
    turns, out_tok, excl = 0, 0, 0
    stop_reasons, tool_errors, last_text = [], 0, ""
    t0 = time.time()
    for _ in range(max_turns):
        body = {"model": "q27", "max_tokens": max_tokens, "system": sysp,
                "messages": messages, "tools": tool_schema()}
        payload = post_messages(base_url, body, timeout)
        turns += 1
        sr = payload.get("stop_reason") or ""
        stop_reasons.append(sr)
        out_tok += (payload.get("usage") or {}).get("output_tokens", 0)
        if sr in ("max_tokens", "length"):
            excl += 1
        blocks = payload.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if text:
            last_text = text
        calls = [b for b in blocks if b.get("type") == "tool_use"]
        messages.append({"role": "assistant", "content": blocks})
        if not calls:
            break  # model is done
        results = []
        for c in calls:
            r = sb.dispatch(c.get("name", ""), c.get("input") or {})
            if isinstance(r, str) and r.startswith("ERROR"):
                tool_errors += 1
            results.append({"type": "tool_result", "tool_use_id": c.get("id", ""),
                            "content": r})
        messages.append({"role": "user", "content": results})
    wall = time.time() - t0
    diff = subprocess.run(["git", "-C", repo_root, "diff"],
                          capture_output=True, text=True).stdout
    files = sorted({ln[6:] for ln in diff.splitlines() if ln.startswith("+++ b/")})
    gold = set(inst.get("gold_files", []))
    hit = bool(set(files) & gold)
    if sr in ("max_tokens", "length"):
        mode = "truncated"
    elif turns >= max_turns and calls:
        mode = "turn_cap"
    elif not files:
        mode = "no_edit"
    elif not hit:
        mode = "wrong_file"
    else:
        mode = "hit"
    return {"instance_id": inst["instance_id"], "repo": inst["repo"],
            "difficulty": inst.get("difficulty", "?"), "wall_s": round(wall, 1),
            "turns": turns, "out_tok": out_tok, "excl_turns": excl,
            "stop_reasons": stop_reasons, "tool_errors": tool_errors,
            "files_changed": files, "gold_files": sorted(gold),
            "gold_hit": hit, "nonempty": bool(files), "failure_mode": mode,
            "last_text": last_text[-400:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True, help="manifest.hard.json")
    ap.add_argument("--work", required=True, help="parent dir of checked-out instance repos")
    ap.add_argument("--out", required=True, help="output JSONL (one row per instance)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8213")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--only", default="", help="run only this instance_id")
    args = ap.parse_args()

    insts = json.load(open(args.manifest))
    n_fail = 0
    with open(args.out, "w") as out:
        for inst in insts:
            if args.only and inst["instance_id"] != args.only:
                continue
            repo = os.path.join(args.work, inst["instance_id"], "repo")
            if not os.path.isdir(repo):
                print(f"SKIP {inst['instance_id']}: no repo at {repo}", file=sys.stderr)
                n_fail += 1
                continue
            try:
                r = run_instance(inst, repo, args.base_url, args.max_turns,
                                 args.max_tokens, args.timeout)
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
                r = {"instance_id": inst["instance_id"], "error": str(e),
                     "gold_hit": False, "nonempty": False, "failure_mode": "runner_error"}
                n_fail += 1
            out.write(json.dumps(r) + "\n"); out.flush()
            print(f"[{inst['instance_id']}] {r.get('failure_mode')} "
                  f"hit={r.get('gold_hit')} turns={r.get('turns')} "
                  f"wall={r.get('wall_s')}s", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
