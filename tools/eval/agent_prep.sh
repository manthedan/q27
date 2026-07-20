#!/bin/bash
# Clone + check out each manifest instance's repo at its base_commit.
# Usage: agent_prep.sh <manifest> <workdir> [instance_id]
# Idempotent: skips repos already checked out at the right commit.
set -u
MAN=${1:?manifest}
WORK=${2:?workdir}
ONLY=${3:-}
mkdir -p "$WORK"
python3 - "$MAN" "$WORK" "$ONLY" <<'PY'
import json, os, subprocess, sys
man, work, only = sys.argv[1], sys.argv[2], sys.argv[3]
for m in json.load(open(man)):
    if only and m["instance_id"] != only:
        continue
    iid, repo, base = m["instance_id"], m["repo"], m["base_commit"]
    dest = os.path.join(work, iid, "repo")
    if os.path.isdir(dest):
        cur = subprocess.run(["git","-C",dest,"rev-parse","HEAD"],
                             capture_output=True,text=True).stdout.strip()
        if cur == base:
            print(f"ok   {iid} (already at {base[:8]})"); continue
        subprocess.run(["rm","-rf",dest])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://github.com/{repo}.git"
    print(f"clone {iid} {repo} @ {base[:8]} ...", flush=True)
    # full clone so any base_commit is reachable; quiet; then checkout
    if subprocess.run(["git","clone","-q",url,dest]).returncode != 0:
        print(f"FAIL clone {iid}", file=sys.stderr); sys.exit(1)
    if subprocess.run(["git","-C",dest,"checkout","-q",base]).returncode != 0:
        print(f"FAIL checkout {iid}", file=sys.stderr); sys.exit(1)
    print(f"done {iid}")
PY
