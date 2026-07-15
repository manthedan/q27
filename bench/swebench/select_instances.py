#!/usr/bin/env python3
# Reproducibly select the pinned SWE-bench_Verified subset used by run.sh.
# Deterministic: fast-test repos, "<15 min fix" difficulty, per-repo quota, sorted by id.
# Requires: pip install datasets   (HF_HOME=/mnt/ai/hf_cache to cache off the small / disk)
import json, re
from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
FAST = {"psf/requests", "pallets/flask", "pytest-dev/pytest", "pylint-dev/pylint", "pydata/xarray"}
QUOTA = {"psf/requests": 4, "pytest-dev/pytest": 3, "pydata/xarray": 2, "pylint-dev/pylint": 2, "pallets/flask": 1}
cand = [x for x in ds if x["repo"] in FAST and x["difficulty"] == "<15 min fix"]
by = {}
for x in sorted(cand, key=lambda r: r["instance_id"]):
    by.setdefault(x["repo"], []).append(x)
sel = sorted((x for repo, n in QUOTA.items() for x in by.get(repo, [])[:n]),
             key=lambda r: r["instance_id"])
man = [{"instance_id": x["instance_id"], "repo": x["repo"], "base_commit": x["base_commit"],
        "problem_statement": x["problem_statement"],
        "gold_files": sorted(set(re.findall(r'^\+\+\+ b/(.+)$', x["patch"], re.M))),
        "FAIL_TO_PASS": json.loads(x["FAIL_TO_PASS"]) if isinstance(x["FAIL_TO_PASS"], str) else x["FAIL_TO_PASS"]}
       for x in sel]
json.dump(man, open("manifest.json", "w"), indent=1)
print(f"wrote manifest.json ({len(man)} instances)")
