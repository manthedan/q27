#!/usr/bin/env python3
"""Native-agent durable-session soak: one session file, N process restarts.
Timestamps each stderr line to attribute cold start / session load /
generation / save time. Prompts cycle; the first is ~4K tokens (README head).
Fails on any non-zero exit; prints per-turn timings for evidence.

Usage: test_bonsai2_session_soak.py AGENT MODEL TOKENIZER CONTEXT TURNS
Q27_SOAK_KV=fp16|turbo3|q8 passes --kv to the agent (default: agent default)."""
import os, subprocess, sys, tempfile, time, re
from pathlib import Path

AGENT, MODEL, TOK = (str(Path(a).resolve()) for a in sys.argv[1:4])
CTX, TURNS = sys.argv[4], int(sys.argv[5])
long_text = (Path(__file__).resolve().parents[1] / "README.md").read_text()[:11000]
prompts = ["Here is a project README:\n\n" + long_text + "\n\nIn one sentence, what does this project do?",
           "Name one supported GPU from the README.",
           "What file format does q27 use for weights? One short phrase.",
           "Reply with exactly: done"]
home = Path(tempfile.mkdtemp(prefix="q27-soak-"))
os.chmod(home, 0o700)
session = home / "work.q27agent"
work = home / "ws"; work.mkdir(); os.chmod(work, 0o700)
env = dict(os.environ, Q27_MODEL_PROFILE="bonsai2-qwen38-v1")
for turn in range(TURNS):
    args = [AGENT, MODEL, TOK, "--context", CTX, "--workspace", str(work), "--session", str(session),
            "--no-think", "--temperature", "0", "--max-tokens", "48", "--prompt", prompts[turn % len(prompts)]]
    if os.environ.get("Q27_SOAK_KV"):
        args[3:3] = ["--kv", os.environ["Q27_SOAK_KV"]]
    t0 = time.time()
    p = subprocess.Popen(args, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    events = []
    for line in p.stderr:
        events.append((time.time() - t0, line.rstrip()))
    out = p.stdout.read(); p.wait()
    total = time.time() - t0
    def when(pat):
        for t, l in events:
            if re.search(pat, l): return t, l
        return None, None
    ready, _ = when(r"Metal model ready|upload path")
    loaded, _ = when(r"session loaded")
    stats_t, stats = when(r"\[q27-agent prompt=")
    saved_t, _ = when(r"session saved")
    size = sum(f.stat().st_size for f in home.iterdir() if f.is_file())
    print(f"turn {turn+1}: rc={p.returncode} total={total:6.1f}s model_ready={ready and round(ready,1)} "
          f"session_loaded={loaded and round(loaded,1)} gen_done={stats_t and round(stats_t,1)} "
          f"saved={saved_t and round(saved_t,1)} save_cost={saved_t and stats_t and round(saved_t-stats_t,1)} "
          f"session_dir={size/1e6:.0f}MB")
    print("   ", stats)
    print("    reply:", out.strip().replace("\n", " ")[:100])
    if p.returncode:
        print("\n".join(l for _, l in events[-15:])); sys.exit(1)
print("files:", sorted((f.name, f.stat().st_size) for f in home.iterdir()))
