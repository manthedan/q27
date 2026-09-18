#!/usr/bin/env python3
"""Model-backed native XML + thinking-budget + durable-prefix regression.

Usage: test_bonsai2_native.py AGENT MODEL TOKENIZER
Runs one engine at a time, context 2048, in an owner-only disposable workspace.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main(agent, model, tokenizer):
    base = [str(Path(p).resolve()) for p in (agent, model, tokenizer)]
    base += ["--context", "2048", "--temperature", "0", "--max-tokens", "128"]
    env = dict(os.environ, Q27_REASONING_EFFORT="xhigh", Q27_TOOL_DIALECT="xml")
    with tempfile.TemporaryDirectory(prefix="q27-bonsai2-native-") as tmp:
        work = Path(tmp)
        (work / "secret.txt").write_text("The secret word is marigold.\n")
        (work / "index.txt").write_text("marigold grows here\n")

        def run(args):
            result = subprocess.run([*base, "--workspace", tmp, *args], cwd=tmp, env=env,
                                    text=True, capture_output=True, timeout=600)
            sys.stderr.write(result.stderr)
            require(result.returncode == 0, f"native exit {result.returncode}: {result.stdout}")
            return result

        tools = run(["--no-think", "--auto-tools", "--max-tool-rounds", "3", "--prompt",
                     "Use read to read secret.txt. Then use search to find the secret word in index.txt. Do not use shell."])
        require("automatic tool=read" in tools.stderr and "automatic tool=search" in tools.stderr,
                "native must execute read then search")
        require(tools.stderr.count("[q27-tool exit=0") >= 2 and
                tools.stderr.count("dialect=xml)") >= 2 and "[toolgram] disengaged:" not in tools.stderr,
                "both native tools must succeed with XML constraints")
        print("Native read/search XML: PASS", flush=True)
        # Thinking is ON by default, with a one-token budget. The prefilled
        # opener must seed BOTH the inline transcript and the budget tracker.
        session = str(work / "thinking.q27agent")
        args = ["--session", session, "--max-think-tokens", "1", "--prompt", "Reply with exactly: ready"]
        for resumed in (False, True):
            result = run(args)
            text = result.stdout.strip()
            require(text.startswith("<think>\n") and text.count("<think>") == 1 and
                    text.count("</think>") == 1, f"thinking stream framing: {text!r}")
            require(text.split("</think>", 1)[1].strip() == "ready", f"thinking final answer: {text!r}")
            require("thinking token budget: forced </think> and continued" in result.stderr,
                    "prefilled think span must trigger the one-token budget")
            require(Path(session).is_file(), "session manifest was not saved")
            if resumed:
                cached = [int(x) for x in re.findall(r"cached=(\d+)", result.stderr)]
                require("session loaded:" in result.stderr and cached and max(cached) > 0,
                        "thinking restart must restore a reusable exact prefix")
                print(f"Native thinking save/restart: PASS (cached={max(cached)})", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
