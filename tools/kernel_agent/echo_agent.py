#!/usr/bin/env python3
"""No-LLM smoke agent for driver.py: echoes the incumbent region back.

Reads the driver prompt on stdin, extracts the incumbent's ```metal fence
(the first one in the prompt — the contract contains no metal fences), and
re-emits it as the candidate. Exercises splice -> build -> gate -> bench ->
ledger mechanics end to end; the expected outcome is one `ok` attempt that
is NOT accepted (same code, no improvement) and an unchanged working tree.

Set Q27_KERNEL_AGENT_CMD='python3 tools/kernel_agent/echo_agent.py'.
"""

import re
import sys

prompt = sys.stdin.read()
m = re.search(r"```metal\s*\n(.*?)```", prompt, re.S)
if not m:
    sys.exit("echo_agent: no metal fence in prompt")
print("RATIONALE: smoke-test echo of the incumbent region (no change expected)")
print("```metal")
print(m.group(1).strip("\n"))
print("```")
