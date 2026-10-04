#!/usr/bin/env python3
"""KernelBench-style agentic optimization loop for one q27 Metal kernel region.

Contract (mirrors kernelbench.com, adapted to this repo's discipline):

  * The optimization surface is ONE marked region of src/metal/q27_kernels.metal
    (between `// [[kernel-agent:begin NAME]]` and `// [[kernel-agent:end]]`).
    Everything outside the region — host dispatch, buffer layouts, other
    kernels — is frozen for the run.
  * Each iteration an agent proposes a complete replacement region. The driver
    materializes it as a full candidate shader file, gates correctness, then
    times candidate and incumbent in an interleaved A/B B/A pattern (the
    repo's accept_ab.sh discipline) via Q27_METAL_SOURCE — run-to-run machine
    drift on this host measured ±60% at 2 chunks and ±30%+ at 10 chunks
    (2026-07-28), so absolute timing across processes cannot separate
    candidates; paired ratios can. A candidate is adopted only if its median
    paired ratio beats --min-improvement. The working tree always holds the
    incumbent; it is rewritten only on acceptance.
  * KernelBench's pass/wrong/build/slow taxonomy maps to
    ok/gate_fail/build_fail/timeout here. Every attempt lands in a JSONL
    ledger plus per-iteration artifacts under logs/kernel-agent-<timestamp>/.

Defaults target q27_matmul_q4_mm: 93.9% of synthetic prefill time on M4
(Q27_METAL_PROFILE=1 ./build/metal_prefill_bench, 2026-07-28), i.e. the one
kernel where an agent win moves the serving number.

Revival note: this is the historical Q4/Q8 prefill research harness, NOT a
Bonsai 2 prefill benchmark (that model currently uses serial float activations).

Gate:   ./build/test-metal-backend --matmul-tiles   (tiled-GEMM envelope, 3e-4 vs
        serial GEMV reference, all four quantized dtypes, widths 1..96,
        partial tiles, production shapes)
Metric: median ms/chunk ratio candidate/incumbent over --ab-pairs interleaved
        pairs of ./build/metal_prefill_bench --prompt N runs (order ABBA per
        two pairs to cancel linear thermal drift). Baseline iteration runs the
        incumbent against itself and records the noise floor.

GPU EXCLUSIVITY: benchmark numbers are only meaningful with an idle GPU and
an idle host — a second GPU-heavy agent (or an actively used desktop) both
skews ratios (observed ±20%+ under contention) and risks memory-pressure
crashes (each bench run allocates ~1.6 GiB). All GPU work (gate + bench
runs) is serialized through an flock on --gpu-lock; point cooperating agents
at the SAME lock file (Q27_GPU_LOCK env or the flag) so only one agent
benchmarks at a time. Run the loop when the machine is unattended.

Usage:
  python3 tools/kernel_agent/driver.py --iterations 10 --model sonnet
  python3 tools/kernel_agent/driver.py --iterations 0          # baseline/noise only
  Q27_KERNEL_AGENT_CMD='python3 tools/kernel_agent/echo_agent.py' \
      python3 tools/kernel_agent/driver.py --iterations 1      # no-LLM smoke

The agent command reads the prompt on stdin and must print one ```metal
fenced block containing the COMPLETE replacement region, preceded by a
`RATIONALE: ...` line. Default agent: `claude -p --model <model>`.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
KERNEL_FILE = REPO / "src/metal/q27_kernels.metal"
BEGIN_RE = re.compile(r"^// \[\[kernel-agent:begin (\w+)\]\]\s*$", re.M)
END_MARKER = "// [[kernel-agent:end]]"

GATE_CMD = ["./build/test-metal-backend", "--matmul-tiles"]
BENCH_CMD = ["./build/metal_prefill_bench"]
BUILD_TARGETS = ["build/test-metal-backend", "build/metal_prefill_bench"]
METRIC_RE = re.compile(r"prefill:\s*([\d.]+) ms/chunk,\s*([\d.]+) tok/s")
CHUNKS_RE = re.compile(r"chunks_ms:\s*([\d. ]+)")

CONTRACT = """\
You are optimizing ONE Metal compute kernel inside the q27 inference engine
(Apple Silicon, here an Apple M4). This is a KernelBench-style loop: you
propose a complete replacement for the kernel region below; a harness gates
its correctness, then times it against the incumbent in interleaved A/B runs.
Only gate-passing candidates that beat the incumbent are kept.

== Frozen context (do NOT rely on changing any of this) ==
- Host dispatch (cannot change): grid = (ceil(rows/32), ceil(x_rows/16), 1)
  threadgroups, 128 threads per threadgroup, one MatmulArgs constant buffer.
- Kernel signature must stay EXACTLY:
    kernel void q27_matmul_q4_mm(
        device const uchar *weights [[buffer(0)]],
        device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]],
        device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]],
        constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]])
- struct MatmulArgs { uint rows; uint cols; uint x_rows; uint simdgroups; };
  (simdgroups is always 1 here; ignore it.)
- Buffer layouts (cannot change):
    weights:       packed 4-bit, row-major [rows, cols/2] bytes; column c of
                   row r is nibble (c&1 ? high : low) of weights[r*(cols/2)+c/2],
                   value = nibble - 8 (signed, exact in f32).
    weight_scales: half [rows, cols/64], one scale per 64-column group, applied
                   as float(scale) AFTER the integer-exact group sum.
    x:             int8 activations, row-major [x_rows, cols].
    x_scales:      float [x_rows, cols/32], one scale per 32-column block.
    out:           float, TOKEN-major [x_rows, rows]: out[t*rows + r].
- Shape guarantees: cols % 64 == 0 (gate only exercises % 128 == 0); rows is
  arbitrary (9, 17, 33, 100, 64, 10240 in the gate; partial 32-row tiles must
  clamp); x_rows in 1..96 arbitrary (partial 16-token tiles must skip stores
  and must not read out of bounds).
- Threadgroup memory budget: the device max is 32 KiB per threadgroup; the
  incumbent uses 14 KiB. Exceeding the device max fails pipeline creation and
  counts as gate_fail.

== Correctness gate (must pass; this is the hard constraint) ==
./build/test_metal --matmul-tiles compares your kernel against the serial
GEMV reference per element with relative tolerance 3e-4, across shapes
{9x256, 9x1024, 17x1152, 17x1152@96tok, 9x256@20tok, 33x5120@17tok,
10240x5120@17tok, 100x1152@33tok, 64x128@64tok} and token widths n in
{1,4,5,8,9,12,17,33,tokens}. The same suite runs for the Q8/T2/B1 sibling
kernels, which live outside your region and must not break.
Numerical freedom: the weight side must stay integer-exact per 64-column
scale group (nibble - 8 is exact in f32); float reassociation is fine — the
incumbent itself is not bit-identical to the reference, it sits inside the
3e-4 envelope. Activation staging may round once per value.

== Metric (lower is better) ==
Your candidate and the incumbent run interleaved (A/B B/A) through
`./build/metal_prefill_bench --prompt 960` — synthetic resident weights,
production 96-token chunk schedule, Apple M4. The score reported to you is
the median ms/chunk RATIO candidate/incumbent; below 1.0 is faster. Your
kernel is ~94% of prefill time, so kernel wins show up ~1:1 in the ratio.
Machine noise moves both arms together, so chase real algorithmic wins
(≥2-3%), not noise.

== Rules ==
- Replace ONLY the region content. Keep the kernel name and signature.
- Metal Shading Language only (metal_stdlib, metal_simdgroup). No host code,
  no Objective-C, no new #include, no function constants (host sets none),
  no environment variables.
- Helpers you define must be file-local: prefix names with `ka_` and mark
  them `static` (the region lives in one translation unit with ~200 other
  kernels — name collisions break the build).
- Do not special-case gate shapes; the same source must serve production
  shapes (rows/cols up to 248320x5120, x_rows up to 96).
- Output format: first a line `RATIONALE: <one sentence: what you changed and
  why it should be faster>`, then exactly ONE ```metal fenced block with the
  COMPLETE replacement region (doc comment plus the full kernel, no
  [[kernel-agent]] markers). Nothing else.
"""

PROMPT_TEMPLATE = """\
{contract}

== Incumbent region (the code to beat) ==
```metal
{incumbent}
```

== Run state ==
- Baseline noise floor (incumbent vs itself, median ratio): {noise:.3f}
- Current incumbent: iteration {incumbent_iter}
- This is iteration {iteration} of {max_iterations}.

== Attempt history (oldest to newest) ==
{history}

Propose a replacement region that passes the gate and lowers the paired
ratio. Prefer one well-motivated change over many guesses. If recent attempts
failed the gate, fix the failure class instead of re-rolling.
"""


def extract_region(source: str) -> tuple[str, str, int, int]:
    m = BEGIN_RE.search(source)
    if not m:
        sys.exit("no kernel-agent region marker in " + str(KERNEL_FILE))
    end = source.index(END_MARKER, m.end())
    return m.group(1), source[m.end():end], m.end(), end


def splice_region(source: str, start: int, end: int, region: str) -> str:
    if not region.endswith("\n"):
        region += "\n"
    return source[:start] + "\n" + region + source[end:]


def with_region(full_source: str, region: str) -> str:
    _, _, s, e = extract_region(full_source)
    return splice_region(full_source, s, e, region)


def run(cmd: list[str], log: Path, timeout: int, env: dict | None = None) -> tuple[int, str]:
    full_env = {**os.environ, **(env or {})}
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           timeout=timeout, env=full_env)
    except subprocess.TimeoutExpired:
        log.write_text("TIMEOUT\n")
        return 124, "timeout"
    log.write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    return p.returncode, p.stdout + p.stderr


def run_gpu(cmd: list[str], log: Path, timeout: int, env: dict,
            lock_path: Path) -> tuple[int, str]:
    """Run a GPU-touching command under the shared inter-agent lock."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return run(cmd, log, timeout, env=env)


def tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text().splitlines()[-lines:])
    except OSError:
        return ""


def parse_metric(output: str) -> float | None:
    """Per-run metric: median of per-chunk ms (robust to transient host
    interference inside a run); falls back to the aggregate line."""
    c = CHUNKS_RE.search(output)
    if c:
        vals = [float(v) for v in c.group(1).split()]
        if vals:
            return statistics.median(vals)
    m = METRIC_RE.search(output)
    return float(m.group(1)) if m else None


def bench_ms(shader: Path, log: Path, prompt: int, lock: Path,
             timeout: int = 1200) -> float | None:
    rc, out = run_gpu([*BENCH_CMD, "--prompt", str(prompt)], log, timeout,
                      env={"Q27_METAL_SOURCE": str(shader)}, lock_path=lock)
    return parse_metric(out) if rc == 0 else None


def gate(shader: Path, log: Path, lock: Path, timeout: int = 900) -> tuple[bool, str]:
    rc, _ = run_gpu(GATE_CMD, log, timeout,
                    env={"Q27_METAL_SOURCE": str(shader)}, lock_path=lock)
    return rc == 0, ("timeout" if rc == 124 else "")


def evaluate_ab(run_dir: Path, tag: str, incumbent: Path, candidate: Path,
                prompt: int, pairs: int, lock: Path) -> dict:
    """Interleaved A/B B/A timing. Ratio < 1 means candidate is faster."""
    inc, cand = [], []
    order = []
    for p in range(pairs):
        seq = [("A", incumbent, inc), ("B", candidate, cand)]
        if p % 2 == 1:
            seq.reverse()  # BA on odd pairs: ABBA cancels linear drift
        for arm, shader, sink in seq:
            ms = bench_ms(shader, run_dir / f"{tag}.pair{p}{arm}.log", prompt, lock)
            order.append(arm)
            if ms is None:
                return {"status": "bench_fail",
                        "detail": tail(run_dir / f"{tag}.pair{p}{arm}.log")}
            sink.append(ms)
    inc_med, cand_med = statistics.median(inc), statistics.median(cand)
    # Median of PAIRED ratios (each pair shares its drift window), not the
    # ratio of independent medians, which can accept a per-pair slowdown.
    ratio = statistics.median(c / i for i, c in zip(inc, cand))
    return {"status": "ok", "incumbent_ms": inc_med, "candidate_ms": cand_med,
            "ratio": ratio, "order": "".join(order),
            "incumbent_all": inc, "candidate_all": cand}


def default_agent_cmd(model: str) -> list[str]:
    return ["claude", "-p", "--model", model]


def call_agent(cmd: list[str], prompt: str, out_file: Path, timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           cwd=REPO, timeout=timeout)
    except subprocess.TimeoutExpired:
        out_file.write_text("TIMEOUT\n")
        return 124, ""
    out_file.write_text(p.stdout + ("\n--- stderr ---\n" + p.stderr if p.stderr else ""))
    return p.returncode, p.stdout


FENCE_RE = re.compile(r"```metal\s*\n(.*?)```", re.S)


def parse_candidate(response: str, kernel: str) -> tuple[str | None, str]:
    blocks = FENCE_RE.findall(response)
    if len(blocks) != 1:
        return None, f"expected 1 metal fence, got {len(blocks)}"
    region = blocks[0].strip("\n")
    if f"kernel void {kernel}(" not in region:
        return None, f"candidate region is missing the {kernel} kernel"
    if "[[kernel-agent" in region:
        return None, "candidate contains region markers"
    rat = re.search(r"^RATIONALE:\s*(.+)$", response, re.M)
    return region, (rat.group(1).strip() if rat else "(no rationale)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--ab-pairs", type=int, default=3,
                    help="A/B pairs per comparison; pair order alternates AB/BA")
    ap.add_argument("--bench-prompt", type=int, default=960,
                    help="synthetic prompt length; 960 = 10 production chunks")
    ap.add_argument("--min-improvement", type=float, default=0.01,
                    help="required ratio improvement (default 1%%)")
    ap.add_argument("--max-fail-streak", type=int, default=5)
    ap.add_argument("--agent-timeout", type=int, default=1800)
    ap.add_argument("--history", type=int, default=8, help="attempts shown to the agent")
    ap.add_argument("--gpu-lock", default=os.environ.get(
                        "Q27_GPU_LOCK", str(REPO / "logs" / "gpu.lock")),
                    help="flock file serializing GPU work across agents")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Absolute: subprocesses run with cwd=REPO, so a caller-relative run dir
    # would point their shader override at a missing file (silent fallback).
    run_dir = (Path(args.run_dir).resolve() if args.run_dir
               else REPO / "logs" / f"kernel-agent-{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu_lock = Path(args.gpu_lock)
    ledger = (run_dir / "ledger.jsonl").open("a")

    def log(entry: dict) -> None:
        ledger.write(json.dumps(entry) + "\n")
        ledger.flush()
        slim = {k: v for k, v in entry.items() if k != "detail"}
        print(json.dumps(slim), flush=True)

    source = KERNEL_FILE.read_text()
    name, region, _, _ = extract_region(source)
    if name != "q27_matmul_q4_mm":
        print(f"warning: region target is {name}, contract text assumes q27_matmul_q4_mm",
              file=sys.stderr)
    incumbent_region = region.strip("\n")
    incumbent_shader = run_dir / "incumbent_full.metal"
    incumbent_shader.write_text(source)
    (run_dir / "incumbent_region.metal").write_text(incumbent_region + "\n")

    print(f"run dir: {run_dir}", flush=True)
    print("building bench + gate binaries (one-time)...", flush=True)
    rc, _ = run(["make", "-j4", *BUILD_TARGETS], run_dir / "build.log", 900)
    if rc != 0:
        sys.exit(f"build failed — see {run_dir / 'build.log'}")

    print(f"region: {name} ({len(incumbent_region.splitlines())} lines); "
          "baseline gate + noise-floor A/A...", flush=True)
    ok, why = gate(incumbent_shader, run_dir / "iter_0000.gate.log", gpu_lock)
    if not ok:
        sys.exit(f"baseline gate failed ({why or 'mismatch'}) — see {run_dir}")
    base = evaluate_ab(run_dir, "iter_0000", incumbent_shader, incumbent_shader,
                       args.bench_prompt, args.ab_pairs, gpu_lock)
    log({"iter": 0, "kind": "baseline", **base})
    if base["status"] != "ok":
        sys.exit(f"baseline bench failed — see {run_dir}")
    noise = base["ratio"]

    env_cmd = os.environ.get("Q27_KERNEL_AGENT_CMD")
    agent_cmd = env_cmd.split() if env_cmd else default_agent_cmd(args.model)

    history: list[dict] = []
    incumbent_iter = 0
    fail_streak = 0
    for it in range(1, args.iterations + 1):
        hist_text = "\n".join(
            f"- iter {h['iter']}: {h['status']}"
            + (f", ratio {h['ratio']:.3f} (cand {h['candidate_ms']:.0f} vs inc "
               f"{h['incumbent_ms']:.0f} ms/chunk)" if "ratio" in h else "")
            + (", ACCEPTED" if h.get("accepted") else "")
            + (f" — {h['rationale']}" if h.get("rationale") else "")
            + (f"\n  error tail:\n    " + "\n    ".join(h["detail"].splitlines()[-12:])
               if h.get("detail") and h["status"] != "ok" else "")
            for h in history[-args.history:]
        ) or "(none yet)"
        prompt_text = PROMPT_TEMPLATE.format(
            contract=CONTRACT, incumbent=incumbent_region, noise=noise,
            incumbent_iter=incumbent_iter, iteration=it,
            max_iterations=args.iterations, history=hist_text)
        (run_dir / f"iter_{it:04d}_prompt.txt").write_text(prompt_text)

        rc, response = call_agent(agent_cmd, prompt_text,
                                  run_dir / f"iter_{it:04d}_response.txt",
                                  args.agent_timeout)
        if rc != 0:
            entry = {"iter": it, "kind": "attempt", "status": "agent_fail",
                     "agent_rc": rc}
            log(entry)
            history.append(entry)
            fail_streak += 1
        else:
            candidate_region, rationale = parse_candidate(response, name)
            if candidate_region is None:
                entry = {"iter": it, "kind": "attempt", "status": "parse_fail",
                         "rationale": rationale}
                log(entry)
                history.append(entry)
                fail_streak += 1
            else:
                cand_shader = run_dir / f"iter_{it:04d}_full.metal"
                cand_shader.write_text(with_region(
                    incumbent_shader.read_text(), candidate_region))
                (run_dir / f"iter_{it:04d}_region.metal").write_text(
                    candidate_region + "\n")
                ok, why = gate(cand_shader, run_dir / f"iter_{it:04d}.gate.log",
                               gpu_lock)
                if not ok:
                    result = {"status": "timeout" if why == "timeout" else "gate_fail",
                              "detail": tail(run_dir / f"iter_{it:04d}.gate.log")}
                else:
                    result = evaluate_ab(run_dir, f"iter_{it:04d}", incumbent_shader,
                                         cand_shader, args.bench_prompt, args.ab_pairs,
                                         gpu_lock)
                    if result["status"] != "ok":
                        result["detail"] = result.get("detail", "")
                accepted = (result["status"] == "ok" and
                            result["ratio"] < 1.0 - args.min_improvement)
                entry = {"iter": it, "kind": "attempt", "rationale": rationale,
                         "accepted": accepted, **result}
                log(entry)
                history.append(entry)
                if accepted:
                    incumbent_region = candidate_region
                    incumbent_shader = cand_shader
                    incumbent_iter = it
                    KERNEL_FILE.write_text(with_region(KERNEL_FILE.read_text(),
                                                       candidate_region))
                    (run_dir / "incumbent_region.metal").write_text(
                        incumbent_region + "\n")
                    fail_streak = 0
                else:
                    fail_streak = 0 if result["status"] == "ok" else fail_streak + 1
        if fail_streak >= args.max_fail_streak:
            log({"iter": it, "kind": "abort",
                 "reason": f"{fail_streak} consecutive failures"})
            break

    summary = {
        "noise_floor_ratio": noise,
        "incumbent_iter": incumbent_iter,
        "attempts": len(history),
        "accepted": sum(1 for h in history if h.get("accepted")),
        "by_status": {s: sum(1 for h in history if h["status"] == s)
                      for s in {h["status"] for h in history}} or {},
        "best_ratio": min((h["ratio"] for h in history if "ratio" in h),
                          default=None),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log({"kind": "summary", **summary})


if __name__ == "__main__":
    main()
