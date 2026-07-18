#!/usr/bin/env python3
"""Agentic replay bench — the product-truth serving number
(docs/plans/2026-07-17-agentic-replay-bench.md). Stdlib only.

Replays a corpus (agentic_replay_extract.py) serially against a running
q27 server via /v1/completions with stream=true: the rendered prompt is
POSTed VERBATIM (no re-rendering — byte-identical to the recorded
traffic), max_tokens = the turn's recorded output_tokens, greedy (no
temperature field; the server's missing-temperature default is 0.0).
Turns run back-to-back (think-time excluded): this measures the
server's agentic capacity, not user-perceived session latency.

Measured per turn: TTFT (request sent -> first content chunk) and wall
(sent -> stream closed). Exact output-token and prefix-hit accounting
joins from the REPLAY server's own trace afterwards (--server-trace,
matched by response id); without it, output_tokens falls back to the
corpus value only for finish=length turns and the turn is marked
approximate.

Headline: agentic effective output tok/s = total output tokens / total
wall. Reported beside it, never instead of it: TTFT median/p95, context
throughput (prompt tokens / TTFT total), per-turn table, prefix-hit
totals.

AUTHORED AND MOCK-TESTED ONLY (tools/agentic_replay_selftest.py): this
is a model consumer — run it ONLY inside a coordinated GPU slot against
a server whose owner expects the traffic (COORDINATION.md).
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def stream_turn(base_url, prompt, max_tokens, timeout):
    """POST one streaming completion; return (ttft_s, wall_s, resp_id,
    finish_reason, bytes_rx)."""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "stream": True}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    resp_id = None
    finish = None
    n_bytes = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            if ttft is None:
                ttft = time.monotonic() - t0
            n_bytes += len(chunk)
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if resp_id is None:
                    resp_id = obj.get("id")
                for ch in obj.get("choices", []):
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    return ttft, time.monotonic() - t0, resp_id, finish, n_bytes


def join_server_trace(path, ids):
    """Map response id -> (output_tokens, prefix_hit) from the replay
    server's trace (completions outcomes only; last boot wins)."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") == "outcome" and e.get("id") in ids:
                out[e["id"]] = (e.get("output_tokens", 0),
                                e.get("prefix_hit", 0))
    return out


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus", help="replay corpus JSONL")
    ap.add_argument("--base-url", default="http://127.0.0.1:8213")
    ap.add_argument("--server-trace", default=None,
                    help="replay server's trace for exact token accounting")
    ap.add_argument("--out", default=None, help="write JSON summary here")
    ap.add_argument("--limit", type=int, default=0, help="replay first N only")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    turns = [json.loads(l) for l in open(args.corpus, encoding="utf-8")
             if l.strip()]
    if args.limit:
        turns = turns[:args.limit]
    if not turns:
        sys.exit("empty corpus")

    rows = []
    for t in turns:
        try:
            ttft, wall, rid, finish, nb = stream_turn(
                args.base_url, t["prompt"], t["output_tokens"], args.timeout)
        except (urllib.error.URLError, OSError) as e:
            sys.exit("turn seq=%d failed: %s (server down mid-replay is a "
                     "run-invalidating event, not a data point)" % (t["seq"], e))
        if ttft is None:
            sys.exit("turn seq=%d: no bytes received" % t["seq"])
        rows.append({"seq": t["seq"], "prompt_tokens": t["prompt_tokens"],
                     "corpus_output_tokens": t["output_tokens"],
                     "ttft_s": ttft, "wall_s": wall, "resp_id": rid,
                     "finish": finish, "bytes": nb})
        print("seq %3d  prompt %6d  ttft %7.2fs  wall %8.2fs  finish %s"
              % (t["seq"], t["prompt_tokens"], ttft, wall, finish),
              flush=True)

    exact = {}
    if args.server_trace:
        exact = join_server_trace(args.server_trace,
                                  {r["resp_id"] for r in rows})
    approx = 0
    for r in rows:
        if r["resp_id"] in exact:
            r["output_tokens"], r["prefix_hit"] = exact[r["resp_id"]]
            r["accounting"] = "trace"
        elif r["finish"] == "length":
            r["output_tokens"], r["prefix_hit"] = r["corpus_output_tokens"], None
            r["accounting"] = "cap"
        else:
            r["output_tokens"], r["prefix_hit"] = None, None
            r["accounting"] = "unknown"
            approx += 1

    priced = [r for r in rows if r["output_tokens"] is not None]
    if not priced:
        sys.exit("no turns with token accounting (pass --server-trace)")
    tot_out = sum(r["output_tokens"] for r in priced)
    tot_wall = sum(r["wall_s"] for r in priced)
    tot_prompt = sum(r["prompt_tokens"] for r in priced)
    tot_ttft = sum(r["ttft_s"] for r in priced)
    ttfts = [r["ttft_s"] for r in rows]
    hits = [r["prefix_hit"] for r in priced if r.get("prefix_hit") is not None]
    summary = {
        "turns": len(rows), "priced_turns": len(priced),
        "unpriced_turns": approx,
        "agentic_effective_output_tok_s": tot_out / tot_wall,
        "context_throughput_tok_s": tot_prompt / tot_ttft if tot_ttft else None,
        "ttft_median_s": pctl(ttfts, 0.5), "ttft_p95_s": pctl(ttfts, 0.95),
        "total_output_tokens": tot_out, "total_prompt_tokens": tot_prompt,
        "total_wall_s": tot_wall,
        "prefix_hit_tokens": sum(hits) if hits else None,
    }
    print("\nagentic replay: %d turns (%d priced exactly, %d unpriced)"
          % (summary["turns"], len(priced), approx))
    print("  agentic effective output tok/s: %.2f  (%d tokens / %.1f s wall)"
          % (summary["agentic_effective_output_tok_s"], tot_out, tot_wall))
    print("  ttft median %.2f s, p95 %.2f s; context throughput %.1f tok/s"
          % (summary["ttft_median_s"], summary["ttft_p95_s"],
             summary["context_throughput_tok_s"] or 0.0))
    if hits:
        print("  prefix hits: %d tokens across %d turns"
              % (sum(hits), len(hits)))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "turns": rows}, f, indent=1)
        print("  summary -> " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
