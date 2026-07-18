#!/usr/bin/env python3
"""Agentic replay bench — the product-truth serving number
(docs/plans/2026-07-17-agentic-replay-bench.md). Stdlib only.

Replays a corpus (agentic_replay_extract.py) serially against a running
q27 server via /v1/completions with stream=true: the rendered prompt is
POSTed VERBATIM (no re-rendering — byte-identical to the recorded
traffic), preserving the source request's max_tokens (or allowing one
extra terminal-token step for legacy corpora), greedy (no temperature
field; the server's missing-temperature default is 0.0). The replay finish
class must match the source turn. Turns run back-to-back (think-time excluded): this measures the
server's agentic capacity, not user-perceived session latency.

Measured per turn: TTFT (request sent -> first content chunk) and wall
(sent -> stream closed). Exact output-token and prefix-hit accounting
joins from the REPLAY server's own trace afterwards (--server-trace,
matched by response id). The replay is invalid unless output-token count and
terminal cause match the source exactly.

Headline: agentic effective output tok/s = total output tokens / total
wall. Reported beside it, never instead of it: TTFT median/p95, context
throughput (prompt tokens / TTFT total), per-turn table, prefix-hit
totals.

AUTHORED AND MOCK-TESTED ONLY (tools/agentic_replay_selftest.py): this
is a model consumer — run it ONLY inside a coordinated GPU slot against
a server whose owner expects the traffic (COORDINATION.md).
"""
import argparse
import hashlib
import ipaddress
import json
import os
import re
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code,
                                    "redirect refused: " + newurl, headers, fp)


DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
PROTOCOL_KEYS = {"context", "kv", "mtp", "suffix", "slots", "prefix_entries",
    "constrain_tools", "snapshots", "snapshot_auto_min", "snapshot_max_bytes",
    "max_tokens_default", "kv_fp16_except",
    "kv_fp16_cell_masks", "kv_side_codec", "gemm_half", "gemm_half_q4",
    "gqa_tile", "gqa_block", "gqa_threshold", "gpu_sample", "resident",
    "bare_system", "tool_strict", "test_failpoints", "tokenizer", "tokenizer_sha1"}
PROTOCOL_BOOLS = {"constrain_tools", "snapshots", "kv_fp16_except", "gemm_half",
    "gemm_half_q4", "gpu_sample", "resident", "bare_system", "tool_strict",
    "test_failpoints"}
PROTOCOL_INTS = {"context", "mtp", "suffix", "slots", "prefix_entries",
    "snapshot_auto_min", "snapshot_max_bytes", "max_tokens_default",
    "gqa_tile", "gqa_block", "gqa_threshold"}


def validate_loopback_url(url):
    u = urllib.parse.urlsplit(url)
    if (u.scheme != "http" or u.username is not None or u.password is not None
            or u.path not in ("", "/") or u.query or u.fragment or u.hostname is None):
        raise ValueError("base URL must be a credential-free HTTP loopback origin")
    try:
        ip = ipaddress.ip_address(u.hostname)
    except ValueError as e:
        raise ValueError("base URL hostname must be a literal loopback IP") from e
    if not ip.is_loopback:
        raise ValueError("base URL must use a literal loopback IP")


def local_eval_host_id():
    root = os.path.expanduser("~/.q27")
    os.makedirs(root, mode=0o700, exist_ok=True)
    path = os.path.join(root, "eval-host-id")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32) + "\n")
    except FileExistsError:
        os.chmod(path, 0o600)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def server_identity(base_url, timeout):
    req = urllib.request.Request(base_url.rstrip("/") + "/health?identity=1")
    with DIRECT_OPENER.open(req, timeout=timeout) as resp:
        d = json.loads(resp.read().decode())
    required = {"model", "artifact_sha1", "boot_id", "runtime"}
    if not required.issubset(d):
        raise ValueError("health response lacks mandatory provenance")
    hex40 = lambda x: isinstance(x, str) and re.fullmatch(r"[0-9a-f]{40}", x) is not None
    if not isinstance(d["model"], str) or not d["model"] or \
       not isinstance(d["boot_id"], str) or not d["boot_id"] or not hex40(d["artifact_sha1"]):
        raise ValueError("health response has invalid model/artifact/boot identity")
    rt = d["runtime"]
    if not isinstance(rt, dict) or rt.get("identity_schema") != 2 or \
       set(rt) != {"identity_schema", "server_sha1", "shader_abi", "shader_sha1", "protocol", "platform"} or \
       not hex40(rt.get("server_sha1")) or not hex40(rt.get("shader_sha1")) or \
       not isinstance(rt.get("shader_abi"), str) or not rt.get("shader_abi"):
        raise ValueError("health response has invalid build/shader identity")
    protocol, platform = rt.get("protocol"), rt.get("platform")
    hkeys = {"sysname", "release", "machine", "metal_device"}
    if not isinstance(protocol, dict) or set(protocol) != PROTOCOL_KEYS or \
       any(type(protocol[k]) is not bool for k in PROTOCOL_BOOLS) or \
       any(type(protocol[k]) is not int for k in PROTOCOL_INTS) or \
       not hex40(protocol.get("tokenizer_sha1")) or \
       re.fullmatch(r"[0-9a-f]{32}", protocol.get("kv_fp16_cell_masks", "")) is None or \
       not isinstance(platform, dict) or set(platform) != hkeys or any(not platform[k] for k in hkeys):
        raise ValueError("health response has incomplete protocol/platform identity")
    return {k: d[k] for k in sorted(required)}


def stream_turn(base_url, prompt, max_tokens, sampling, stops, tool_names, snapshot, timeout):
    """POST one streaming completion; return (ttft_s, wall_s, resp_id,
    finish_reason, bytes_rx)."""
    body_obj = {"prompt": prompt, "max_tokens": max_tokens, "stream": True}
    body_obj.update(sampling or {})
    if stops:
        body_obj["stop"] = stops
    if tool_names:
        body_obj["tools"] = [{"name": name} for name in tool_names]
    if snapshot:
        body_obj["snapshot"] = True
    body = json.dumps(body_obj).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    resp_id = None
    finish = None
    n_bytes = 0
    with DIRECT_OPENER.open(req, timeout=timeout) as resp:
        # SSE is line-framed. HTTPResponse.read(4096) may wait for the whole
        # buffer or EOF, turning TTFT into completion wall on short turns.
        # readline() returns each flushed event promptly; timestamp the first
        # non-empty content delta, not metadata/keepalive bytes.
        while True:
            line = resp.readline()
            if not line:
                break
            n_bytes += len(line)
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
                content = ch.get("text") or ch.get("delta", {}).get("content", "")
                if content and ttft is None:
                    ttft = time.monotonic() - t0
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return ttft, time.monotonic() - t0, resp_id, finish, n_bytes


def join_server_trace(path, ids):
    """Map response id -> (output_tokens, prefix_hit, terminal) from the replay
    server's trace (completions outcomes only; IDs are boot-unique)."""
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
                ot,ph=e.get("output_tokens"),e.get("prefix_hit")
                if (not isinstance(ot,int) or isinstance(ot,bool) or ot < 0 or
                    not isinstance(ph,int) or isinstance(ph,bool) or ph < 0 or
                    not isinstance(e.get("terminal"),str)):
                    continue
                out[e["id"]] = (ot,ph,e["terminal"])
    return out


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus", help="replay corpus JSONL")
    ap.add_argument("--base-url", default="http://127.0.0.1:8213")
    ap.add_argument("--server-trace", required=True,
                    help="replay server trace; required for exact work/terminal validation")
    ap.add_argument("--out", default=None, help="write JSON summary here")
    ap.add_argument("--limit", type=int, default=0, help="replay first N only")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()
    try:
        validate_loopback_url(args.base_url)
        provenance = server_identity(args.base_url, args.timeout)
        provenance["eval_host_id"] = local_eval_host_id()
    except (ValueError, urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        sys.exit("invalid/unverifiable replay server: %s" % e)

    turns = [json.loads(l) for l in open(args.corpus, encoding="utf-8")
             if l.strip()]
    if args.limit:
        turns = turns[:args.limit]
    if not turns:
        sys.exit("empty corpus")
    source_identity = turns[0].get("source_identity")
    if not isinstance(source_identity, dict) or not source_identity:
        sys.exit("corpus lacks mandatory source boot provenance")
    if any(t.get("source_identity") != source_identity or not t.get("source_boot_id")
           for t in turns):
        sys.exit("corpus mixes or omits source boot provenance")
    source_boot_ids = sorted({t["source_boot_id"] for t in turns})

    rows = []
    length_finishes = {"length", "max_tokens", "incomplete"}
    for t in turns:
        source_terminal = t.get("terminal")
        source_length = (source_terminal == "length" if source_terminal is not None
                         else t.get("finish") in length_finishes)
        cap = t.get("source_max_tokens")
        if not isinstance(cap, int) or cap <= 0:
            cap = t["output_tokens"] + (0 if source_length else 1)
        try:
            ttft, wall, rid, finish, nb = stream_turn(
                args.base_url, t["prompt"], cap, t.get("sampling", {}),
                t.get("stops", []), t.get("tool_names", []),
                bool(t.get("snapshot", False)), args.timeout)
        except (urllib.error.URLError, OSError) as e:
            sys.exit("turn seq=%d failed: %s (server down mid-replay is a "
                     "run-invalidating event, not a data point)" % (t["seq"], e))
        if ttft is None:
            if t["output_tokens"] == 0:
                # Successful immediate-EOS turns have no content delta. Their
                # terminal wall is the observable first/only response time.
                ttft = wall
            else:
                sys.exit("turn seq=%d: no content bytes received" % t["seq"])
        expected_finish = "length" if source_length else "stop"
        if finish != expected_finish:
            sys.exit("turn seq=%d: replay finish %r does not match source class %r"
                     % (t["seq"], finish, expected_finish))
        rows.append({"seq": t["seq"], "prompt_tokens": t["prompt_tokens"],
                     "replay_max_tokens": cap, "source_finish": t.get("finish"),
                     "source_terminal": source_terminal or
                         ("length" if source_length else
                          "stop_sequence" if t.get("finish")=="stop_sequence" else "eos"),
                     "corpus_output_tokens": t["output_tokens"],
                     "ttft_s": ttft, "wall_s": wall, "resp_id": rid,
                     "finish": finish, "bytes": nb})
        print("seq %3d  prompt %6d  ttft %7.2fs  wall %8.2fs  finish %s"
              % (t["seq"], t["prompt_tokens"], ttft, wall, finish),
              flush=True)

    try:
        final_provenance = server_identity(args.base_url, args.timeout)
    except (ValueError, urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        sys.exit("final replay server identity failed: %s" % e)
    if final_provenance != {k: provenance[k] for k in final_provenance}:
        sys.exit("replay server/model/runtime changed during run")

    exact = join_server_trace(args.server_trace, {r["resp_id"] for r in rows})
    approx = 0
    for r in rows:
        if r["resp_id"] not in exact:
            sys.exit("turn seq=%d: response id %r missing from replay server trace"
                     % (r["seq"], r["resp_id"]))
        r["output_tokens"], r["prefix_hit"], r["terminal"] = exact[r["resp_id"]]
        if r["output_tokens"] != r["corpus_output_tokens"]:
            sys.exit("turn seq=%d: replay output_tokens %d != source %d"
                     % (r["seq"], r["output_tokens"], r["corpus_output_tokens"]))
        if r["terminal"] != r["source_terminal"]:
            sys.exit("turn seq=%d: replay terminal %r != source %r"
                     % (r["seq"], r["terminal"], r["source_terminal"]))
        r["accounting"] = "trace"

    priced = [r for r in rows if r["output_tokens"] is not None]
    if not priced:
        sys.exit("no turns with token accounting (pass --server-trace)")
    # Every priced row is trace-joined and source-equivalent; mismatched or
    # missing work/terminal evidence exits above rather than becoming an
    # approximate data point.
    n_trace = sum(1 for r in priced if r["accounting"] == "trace")
    n_cap = sum(1 for r in priced if r["accounting"] == "cap")
    tot_out = sum(r["output_tokens"] for r in priced)
    tot_wall = sum(r["wall_s"] for r in priced)
    tot_prompt = sum(r["prompt_tokens"] for r in priced)
    tot_ttft = sum(r["ttft_s"] for r in priced)
    ttfts = [r["ttft_s"] for r in rows]
    hits = [r["prefix_hit"] for r in priced if r.get("prefix_hit") is not None]
    summary = {
        "turns": len(rows), "priced_turns": len(priced),
        "trace_priced_turns": n_trace, "cap_priced_turns": n_cap,
        "unpriced_turns": approx,
        "agentic_effective_output_tok_s": tot_out / tot_wall,
        "context_throughput_tok_s": tot_prompt / tot_ttft if tot_ttft else None,
        "ttft_median_s": pctl(ttfts, 0.5), "ttft_p95_s": pctl(ttfts, 0.95),
        "total_output_tokens": tot_out, "total_prompt_tokens": tot_prompt,
        "total_wall_s": tot_wall,
        "prefix_hit_tokens": sum(hits) if hits else None,
        "provenance": {"source": source_identity,
                       "source_boot_ids": source_boot_ids,
                       "target": provenance},
    }
    print("\nagentic replay: %d turns (%d priced from trace, "
          "%d cap-priced from corpus [approx], %d unpriced)"
          % (summary["turns"], n_trace, n_cap, approx))
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
