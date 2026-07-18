#!/usr/bin/env python3
"""Agentic replay corpus extractor
(docs/plans/2026-07-17-agentic-replay-bench.md). Stdlib only, no model.

Reads a server trace (--trace JSONL) and emits a replay corpus: one line
per completed generation turn, in trace order, carrying the RENDERED
prompt verbatim. Replay is teacher-forced by construction — turn N's
rendered prompt embeds the RECORDED prior assistant turns, not a live
model's — which is what makes replays deterministic and comparable
across packs and commits.

Pairing: request -> outcome by id, scoped per boot. A mandatory boot identity
(model, resident artifact, build/shader/tokenizer/config/platform) must precede
requests; compatible restarts are retained with per-turn boot IDs, while mixed
source identities fail extraction. Dropped and counted, never silent: truncated rendered
prompts (the 64 KB trace cap — cannot be replayed faithfully),
cancelled/errored turns (no completed generation to price), sub-floor
prompts (--min-prompt-tokens, default 512: gate fixtures are not agent
traffic), and APIs outside chat/messages/completions/responses.

Corpus JSONL fields: seq, source_api, source_id, prompt (rendered
text), prompt_tokens, output_tokens, orig_prefix_hit, finish.
"""
import argparse
import json
import os
import re
import secrets
import stat
import sys

REPLAY_APIS = {"chat", "messages", "completions", "responses"}
PROTOCOL_KEYS = {"context", "kv", "mtp", "suffix", "slots", "prefix_entries",
    "constrain_tools", "snapshots", "snapshot_auto_min", "snapshot_max_bytes",
    "snapshot_spine_pin", "max_tokens_default", "kv_fp16_except",
    "kv_fp16_cell_masks", "kv_side_codec", "gemm_half", "gemm_half_q4",
    "gqa_tile", "gqa_block", "gqa_threshold", "gpu_sample", "resident",
    "bare_system", "tool_strict", "test_failpoints", "tokenizer", "tokenizer_sha1"}
PROTOCOL_BOOLS = {"constrain_tools", "snapshots", "snapshot_spine_pin",
    "kv_fp16_except", "gemm_half",
    "gemm_half_q4", "gpu_sample", "resident", "bare_system", "tool_strict",
    "test_failpoints"}
PROTOCOL_INTS = {"context", "mtp", "suffix", "slots", "prefix_entries",
    "snapshot_auto_min", "snapshot_max_bytes", "max_tokens_default",
    "gqa_tile", "gqa_block", "gqa_threshold"}


def write_private_jsonl(path, rows):
    if os.path.lexists(path):
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_mode & 0o077:
            raise RuntimeError("refusing unsafe existing corpus target: %s" % path)
    parent = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(parent, ".%s.tmp.%s" %
                       (os.path.basename(path), secrets.token_hex(8)))
    fd = None
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None
            for item in rows:
                f.write(json.dumps(item) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if fd is not None:
            os.close(fd)
        try: os.unlink(tmp)
        except FileNotFoundError: pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", required=True, help="server trace JSONL")
    ap.add_argument("--out", required=True, help="replay corpus JSONL")
    ap.add_argument("--min-prompt-tokens", type=int, default=512,
                    help="drop turns below this (gate fixtures)")
    args = ap.parse_args()

    events = []
    with open(args.trace, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError as err:
                sys.exit("%s:%d: bad JSON: %s" % (args.trace, ln, err))
            events.append(e)

    dropped = {"truncated": 0, "no_outcome": 0, "cancelled": 0,
               "small": 0, "other_api": 0, "errored": 0}
    corpus = []
    reqs = {}
    cancelled = set()
    errored = set()
    current_boot = None
    source_identity = None

    def parse_boot(e):
        required = ("boot_id", "model", "artifact_sha1", "runtime")
        hex40 = lambda x: isinstance(x, str) and re.fullmatch(r"[0-9a-f]{40}", x) is not None
        if any(k not in e for k in required) or not isinstance(e.get("boot_id"), str) \
                or not e["boot_id"] or not isinstance(e.get("model"), str) or not e["model"] \
                or not hex40(e.get("artifact_sha1")):
            return None
        rt = e.get("runtime")
        if not isinstance(rt, dict) or rt.get("identity_schema") != 3 or \
                set(rt) != {"identity_schema", "server_sha1", "shader_abi", "shader_sha1", "protocol", "platform"} or \
                not hex40(rt.get("server_sha1")) or not hex40(rt.get("shader_sha1")) \
                or not isinstance(rt.get("shader_abi"), str) or not rt.get("shader_abi"):
            return None
        protocol, platform = rt.get("protocol"), rt.get("platform")
        hkeys = {"sysname", "release", "machine", "metal_device"}
        if not isinstance(protocol, dict) or set(protocol) != PROTOCOL_KEYS \
                or any(type(protocol[k]) is not bool for k in PROTOCOL_BOOLS) \
                or any(type(protocol[k]) is not int for k in PROTOCOL_INTS) \
                or not hex40(protocol.get("tokenizer_sha1")) \
                or re.fullmatch(r"[0-9a-f]{32}", protocol.get("kv_fp16_cell_masks", "")) is None \
                or not isinstance(platform, dict) or set(platform) != hkeys \
                or any(not platform[k] for k in hkeys):
            return None
        return {"model": e["model"], "artifact_sha1": e["artifact_sha1"],
                "runtime": rt}

    def flush_boot():
        # Streaming engine failures/cancellations can terminate without an
        # outcome event. Attribute those requests to their explicit terminal
        # event rather than misreporting them as unexplained no_outcome rows.
        for rid in reqs:
            if rid in cancelled:
                dropped["cancelled"] += 1
            elif rid in errored:
                dropped["errored"] += 1
            else:
                dropped["no_outcome"] += 1
        reqs.clear()
        cancelled.clear()
        errored.clear()

    for e in events:
        kind = e.get("kind")
        if kind == "boot":
            flush_boot()
            identity = parse_boot(e)
            if identity is None:
                print("trace boot lacks mandatory model/artifact/runtime identity", file=sys.stderr)
                return 2
            if source_identity is None:
                source_identity = identity
            elif identity != source_identity:
                print("trace combines incompatible source boot identities", file=sys.stderr)
                return 2
            current_boot = e["boot_id"]
            continue
        if kind == "request":
            if current_boot is None:
                print("trace contains request before mandatory boot identity", file=sys.stderr)
                return 2
            if e.get("api") not in REPLAY_APIS:
                dropped["other_api"] += 1
                continue
            e = dict(e)
            e["_source_boot_id"] = current_boot
            reqs[e.get("id")] = e
        elif kind == "cancel":
            cancelled.add(e.get("id"))
        elif kind == "error":
            errored.add(e.get("id"))
        elif kind == "outcome":
            req = reqs.pop(e.get("id"), None)
            if req is None:
                continue
            if e.get("id") in cancelled:
                dropped["cancelled"] += 1
                continue
            if e.get("id") in errored:
                dropped["errored"] += 1
                continue
            rendered = req.get("rendered")
            if isinstance(rendered, dict):  # {"truncated":true,...}
                dropped["truncated"] += 1
                continue
            if not isinstance(rendered, str) or not rendered:
                dropped["truncated"] += 1
                continue
            if req.get("prompt_tokens", 0) < args.min_prompt_tokens:
                dropped["small"] += 1
                continue
            out_tokens = e.get("output_tokens")
            if not isinstance(out_tokens, int) or isinstance(out_tokens, bool) or out_tokens < 0:
                dropped["errored"] += 1
                continue
            corpus.append({
                "seq": len(corpus),
                "source_api": req.get("api"),
                "source_id": req.get("id"),
                "source_boot_id": req.get("_source_boot_id"),
                "source_identity": source_identity,
                "prompt": rendered,
                "prompt_tokens": req.get("prompt_tokens"),
                "output_tokens": out_tokens,
                "source_max_tokens": req.get("max_tokens"),
                "sampling": req.get("sampling", {"temperature": 0.0, "top_p": 1.0,
                                                   "top_k": 0, "seed": 0}),
                "stops": req.get("stops", []),
                "tool_names": req.get("tool_names", []),
                "snapshot": bool(req.get("snapshot", False)),
                "orig_prefix_hit": e.get("prefix_hit", 0),
                "finish": e.get("finish"),
                "terminal": e.get("terminal"),
            })
    flush_boot()

    if not corpus:
        print("no replayable turns (dropped: %s)" % dropped, file=sys.stderr)
        return 2
    try:
        write_private_jsonl(args.out, corpus)
    except (OSError, RuntimeError) as e:
        print("cannot write replay corpus: %s" % e, file=sys.stderr)
        return 2
    tp = sum(c["prompt_tokens"] for c in corpus)
    to = sum(c["output_tokens"] for c in corpus)
    print("replay corpus: %d turns, %d prompt tokens, %d output tokens "
          "-> %s" % (len(corpus), tp, to, args.out))
    print("dropped: " + ", ".join("%s=%d" % kv for kv in sorted(dropped.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
