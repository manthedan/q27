#!/usr/bin/env python3
"""Agentic replay corpus extractor
(docs/plans/2026-07-17-agentic-replay-bench.md). Stdlib only, no model.

Reads a server trace (--trace JSONL) and emits a replay corpus: one line
per completed generation turn, in trace order, carrying the RENDERED
prompt verbatim. Replay is teacher-forced by construction — turn N's
rendered prompt embeds the RECORDED prior assistant turns, not a live
model's — which is what makes replays deterministic and comparable
across packs and commits.

Pairing: request -> outcome by id, scoped per boot (id counters reset
each boot). Dropped and counted, never silent: truncated rendered
prompts (the 64 KB trace cap — cannot be replayed faithfully),
cancelled/errored turns (no completed generation to price), sub-floor
prompts (--min-prompt-tokens, default 512: gate fixtures are not agent
traffic), and apis outside chat/messages/completions.

Corpus JSONL fields: seq, source_api, source_id, prompt (rendered
text), prompt_tokens, output_tokens, orig_prefix_hit, finish.
"""
import argparse
import json
import sys

REPLAY_APIS = {"chat", "messages", "completions"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", required=True, help="server trace JSONL")
    ap.add_argument("--out", required=True, help="replay corpus JSONL")
    ap.add_argument("--min-prompt-tokens", type=int, default=512,
                    help="drop turns below this (gate fixtures)")
    args = ap.parse_args()

    boots = [{}]  # per-boot: id -> request event
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

    def flush_boot():
        dropped["no_outcome"] += len(reqs)
        reqs.clear()
        cancelled.clear()
        errored.clear()

    for e in events:
        kind = e.get("kind")
        if kind == "boot":
            flush_boot()
            continue
        if kind == "request":
            if e.get("api") not in REPLAY_APIS:
                dropped["other_api"] += 1
                continue
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
            out_tokens = e.get("output_tokens", 0)
            if not out_tokens:
                dropped["errored"] += 1
                continue
            corpus.append({
                "seq": len(corpus),
                "source_api": req.get("api"),
                "source_id": req.get("id"),
                "prompt": rendered,
                "prompt_tokens": req.get("prompt_tokens"),
                "output_tokens": out_tokens,
                "orig_prefix_hit": e.get("prefix_hit", 0),
                "finish": e.get("finish"),
            })
    flush_boot()

    if not corpus:
        print("no replayable turns (dropped: %s)" % dropped, file=sys.stderr)
        return 2
    with open(args.out, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item) + "\n")
    tp = sum(c["prompt_tokens"] for c in corpus)
    to = sum(c["output_tokens"] for c in corpus)
    print("replay corpus: %d turns, %d prompt tokens, %d output tokens "
          "-> %s" % (len(corpus), tp, to, args.out))
    print("dropped: " + ", ".join("%s=%d" % kv for kv in sorted(dropped.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
