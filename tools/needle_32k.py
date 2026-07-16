#!/usr/bin/env python3
# 32K needle retrieval smoke for the Metal ternary tier (quality closure).
# ~30.5K-token haystack (War and Peace head), needles at 10/35/60/70/78/92%
# depth, deepest-first so each shallower request reuses the prefix cache.
# Greedy answers (server default), exact-substring check. Usage:
#   tools/needle_32k.py [http://127.0.0.1:8080]
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
BODY = open("data/wandp.txt", encoding="utf-8").read()
HAY = BODY[:125000]  # ~30.5K tokens at ~4.1 chars/tok, question + answer stay under 32K

NEEDLES = [  # (depth, needle, question, expected substring) — deepest first
    (0.92, "The calibration constant for the tidal array is 88231.",
     "What is the calibration constant for the tidal array?", "88231"),
    (0.78, "The secret passphrase for the blue vault is 'cobalt-fern-42'.",
     "What is the secret passphrase for the blue vault?", "cobalt-fern-42"),
    (0.70, "The maintenance code for the orbital elevator is 917403.",
     "What is the maintenance code for the orbital elevator?", "917403"),
    (0.60, "Dr. Imura's favorite lichen species is Xanthoria parietina.",
     "What is Dr. Imura's favorite lichen species?", "Xanthoria parietina"),
    (0.35, "The registration number of the airship Peregrine is ZK-4471.",
     "What is the registration number of the airship Peregrine?", "ZK-4471"),
    (0.10, "The harbor master's cat is named Bramblewick.",
     "What is the name of the harbor master's cat?", "Bramblewick"),
]


# All needles embedded in ONE haystack so the document prefills once and
# every question reuses the full cached prefix (~3x cheaper than per-needle
# documents at 32K; multi-needle NIAH keeps the per-depth signal).
def build_all():
    doc = HAY
    for depth, needle, _, _ in sorted(NEEDLES, reverse=True):  # deepest first: offsets stay valid
        at = int(len(HAY) * depth)
        nl = doc.rfind("\n", 0, at)
        if nl < 0:
            nl = at
        doc = doc[:nl] + "\n" + needle + "\n" + doc[nl:]
    return doc


DOC = build_all()
ok = 0
for depth, needle, q, want in NEEDLES:
    doc = DOC
    body = {"model": "q27",
            "messages": [{"role": "user",
                          "content": doc + "\n\nBased on the document above: " + q +
                                     " Answer in one short sentence."}],
            "max_tokens": 64}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        out = json.load(r)
    text = out["choices"][0]["message"]["content"]
    usage = out.get("usage", {})
    hit = want.lower() in text.lower()
    ok += hit
    print(f"depth {depth:.2f}: {'HIT ' if hit else 'MISS'} "
          f"({usage.get('prompt_tokens', '?')} prompt tok, "
          f"{time.time() - t0:.1f}s) answer: {text.strip()[:100]!r}", flush=True)

print(f"needle 32K: {ok}/{len(NEEDLES)}")
sys.exit(0 if ok == len(NEEDLES) else 1)
