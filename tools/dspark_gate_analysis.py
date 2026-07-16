#!/usr/bin/env python3
"""DSpark Phase-3 gate arithmetic from a test-dspark-real-eval log.

Parses the fork harness's accept-by-depth table + per-prompt lines and prices
the pre-registered gate (2026-07-16-dspark-port-phase0.md): the post-lever-2
oracle curve gives break-even BE(w) = round(w)/G; the gate fires iff S at the
chained width >= 1.3x at the acceptance-weighted committed rate.

Chained blocks were never run by the fork (one block/round), so this computes
a measured single-block S plus a chain UPPER bound: block k+1 can only help
when block k fully accepted, and its features are strictly degraded (no
target rows for unverified tokens), so per-block acceptance in a chain cannot
exceed the measured single-block rates:
    E_chain(K) <= (tau - 1) * (1 - p^K)/(1 - p) + 1,   p = P(full block)
If even the upper bound misses the bar, the port parks decisively; if it
clears, the true chained measurement (fork patch) is required residue.

Usage: dspark_gate_analysis.py legB-accept-decay.log
"""
import re
import sys

# Lever-2 measured round costs (2026-07-16-lever2-verify-width.md) and the
# serial bracket G from the same sweep.
ROUND_MS = {12: 356.3, 16: 369.2, 24: 622.6, 48: 939.4}
G_MS = 89.4  # serial bracket midpoint (87-92 measured)
BAR = 1.30   # pre-registered Phase-3 gate


def main(path):
    text = open(path).read()

    depth = {}
    for m in re.finditer(r"^(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s*$", text, re.M):
        depth[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
    overall = re.search(
        r"OVERALL: prompts=(\d+) n_predicted=(\d+) rounds=(\d+) drafted=(\d+) "
        r"accepted=(\d+) accept=([\d.]+) tau=([\d.]+)", text)
    if not overall or not depth:
        sys.exit("log incomplete: missing OVERALL line or depth table")
    rounds = int(overall.group(3))
    tau = float(overall.group(7))

    cats = {}
    for m in re.finditer(
            r"^\[\s*\d+\]\[([\w-]+)\s*\] n_predicted=\S+\s+rounds=(\d+)\s+drafted=(\d+)\s+"
            r"accepted=(\d+)", text, re.M):
        c = cats.setdefault(m.group(1), [0, 0, 0])
        c[0] += int(m.group(2)); c[1] += int(m.group(3)); c[2] += int(m.group(4))

    block = max(depth)
    p_full = depth[block][1] / depth[1][0]  # P(all `block` positions accepted)

    print(f"rounds={rounds} tau={tau:.3f} (committed/round, single block + bonus)")
    print(f"conditional accept by depth: " +
          " ".join(f"d({i})={depth[i][2]:.3f}" for i in sorted(depth)))
    print(f"P(full {block}-block) = {p_full:.3f}")
    for name, (r, d, a) in sorted(cats.items()):
        print(f"  category {name:12s}: rounds={r:5d} accept={a/d if d else 0:.3f} "
              f"tau={a/r + 1 if r else 0:.3f}")

    print(f"\ngate arithmetic (lever-2 curve, G={G_MS} ms):")
    for w in (12, 16):
        be = ROUND_MS[w] / G_MS
        k = w // block
        e_chain = (tau - 1) * (1 - p_full ** k) / (1 - p_full) + 1 if p_full < 1 else (tau - 1) * k + 1
        s_single = tau / be
        s_chain = e_chain / be
        print(f"  w={w:2d} (K={k} chained blocks): BE={be:.2f} tok/round | "
              f"single-block S={s_single:.3f} | chain UPPER bound E<={e_chain:.2f} -> S<={s_chain:.3f}")

    be16 = ROUND_MS[16] / G_MS
    k16 = 16 // block
    e16 = (tau - 1) * (1 - p_full ** k16) / (1 - p_full) + 1 if p_full < 1 else (tau - 1) * k16 + 1
    verdict = e16 / be16
    print(f"\nVERDICT vs pre-registered bar {BAR}x at w=16:")
    if verdict < BAR:
        print(f"  S_upper = {verdict:.3f} < {BAR} -> chain upper bound misses the bar: PARK "
              f"(decisive — true chained acceptance can only be lower)")
    elif tau / be16 >= BAR:
        print(f"  single-block S = {tau/be16:.3f} >= {BAR} already: GATE FIRES on measured "
              f"single-block data alone")
    else:
        print(f"  S_upper = {verdict:.3f} >= {BAR} but single-block S = {tau/be16:.3f} < {BAR}: "
              f"INCONCLUSIVE — the true chained measurement (fork chain patch) is required")


if __name__ == "__main__":
    main(sys.argv[1])
