# Draft: upstream issue (not yet sent)

Refresh of the 2026-07-16 draft (which lived in a temp job dir and had stale
scope numbers). Held per [MERGE-BACK.md](MERGE-BACK.md) until the stages are
branched. Edit here, send by hand.

---

**Title:** Interested in a Metal backend merge-back?

Hi — I've been running a fork at https://github.com/manthedan/q27 (branch
`metal`) that adds an Apple Silicon / Metal backend for the same narrow
mission: Qwen3.6-27B-MTP official tier, plus a ternary Bonsai-27B tier that
fits a 16 GB Mac. (Thanks for landing the MIT LICENSE — our fork carries it
with your copyright line plus a fork-additions line.)

**The question: are you interested in the Metal work merging back**, so q27
stays canonical and covers both CUDA and Metal? If you'd rather keep q27
CUDA-only and single-target, genuinely no hard feelings — I'll keep the fork
clearly labelled as a downstream Metal port and keep pulling from you.

Honest scope. The raw branch diff looks alarming (973 files) but most of it
is run logs and local experiments that would never be in a PR. The surface
I'd actually propose is about **24k lines**:

| | lines | note |
|---|---|---|
| `src/metal/` | 20.5k | the port itself: engine, shaders, server, CLI, tests |
| shared `src/*` | ~2.9k | the backend seam + serving hardening |
| docs | small | a condensed Metal section, not our lab notebook |

I wouldn't propose that as one PR. Suggested order, smallest first:

1. **`server.cu` CUDA 12.0 compat** — nvcc 12.0 rejects lambda-captured
   structured bindings; behavior-identical rewrite to named tuple
   references. ~10 lines.
2. **README note for 24 GB cards** — the default `Q27_W_MAX=12` graph zoo
   runs ~2.7 GB and doesn't fit next to 17.7 GB of weights (dies in
   `cudaGraphInstantiate`); `-DQ27_W_MAX=8` fits with ~0.9 GB headroom,
   exactly as your width-knob comment predicts. One paragraph.
3. **The backend seam** (`src/backend.h` + loader/tokenizer generalizations,
   no Metal code) — this is the real decision point. Everything after it is
   additive.
4. Metal core, then Metal serving, as separate PRs.

**Two things I should flag rather than bury.**

*I cannot compile CUDA.* Both my machines are Apple Silicon, so anything
touching your CUDA path is untested by me — I'd rather say that up front
than have you find out in review. The two CUDA items above are a compile fix
and a doc note for that reason.

*One behavior change, offered carefully.* I have a fix making CUDA's argmax
resolve exact-value ties to the **lowest** index, so it agrees with Metal's
argmax and CPU `max_element` (with a test). It's what makes the two backends
byte-comparable, and our 16-token canonical gate is byte-exact across both.
But it changes kernel behavior, and your canonical anchors are byte-exact
hashes — so you'd want to re-bless those yourself. Happy to send it, hold
it, or drop it.

Optional extra: a `Q27_GRAPH_TRACE=1` instrument that attributes graph
memory per family on instantiate failure, so an OOM report self-attributes
(measured: `verify_w` dominates at 855 MB at w8, sampled family 577 MB).
Only if it's useful to you.
