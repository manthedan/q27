# Draft: upstream issue (not yet sent)

Rewritten 2026-07-27. **The previous revision must not be sent** — it
asserted "I cannot compile CUDA" (wrong: yukon, RTX 3090 / nvcc 12.0), it
still listed the two items since overtaken by upstream (README note,
`Q27_GRAPH_TRACE`), and its scope numbers predated the staged branches.

Send per [PR-PLAYBOOK.md](PR-PLAYBOOK.md) wave 0. Everything below the line
is the message.

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
clearly labelled as a downstream Metal port and keep pulling from you. The
small fixes below I'd like to send either way.

### Scope, measured

The raw branch diff looks alarming (973 files) but almost all of it is run
logs and local experiments that would never be in a PR. I staged the actual
contribution as separate branches off your `master` and measured it:

**23 files, 21,692 insertions, 3 deletions, zero `.cu` files touched.**

The three deletions are one `DType` enum line extended in place, and
`emit_head` gaining a `bool` return. That's the shape of the ask: it is
almost entirely new files plus about a dozen additive lines in four shared
headers. Nothing of yours gets restructured.

I verified that isn't wishful thinking — I built the Metal backend against
pristine `master` with only our new files copied in. All 48 errors were
missing additions; not one was a CUDA dependency. `blocks.cu`, `server.cu`,
`test_kernels.cu` and `inspect.cpp` aren't required by the port at all.

### Suggested order

**Standalone fixes to your code** — independent of the Metal question, happy
to send these regardless:

1. **`server.cu` CUDA 12.0 compat.** On CUDA 12.0.140 (`sm_86`, RTX 3090)
   `master` fails: `src/server.cu(2331): error: structured binding cannot be
   captured` ×3. C++20 relaxed this; 12.0's nvcc predates the relaxation.
   Behaviour-identical rewrite to named tuple references, +8/−2, compiles
   clean on that toolchain.
2. **Tokenizer lifetime.** `Tokenizer` holds `Impl*` as an owning raw
   pointer with no destructor, so each instance leaks the vocab, merge table
   and special-token list. It's also implicitly copyable, and that copy would
   double-free once a destructor exists — so the destructor and the
   copy-deletes ship together. +9/−1.
3. **Adjacent tool calls emit no boundary segment.** `</tool_call><tool_call>`
   produces nothing between the two, so a consumer buffering one TOOL segment
   at a time folds both calls into one buffer. A well-formed call followed by
   a malformed one loses the malformed one's raw text entirely. Inert for
   consumers that don't buffer tool calls.
4. **Argmax tie-break** — the one that changes kernel behaviour, so it gets
   its own section below.

**Then, only if you're interested in the port:** the backend seam (730 lines,
no Metal code — this is the real decision point), then Metal core, then Metal
serving, as separate PRs.

### The argmax tie-break, with the measurement

CUDA's argmax currently resolves exact-value ties to an arbitrary index; this
makes it resolve to the **lowest**, agreeing with CPU `max_element` and with
Metal. It's what makes two backends byte-comparable.

I know your anchors are byte-exact hashes, so I measured it rather than
warning you about it. On a 3090 against the official artifact (md5-verified
against your `CHECKSUMS.md5`):

| arm | canonical md5 |
|---|---|
| `master` (`c2d2116`) | `a2982c5197c627551b27d76a0a94b220` |
| with the fix | `a2982c5197c627551b27d76a0a94b220` |
| your published anchor | `a2982c5197c627551b27d76a0a94b220` |

Identical, and equal to your published value — so there's nothing to
re-bless. The full `test_kernels` battery passes on both arms with every
reported error value identical between them. On a synthetic tie harness,
`master` fails 3 of 8 cases and the fix passes 8 of 8.

Stated precisely, because it's one sequence and not a proof: this shows the
*published* anchor is unaffected. It doesn't prove no prompt can ever hit an
argmax tie — the 128-token greedy canonical simply doesn't. With the non-tie
cases bit-identical, the residual risk is confined to a generation landing on
exact float equality at the argmax.

I'll include a 43-line model-free harness that builds against `blocks.cu`
alone, so you can reproduce the tie cases in seconds without the artifact.

### One thing I should flag rather than bury

The Metal server does minutes of GPU work before writing anything (measured
322 s prefill for an 8 K prompt on a base M4), so it needs to know whether
the client is still there before it burns a slot. cpp-httplib has no
supported path to the request fd before the first write, so our vendored copy
carries a 7-line patch adding a `socket_t sock` field to `Request`. It's
disclosed here rather than discovered in review, and it's an open question
I'd rather you answer before the serving PR — details when we get there, and
a "no" is a fine answer.

### Maintenance

If any of this lands: we maintain the Metal path, run its gates, and a Metal
break should never block a CUDA release. Happy to be `CODEOWNERS` for
`src/metal/`.
