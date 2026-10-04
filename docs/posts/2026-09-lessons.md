# Quasar, three months in: most of my bugs were in the measuring

*(draft -- not published. Quasar is the engine, codename q27. The July
methodology post covers the first two weeks; this one covers what the
next ten taught me. Every number below has a dated entry in
[docs/BUILDLOG.md](../BUILDLOG.md).)*

I started q27 on July 2nd to see how fast one model could go on one GPU if
the engine did nothing else. Three months later it serves Qwen3.6 and
Qwen3.8 27B at about 230 t/s aggregate on Claude Code traffic from a 5090,
runs PrismML's ternary Bonsai 2 on an 8 GB card, and has a build log north
of sixteen thousand lines.

Most of those lines are about measurement. The kernels were the easy part.
The mistakes that cost me days, and a couple of retracted claims, were
almost all in how I decided whether something was better. So
this is the list I wish I'd had in July, with the incident that taught each
one.

## Defaults are configuration

For two days in August q27 lost to llama.cpp on an agentic benchmark, 0.759
to 0.854. It survived five q27 configurations and three investigations
(renderer bytes, KV precision, tool-call parser shapes). The engine was fine.
Claude Code sends no sampling fields at all (0 of 29 captured request
bodies carried temperature, top_p or top_k), so q27 fell back to greedy argmax,
while llama-server applied the sampler baked into the GGUF metadata. I was
comparing greedy to sampled.

The tell was sitting in the data the whole time: two independent trials
produced 1,068,470 and 1,068,407 tokens with identical scores, and two others
were byte-identical at 75,760. Sampled runs don't do that.

The same investigation turned up a second default. q27 capped reasoning at
half the client's max_tokens, Claude Code asks for 64,000, so thinking got
force-closed at 32,000. When that fires the model can end the turn with no
answer, Claude Code treats the empty turn as done, and the session ends: 5
of 6 trials on one task wrote no files and scored exactly 0.000. Fixing both
defaults took the mean from 0.758 to 0.847 against llama.cpp's 0.878, and no
task separated the engines afterwards.

Before you attribute a difference to the thing you're studying, list every
field the client doesn't send and what each side does with its absence.

## Check token ids, never just text

From July 1st to September 10th, q27's tokenizer never matched the
`<tool_call>` and `<tool_response>` added tokens. Every agentic prompt showed
the model its own tool history spelled as `<`, `tool`, `_call`, `>`, an
encoding it never trained on, while the model itself emitted the single
tokens. My template tests compared rendered text byte for byte against
llama.cpp's renderer and passed the entire time, because the text was right.

What finally caught it was llama.cpp's `/tokenize`, which returned 26,544
tokens for a prompt q27 made 26,578. The fix cut per-turn reasoning 43% on Claude Code,
raised decode from 214 to 232 t/s (the draft model liked the right tokens
too), and dropped wall time per task from 106 to 75 seconds. Ten weeks of
benchmarks had run against a subtly wrong prompt.

Parity checks now compare ids against AutoTokenizer on recorded bodies and
report N/N identical. Matching text told me nothing about this bug.

## A failing gate means nothing until the control passes

In August I wrote a gate that compared full responses served solo versus
concurrently, pointed it at a new shared prefill arena, and watched it fail
2/6 and 3/6. I reported two proven bugs. Then I ran the same gate with the
feature switched off entirely, and it failed 1/6. The gate was measuring
something else: under concurrency the engine swaps between two attention
kernels that agree to rounding, a near-tie flips, and the whole continuation
forks. Solo-versus-concurrent text identity had never been an invariant.

Two days later I got a sharper version. I killed a 4-bit fp4 weight tier
because it scored 347 catastrophic positions against a bar, after passing
three careful controls. The bar was wrong: it priced an 8-bit KV cache
format, and the 4-bit weight tier I was already shipping scored 455 on the
same instrument, worse than the thing I was killing. The dump proving it had
been on disk for hours.

The feature's own kill switch is the cheapest control, and the incumbent is
the one I should have run first.

## The instrument has state too

Profiling said the decode round spent 1.5-2.2 ms in host synchronization,
which made restructuring the host loop the obvious next lever. A graph-level
trace said 0.22 ms. The difference was nsys node tracing, which added 1.27 ms
per round of its own overhead that I was reading as engine time. Two levers
died that evening on re-measurement.

Same class, different week: most of the wrong numbers on August 18th came
from a stale instrument (an old binary, a leftover cache root, a profiler
bracket that only existed on one code path), never from the engine. Before
believing a surprising number, re-run the instrument's own control: same
config twice should be identical.

## Checksum the bytes before you trust the upload

q27's greedy output drifted every so often on the 5090, and I spent a day
hunting an engine race. Roughly 1% of model loads arrived on the GPU with
bit flips, scaling with the bytes copied (0.67% for a 16 GB model, 1.4% for
21 GB). I blamed host DRAM first, then the 5090 itself. Both were wrong. With
the cards physically swapped between slots it hit both GPUs at the same rate
(6/250 and 4/250), llama.cpp corrupted its loads just as often as q27 did, and
the same 22.5 GB buffer came through clean on every round copied from pinned
host memory while pageable copies flipped bits. The fault is in this host's
pageable DMA path, which q27 still loads through; what changed is that q27
now catches it.

It went unnoticed for months because token gates can't see it: 2 of 3 corrupt
runs produced bitwise-identical logits, since the flipped bits landed in
tensors that prompt never read. q27's own weight check passed too, because it
checksummed the device copy after the upload and blessed whatever landed. The
fix is boring. Sum the bytes on the host right before the copy, sum them on
the device after, and print both on every load. Every gate run since starts
by checking that pair.

## Averages hide the tail you care about

When I added a 5-bit key format for the KV cache (turbo5k), perplexity said
it was worse than the 3-bit one it was meant to replace: +0.983% versus
+0.804% against full precision. Every tail metric said the opposite: 43%
fewer catastrophic positions (65 vs 114, where full precision was confident
and the cache made it wrong), 23% fewer large errors. 97% of the 3-bit
format's absolute error cancels in the signed mean, so perplexity barely
sees it. Had I gated on perplexity, I'd have shipped the worse format as the
Ampere default.

Same lesson from a borrowed claim: a forum post said precision loss clusters
at tool calls. On our traffic, tool positions flip at 0.2-0.4x the rate of
ordinary prose, every row. JSON keys and closing tags are the tokens the model
is surest about. Measure the distribution you care about, on your own
traffic, before you repeat someone else's.

## Condition before you compare

After the defaults fix, a remaining quality gap between q27 and llama.cpp
still looked real. Decomposed, it was one task, and inside that task one
unspecified API detail: the test harness constructs the class with a bare
file path, so an implementation that takes an options object fails 12 of 25
tests. Conditioned on which form the model happened to write, q27 scored
0.996 and llama.cpp 0.973. q27 had drawn the losing form 6 times in 15,
llama.cpp 0 in 3, and P(0 of 3 at 40%) is 0.22. Five measurements of one arm
and three of the other isn't a comparison.

## Decompose the competitor before chasing it

ninfer decoded faster than q27 on the same model. I assumed kernels and spent
a morning on a plan. Then I profiled ninfer directly, and their round wall was
18.2 ms against q27's 17.9, which is a tie. Their whole lead was tokens
accepted per round, and inside that, positions 5-7 of the draft, which their block
drafter produces for free where our ladder paid for each step. The actual
defect on our side was a hard cap: the sampled ladder never drafted past
depth 4, and live traffic always samples. Lifting that cap onto the adaptive
ladder and adding a sampled path for the block drafter closed the gap to
decode parity (218-222 vs 218 t/s). The matmuls were already as fast as
theirs; the difference was drafting policy.

## Build equality into the design, then test for equality

The 8 GB Bonsai work needed a new weight container: five ternary digits per
byte instead of four 2-bit codes, 1.6 bits per weight instead of 2.125. I
could have gated it against a tolerance. Instead the device layout is chosen
so the new kernel sums exactly the chunks the old 2-bit kernel sums, in the
same lane order, with the same float chain. That makes the test "bitwise
equal", and it held: 400 of 400 matrices at every verify width, the same
wikitext NLL to six digits, identical server output.

A tolerance gate would have passed a layout bug that shifted one chunk; a
bitwise gate fails on it immediately and names the matrix. Where equality
can't be the contract, I write that down instead: q27's multi-lane verify
at width 4 and up flips near-tie words against plain decode on both cards and both models, it's
logged as a tolerance class, and width-2 rounds are the only fused shape that
is bitwise.

## Count instructions, then bytes

That five-trit container decoded at the 2-bit container's speed on a 3090
despite 24% fewer bytes. Two plausible fixes (a cheaper digit gather, fully
contiguous loads) moved nothing. The SASS said why: the inner loop issued
72.6 instructions per 32 weights against 41.5 for the old kernel. A digit
extraction without a dependency chain (digit = T(r+1) - 3T(r), each term read
straight from the original byte, the four-lane subtract one integer
multiply-add) plus hoisted addressing got it to 56.4, and decode went from
72-74 to 79-82 t/s, past the 2-bit pack. I had assumed a GEMV at 1.6 bits per
weight was memory-bound. Dumping the SASS would have told me otherwise on day
one, and it takes five minutes.

## Run the default configuration, and a real device, at least once

The 8 GB container passed every gate I wrote. Then I wrote a one-shot
installer for a friend's card, ran it end to end here first, and its first
request crashed: with the server's defaults, a multi-slot conductor sized a
workspace from 4-bit, 8-bit and 2-bit tensors only, found none in the new
pack, and asserted on a null pointer. Every gate had set the flag that turns
the conductor off.

The friend's card then reported 36,864 tokens of context where my simulation
(a process holding VRAM down on a 3090) had predicted 45,056. A real 8 GB card
exposes about 7.9 GB, and I'd simulated 8.3. The simulation was a fine
instrument for relative numbers and a bad one for the absolute that the
README quoted. It says 36.9K now.

## Measure outside ideas on your own traffic first

A recent post on llama.cpp's n-gram drafter cut its lookup cost from 165 us
to 1.2 us, and it's good work. q27's suffix drafter proposes in 0.05 us
against a 14 ms verify round, so none of that transfers; what did transfer
was noticing that we rebuilt the drafter's index over the whole prompt on
every request, 2-34 ms at 32-200K tokens. That's now incremental: 0.2 ms, same
index. I also replayed the post's two drafting ideas over 786K output tokens
of recorded Claude Code sessions. A cross-session cache added 0.5% coverage.
A frequency fallback fired on 28% of rounds at 1.2 accepted tokens each,
which would replace model drafts worth 3-4 tokens. Neither got built, and
the replay took an hour.

## Stop when the bar says stop

I built a 4-bit-weight, 8-bit-activation prefill GEMM spike, got its
bitwise-exact design reviewed, and measured 1.30-1.41x against a declared bar
of 1.6x. Then a second attempt with TMA loads landed at 1.33-1.39x. The port
stopped there. Declaring the bar before the first measurement is what made
stopping cheap; without it, 1.4x would have looked close enough to keep
going.

## The rules, as they stand

- List every field the client omits and each side's default for it.
- Compare token ids against the reference tokenizer, never only text.
- Run the feature-off control and the incumbent through any new gate.
- Re-run the instrument's own identity control before believing it.
- Sum weights on the host before upload and on the device after, every load.
- Gate on tail metrics; treat perplexity as a sanity check.
- Condition on the discriminating variable and check the other side's n.
- Profile the competitor directly before planning to catch it.
- Design for bitwise equality where you can, and label tolerance classes where you can't.
- Count instructions per byte before calling a kernel memory-bound.
- Run the default configuration and a real device before the README quotes a number.
- Replay borrowed ideas on your own recorded traffic before porting them.
- Write the kill bar down before the first measurement.

None of these are new. I had to pay for each one anyway, and the build log
has the receipts. Write your kill criteria down, and keep the log.
