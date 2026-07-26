# A7 — the first request in a fresh process pays ~9–10 s of one-time cost

Measured 2026-07-25 (base M4, default tier, GPU slot). **Open. One fix
attempted and measured as NOT working; the cost itself is confirmed and
reproducible.**

This supersedes the first framing of A7 in the A5 NO-GO commit
(`b5923e3`), which attributed the cost to *residual prefill / chunk
granularity*. That was wrong — see "What it is not".

## The measurement

Three **identical** disk-restore requests inside one process, from
`--trace` (`tms` = ms since boot):

| # | tier | hit | residual | restore (`request`→`prefix`) | rest of request |
|---|---|---|---|---|---|
| 1 | disk | 6912 | 95 | 231 ms | **13778 ms** |
| 2 | disk | 6912 | 95 | 421 ms | **4687 ms** |
| 3 | disk | 6912 | 95 | 220 ms | **4637 ms** |

Same tier, same hit, same residual, same work — the first request costs
**~9.1 s more** than the identical requests that follow it. The cost is
attached to *being first in the process*, not to restoring, not to the
prompt, and not to the disk.

Confirmed by absorbing it deliberately: in a fresh process, a trivial
2-token `{"prompt":"hi","max_tokens":1}` request cost **10362 ms**, after
which the big restore request cost **4931 ms** instead of the 14081 ms it
costs when it goes first.

## What it is not

- **Not residual prefill.** 95 tokens is one full chunk
  (`PREFILL_CHUNK_MAX = 96`), and the steady-state cost of exactly that
  work is 4.6–4.7 s. The extra 9.1 s is not in it.
- **Not chunk granularity.** Ruled out by the same numbers.
- **Not the disk read.** That is 220–231 ms, separately traced (see the
  [A5 NO-GO](2026-07-25-boot-prefetch.md)). This *reinforces* that kill:
  boot prefetch was chasing ~194 ms of a 14 s first request whose real
  content was ~9 s of warm-up.
- **Not restore-specific.** It is the first request of any kind. The
  322 s cold-prefill baseline carries it too.

## Leading hypothesis

Weights are mmap'd, so the first pass that genuinely reads them faults in
the whole ~17 GB artifact — ~6 s at this box's measured 2.8 GB/s — and the
production PSOs build lazily on first dispatch on top of that. Together
that is the right order for ~9–10 s. **Not yet confirmed by instrumentation**
(no per-phase timing inside the first request).

## Fix attempt #1 — decode-step warm-up: NO-GO

Ran at the end of `Runtime` construction, on the engine rather than through
`run()` so it could not pollute the prefix cache or snapshot store:

```cpp
s->engine.reset();
s->engine.step(tok);     // one decode step reads every weight
s->engine.reset();
```

| arm | boot | first request | second |
|---|---|---|---|
| `--no-warmup` | 32659 ms | 16594 ms | 209 ms |
| warm-up on | 42886 ms (+10227) | **18780 ms** | 74 ms |

The warm-up itself took 10045 ms and the first request **did not get
faster**. Pure loss: +10 s of boot for nothing. **Reverted** (not shipped).

Why it presumably failed: a serial `step()` exercises the decode path, but
the expensive first-request work lives in the *request* path — chunked
prefill and its head projection have their own PSOs and access pattern. A
real 2-token request warms it (measured above); a bare decode step does not.

## Next step

Warm through the **actual request path** rather than `engine.step()` — i.e.
issue a real minimal generation at boot, the programmatic equivalent of the
`{"prompt":"hi","max_tokens":1}` that is measured to work, while still
keeping it off the prefix cache and snapshot store. Gate on the same two
numbers: boot delta vs first-request delta. Ship only if the first-request
saving is real; the previous attempt shows that must be measured, not
assumed.

## Side result worth keeping

Steady-state disk restore is **~4.9 s** for a 6912-token prefix (+95
residual) against **~312 s** of cold prefill for a comparable prompt
(5606 tokens at 55.7 ms/token, warm-up excluded) — roughly **64×**. The
~21.8× recorded in the A5 doc was measured on *first* requests, so it was
diluted by exactly the warm-up described here. The snapshot tier is worth
even more on Metal than that figure suggested.
