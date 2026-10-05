# q27 metal-v0.7.0 — Bonsai 2 on Apple silicon, native agent, suffix speculation

First prebuilt release of the Metal revival. The default model changes from
Bonsai 1 to **Bonsai 2 27B** (Prism ML's ternary Qwen3.8-trained model), which
runs on a **16 GB** Mac. Upstream q27 is merged through v0.14.3.

```sh
brew upgrade manthedan/tap/q27      # or: brew install manthedan/tap/q27
q27 pull b2                         # 6.7 GB, SHA-256 verified
q27 agent                           # native coding agent (Ratatui TUI on a terminal)
q27 serve                           # OpenAI/Anthropic/Responses API on :8080
```

Bonsai 2 weights: **Created using Bonsai by Prism ML** (Apache 2.0).

## Behavior changes (defaults)

- **Default pack is `b2`** (Bonsai 2 T2 slim, 6.7 GB) for `q27 agent`,
  `q27 serve` and `q27 recommend`. `b2-t3` (5.6 GB) is the smaller option;
  it prefills serially, so long prompts start slower and suffix bursts are off.
- **Bonsai 2 profile** (`bonsai2-qwen38-v1`): thinking on with an unbounded
  budget, sampling T=1.0 / top-p 0.95 / top-k 20, agent context 16K, server
  one slot at 16K (`Q27_SERVE_CONTEXT`, `Q27_SERVE_SLOTS`). 16K is the fp16-KV
  ceiling measured on a 16 GiB Mac; use `--kv q8` for up to 64K.
- **Suffix-burst speculation on by default** for Bonsai 2 (`--suffix 16`):
  n-gram drafts from the context verified in one batch, exact for greedy and
  sampled decode. Speeds up copying/editing text already in context (file copy
  in the agent: 63 s -> 41 s); no measurable cost on new prose. `Q27_SUFFIX=0`
  disables it.
- **Older packs** (Qwen3.6 tiers, Qwen3.8 `q38`, Bonsai 1 `b1`/`t2`) stay in
  the registry but are marked **experimental: not re-validated in this
  release**. If you depend on one, stay on metal-v0.6.1 for now. Qwen3.8 needs
  24 GB and will be re-gated in v0.7.1.
- `q27 serve PACK [args]` now forwards extra arguments to the server.
- TUI toggles moved to **Ctrl-T** (thinking), **Ctrl-Y** (theme) and
  **Ctrl-O** (markdown), so plain letters always type into the prompt.

## Also shipped

- Bonsai 2 on Metal: reads upstream's slim T2/T3 packs; reference parity
  against Prism's own implementation (t2-slim 339/340 argmax, the one flip a
  2e-5-logit tie; t3-slim 340/340; worst KL < 3.3e-7 nats).
- Batched float-activation prefill for T2 packs (~3x faster than serial:
  41 tok/s at 512 tokens, 27 tok/s at 13K).
- Prefix snapshots for T2 packs (a 3.3K-token system prompt answers in 5.2 s
  instead of 95 s when reused).
- `--kv q8`: rotated int8 KV cache, ~1,000x closer to the reference than
  turbo3 and faster than fp16 at depth.
- Native agent and Ratatui TUI hardening from four model reviews (tool-call
  parsing inside markdown fences, UTF-8 streaming, quit/drain races, session
  restore of a sampled pending token, and more).
- Upstream merges v0.14.1-v0.14.3, including the CUDA Bonsai 2 path and the
  `FP4_G16` loader contract.
- Packaging: relocatable shader lookup (release archive and Homebrew), binaries
  built for macOS 13+, `q27-lock-exec` lifetime lock, SHA-256-pinned pulls
  from immutable Hugging Face commits.

## Gates

Tier B (engine/serving delta since metal-v0.6.1) on the M4 / 16 GiB mini,
binaries built with `MACOSX_DEPLOYMENT_TARGET=13.0`, `tools/release_gates.sh`:

- **Tier C, 7/7** at the release commit: `test-tools`, `test-inspect`,
  `test-agent`, `test-repack` (+ Bonsai 2 repack), `test-packaging`,
  `test-shader-discovery`, TUI `cargo test`.
- **Metal + live Bonsai 2, 13/13:** `test-metal-backend`; t2-slim serving,
  native agent, Metal recovery, snapshot reuse; t3-slim serving, native agent.
  T3 recovery/snapshot-reuse are N/A by design (no chunked prefill, so no
  prefix snapshots).
- **Prism reference oracle:** not rerun for this release (needs a pinned
  Prism build). The 2026-10-01 evidence stands: t2-slim 339/340, t3-slim
  340/340. Later engine changes (suffix bursts, `--kv q8`) carry their own
  exactness checks: greedy suffix output byte-identical, sampled walk
  verified by Monte Carlo, q8 KV 126/127 against the reference.
- **Release smokes on the published archive:** fresh-HOME tarball smoke and
  Homebrew smoke (local tap, `brew test`), each with a real `q27 pull b2`
  from `manthedan/q27-bonsai-packs@6d59ea13`, `serve` + OpenAI/Anthropic/
  Responses round trips, and a native-agent tool loop: PASS.

Reviews: Sol 6.1 xhigh, Astra xhigh and Opus 5.5 on the release branch;
every finding verified and fixed (Homebrew symlink chains for
`q27-fetch`/`q27-bench`/`q27-report`, resumable downloads after transport
drops, per-pack pull lock, TUI shader discovery no longer reads the cwd,
`prism_gguf.py` shipped with the tokenizer exporter, Bonsai 2 bench route,
smoke-test harness fixes). Two Sol re-review rounds on the fix commits found
a stale-lock takeover race (pull lock is now a kernel flock) and test-harness
robustness issues; all fixed. The final follow-up commit (test/Makefile only)
was not re-reviewed.

## Known limitations

- Qwen3.8 (`q38`) and the other inherited packs were not run in this release.
- No MTP on Bonsai 2 (the checkpoint has no MTP layer); suffix bursts are the
  speculation path. T3 packs decode serially.
- Text only (no vision). The server accepts image parts and fields it does
  not implement (`min_p`, presence/frequency/repetition penalties) and
  ignores them, like upstream, rather than rejecting the request.
- Homebrew 7 asks you to trust third-party taps; installing by full name
  (`brew install manthedan/tap/q27`) trusts the formula.
- See [BONSAI2.md](../BONSAI2.md) for context limits, speeds and evidence.
