This release re-validates Qwen3.8 (`q38`) for 24 GB Macs and speeds up
Bonsai 2 decode, prefill and startup. The TUI's frame cost no longer grows
with transcript length. Upstream q27 is merged through 5dcf4df.

```sh
brew upgrade manthedan/tap/q27      # or: brew install manthedan/tap/q27
q27 agent                           # Bonsai 2 (b2), the default on every Mac
q27 pull q38 && q27 agent q38       # 24 GB Macs: Qwen3.8 27B c-small, 14.6 GB
```

## Qwen3.8 is back (24 GB)

- `q38` (Qwen3.8 27B MTP, `q38-c-small-v1`) passes the Metal gates on a
  24 GB M4 and is no longer marked experimental. The gates are:
  - policy negative contracts, with all six malformed fixtures rejected;
  - the published 128-token canonical digest;
  - speculative vs. plain prefix identity;
  - OpenAI Chat, Responses and Anthropic tool round trips;
  - the native agent through the packaged wrapper.
- **Defaults are unchanged.** `b2` stays the recommended pack on every Mac it
  fits, for `agent`, `serve`, `pull` and `recommend`. `q38` is listed as a
  validated 24 GB option; choose it by name.
- `tools/qwen38_laptop_gate.sh` runs the whole Qwen3.8 gate in one command.
  It pulls the pack when it is missing.
- The canonical registry carries per-model digests: `canonical_md5_for ARCH
  TIER [MODEL]`. Upstream's two-argument callers still get the Qwen3.6
  digests.

## Faster (Bonsai 2 on an M4 16 GB Mac, outputs bit-identical)

- **Decode +2.2%**:
  - a shared projection input is rotated once instead of once per weight
    (−144 GPU dispatches per token);
  - the residual add is folded into the T2 matvec (−128 dispatches per
    token);
  - RMSNorm runs float-only on all-Bonsai packs.
- **Prefill ~43 → ~50 tok/s**. The T2 GEMM stages weights as half (exact
  for trits), which cuts GEMM time by 17%, and it flushes once per 128
  columns.
- **Short prompts**: batches of up to 8 tokens use an 8-token tile. A 6-token
  follow-up turn takes 264 ms of GPU time instead of 386 ms.
- **Startup**: opening a cached pack takes 0.3–0.8 s instead of 11–13 s
  (~5–7 s cold). Packs are mapped shared with read-ahead, and the T2 payload
  scan is vectorized and runs in parallel.
- **TUI**: a per-block render cache and windowed scrollback bring a frame at
  400 turns from 94 ms to 0.4 ms. Very long lines are clipped with an
  elision note.

## For contributors

- **Perf ratchets.**
  - `q27-metal --counters` reports command buffers, GPU dispatches and GPU
    time.
  - `tools/perf_ceilings.sh` gates the deterministic counts exactly against
    `perf/ceilings.tsv`, which now includes the first `q38` rows. Its
    `--ratchet` flag only lowers limits.
  - `tools/perf_journeys.sh` gates wall clock against a per-machine baseline
    (best of 3, 10% band).
  - Both gates run in `release_gates.sh` tier B; see `perf/README.md`.
- **DFlash2 speculative decoding is not pursued on M4.** Verifying drafted
  tokens is bound by the matrix units, whose peak measures 3.85 TFLOP/s on M4.
  That caps the projected speedup at 1.1–1.3×. See
  `docs/metal/plans/2026-10-06-dflash2-metal.md`.
- **From upstream:** `repack --name` for fine-tunes, and the split MTP join
  accepts the newer per-layer tensor names.

## Gates

All gates ran on release candidate f360af2. The final commit only changes the
registry, the recommendation rule, docs and the ceilings table; the smokes
were rerun on the final archive.

- **Mini, M4 / 16 GiB.**
  - `release_gates.sh` tier B passed 17/17, including the perf ceilings. T3
    recovery and snapshot reuse are N/A by design.
  - Tarball and Homebrew smokes passed. The Homebrew smoke used a local tap
    and ran `brew test`. Each smoke did a real `q27 pull b2` from
    `manthedan/q27-bonsai-packs@6d59ea13`, the serve + API round trips and a
    native-agent tool loop.
- **Laptop, M4 / 24 GiB, macOS 26.4.**
  - `tools/qwen38_laptop_gate.sh` passed 6/6. Pack and tokenizer SHA-256
    matched the registry.
  - Tier C passed 9/9, and Bonsai tier B passed 17/17.
  - Independent Prism reference, T2, rebuilt from pinned sources at
    `1a07bfa5`: 339/340 argmax matches. The one miss is a near-tie, with zero
    high-margin mismatches; max KL is 3.3e-7 nats, the same as v0.7.0.
- **Perf journeys** (wall clock) were skipped on both machines because
  neither has a validated quiet-machine baseline yet.

Reviews: Sol 6.1 xhigh reviewed every change. Astra and Opus also reviewed
the merges, the loader and the fix commits. Every finding was verified
before it was fixed.

## Known limitations

- Qwen3.6 tiers, Bonsai 1 packs and `q38-q4s` remain experimental (not
  re-validated).
- The Qwen3.8 canonical digest is published for M4 / M4 Pro only.
- No MTP on Bonsai 2; suffix bursts remain its speculation path.
- No new long-context soak was run for this release.
