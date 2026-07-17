# Homebrew distribution — Metal fork packaging for end users

**Status: DESIGN (2026-07-16). No packaging work has landed.** This plan
covers getting `q27-metal-server` onto an end user's Apple Silicon Mac via
`brew install manthedan/tap/q27`, with weight artifacts pulled separately
from Hugging Face. Target user: M-chip MacBook, 16 GB unified memory, no
dev toolchain beyond Command Line Tools, wants a local Anthropic endpoint
for Claude Code.

The end state:

```bash
brew install manthedan/tap/q27   # small bottle: binaries + wrapper
q27 pull                         # detects RAM -> recommends tier -> downloads + md5-verifies
q27 serve                        # loopback server, working-set-aware
export ANTHROPIC_BASE_URL=http://localhost:8080 && claude
```

## Decisions (with reasons)

### D1. Tap: `manthedan/homebrew-tap`

The tap exists, is public, and already has the standard layout
(`Formula/macroscope.rb` + `.github/workflows`). Homebrew namespaces
formulas per-tap, so `manthedan/tap/q27` can never collide with an
upstream `signalnine/tap/q27` if one ever appears. Publishing under our
own namespace is also correct fork etiquette; if the Metal work merges
upstream later, migration is a one-line `deprecate! because: :repo_moved`
pointing at the upstream formula.

### D2. Binary via formula; weights explicitly NOT

The engine is a self-contained binary with zero runtime deps — ideal brew
material. The weights (1.8–17 GB) are hostile brew material: no resumable
downloads, cache duplication, full re-download on every formula upgrade.
Weights live in `~/.q27/models/`, managed by the wrapper (D4), downloaded
from HF with `curl -L -C -` (resumable) and verified against the
`CHECKSUMS.md5` we already ship in every model directory.

### D3. From-source formula is viable — runtime shader compilation saves us

`metal_backend.mm` compiles shaders at runtime via `newLibraryWithSource:`
(metal_backend.mm:508); the Makefile has **no `xcrun metal` step**. So the
build needs only clang++ from the Command Line Tools, which Homebrew
guarantees on every machine. A from-source formula in the
`macroscope.rb` mold works today (modulo the shader-path blocker, B1
below). Bottles are a Phase-3 optimization, not a prerequisite — the
build is seconds of clang, not a Rust dependency tree.

Runtime compilation is also the compatibility escape hatch: no
metallib-vs-macOS-version matrix, one artifact per release.

### D4. Supervisor: a shell wrapper, not Go/Rust, not the engine

Audited 2026-07-16: **no downloader, registry, or supervisor exists
anywhere in the repo** — `tools/` is bench rigs and gate scripts. It needs
building, and v1 should be ~150 lines of shell (`bin/q27`) installed by
the formula:

- `q27 pull [name]` — `sysctl hw.memsize` → tier recommendation table →
  resumable HF download → `md5 -c` verify → `~/.q27/models/<name>/`
- `q27 serve [name]` — resolve name → path, refuse if another
  `q27-metal-server` is already running (`pgrep` guard — this bakes in the
  one-model-loading-process-at-a-time discipline instead of trusting users
  to know it), then `exec` the server
- `q27 ls` / `q27 rm <name>` — cache management
- Model switching = process replacement. Weights are mmap'd
  (loader.cpp:153) and wrapped zero-copy (metal_backend.mm:728), so a
  warm-cache load is seconds; a keep-alive multi-model daemon is
  complexity that has not earned itself. Graduate to Go only if we later
  want on-demand switching or download progress UI.

The engine binary grows none of this. Its narrow/no-deps character is a
feature we are protecting.

### D5. Tier selection: auto by memory, ternary is the 16 GB answer

macOS caps GPU-usable unified memory at roughly 65–75% of RAM;
`recommendedMaxWorkingSetSize` reports it exactly and is already plumbed
(metal_backend.h:211). Selection at `q27 pull` time: recommend the largest
artifact whose weights + a starter KV budget fit under the working-set
cap, print the table, `--tier`/explicit name overrides.

| machine | ~usable working set | recommended artifact | size |
|---|---|---|---|
| 16 GB | ~10.5 GB | `ternary-bonsai-27b-t2` | 6.7 GB |
| 24 GB | ~17 GB | `ternary-bonsai-27b-t2` default; `qwen36-27b-mtp` (Q4, 17 GB) opt-in — it fits but is exactly borderline and needs an otherwise-idle machine | 6.7 / 17 GB |
| 32 GB+ | ~24 GB | `qwen36-27b-mtp` | 17 GB |

Honesty note for the table the wrapper prints: the ternary artifact is a
different *model* (Bonsai-27B ternary tier), not a quant of Qwen3.6 —
"smaller" here trades model, not just bits. The mmap + `bytesNoCopy`
path means weights are file-backed and pageable: under memory pressure a
16 GB machine degrades (page faults) instead of dying, which is the right
consumer failure mode.

## Blockers found in the current tree

- **B1 (hard): hardcoded shader path.** The shader loader searches
  `src/metal/q27_kernels.metal` relative to cwd (metal_backend.mm:73) and
  throws if absent (metal_backend.mm:98). An installed binary run from any
  directory but a repo checkout fails at startup. Fix: embed the `.metal`
  source into the binary at build time (xxd-style generated header;
  Makefile rule). Keeps the one-file-no-assets property; the on-disk
  candidate paths stay as a dev override. This is the only engine-code
  change the whole plan requires.
- **B2 (soft): no auto-ctx.** The server defaults to a fixed
  `--ctx 8192` (metal_server.cpp:658). The working-set *guard* exists —
  cache allocation refuses to exceed half the working set
  (metal_engine.cpp:238, metal_server.cpp:235) — but ctx does not
  auto-size to it the way the CUDA server sizes to free VRAM. v1 ships
  with the fixed default (safe everywhere in the table above); auto-ctx =
  solve ctx from `(working_set/2 − resident weights)` is Phase 3. On
  unified memory, size to the budget with 2–3 GB headroom, never to the
  cap — the OS and the user's browser share this pool.
- **B3: no HF artifacts under our namespace.** The ternary/binary Bonsai
  `.q27` artifacts exist only on this machine (`models/`, with
  `CHECKSUMS.md5`). They need publishing (HF repo per model, Apache-2.0
  inherited) before `q27 pull` has anything to pull. The Qwen tiers can
  point at the existing upstream `signalnine/Qwen3.6-27B-MTP-q27` repo.
- **B4: no tagged releases on `manthedan/q27`.** The formula `url` pins a
  tag tarball + sha256 (macroscope pattern). Needs a `v0.1.0` tag and a
  release-cut habit.

## Formula sketch

```ruby
class Q27 < Formula
  desc "Narrow Metal inference engine for Bonsai/Qwen3.6-27B-MTP; local Anthropic endpoint"
  homepage "https://github.com/manthedan/q27"
  url "https://github.com/manthedan/q27/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "..."
  license "MIT"          # confirm — match repo LICENSE

  depends_on :macos
  depends_on arch: :arm64

  def install
    system "make", "build/q27-metal-server", "build/q27-metal"
    bin.install "build/q27-metal-server", "build/q27-metal"
    bin.install "packaging/q27"           # the wrapper (D4)
  end

  test do
    # loads Metal device + compiles shaders, no model needed
    assert_match "usage", shell_output("#{bin}/q27-metal-server 2>&1", 1)
  end
end
```

Name: `q27` (formula and wrapper), keeping the engine binaries'
`q27-metal*` names as-installed. If upstream ships CUDA binaries under a
brew name someday, per-tap namespacing (D1) keeps us collision-free.

## Model registry

A flat manifest, versioned in the engine repo (`packaging/models.tsv`),
baked into the wrapper at install time:

```
name                    hf_repo                                 files                              min_ws_gb
ternary-bonsai-27b      manthedan/Ternary-Bonsai-27B-q27        ternary-bonsai-27b-t2.q27 *.tok    8
qwen36-27b-mtp          signalnine/Qwen3.6-27B-MTP-q27          qwen36-27b-mtp.q27 *.tok           18
```

Living in the repo (not the tap) means a model release rides an engine
release, and `q27 pull` never needs a network call to know what exists.

## Phases and gates

- **Phase 0 — embed shader source (B1).** Gate: `build/q27-metal
  --version` (or the canonical 16-token continuation) succeeds from an
  arbitrary cwd with the repo deleted from the path. Engine change is a
  Makefile rule + one loader fallback; the runtime-compile path itself is
  already what ships.
- **Phase 1 — wrapper + manifest + HF publish (D4, B3).** Gate:
  fresh-account simulation (`HOME=$(mktemp -d)`): `q27 pull` recommends
  correctly for this machine's `hw.memsize`, downloads, md5-verifies;
  `q27 serve` boots; `curl localhost:8080/v1/messages` returns tokens;
  second concurrent `q27 serve` is refused by the pgrep guard.
- **Phase 2 — formula in the tap (B4).** Tag `v0.1.0`, land `q27.rb`.
  Gate: on a machine (or clean user account) that has never seen the
  repo: `brew install manthedan/tap/q27 && brew test q27`, then the full
  Phase-1 gate sequence, then the Claude Code smoke
  (`ANTHROPIC_BASE_URL` + one tool-use turn).
- **Phase 3 — polish (all optional, demand-driven).** Bottles via the
  tap's existing GH workflow (arm64 monterey+); `brew services` launchd
  plist for always-on serving; auto-ctx (B2); download progress UI
  (this is the "graduate the wrapper to Go" trigger, if it ever fires).

## Open questions

- **Q1:** Does the 1.8 GB dspark drafter pack ship as a user-facing
  artifact or stay a dev fixture? (Ride the DSpark port decision —
  2026-07-16-dspark-port-phase0.md; the manifest format doesn't care.)
- **Q2:** Server identity: today the single-model server accepts any
  model name in requests. The wrapper should at minimum write the loaded
  model name into `/health` so clients can tell what's resident. Engine
  or wrapper responsibility? (Lean engine: one string, one endpoint.)
  Adjacent finding (2026-07-17 e2e smoke): codex 0.144's models-manager
  probes `/v1/models` and fails to decode our (OpenAI-standard,
  CUDA-identical) `{"object":"list","data":[...]}` — it wants its own
  ModelInfo shape (`missing field models`). Non-fatal: falls back to
  default metadata and proceeds. Codex-side nicety, not a wire bug;
  candidates for its config docs, not our endpoint.
- **Q3:** LICENSE — **RESOLVED (verified 2026-07-17).** The repo now has
  an MIT LICENSE with the upstream attribution line; GitHub's API
  confirms upstream `signalnine/q27` declares MIT (`license.spdx_id`,
  LICENSE on `master`), so the formula's `license "MIT"` is correct
  as sketched.
