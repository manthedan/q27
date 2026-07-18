# QA before releases — the single release checklist

**Status: v1 (2026-07-17, triage item I3).** One place that collects the
gates currently scattered across the chronicle, `make` targets, and
`tools/`. Run the whole list before any tagged release (`v0.1.0` cuts the
homebrew formula's tarball — a tag says "someone ran this list"). A gate
with a stale or skipped leg is recorded in the release notes with its
reason; an unrecorded skip is a process failure (masked-failure lesson:
the checklist must be able to fail).

Box legend: **M4** = the 24 GB serving Mac (this repo's metal lane),
**yukon** = the CUDA 5090 box, **mini** = the 16 GB secondary mac.

## 0. Preconditions

- [ ] Working tree clean: every round committed or explicitly HELD with
      its plan doc saying why (`git status --short` has no surprises;
      stage explicitly — never `git add -A`, tasks/lessons.md).
- [ ] Quiet machine for every timed leg: one model resident at a time,
      input-idle, `caffeinate` wraps each leg
      (logs/q4port-20260717/COORDINATION.md — two OOM crashes proved why:
      never two 17 GB residents; pgrep+curl health-check before any
      server launch).
- [ ] `docs/METAL_PROGRESS.md` chronicle current through HEAD (grep-count
      each new entry's headline == 1 after any merge).

## 1. Build + unit (M4, ~5 min)

- [ ] `make clean && make -j` (or `make all`) — zero warnings tolerated
      in the server/engine TU.
- [ ] `make test-cpu` — ALL PASS (toolconstrain, suffixdraft, sampling,
      kl, artifacts, depthctl).
- [ ] `make test-metal` — matvec, ops (incl. turbo3 attention, fp16
      exception window, e4m3 store goldens), stream format + splitter.
- [ ] `./build/test_tokenizer models/qwen36-27b-mtp/qwen36-27b-mtp.tok /dev/null`
      — all artifact-backed tokenizer/API self-tests PASS (including bare-call
      quote repair and incremental tool-call streaming).

## 2. Canonical correctness (box-marked)

- [ ] **yukon:** CUDA 16-token byte gate — canonical CLI run matches the
      recorded SHA (the merge-time reminder: mandatory when loader.cpp,
      tokenizer, or shared headers — api_common.h, stream_split.h,
      tool_preamble.h — were touched; the byte-exact contract lives on
      the CUDA side).
- [ ] **M4:** `./build/q27-metal --chunk-parity 384` on the official
      artifact — widths 17/48/96 bit-identical to width 12.
- [ ] **M4:** `tools/snapshot_gate.sh` — Phase-1 byte identity both KV
      dtypes + reject matrix + crash legs + the l7full exception arm.
- [ ] **M4:** `tools/failpoint_gate.sh <model.q27>` — post-throw engine
      state (position held; reset+regenerate byte-identical).
- [ ] **M4:** `tools/ckpt_gate.sh` — divergence-replay cache aliasing.
- [ ] **M4:** `tools/constrain_gate.sh` — grammar engage/disengage,
      split-brain rebind, pool-full sticky off.

## 3. Live-server API gates (M4, one server, serialized)

Per COORDINATION.md: pgrep + `curl :8213/health` before launch; one
17 GB resident; gates against the running server only.

- [ ] `tools/agentic_parity_gate.sh` — Anthropic + OpenAI chat tool
      loops, count_tokens, ctx-limit 400, prefix hit (G2–G7).
- [ ] `tools/responses_parity_gate.sh` — /v1/responses function_call
      items, stream lifecycle, round-trip, custom tool, 400 class
      (G8a–G8d, G9).
- [ ] `tools/trace_gate.sh <trace.jsonl>` — start this release-trace server
      with `Q27_METAL_TEST_FAILPOINTS=1` and `--trace`; the suites above plus
      deterministic malformed-recovery, disconnect, and Responses engine-error
      legs leave a complete correlated stream: boot, all 5 API families,
      outcomes, 400/500s, recovery/cancel IDs, and per-boot monotonic tms.
- [ ] Cancel smoke: kill a long prefill client-side; slot frees within
      one chunk; follow-up probe answers (the 2026-07-17 cancellation
      gates' standing form).
- [ ] pi/Claude Code smoke: one real read-tool loop against the server.

The next gates **self-launch or load the artifact themselves**. Stop the
resident API server first, verify no `q27-metal-server`/`q27-metal` process
remains, and run them strictly serially under the same coordinated slot:

- [ ] Set `Q27_GATE_MODEL=<official-release-model.q27>` and
      `Q27_GATE_TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok`, then run both
      `tools/multislot_gates.py` **and** `tools/multislot_gates.py mtp` —
      self-launching Metal servers; greedy constraints/admission/503 plus MTP
      scheduling, live speculation counters, fairness, and cancellation.
- [ ] `tools/suffix_burst_gates_2026-07-16.sh` — Metal suffix-burst committed
      byte identity and actual wide dispatch (not the legacy CUDA
      `tools/suffix_gate.sh`).

## 4. Quality + perf spot checks (quiet M4 or as noted)

- [ ] Official 8K NLL leg (`--nll-long 8192 --ctx 8192`, wikitext-2)
      within ±0.5% of the recorded release-baseline number.
- [ ] Decode + prefill spot numbers recorded in the release notes
      (frontier shape: instantaneous tok/s at 2k/4k/8k from one load —
      triage I4 — not whole-run averages).
- [ ] **yukon:** `tools/metal_cuda_gate.py` parity run when
      kernel/numerics rounds rode the release window.

## 5. Release mechanics

- [ ] SHADER_ABI audit: any kernel dispatch-topology or argument-struct
      change since the last tag bumped SHADER_ABI (the stale-shader
      lesson: runtime-compiled source from another checkout served stale
      kernels; tag-relevant even when args are unchanged).
- [ ] Docs consistency sweep: README endpoint/env-knob list matches
      `metal_server.cpp` reality; shipped-semantics knobs promoted to
      documented flags (triage note: snapshot auto, KV fp16 cells,
      max-tokens default); METAL_PROGRESS risk register current.
- [ ] LICENSE: inherited terms from upstream `signalnine/q27` and the
      Bonsai/PrismML artifacts verified compatible for distribution
      (homebrew plan B-blockers; record the resolution).
- [ ] Weights manifest: `CHECKSUMS.md5` per published artifact; HF repos
      reachable (homebrew plan B3).
- [ ] `git tag vX.Y.Z` + release notes listing: shipped rounds (chronicle
      headlines), held/parked items with reasons, every gate above with
      its evidence log path, and known limitations.
- [ ] Homebrew formula `url`/`sha256` updated on the tag tarball;
      `brew install --build-from-source` smoke on a clean CLT machine
      (the no-toolchain-beyond-CLT target).

## 6. Cross-feature and environment (expert review 2, 2026-07-18)

- [ ] **Feature-interaction matrix:** suffix bursts × disk snapshots ×
      two-slot scheduling × KV exception cells — each gated alone to
      date; one seeded end-to-end matrix run (each pair, one seed) as
      "pass alone, fail together" insurance before the mixed-pack ship.
- [ ] **Thermal governance:** every bench artifact carries a powermetrics
      thermal-pressure log line; benches that gate ship decisions record
      power state (twice-burned: gate-3 wall anomaly, the 12.66→10.57
      ceiling drift).
- [ ] **Temperature coverage statement:** release notes state that
      suffix bursts / MTP / tool-constraint engage at temperature 0 only
      (metal_server.cpp:660-661,952) until the sampled-path acceptance
      battery lands (pre-registered in the suffix plan).
