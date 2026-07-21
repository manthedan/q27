# Metal sampled MTP — CUDA Phase-2 rejection sampling on the host path

Status: **READY FOR OVERNIGHT** — Phase 0–1 + top-k polish landed; run `tools/overnight_sampled_mtp.sh` on a quiet machine for Phase 2 numbers.  
Branch: `agent-fenced-body-write` (or a follow-on branch if this stack splits).  
Machine: any Metal Mac with an official MTP pack for live gates; unit tests are CPU-only.  
Upstream design: `docs/sampling-design.md`, `docs/sampling-phase2-impl.md` (CUDA DONE 2026-07-05).

## Goal

When a Metal request runs **temperature > 0** on an artifact with an MTP layer,
keep **batched MTP draft + verify** and accept drafts via **Leviathan/Chen
rejection sampling** against the served target distribution — not exact-match
argmax. Output distribution matches plain `sample_from_logits`; speculation is
only a speedup.

Today Metal does:

```
temp == 0 && mtp  → mtp_round (greedy drafts, equality accept)   // fast
temp > 0          → sample_from_logits + step                    // correct, slow
```

Target:

```
temp == 0 && mtp  → mtp_round                                    // unchanged, bitwise
temp > 0  && mtp  → mtp_sample_round (greedy drafts, RS accept)  // NEW
temp > 0  && !mtp → sample_from_logits + step                    // bonsai / no layer
```

## Non-goals (v1)

- Sampled *drafts* (CUDA design 2b) — greedy drafts only.
- GPU Philox / Gumbel shaders — host rejection walk over readback logits first.
- Tool grammar under sampled MTP (CUDA also disables tools under sampling).
- Sampled suffix-burst (separate track; suffix is ngram, not the MTP head).
- Changing sampling defaults / making sampling default-on (exit criterion below).
- Touching CUDA sources (serving adoption dropped; CUDA is oracle-only).
- Bonsai packs (no MTP layer) — they stay on plain sample.

## Algorithm (CUDA-faithful, host-orchestrated)

After the same draft + verify as `mtp_round`:

1. **lanes[0] = pending** (always committed). **lanes[1..live-1]** = greedy MTP drafts.
2. Verify fills `clogits_[live × VOCAB]` — lane *k* predicts the token after `lanes[k]`.
3. For draft index *k* = 0 .. live-2, test draft `d = lanes[k+1]` under lane *k*’s
   **served** distribution *p* (temperature, top_p, top_k — same construction as
   `sample_logits_cpu`):
   - accept with probability `p(d)` (0 if *d* is outside the nucleus);
   - on first reject: stop; **resample** next pending from lane *k* with *d* excluded.
4. If all drafts accept: free bonus sample from the last verify lane (no exclude).
5. Commit *n* = 1 + accepted drafts (clamped by `remaining` / EOS like greedy).
6. `gdn_replay` + copy last hidden/logits — same commit path as greedy.

**Mass rule:** accept prob is `softmax_full(d) / nucleus_mass`, not raw softmax.
With top_p < 1, using softmax alone under-accepts by ~1/mass (CUDA gate caught this).

**Drafts stay greedy.** Only the verify tail + pending sample change.

## Metal shape vs CUDA

| Concern | CUDA | Metal v1 |
|--------|------|----------|
| Draft | fused graph, greedy | serial `mtp_forward` (unchanged) |
| Verify | fused multi-lane | `chunk_forward` + multi-head (unchanged) |
| Accept | `k_spec_accept` + Philox | **CPU** rejection walk + `mt19937_64` |
| Next pending | `k_sample_stop` Gumbel | **CPU** `sample_logits_cpu` (+ exclude) |
| Logits | device-resident | readback `clogits_` (≈1 MB/lane; OK for batch-1) |
| Graphs | 2nd sampled perm set | none — host quantum like today |

Later optimization (optional Phase 4): per-lane GPU top-k over-set to cut
readback; device nucleus only if profiling says host is hot.

## Parallelization map

Workstreams and what can run concurrently:

```
                ┌─── Phase 0: sampling.h helpers + test_sampling  ──┐
                │         (CPU-only, no Metal, no server)            │
                └──────────────────────┬─────────────────────────────┘
                                       │ API frozen
          ┌────────────────────────────┼────────────────────────────┐
          ▼                            ▼                            ▼
   Phase 1a: engine              Phase 1c: CLI flags           Phase 1d: gate
   mtp_sample_round              allow temp+mtp                script draft
   (metal_engine.*)              (metal_cli.cpp)               (tools/*)
          │
          ▼
   Phase 1b: server route
   temp>0 && mtp → sample round
   (metal_server.cpp — CAREFUL:
    other agent may have dirty edits)
          │
          ▼
   Phase 2: live gates (needs MTP pack)
          │
          ▼
   Phase 3: polish / defaults decision
```

| Stream | Depends on | Parallel with | Owner notes |
|--------|------------|---------------|-------------|
| **P0 helpers+tests** | nothing | doc only | land first; unlocks all |
| **P1a engine** | P0 API | P1c, P1d | main path |
| **P1b server** | P1a signature | — | serial after P1a; respect dirty `metal_server.cpp` |
| **P1c CLI** | P0 semantics | P1a | small; mutual-exclusion flip |
| **P1d gate script** | design only | P1a–c | shell can draft early |
| **P2 live gates** | P1a+b landed binary | — | needs official MTP pack |
| **Codex autoreview** | each landed commit | focused tests via `--parallel-tests` | default engine |

**Multiple agents:** after P0 lands, one agent can implement **P1a engine** while
another drafts **P1d gate script** and/or **P1c CLI** in a worktree. Do **not**
parallel-edit `metal_server.cpp` until the dirty sampling-defaults work is
reconciled. Prefer: write → commit → `autoreview --mode commit` → fix → clean.

## Phases and checklist

### Phase 0 — Host rejection-sampling helpers  `[x]`

**Files:** `src/sampling.h`, `src/test_sampling.cpp`  
**No Metal, no model.**

- [x] `build_served_distribution` matching `sample_logits_cpu` nucleus (temp/top_p/top_k, boundary ties)
- [x] `served_probability(token)` = weight/total or 0
- [x] `sample_served(..., exclude)` never emits exclude; renormalizes
- [x] `spec_rejection_accept(...)` → `{n, stop_lane, exclude, pending}`
- [x] Unit gates (identity, exclude, all-accept, cold draft, MC accept rate, residual MAE)
- [x] Full vs candidates reject-walk parity under top_k (same seed → same SpecRejectResult)
- [x] Commit + Codex autoreview clean (`d58ea12` + `4d8757f`)

### Phase 1a — `MetalEngine::mtp_sample_round`  `[x]`

**Files:** `src/metal/metal_engine.{h,cpp}`

- [x] New method + `generate_mtp_sampled` whole-run driver
- [x] GPU top-k / full-logits accept; commit path shared with greedy
- [x] Bootstrap first pending sampled; live&lt;2 serial fallback
- [x] Greedy `mtp_round` **untouched**
- [x] Commit + Codex clean (`27ed0d4` + follow-ups)

### Phase 1b — Server route  `[x]`

**Files:** `src/metal/metal_server.cpp`

- [x] Sampled MTP when temp&gt;0 && has_mtp && !Q27_SAMPLE_PLAIN
- [x] Quantum loop under lease; tool constrain still greedy-only
- [x] Folded sampling-default flags; Codex clean as part of Phase 1 stack

### Phase 1c — CLI  `[x]`

**Files:** `src/metal/metal_cli.cpp`

- [x] temp+MTP allowed; suffix/oracle still reject sampling
- [x] `generate_mtp_sampled`; `Q27_SAMPLE_PLAIN=1` force plain
- [x] has_mtp + chunked_prefill guard (codex P1)

### Phase 1d — Gate harness  `[x]` (script ready; live run = Phase 2)

**Files:** `tools/metal_sampled_mtp_gate.sh`, `tools/overnight_sampled_mtp.sh`

- [x] Seeded identity / seed varies / sampled≠greedy / plain force
- [x] Capability proof: mtp sample round traces + drafts&gt;0
- [x] Acceptance-vs-temp legs (short in gate; long in overnight)
- [x] Overnight orchestrator (units + gate + wall A/B + temp curve → OUTDIR/SUMMARY.md)

### Phase 2 — Live gates on official MTP pack  `[ ]` **← overnight**

**Run once on a quiet machine** (no other model loaded):

```bash
MODEL=models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok \
  tools/overnight_sampled_mtp.sh
# optional: N_AB=128 N_TEMP=48 OUTDIR=logs/overnight-...
```

Checklist after the run:

- [ ] `SUMMARY.md` Leg 0–3 all green
- [ ] Paste wall t/s (greedy / sample-mtp / plain) into progress log
- [ ] Paste acceptance-vs-temp speculation lines
- [ ] Expect accept hold ~through T≲0.7 (CUDA prior); document sag at T≥1
- [ ] Ship bar: sample-mtp ≫ plain sample at moderate T (≥ ~1.15×) or record miss

### Phase 3 — Defaults / product  `[ ]` (blocked on exit criterion)

- [ ] **Do not** default sampling on until Thunderdome-class quality A/B + drift catalog under production sampling (see `docs/sampling-design.md` exit criterion)
- [ ] Optional: pack-split defaults (bonsai sample, MTP greedy-or-sample with MTP on)
- [ ] Agent path: optional later expose MTP sample through session (agent is serial today)

### Phase 4 — Optional polish  `[~]`

- [x] Per-lane GPU top-k when `top_k` in 1..256 (`build_served_from_candidates` + mtp_sample_round)
- [x] Offset bind on `clogits_` rows (no per-lane copy into `logits_`)
- [x] Agent recipe: temp set ⇒ default top_p=0.95 top_k=20 if omitted (`packaging/bin/q27`)
- [x] `generate_mtp_sampled` CLI/engine helper
- [x] User-facing docs (GETTING-STARTED §3b, README sampling notes)
- [ ] Device nucleus/Gumbel only if profiled hot
- [ ] Sampled suffix-burst
- [ ] Constrained + sampled MTP (CUDA Phase 3 analogue)
- [ ] Native agent MTP sample path (still serial sample today)

## Overnight run (quiet machine)

**One command covers Phase 2:**

```bash
# Stop all other q27/agent/server processes first.
MODEL=models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok \
  nohup tools/overnight_sampled_mtp.sh \
  >logs/overnight-nohup.out 2>&1 &
# Results: logs/overnight-sampled-mtp-*/SUMMARY.md
```

| Leg | What | Needs GPU pack? |
|-----|------|-----------------|
| 0 | `test_sampling` + `test_metal_ops` | no |
| 1 | `metal_sampled_mtp_gate.sh` | yes |
| 2 | wall t/s: greedy / sample-mtp / plain | yes |
| 3 | acceptance vs T ∈ {0,0.3,0.7,1.0,1.5} | yes |

Offline-only dry run: `SKIP_LIVE=1 tools/overnight_sampled_mtp.sh`.

## Progress log

| Date | What | Result |
|------|------|--------|
| 2026-07-21 | Plan written; parallel map; phases checked | this doc |
| 2026-07-21 | Phase 0 helpers+tests | `d58ea12`; tests PASS |
| 2026-07-21 | Codex P2 empty-residual throw | `4d8757f`; autoreview **clean** |
| 2026-07-21 | Phase 1a–1c engine+CLI+server | `27ed0d4` |
| 2026-07-21 | Codex P1 CLI has_mtp/chunked guard + gate script | `c545d81` |
| 2026-07-21 | Live smoke official MTP pack, T=0.7 top_k=20 n=16 | **works**: 5 rounds, 13 drafted, 10 accepted (**76.9%**). Wall ~0.17 t/s is **not** a perf baseline — operator had another model resident (memory pressure); re-bench on a quiet machine. Host full-vocab nucleus may still matter at scale (Phase 4) but was not the measured cause here. |
| 2026-07-21 | CLI has_mtp guard + gate script harden | `c545d81`…`5547704`; gate Codex **clean** |
| 2026-07-21 | GPU top-k + offset bind + agent recipe | `96a717f` `be98a1c` clean |
| 2026-07-21 | Overnight orchestrator + docs + reject-walk parity tests + `generate_mtp_sampled` | *this commit* |
| | Phase 2 full `metal_sampled_mtp_gate.sh` on quiet machine | *pending* (script ready) |
| | Phase 4 GPU top-k / cut readback for speed | *pending* (correctness first) |

## Review workflow

Per landed slice (not once at the end):

1. Implement + focused tests green.
2. Commit (leave unrelated dirty files out — especially foreign `metal_server.cpp` edits).
3. Codex autoreview:

```bash
~/.pi/agent/skills/autoreview/scripts/autoreview \
  --mode commit --commit HEAD \
  --parallel-tests "make build/test_sampling && ./build/test_sampling"
```

4. Fix accepted findings; retest; re-review until clean.
5. Tick checkboxes + append progress log row.

## References

- `docs/sampling-design.md` — design + exit criterion  
- `docs/sampling-phase2-impl.md` — CUDA kernel/accept contract  
- `src/engine.cuh` `spec_sample_round` / `src/blocks.cu` `k_spec_accept`  
- `src/metal/metal_engine.cpp` `mtp_round`  
- `src/metal/metal_server.cpp` temp gate (`mtp = mtp_width && temperature==0`)  
- Qwen3.6 card: recommended T/top_p/**top_k=20** (quality), orthogonal to this speed path  
