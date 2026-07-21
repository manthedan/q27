# Metal sampled MTP — CUDA Phase-2 rejection sampling on the host path

Status: **IN PROGRESS** — Phase 0 clean (Codex); Phase 1a–1c implemented, build green; live gates pending.  
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

### Phase 0 — Host rejection-sampling helpers  `[x]` (landed locally; autoreview pending)

**Files:** `src/sampling.h`, `src/test_sampling.cpp`  
**No Metal, no model.**

- [x] `build_served_distribution` matching `sample_logits_cpu` nucleus (temp/top_p/top_k, boundary ties)
- [x] `served_probability(token)` = weight/total or 0
- [x] `sample_served(..., exclude)` never emits exclude; renormalizes
- [x] `spec_rejection_accept(lane_logits[], drafts[], params, rng)` → `{n, stop_lane, exclude, pending}`
- [x] Unit gates:
  - [x] seeded identity
  - [x] exclude never re-emitted on reject
  - [x] all-accept path samples last lane
  - [x] p=0 draft (outside nucleus) always rejects
  - [x] composition: histogram of pending ≈ residual `sample_served` (MAE)
  - [x] accept rate ≈ analytic p_served (Monte Carlo, |emp−p|<0.04 @ N=4000)
- [x] `make build/test_sampling && ./build/test_sampling` green
- [x] Commit + Codex autoreview clean (`d58ea12` + fix `4d8757f`)

### Phase 1a — `MetalEngine::mtp_sample_round`  `[x]` (pending commit/review)

**Files:** `src/metal/metal_engine.{h,cpp}`

- [x] New method: same draft/verify as `mtp_round`, accept via `spec_rejection_accept`
- [x] Readback multi-lane logits from `clogits_`
- [x] Commit path shared with greedy (`gdn_replay`, EOS clamp, encode rules, live_width adapt)
- [x] Bootstrap: first pending after prefill is **sampled** (CLI/server)
- [x] Fallback when live < 2 / context tight: one serial sample+step (not greedy step)
- [x] Greedy `mtp_round` **untouched** (bitwise)
- [x] Unit / engine self-check if any; else deferred to P2
- [ ] Commit + Codex autoreview clean

### Phase 1b — Server route  `[x]` (pending commit/review)

**Files:** `src/metal/metal_server.cpp`  
**Note:** folded with pre-existing uncommitted sampling-default flags (other agent); kept and extended.

- [x] `mtp` resolved post-slot with `has_mtp()`; sampled MTP when temp>0 && !Q27_SAMPLE_PLAIN
- [x] Quantum loop: `mtp_sample_round` under one lease (mirror greedy MTP quantum)
- [x] Tool constraining stays off under sampling (existing rule)
- [x] Boot stderr notes sampled-MTP vs suffix-disengage
- [ ] Commit + Codex autoreview clean

### Phase 1c — CLI  `[x]` (pending commit/review)

**Files:** `src/metal/metal_cli.cpp`

- [x] Drop hard error for temp+MTP; keep reject for suffix/oracle
- [x] `--temperature` + `--mtp` routes to sample path
- [x] Optional `Q27_SAMPLE_PLAIN=1` force plain sample (CUDA parity for A/B)
- [ ] Commit + Codex autoreview clean

### Phase 1d — Gate harness (draft anytime, run after P1)  `[ ]`

**Files:** `tools/metal_sampled_mtp_gate.sh` (new)

- [ ] Seeded identity (same seed → same tokens)
- [ ] Seed varies + sampled ≠ greedy
- [ ] Spec vs plain both valid trajectories (chi² optional if too heavy live)
- [ ] Acceptance-vs-temp tokens/round curve published
- [ ] Greedy MTP control still bitwise on a fixed prompt (vacuous-gate lesson: assert MTP pack)

### Phase 2 — Live gates on official MTP pack  `[ ]`

- [ ] Quiet machine, official-tier artifact with `has_mtp`
- [ ] Run gate script; record numbers in this doc
- [ ] Acceptance expectation (CUDA prior): hold near greedy through T≲0.7, sag at T≥1.0
- [ ] Speed: sampled MTP ≫ plain sample at moderate T (ship if ≥ ~1.15× or document miss)

### Phase 3 — Defaults / product  `[ ]` (blocked on exit criterion)

- [ ] **Do not** default sampling on until Thunderdome-class quality A/B + drift catalog under production sampling (see `docs/sampling-design.md` exit criterion)
- [ ] Optional: pack-split defaults (bonsai sample, MTP greedy-or-sample with MTP on)
- [ ] Agent path: optional later expose MTP sample through session (agent is serial today)

### Phase 4 — Optional polish  `[ ]`

- [ ] Per-lane GPU top-k readback instead of full vocab
- [ ] Device nucleus/Gumbel only if profiled hot
- [ ] Sampled suffix-burst
- [ ] Constrained + sampled MTP (CUDA Phase 3 analogue)

## Progress log

| Date | What | Result |
|------|------|--------|
| 2026-07-21 | Plan written; parallel map; phases checked | this doc |
| 2026-07-21 | Phase 0 helpers+tests | `d58ea12`; tests PASS |
| 2026-07-21 | Codex P2 empty-residual throw | `4d8757f`; autoreview **clean** |
| 2026-07-21 | Phase 1a–1c engine+CLI+server (uncommitted) | builds green |
| | Phase 1 commit + Codex autoreview | *pending* |
| | Phase 2 live gates on MTP pack | *pending* |

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
