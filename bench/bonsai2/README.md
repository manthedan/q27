# Bonsai 2 gates (2026-09-18 .. 09-20)

Scripts behind BUILDLOG (as) through (aw). They run servers as transient user
units, stop production (`q27-38`) on the 5090 for the duration and relaunch
it, and stop the transcribers for 3090 work. Output dir: `OUT` (default
`/tmp/bonsai2-gates`).

- `fused_gate_3090.sh` -- identity gate for fused multi-slot rounds on the
  pure-T2 pack: five server configs (conductor-free reference, 1-slot fused,
  2-slot fused, solo-pinned, FIFO), four greedy prompts as concurrent pairs
  and alone; every text must match the reference. sm_86 only: the 5090's
  greedy is not width-invariant (BUILDLOG 2026-09-07 (f), 2026-09-18 (as)).
- `ladder_5090.sh` -- the 08-14 concurrency ladder (8 slots / 16K, C=1/2/4/8,
  temp 0.6) for the fused path vs the solo-pinned control
  (`Q27_BONSAI_FUSED=0`).
- `d2ab_5090.sh` -- DFlash2 drafter pack A/B on the Bonsai target: exact-mode
  identity vs plain (an sm_86 instrument; see above) and vgemm timing per
  prompt. `MODEL`, `PACKB`, `RUNS`, `TAG`, `NORELAUNCH` env overrides.
- `mtp_provenance.py` -- layout check of an HF-named MTP head checkpoint
  against the Qwen3.8 pack's blk.64 (row correlation, norm offset).
- `mtp_gate_3090.sh` -- CLI canonical plain vs `--spec` on the T2+MTP pack,
  then server identity (pure-T2 plain vs 1-slot ladder vs 2-slot fused).
- `mtp_bisect_3090.sh` -- CLI ladder variants over 1500 tokens vs plain.
- `width_probe_3090.sh` -- runs `build/width_probe` (built by hand from
  `tools/width_probe.cu`, the nvcc line is in its header; bitwise multi-lane vs
  plain: widths, folds, rejections, truncations, graph replay) at a deep
  position: the code prompt plus 550 plain-decoded tokens.
- `gate12g_3090.sh` -- the 12 GB-card simulation (BUILDLOG (av)): `vram_hog.py`
  holds VRAM on the 3090 (HOG GB, default 11.4) so the server sees a 3060's
  free memory, then `build/q27-server-12g` (`make build/q27-server-12g`,
  sm_86 only) boots a `--slim` pack (LEGS=plain|fusedplain|mtp|d2, default
  `plain d2`; SLIM=<pack>; FIXED=<Q27_FIXED_STACK_GB>, default 0.9; KV, CTX,
  BIN, PFX, REF) and its texts are compared with the plain reference when one
  is present.
- `vram_hog.py` -- `vram_hog.py <GB> [cuda:N]`: allocates and holds that much
  VRAM until killed; prints `hog: holding ...` once it has it.
- `g8_3090.sh` -- the 8 GB-card simulation for the T3 slim packs (BUILDLOG
  (aw)): drives `gate12g_3090.sh` once per SPECS item ("leg free_gb pack
  prefix", default plain at 8.3/8.0/7.6 GB free and mtp at 8.0) with
  Q27_FIXED_STACK_GB=0.6 and turbo5k KV; texts must match the full-memory
  turbo5k reference (`t3s_t3k5.json` from `t3_gate_3090.sh`). BASE_USED is
  what the 3090 holds at rest.
- `t3_gate_3090.sh` -- T3_G128 container gates, T2 slim vs T3 slim pack
  (BUILDLOG (aw)); LEGS from `gate kernels ninv cli server nll`: `build/t3_gate
  --bench` (every T3 matrix bitwise vs the T2 pack), `build/test_kernels` and
  `build/ninv_test` on the T3 pack, CLI canonical identity, 1-slot servers on
  the four greedy prompts (turbo5k), and `--nll` chunk 512 / ctx 512 on both.

Related tools outside this directory:

- `tools/t3_gate.cu` (`make build/t3_gate`) -- the T3 pack gate:
  `build/t3_gate <t2pack> <t3pack> [--layers N] [--bench] [--only blk.0.]`,
  and `build/t3_gate --synthetic`, a no-pack fuzz of random ternary matrices
  at off-model shapes (added in BUILDLOG (az) after a tail-window bug).
- `tools/install-bonsai2-8gb.sh` -- end-user installer for 8/12 GB cards:
  builds `q27-server-12g`, downloads the T3 slim pack (`--mtp` for the MTP
  one) and tokenizer, checks md5s, writes `run.sh`, does one test request.
- `tools/bench-bonsai2-8gb.sh` -- the matching end-user bench: boots each
  installed pack, sends the four greedy prompts (md5s compared with ours) and
  one ~5K-token prefill, writes `~/bonsai2/bench-<date>.txt`.
