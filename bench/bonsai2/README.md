# Bonsai 2 gates (2026-09-18)

Scripts behind BUILDLOG (as) and (at). They run servers as transient user
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
- `width_probe_3090.sh` -- runs `tools/width_probe` (bitwise multi-lane vs
  plain: widths, folds, rejections, truncations, graph replay) at a deep
  position: the code prompt plus 550 plain-decoded tokens.
- `gate12g_3090.sh` + `vram_hog.py` -- the 12 GB-card simulation: a torch
  process holds VRAM on the 3090 (HOG GB) so the server sees a 3060's free
  memory, then `build/q27-server-12g` boots a `--slim` pack (LEGS=plain|mtp|d2,
  SLIM=<pack>, FIXED=<Q27_FIXED_STACK_GB>) and its texts are compared with
  the plain reference when one is present.
