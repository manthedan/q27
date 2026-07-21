# GPU-resident greedy decode (ds4-survey item 4)

Phase 4 attribution pinned the ternary tier's decode gap (8.2–9.4 observed
vs 10.57–12.66 same-moment ceiling) on per-step overhead: the argmax
readback sync and command-buffer turnaround, not kernels. ds4's pattern —
only 4 bytes cross back per token — stops one step short of the fix both
engines can make: **feed the argmax token id back into the next embedding
lookup on the GPU**, so K greedy steps encode into one command buffer with
zero CPU syncs, and the host reads K ids per batch instead of 1.

## Design

- `q27_embedding_{q8,t2}_dev`: identical to the host-token kernels but
  buffer(3) is `device const uint*` — the token id the previous step's
  argmax wrote into `token_out_`. No other kernel changes: positions are
  host-known at encode time (`position_` advances per encoded step), so
  rope/KV args stay ordinary dispatch constants.
- `token_ring_` (K × u32 device buffer): after each step's argmax, an
  in-batch `backend_.copy` archives `token_out_` → `ring[i]`. The host
  seeds `token_out_` with the pending token before the batch and reads the
  ring once after `batch.finish()`.
- `decode_resident(pending, out, k)`: one CommandBatch of k chained steps.
  Wired into `generate_from_pending`'s greedy branch in K=8 slices
  (`Q27_METAL_RESIDENT=0` opts out — the A/B lever). Serving
  (`stream_from_pending`), MTP/suffix, and sampled paths keep per-step
  rounds for v1: EOS/stop overshoot would advance KV/GDN state past the
  last emitted token (the exact hazard the codex sweep flagged for batched
  MTP), and tool constraints require host grammar feeding between tokens —
  the resident path refuses to engage while a constraint mask is active.

## Gates

1. Committed tokens byte-identical vs the per-step path (env A/B) over the
   canonical prompts, 128 tokens.
2. `make test-metal` green.
3. Artifact decode wall re-measured back-to-back with the resident-weight
   ceiling at matched thermal state; expect the observed-vs-ceiling gap to
   close materially (attribution says it is mostly this overhead).

## Kill criteria

< 5% wall improvement at K=8 (the attribution would then be wrong and the
entry records that), or any committed-token divergence.
