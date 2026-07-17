# Fingerprint migration (2026-07-16, evening)

Cells 000–031 were measured under the original fingerprint:

    9d2239427755a3fda163ccf1a48c8f026d59a7d2 b726efb6f89728fac53986cbb611a77c ...

Cells 032–127 run under the migrated fingerprint (HEAD dc43c52, binary
md5 7baa6ede1beb7bff2c00b6db3eb58af5). Model, corpus, and all four route
pins are unchanged.

Why the identity moved mid-census:
- The repo history was force-rewritten (trailer strip) and this checkout
  was hard-reset + cherry-picked while the census ran, moving HEAD from
  9d22394 to dc43c52. The new tree additionally contains the step-2
  instrument staging (flags/scale_off/aux extension of the attrib store),
  which the census does not exercise (flags = 0).
- The reset refreshed source mtimes, so the resume's rebuild-at-top
  relinked the binary (new Mach-O UUID → new md5) from the dc43c52 tree.

Why mixing is sound — measured, not assumed:
- Cell 0 was re-run under the dc43c52 binary with the same pins and is
  DIGIT-IDENTICAL to cell_000.log on every distribution line: mean
  0.000234749 nats, max 0.00799207, all tail quantiles, both run
  structures, and both position-bucket lines. Evidence preserved at
  equiv_cell000_dc43c52.log (only startup/wall timing lines differ).

Also recorded here: cell 32's first attempt under the old binary died at
~pos 1920 with a GPU command-buffer page fault. The raw log was
overwritten by the resume's re-run; its full content was:

    Metal model ready on Apple M4 in 5.76 s (two engines, one mapping:
      fp16 baseline vs turbo3 cell L19:h0:K round-trip)
    kl-kv: 2048 positions, single pass, no resets
      kl pos 384/2048  kl pos 768/2048  kl pos 1152/2048
      kl pos 1536/2048  kl pos 1920/2048q27 Metal: command batch failed:
      Caused GPU Address Fault Error
      (0000000b:kIOGPUCommandBufferCallbackErrorPageFault)

The driver now retries a failed cell once before aborting; whether the
fault reproduces is answered by cell_032.log from the resumed run.

Post-launch HEAD move #2 (2026-07-16, while cells 32+ run): fast-forward
dc43c52 → cdff4b8 (yukon's E1/audit/B1 batch + the E2 task assignment).
build/q27-metal is NOT rebuilt (md5 stays 7baa6ede...) — the binary term
is the real measurement identity, so if a resume is ever needed, a
HEAD-term-only fingerprint rewrite is sound by construction PROVIDED the
binary md5 still matches; if the binary ever changes, re-verify with a
digit-identical completed-cell re-run as before.
