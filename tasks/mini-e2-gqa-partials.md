# Task for mac-mini: E2 — per-engine gqa_partials + shrink (P2)

From the 2026-07-17 metal audit triage
(docs/plans/2026-07-17-metal-review-triage.md, items B1/E2/C3-fold);
assigned to the mini because it owns the multislot/admission lane (G6
accounting, snapshots Phase 2). Daniel-authorized 2026-07-16 night
("make the fixes... split the work between local and mac-mini").

**Problem (verified in source):** `gqa_partials` lives on the backend
`Impl`, grows monotonically to the peak context ever decoded, is never
reclaimed, and is shared across engines. On a multislot server cycling a
32K and a 2K conversation it pins the 32K-sized partials forever, and
the G6 admission accounting treats it as backend-shared peak rather than
per-slot.

**Scope:**
1. Move the partials buffer per-engine (same shape as the KV-budget
   per-engine move the codex `Shared` review already did), sized at
   engine construction from the engine's own max_ctx — this also removes
   the growth-check from the attention hot path (audit C3).
2. Shrink/free on engine teardown; optional shrink-below-25%-utilization
   only if it falls out naturally — the per-engine move already bounds
   the waste to each slot's own ctx.
3. Update the G6 admission accounting to charge the partials to the slot
   footprint (it currently rides the backend-shared GQA partial peak
   term — that term changes meaning with this move).
4. Gates: unit suites; the multislot suite (G1..G6) green with the moved
   buffer; one A/B showing byte-identical decode pre/post move (any
   T2 prompt, chunked + serial attention classes both exercised).

**Coordination:** the local machine's same-night fix batch touches
metal_backend.{h,mm} (tensor-extent validation E1, comments) but
deliberately does NOT touch gqa_partials — the surfaces are disjoint,
but pull before starting; local pushes land tonight. Delete this file in
the commit that lands the work.
