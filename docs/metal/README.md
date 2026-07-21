# Metal development docs

This directory holds the **q27 Metal (Apple Silicon) development
documentation** — the records of *why* the Metal engine, tiers, serving
stack, and native agent are the way they are. It is separate from the
user-facing docs (at [docs/](../)) and from the upstream-CUDA records (also
at [docs/](../), left in place so upstream merges stay clean).

If you are a **user**, start at [docs/GETTING-STARTED.md](../GETTING-STARTED.md)
and [docs/MODELS.md](../MODELS.md) instead.

## The ledger

- [METAL_PROGRESS.md](METAL_PROGRESS.md) — the maintained **Current state**
  table (release lines, shipped-and-gated features, parked levers, active
  work) on top of an append-only dated chronicle. The authoritative record;
  every claim cites a chronicle entry. Start here.
- [METAL.md](METAL.md) — the Metal backend's scope, design constraints, and
  implementation order (the `src/backend.h` boundary, kernel choices).

## Session handoffs and reviews

Working notes passed between sessions, and the external/self reviews that
shaped the work. Dated; read the newest first.

- [HANDOFF-2026-07-17.md](HANDOFF-2026-07-17.md) · [HANDOFF-2026-07-15.md](HANDOFF-2026-07-15.md)
- [EXPERT-BRIEF-2026-07-17.md](EXPERT-BRIEF-2026-07-17.md) · [EXPERT-BRIEF-2026-07-15.md](EXPERT-BRIEF-2026-07-15.md)
- [metal-review-2026-07-17.md](metal-review-2026-07-17.md) · [k3-review-2026-07-15.md](k3-review-2026-07-15.md)
- [NOTEBOOK.md](NOTEBOOK.md) — the upstream scratch pad, preserved verbatim
  (working notes, performance model, risk register). No longer maintained.

## Experiment plans

[plans/](plans/) — our pre-registered experiment plans (2026-07-14 onward).
Each has gates and kill lines up front, with measured results appended below
the line. The chronicle in METAL_PROGRESS.md links into these. Upstream's
earlier plans (through 2026-07-14) live at [docs/plans/](../plans/).
