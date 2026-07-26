# Metal development docs

Records of *why* the Metal (Apple Silicon) engine, tiers, serving stack, and
native agent are the way they are. Separate from the user-facing docs at
[docs/](../) and from the upstream-CUDA records also kept there, so upstream
merges stay clean.

If you are a **user**, start at [docs/GETTING-STARTED.md](../GETTING-STARTED.md)
and [docs/MODELS.md](../MODELS.md) instead.

## Start here

| doc | what it is |
|---|---|
| [BUILDLOG.md](BUILDLOG.md) | the authoritative ledger — a maintained **Current state** table on top of an append-only dated chronicle. Every claim elsewhere should cite an entry here. |
| [DECISIONS.md](DECISIONS.md) | one line per experiment: what was tried, what the measurement said, and the number that decided it. The fastest way to find out whether something was already ruled out. |
| [METAL.md](METAL.md) | the backend's scope, the `src/backend.h` boundary, and implementation order. |
| [PARITY-2026-07-25.md](PARITY-2026-07-25.md) | current CUDA↔Metal parity plus the standing audit findings (A1–A9). |
| [MERGE-BACK.md](MERGE-BACK.md) | the staged plan for contributing back upstream, and what blocks it. |

## Also here

- [plans/](plans/) — pre-registered experiment plans, 2026-07-14 onward.
  Each carries its gates and kill lines up front, with measured results
  appended below the line. `DECISIONS.md` is the curated index into these;
  read that first unless you want a specific pre-registration. Upstream's
  earlier plans (through 2026-07-14) live at [docs/plans/](../plans/).
- [releases/](releases/) — per-release notes.
- [archive/](archive/) — session handoffs, external expert briefs, one-off
  code reviews, the original upstream scratch NOTEBOOK, and superseded
  parity docs. Kept verbatim for provenance; none of it is maintained.
- [upstream-issue-draft.md](upstream-issue-draft.md) — unsent draft of the
  merge-back issue.

## The rule this directory follows

Experiments are pre-registered with kill lines *before* they run, and
negative results are recorded with their mechanisms. A parked lever with a
measured reason is worth more than an untested idea, because it stays
parked. If you are about to try something, check `DECISIONS.md` first.
