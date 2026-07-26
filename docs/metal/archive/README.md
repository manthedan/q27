# Archive

Working documents kept verbatim for provenance. **None of it is
maintained**, and some of it was already stale when it was archived. Do not
cite these as current state — use [../BUILDLOG.md](../BUILDLOG.md) for what
is true now and [../DECISIONS.md](../DECISIONS.md) for what was decided.

| file | what it was |
|---|---|
| `HANDOFF-2026-07-15.md`, `HANDOFF-2026-07-17.md` | notes passed between working sessions: live serving state, machine queues, priority order at the time. `HANDOFF-2026-07-17.md` is still the reference for the quiet-gap deploy protocol. |
| `EXPERT-BRIEF-2026-07-15.md`, `EXPERT-BRIEF-2026-07-17.md` | briefs written *for* outside reviewers, describing the project as it stood. |
| `k3-review-2026-07-15.md`, `metal-review-2026-07-17.md` | the reviews that came back. Their triage — which findings were confirmed, fixed, or rejected, and why — is in `../plans/2026-07-17-k3-audit-triage.md` and `../plans/2026-07-17-metal-review-triage.md`. |
| `NOTEBOOK.md` | the original scratch pad: working notes, an early performance model, a risk register. Superseded by `BUILDLOG.md`. |
| `PARITY-2026-07-22.md` | the previous CUDA↔Metal parity evaluation. All of its action items were closed; superseded by `../PARITY-2026-07-25.md`. |

Documents elsewhere still refer to these by their old top-level paths, and
to `METAL_PROGRESS.md` (now `../BUILDLOG.md`). Those references were left
alone rather than rewritten: they are historical records, and editing what a
past document said to match a later reorganization is how a ledger stops
being trustworthy.
