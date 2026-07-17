# Lessons

- **METAL_PROGRESS merge conflicts (2026-07-16):** both machines append
  ledger entries at the same anchor, so conflicts are routine. Keep-both
  resolution duplicates an entry whenever one side already contains the
  other's pushed text (two dedup fixups on the chronicle: f4f9ea1,
  0135c14). After every ledger merge, grep-count each new entry's
  headline and assert 1 before pushing.
- **Docs edits via python replace (2026-07-16):** an AssertionError in a
  heredoc python edit does not stop a `&&`-chained commit that follows a
  separate command — commit and verification must be coupled to the edit
  actually succeeding (codex caught a committed doc missing its promised
  verdict, c77c504). Prefer the Edit tool for docs; if scripting, gate
  the commit on the script's exit.

## Fingerprinted batches pin HEAD — hold ALL commits until the batch completes
A running fingerprinted experiment (kv census pattern: FP = HEAD + binary
md5 + pins) means ANY commit — even a docs-only one — breaks a future
resume. Batch every commit (code, docs, driver fixes) until the summary
exists, then land them together. Corollary: rebuild-at-top + resume is
only coherent if `make` is a no-op on resume; a git reset/checkout that
refreshes mtimes forces a relink and a new binary md5, so after any git
surgery expect the fingerprint to refuse — verify measurement equivalence
empirically (digit-identical re-run of a completed cell), document the
migration next to the data, and only then rewrite the fingerprint.
Preserve a failed run's log BEFORE relaunching: the resume's `> "$log"`
redirect truncates it at process start, not at first output.

## Background jobs must be caffeinated end-to-end (2026-07-16)
Two rounds of background-task kills (~10 min apart, idle unattended
mini) hit exactly the jobs launched WITHOUT caffeinate (E2 gate suite,
codex reviews); the 4-hour census survived because its driver wrapped
every cell in `caffeinate -i`. System sleep reaps background children.
- The METAL_PROGRESS protocol rule ("caffeinate every long run") applies
  to ALL background work, including codex reviews and gate suites, not
  just timed measurement runs.
- Wrap the LAUNCH (`caffeinate -i script.sh`), not just inner commands —
  the gap between inner caffeinated children is enough to sleep in.
- Never edit a shell script while zsh is executing it (incremental
  read) — fix drivers between runs, not during.

## Stage files explicitly, never by directory or -A (2026-07-17)
Twice in one night a broad `git add` swept in unintended files: `git add
-A` grabbed stray root logs plus the 1.2 MB corpus binary (caught only
because the commit stat looked wrong — amended), and `git add
logs/kv_step4b/` grabbed a stale ABORTED marker next to the real
results (caught by codex, contradicted the summary for any tooling
reading the marker convention). Measurement directories accumulate
driver droppings (ABORTED, .tmp, .attempt1); name every file in the
`git add`, and glance at `git status --short` of what got staged
before committing.
