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
