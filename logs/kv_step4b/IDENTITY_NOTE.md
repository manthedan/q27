HEAD-term-only fingerprint migration (2026-07-17 ~02:00): the batch
launched at 7c284d3; the 4b results commit (022c59f) and the
agentic-parity merge moved HEAD, but `make build/q27-metal` is a no-op
and the binary md5 term (ff969b25b2487293032fdf00d746f3f0) is
UNCHANGED — the measurement identity is the binary, HEAD is a proxy
(precedent: logs/kv_census/IDENTITY_MIGRATION.md). Migrated so the
driver rerun could regenerate step4b_summary.txt with the l7h1v
amendment arm (codex P2 on 022c59f: the manually-run arm was not
reproducible through the batch workflow).
