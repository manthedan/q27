# Q4 quiet-bench evidence status

The captured 2026-07-17 run is **rejected as quiet-gate evidence**. Its
watcher log records that the decisive invocation was forced with
`idle>=0s`, even though the generated output used the pre-registered
`machine idle>=10min` label.

The original bytes are retained, but explicitly quarantined as:

- `quiet-watcher-nonidle-rejected.log`
- `quiet_bench-nonidle-rejected.txt`
- `quiet_{float,half}*.log` (raw legs from the same rejected invocation)

The observed 0.855 ratio is diagnostic only. It cannot substantiate either
a SHIP or PARK quiet-gate decision. The Q4 half route remains default-off,
and the pre-registered 600-second-idle gate remains pending a valid rerun.

`tools/quiet_q4_bench.sh` now refuses missing/non-finite legs, contaminated
baseline measurements, concurrent model residents, unbound restart health,
and serving-restart failure. Only a successful run may publish
`quiet_bench.verdict`.
