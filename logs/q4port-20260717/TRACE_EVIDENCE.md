# Trace-gate evidence status

`trace-gate-pre-correlation.jsonl` and
`trace-gate4-pre-correlation.log` are retained as historical evidence from the
older trace gate. They predate request-correlated validation errors and the
deterministic recovery/cancellation/engine-error legs, and therefore do **not**
substantiate the current release trace gate.

Current-gate evidence is intentionally pending a coordinated live model slot.
Run `tools/trace_gate.sh` against a server started with
`Q27_METAL_TEST_FAILPOINTS=1 --trace <new-path>` and save the new trace/verdict
under distinct current-gate names; never overwrite these historical files.
