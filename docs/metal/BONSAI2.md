# Bonsai 2 on q27 / Apple Silicon

This is the **source-checkout revival**: upstream v0.14.1 (which serves Bonsai 2
on CUDA) plus Bonsai 2 Metal support and the recovered native agent/TUI. Metal
reads **upstream's own Bonsai 2 packs** (`tools/repack.py --slim`, FORMAT.md
"Bonsai 2 packs"); the revival's earlier `bonsai2-t2-hadamard-v1` container is
retired and rejected at load. It does not replace your Homebrew installation or
publish a release. Reference parity and the short-context runtime gates pass on
an M4 / 16 GiB mini (one engine/slot, context 2048); the original revival
parity run was on a base M4 / 24 GiB laptop.

## Use the prepared checkout

```sh
cd "$HOME/projects/q27-revival"
export Q27_CONTEXT=2048            # validated mini context; launcher default is 8192
./q27 build
./q27 agent                        # classic native agent; no model-driven tools
./q27 tui                          # Rust/Ratatui frontend, same native engine
./q27 agent --auto-tools           # explicit opt-in to local tools
./q27 serve                        # local HTTP, one slot, Q27_CONTEXT tokens
```

Invoke the launcher by absolute path from a project directory to use that
directory as the agent workspace. Tool-enabled runs can read/write files and
run bounded shell jobs there. Start in a disposable directory when evaluating
model behavior. `--workspace DIR` overrides the default. The recovered tool
sandbox and durable-session rules are described in
[the native-agent README](../../experiments/ds4-agent/README.md).

The launcher uses thinking-mode sampling: temperature 1.0, top-p 0.95, top-k 20
(the Metal sampler has no min-p filtering, matching this model's min-p=0).
The native transcript/tool protocol remains experimental, not a claim of
model-card task quality. Non-thinking mode's recommended presence penalty is
not implemented: `--no-think --temperature 0` is a smoke/debug control, not
full sampler-recipe parity. Native Qwen3.8/Bonsai 2 turns now share the current
profile-aware renderer: default xhigh effort, an open thinking prefix, and
normalized reasoning/history. `Q27_REASONING_EFFORT=low|medium|xhigh` overrides
the effort. Saved prefixes are validated against this rendering; incompatible
older sessions must be recreated rather than silently resumed.

Useful overrides: `Q27_CONTEXT=4096`, `Q27_BONSAI2_DIR=/path/to/models`, and
`Q27_METAL_SOURCE=/path/to/q27_kernels.metal`. Agent arguments follow the
launcher command; server arguments likewise follow `serve`.

```sh
# Exact restart persistence (snapshot files can be hundreds of MB).
# The session parent must be owned by you and not group/other writable.
mkdir -m 700 -p "$HOME/.q27-revival-sessions"
./q27 agent --session "$HOME/.q27-revival-sessions/work.q27agent"
```

Keep the shader/runtime unchanged when resuming: snapshots deliberately bind
the model, tokenizer/transcript, and runtime identities and reject mismatches.
Cold load and first snapshot latency still need soak/attribution; one initial
save exceeded an external 180-second timeout, although the subsequent save and
restart/reuse checks passed. This is not yet a packaged-release reliability claim.

## Reproduce the pack on another checkout

Requires Apple command-line developer tools for q27, Python 3.12–3.14 for the
conversion, and `hf` for the download. Rust/Cargo is needed only for the TUI.
Allow roughly **22 GB free disk** during conversion (source + spool + output).
No full-precision model download is needed.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-deps --require-hashes \
  -r tools/requirements-repack-test.txt

hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-PQ2_0.gguf LICENSE NOTICE.txt \
  --revision 6ed5e12bf84b7a63069882c91dd9e9218647d17b \
  --local-dir models/bonsai2

printf '%s  %s\n' \
  3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1 \
  models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf | shasum -a 256 -c -

.venv/bin/python tools/repack.py --bonsai2-container t2 --slim \
  models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf models/bonsai2/bonsai2-27b-t2-slim.q27
# optional, 1.1 GB smaller but ~30% slower decode on M4 (see below):
.venv/bin/python tools/repack.py --bonsai2-container t3 --slim \
  models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf models/bonsai2/bonsai2-27b-t3-slim.q27
.venv/bin/python tools/export_tokenizer.py \
  models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf models/bonsai2/bonsai2.tok
./q27 build
```

Expected SHA-256:

| file | SHA-256 |
|---|---|
| `bonsai2-27b-t2-slim.q27` | `b85094a0f53c68c25fc73c28e70e05462620417698c7b08a8977d475391f0aeb` |
| `bonsai2-27b-t3-slim.q27` | `e548b5208c693f087cb00fe6c73a5da4942982612d4923374a2031bf653e669d` |
| `bonsai2.tok` | `f8adac2001668f3ba68e040e5c13f0982d6a194e2b4d9047cc050fdc6d7c93f2` |

The t2-slim pack is **7,195,341,824 bytes**; t3-slim is **6,055,015,680**. Both
are upstream's containers byte for byte: the t3-slim hash equals
`signalnine/Bonsai-2-27B-q27`'s published `bonsai2-27b-t3-slim.q27`, which can
be downloaded instead of converted (its `qwen38-27b-mtp.tok` is identical to
`bonsai2.tok`). The 402 ternary tensors keep every source code and scale; the
96 BF16 GDN gates are stored as F16 (8,016 of 23.6M values, all below 3e-8 in
magnitude, round or flush). Conversion peaks at about 8.7 GB RAM. Select the T3
pack with `Q27_BONSAI2_PACK=models/bonsai2/bonsai2-27b-t3-slim.q27`. Non-slim
upstream packs (Q8 embedding/head, ~9.4 GB) and MTP packs are rejected on Metal.

## Gates and evidence

```sh
make test-tools test-inspect test-agent
make test-repack test-bonsai2-repack PYTHON=.venv/bin/python
make test-metal-backend             # run GPU jobs serially
cargo test --manifest-path experiments/q27-tui/Cargo.toml --locked
python3 tools/test_bonsai2_serving.py build/q27-metal-server \
  models/bonsai2/bonsai2-27b-t2-slim.q27 models/bonsai2/bonsai2.tok
python3 tools/test_bonsai2_native.py build/q27-agent \
  models/bonsai2/bonsai2-27b-t2-slim.q27 models/bonsai2/bonsai2.tok
```

For the independent oracle, build Prism's `prism` branch at
`1a07bfa5f4144274c8f1c9963821dd9d9a51854b`:

```sh
# In that separately pinned Prism checkout:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_CXX_FLAGS='-include cerrno'
cmake --build build --target llama -j2
# Back in q27 (Python needs numpy):
PRISM_DIR=/absolute/path/to/prism-llama tools/bonsai2_gate.sh
```

`-include cerrno` works around a missing `errno` declaration in the pinned
reference's `gguf.cpp`; its source and math were not modified. The gate runs
reference and q27 **sequentially**, never two resident models together.

[Current evidence](evidence/bonsai2-upstream-2026-10-01.json), on the 16 GiB
mini with upstream's packs: t2-slim **339/340** argmax agreement (one flip on a
2e-5-logit tie, traced to the F16 gate storage), t3-slim **340/340**, worst KL
below `3.3e-7` nats, no high-margin mismatches; serving, native and launcher
gates pass on t2-slim. Measured decode at context 2048: t2-slim ~11 tok/s,
t3-slim ~7.8 tok/s.

[Original revival evidence](evidence/bonsai2-2026-09-18.json), retired pack: 340 teacher-forced
positions over prose, code, and tool-history/Unicode inputs; **340/340 argmax
agreement**, worst KL below `8.4e-8` nats. The separate dense-CPU/GPU transform
tests cover widths 1024/5120/6144/17408, signs, inverse order, grouped-GDN
permutation, batched dispatch, red zones and invalid arguments. Legacy B1,
T2, and q4s artifacts still validate. Native read-tool round trip and persisted
restart with a 41-token exact prefix passed. Consecutive native `read` then
`search` calls also passed. Review-driven regressions cover cross-tool XML mask
keys and bounded device-pool reuse; live `--constrain-tools` tests engage XML
on all six streaming/non-streaming tool routes, with per-request engagement
and exact tool-result assertions. XML constraints cover parameter names and
required keys, not arbitrary JSON Schema value semantics. Serial mask pools
recycle safely within a request rather than imposing a 64-state call limit.
The [separate mini evidence](evidence/bonsai2-mini-2026-09-18.json)
records ten native CPU suites, real-tokenizer stable-prefix checks, native
read/search, and default-thinking budget/save/restart (83 cached tokens).
The 340-position reference comparison was **not rerun on the mini**. This is
short-context correctness, not a long-context or task-benchmark claim.

## What is not enabled

- **PTQ1_0 / the 5.95 GB dense pack**: not supported by this converter yet.
- **Bonsai chunked prefill**: remains off. Old chunk GEMM activation
  quantization is not equivalent to the correct float-activation path.
- **MTP**: the checkpoint has no MTP layer; requesting it fails.
- **Vision**: this port is text-only; no vision tower/mmproj is loaded.
- **HTTP snapshot hints**: upstream ignores explicit hints for serial-only
  Bonsai. Its q4s snapshot-recovery suite therefore does not pass unchanged;
  do not substitute the endpoint smoke for a full recovery gate.
- **Homebrew/prebuilt release**, larger-context 16 GiB validation, and cold
  start/persistence soak: still pending. The mini gates use context 2048, not
  the launcher's default 8192.

## Recovered experiments

`tools/kernel_agent/driver.py` and the synthetic prefill bench are restored from
`metal-kernel-agent`. Their current Q4 GEMM correctness gate covers four dtypes,
widths 1–96, partial tiles and production-width rows. The no-change baseline
smoke passed; no optimization was attempted. This is **not a Bonsai 2 prefill
benchmark**. Run only on an idle machine, without any model server:

```sh
python3 tools/kernel_agent/driver.py --iterations 0   # baseline/noise only
```

The [MLX/inferno survey](plans/2026-07-31-mlxfast-survey.md) is preserved as a
research record. Its head-pruning kernel is not promoted here. Other donor
branches remain intact; see the [revival inventory](REVIVAL-2026-09-18.md).
