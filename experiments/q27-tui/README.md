# q27-tui

Rust Ratatui frontend for `q27-agent` over **Frontend Protocol v1** (FP1).

Spec: `docs/metal/plans/2026-07-21-frontend-protocol-v1.md`

## Build

```sh
# from repo root
make build/q27-agent
cd experiments/q27-tui && cargo build --release
```

Binary: `experiments/q27-tui/target/release/q27-tui` (or `target/debug/q27-tui`).

## Run

```sh
# from repo root
make build/q27-agent
cd experiments/q27-tui && cargo build --release

export Q27_AGENT="$PWD/../../build/q27-agent"   # optional; otherwise walks up for build/q27-agent
./target/release/q27-tui -- /path/to/MODEL.q27 /path/to/MODEL.tok --auto-tools --workspace "$PWD"
```

Spawns:

```text
q27-agent MODEL TOKENIZER --frontend-proto 1 [extra agent args…]
```

Startup prints load progress on **stderr** and only enters the full-screen UI after `hello`+`idle`.  
Child agent **stderr** is redirected to `$TMPDIR/q27-tui-stderr-<pid>.log` (shown in the pre-UI banner).

### If it “crashes immediately”

Usually one of:

1. **Not a TTY** — must run in a real terminal (not a pipe / some IDE runners).
2. **Bad model/tokenizer paths** — files must exist; check the printed paths.
3. **Agent failed to load Metal** — read the stderr log path printed on startup.
4. **Old binary** — rebuild: `cargo build --release` and `make build/q27-agent`.

The TUI restores the terminal on error/panic; if a prior version left raw mode on, run `reset`.

## Keys

| Key | Action |
|-----|--------|
| Enter | send prompt (while busy, enqueues when `hello.features` has `queue`) |
| Ctrl-C | cancel active turn, or clear input when idle |
| Ctrl-D / Ctrl-Q | quit (`op: quit` → backend `bye`) |
| PgUp / PgDn | scroll stream |
| `t` | toggle thinking (`<think>…</think>`) expand/collapse |
| `T` | cycle theme (`dark` → `ocean` → `ember` → `mono`) |
| `m` | toggle markdown rendering on finalized assistant turns |

Slash → structured FP1 ops: `/help` `/save` `/compact` `/session` `/new` `/read` `/search` `/shell` `/queue_clear`.

## Layout

- **Stream** — user / assistant / tool cards / notices  
- **Input** — enabled only after event `type: idle` (not merely `state: idle`)  
- **Footer** — `ctx used/size | phase · status`

Gates on `hello.features` (e.g. no `queue` until backend P4).

## Dev tests

```sh
cargo test
```

Proto tests cover v0 jsonl decode (base64 payload) and v1 `hello` parse.
