Hi — I've been running a fork at https://github.com/manthedan/q27 (branch
`metal`) with an Apple Silicon / Metal backend for the same narrow mission:
Qwen3.6-27B-MTP official tier, plus a ternary Bonsai-27B tier that fits a
16 GB Mac.

**Are you interested in the Metal work merging back**, so q27 stays canonical
across CUDA and Metal? If you prefer q27 to stay CUDA-only and single-target,
that is a perfectly good answer. I'll keep the fork clearly labelled as a
downstream port and continue pulling from you. The standalone fixes below are
useful either way.

### Prepared scope

I did not use the raw fork diff as the proposal; it includes logs,
experiments, packaging, and project history. I rebuilt the contribution as a
serialized stack from your current `master`:

| stage | measured diff |
|---|---:|
| backend seam, validation, and CUDA dtype guard | 11 files, +1,145/-19 |
| Metal core: backend, engine, shaders, CLI, tests, README | 13 files, +16,701/-2 |
| Metal serving: HTTP, snapshots, recovery, streaming, shared API parity | 18 files, +7,462/-85 |

The seam contains no Metal implementation and no `.cu` change. Its one
CUDA-header edit is a four-line validation guard required by the expanded
dtype set: token embeddings must remain Q8 before the existing Q8-only row
kernel can launch.

The prepared Metal-only stack, excluding the standalone stream-boundary fix,
is **36 files, +25,293/-91**. The core stage touches no `.cu` file. The serving
stage does deliberately update `src/server.cu` for shared OpenAI API parity; I
want to state that plainly rather than sell the aggregate as "zero CUDA
changes." Core and serving are separate so you can take the engine without our
HTTP opinions.
The serving stage also carries the small Metal-backend health contract used by
server recovery: an atomic poison flag and a `healthy()` query. It does not
revert or duplicate lower-layer work, but it is not HTTP-files-only.


### Four standalone fixes

I would like to send these independently of the Metal decision:

1. **CUDA 12.0 compatibility.** On nvcc 12.0.140 (`sm_86`, RTX 3090), pristine
   `master` fails three captures of lambda-captured structured bindings in
   `server.cu`. Rebinding the tuple results to named references is +8/-2 and
   compiles on the same toolchain without changing behavior.
2. **Tokenizer ownership.** `Tokenizer` owns a PImpl but had no destructor;
   adding only a destructor also exposes copying and throwing-constructor
   hazards. The prepared fix makes ownership explicit, deletes copies, uses
   construction-local RAII, and includes a repeated malformed-construction
   lifetime gate.
3. **Argmax ties.** Exact maxima, including `-0.0f` versus `+0.0f`, resolve to
   the lowest index in both CUDA argmax paths, matching CPU and Metal. On
   nvcc 12.0.140 (`sm_86`, RTX 3090), the earlier exact-tie tip changed the
   synthetic gate from 3/8 failures to 8/8 passes. The final signed-zero tip
   passes both signed-zero orders in plain and fused paths, the full CUDA
   kernel battery reports `ALL PASS`, and the canonical md5 remains your
   published `a2982c5197c627551b27d76a0a94b220`.
4. **Adjacent tool-call boundary.** `</tool_call><tool_call>` currently emits
   no separating segment, so a consumer buffering one TOOL segment can merge
   two calls or lose a malformed second call's raw text. The fix has full,
   bytewise, empty-think, text-between, and flush regressions.

### Proposed order

If you're interested:

1. the four standalone fixes above;
2. the backend-neutral seam — no Metal implementation and no `.cu` changes;
3. Metal core after the seam merges;
4. Metal serving after core merges and after one explicit dependency question
   below.

The seam is the real architecture decision. If you decline it, I stop there;
I will not hide it inside a later Metal PR.

### One dependency question to decide before serving

The Metal server can spend minutes in queue wait and prefill before writing
headers. To avoid burning a slot for a client that already disconnected, our
vendored cpp-httplib carries seven additive lines exposing the accepted socket
as a borrowed `Request::sock` field before the first write.

There is no supported pre-header per-request liveness hook: `DataSink` exists
after headers, the connection `Stream` does not leave `process_request`, and
accept-time socket options have no request correlation. Moving the work into a
content provider would commit HTTP 200 before overload or engine errors are
known.

The server uses its own small POSIX liveness probe rather than httplib
internals. Call sites are intentionally unguarded, so removing the field fails
loudly at compile time instead of silently disabling cancellation. If you
would prefer a different accessor or hook, the patch is small enough to adopt
that shape. If you do not want the vendored change at all, the serving PR stays
out and the Metal core can still land.

### Maintenance

If the Metal path lands, we maintain it and run its gates. A Metal break should
not block a CUDA release. Happy to own `src/metal/` through `CODEOWNERS` or the
project's preferred equivalent.
