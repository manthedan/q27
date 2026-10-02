# /metrics -- Prometheus text exposition

**Status:** implemented 2026-08-30 (`feat/metrics-endpoint`, shipped since
v0.11.0). Opt-in via
`--enable-metrics` (default off). Auth-exempt like `/health`. This document
is the reference for every series, the observation semantics, and the
consumer contract; `src/metrics.h` is the implementation and the comment
blocks there mirror this document.

## Posture

- **Off by default.** Without `--enable-metrics` the handler is never
  registered (`GET /metrics` -> 404; `/health` stays 200) and the per-request
  bookkeeping is skipped entirely -- a deployment that does not want metrics
  keeps the zero-overhead shape. With the flag, behavior is identical to what
  is documented here.
- **Auth-exempt when enabled**, same rule as `/health`: infra pollers
  (Prometheus and any text-exposition scraper) reach it without a key. A
  401/403 would make a poller mark the backend unreachable and blank every
  tile. The
  endpoint carries metadata-scale telemetry only -- counts, latencies,
  occupancy, uptime -- never prompt text or generated content.
- **One scrape = one consistent picture.** Histograms are seqlocked so a
  scrape can never see a torn update (the `+Inf == _count` invariant
  Prometheus consumers rely on holds under traffic; the reader retries on a
  torn read, at worst one poll shows a null quantile and the next heals).
  Slot/KV gauges are snapshotted under the server's `route_m`.

## The series

Per-API label `api=` -- `chat` (OpenAI `/v1/chat/completions`),
`completions` (`/v1/completions`), `messages` (Anthropic `/v1/messages`),
`responses` (OpenAI Responses; the internal spellings oai/cmpl/anth/resp
map via `api_from_str` in `metrics.h`):

| Series | Type | Meaning |
|---|---|---|
| `q27_requests_total` | counter | Generation requests ([req] universe) |
| `q27_requests_errors_total` | counter | Requests ending `end=error` |
| `q27_prompt_tokens_total` | counter | Prompt tokens sent to prefill (incl. cached) |
| `q27_prefill_computed_tokens_total` | counter | Prompt tokens actually computed |
| `q27_prefill_cached_tokens_total` | counter | Prompt tokens served from cache (RAM/ckpt/disk) |
| `q27_decode_tokens_total` | counter | Generated tokens delivered |
| `q27_queue_wait_seconds_total` | counter | Sum of slot-claim wait (µs-resolution, `%.6f`) |
| `q27_ttft_seconds` | histogram | Pre-fill wall per request (first-token proxy) |
| `q27_e2e_seconds` | histogram | Request arrival -> generation complete |
| `q27_itl_seconds` | histogram | Per-request mean inter-token latency |
| `q27_requests_inflight` | gauge | Slots currently claimed |
| `q27_slots_total` | gauge | Configured serving slots |
| `q27_kv_usage_perc` | gauge | Paged KV pool occupancy, 0-1 (omitted if pool off) |
| `q27_prefix_cache_entries` | gauge | Persistent prefix cache entries (omitted if off) |
| `q27_spec_draft_tokens_total` | counter | MTP draft lanes fired (accept gate) |
| `q27_spec_accepted_tokens_total` | counter | MTP draft lanes accepted |
| `q27_spec_accept_ratio` | gauge | `accepted / drafted` (0 when nothing speculated) |
| `q27_decode_tokens_processed_total` | counter | Live decode tokens (unlabeled) |
| `q27_prefill_computed_tokens_processed_total` | counter | Live prefill computed (unlabeled) |
| `q27_prefill_cached_tokens_processed_total` | counter | Live prefill cached (unlabeled) |
| `q27_preemptions_total` | counter | Constant 0 -- see "Why preemptions is 0" |
| `q27_uptime_seconds` | gauge | Server uptime |

Sample scrape:

```
# HELP q27_ttft_seconds Pre-fill wall per request (first-token latency proxy).
# TYPE q27_ttft_seconds histogram
q27_ttft_seconds_bucket{api="chat",le="0.010"} 0
...
q27_ttft_seconds_bucket{api="chat",le="60.000"} 11
q27_ttft_seconds_bucket{api="chat",le="+Inf"} 11
q27_ttft_seconds_sum{api="chat"} 53.123456
q27_ttft_seconds_count{api="chat"} 11
```

## Observation semantics

**Universe.** One `observe()` per generation request, at the point the
`[req]` telemetry line fires -- when `GenStats` is final, errors included
(`end=error` lands in both `requests_total` and `errors_total`). Counts
derived from `/metrics` therefore match the `[req]` log line-for-line.

**The computed/cached split** follows the ds4 convention: the headline
prefill number is *computed* tokens (`prompt - hit - pfx`); cached tokens
(snapshot/ckpt/disk restores) are the complementary series. Both are exposed
because they answer different questions: computed = GPU work done, cached =
prefix reuse working.

**Latency definitions** (and why they are not interchangeable with other
engines'):

- **TTFT = `pf_ms`**, the prefill wall. Under prompt-seeded thinking the
  first *visible* token can arrive later than prefill + one decode step;
  this measures the engine's prefill, which is the quantity that scales
  with context and cache state.
- **E2E = arrival -> generation complete**, stamped at the `[req]` point
  (generation done), *not* last-byte: client write time is not model work.
- **ITL = `dec_ms / dec` per request**, observed only when `dec > 0`
  (a zero-token request has no inter-token latency to average). The
  histogram's `_count` is therefore the count of requests with at least one
  decoded token, not the request count.

**Histogram shape.** Fixed finite edges in seconds plus the implicit `+Inf`
bucket, rendered cumulative as Prometheus expects. The tails are sized for
real observations, not for symmetry: TTFT to 60 s, E2E to 300 s, ITL to
100 ms -- a 60k-token prefill takes ~25 s and must land in a finite bucket,
otherwise p95 quantiles fall into `+Inf` and consumers report "no data".
`_sum` is integer microseconds at rest and rendered `%.6f`: `%g`'s six
significant digits would round away microsecond deltas once totals grow,
corrupting `rate()` on long-running servers.

## Live vs completion counters

The per-api totals above stamp at request completion. That is the correct
accounting for `rate()` over idle-then-burst traffic, but it is a step
function: during a 2000-token decode every poll reads the same value and a
dashboard shows 0 t/s, then +2000 at the end. The `*_processed_total`
series exist for dashboards instead: the engine bumps
`live_decoded` per delivered token at the single delivery point
(`post_round`, covering the solo, sampled and conductor-fused paths) and
`live_prefill_{computed,cached}` at `generate_prefill`'s success exit, so a
poller sees progress *during* generation (measured: 79 -> 115 t/s
real-time where the step counter showed 0 then 1023).

They are unlabeled (the engine does not know the API shape) and cumulative;
at rest they equal the sum of the labeled per-api totals, so consumers can
diff them against the labeled counters to detect in-flight work.

## MTP accept telemetry

`q27_spec_draft_tokens_total` / `q27_spec_accepted_tokens_total` count
confidence-gate lanes: lane `j` is "fired" when the round drafted through
depth `j` under the margin gate, "accepted" when the round committed at
least `j+1` tokens. The ratio gauge is the cumulative acceptance.

Both decode modes feed it. The greedy path (`spec_round`) always did; the
sampled path (`spec_sample_round`, and the conductor's fused
`commit_outcome(sampled=true)`) deliberately updated nothing, which made the
ratio read a constant 0 on sampled-only serving profiles -- the measured
recipe runs `--temp 1.0`, so that was the normal case, not an edge case.
Both sampled paths now mirror the greedy lane accounting, monitoring-only:
no depth-controller or EMA feed, because the sampled ceiling is the fixed
`gate_maxd` (7 under the server's default `Q27_MAXD=auto7`) and adaptive
depth is greedy-only. Expect 0.4-0.75 depending on request mix;
0 with traffic flowing means the gate never fired (e.g. `Q27_PMIN=0`).

**Why preemptions is an honest 0.** q27's admission is a FIFO queue: a
request that finds no free slot waits (visible in
`q27_queue_wait_seconds_total`), and a running request is never evicted.
`q27_preemptions_total` is therefore a constant-0 counter, not a stub:
dashboards can read it and a non-zero value would mean a code path regressed
into preemption. Dashboards should treat 0 as healthy, not as missing data.

## Consumers

**Prometheus** (scrape every 5-15 s):

```yaml
scrape_configs:
  - job_name: q27
    static_configs:
      - targets: ["gpu-host:8080"]
```

Useful queries:

```promql
rate(q27_decode_tokens_processed_total[1m])            # live decode tok/s
rate(q27_prefill_computed_tokens_processed_total[1m])  # live prefill tok/s
histogram_quantile(0.95, rate(q27_ttft_seconds_bucket[5m]))
q27_spec_accept_ratio                                   # MTP acceptance
1 - (rate(q27_prefill_cached_tokens_total[5m]) /
     rate(q27_prompt_tokens_total[5m]))                 # cache miss fraction
```

**Custom dashboards / scrapers.** The intended mapping: live tok/s from
`rate()` of the `*_processed_total` counters (they move during generation;
the per-api totals are step functions that only stamp at request end),
TTFT/E2E/ITL percentiles from the histograms via `histogram_quantile`, cache
health from the computed/cached prefill split, and preemptions read as a
regression alarm -- 0 is the healthy steady state, q27 never preempts.

**Raw:**

```
curl -s http://127.0.0.1:8080/metrics | grep ^q27_
```

## Precision notes

- Counters are `unsigned long long` atomics; a scrape can at worst miss one
  in-flight increment on a single series (self-corrects next poll). No
  cross-series invariant exists between them, so partial reads are harmless.
- The MTP lane counters (`gate_lane_fired/acc`) are plain per-engine longs
  written by decode threads without atomics; reads are approximate-atomic on
  x86-64 and monotonic, accepted for monitoring (documented at the handler).
- The histogram seqlock guarantees the `+Inf == _count` invariant and a
  coherent `sum/count` pair per scrape; it never blocks the decode path
  (writer critical section: three relaxed atomics).
