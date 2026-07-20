#!/bin/bash
# Pre-registered Q4 half-GEMM ship-line bench, serverless variant
# (docs/plans/2026-07-17-t2-prefill-throughput.md, "Q4/Q8 chunk-GEMM
# schedule port"). The standing watcher (tools/quiet_q4_bench.sh) hard-requires
# a canonical T2 server to stop/restore; since 2026-07-18 there is no live
# pi traffic and no resident server, so the service lifecycle is moot. This
# runs the EXACT pre-registered measurement: float vs half at 8352 and 960,
# quiet machine, one synthetic-weight bench resident at a time, each leg
# gated on >=600 s input idle, verdict computed by the same ratio/ship-line.
set -u -o pipefail
cd "$(dirname "$0")/../.."
LOGDIR=logs/q4port-20260717
BENCH=./build/metal_prefill_bench
IDLE_NEED=600

idle_s() { ioreg -c IOHIDSystem 2>/dev/null | awk '/HIDIdleTime/{printf "%d",$NF/1000000000; exit}'; }
wait_idle() {
    local t0; t0=$(date +%s)
    while :; do
        local i; i=$(idle_s)
        [ -n "$i" ] && [ "$i" -ge "$IDLE_NEED" ] && return 0
        sleep 20
    done
}
rate() { grep -oE '[0-9]+\.[0-9]+ tok/s' "$1" | tail -1 | grep -oE '^[0-9.]+'; }

run_leg() { # name half prompt
    local name="$1" half="$2" prompt="$3" log="$LOGDIR/quiet_$1.log"
    wait_idle
    env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin Q27_METAL_GEMM_HALF_Q4=$half \
        caffeinate -dims "$BENCH" --dtype q4q8 --prompt "$prompt" > "$log" 2>&1 || return 1
    rate "$log"
}

echo "quiet ship bench: waiting for idle>=${IDLE_NEED}s (serverless; no live traffic)"
wait_idle
echo "idle reached $(date '+%H:%M:%S'); fingerprinting"
make -B build/metal_prefill_bench >/dev/null 2>&1 || { echo "FAIL rebuild"; exit 1; }
BSHA=$(shasum -a1 "$BENCH" | awk '{print $1}')
SSHA=$(shasum -a1 src/metal/q27_kernels.metal | awk '{print $1}')

F1=$(run_leg float8352_a 0 8352); H1=$(run_leg half8352_a 1 8352)
F2=$(run_leg float8352_b 0 8352); H2=$(run_leg half8352_b 1 8352)
F9=$(run_leg float960 0 960);    H9=$(run_leg half960 1 960)

/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 - "$F1" "$F2" "$H1" "$H2" "$F9" "$H9" "$BSHA" "$SSHA" <<'EOF'
import math,sys
f1,f2,h1,h2,f9,h9=[float(x) for x in sys.argv[1:7]]
bsha,ssha=sys.argv[7:9]
fmax=max(f1,f2); hmin=min(h1,h2)
ratio=hmin/fmax if fmax>0 else float('nan')
print(f"quiet ship-line bench (8352/96, dtype q4q8, idle>=10min, serverless)")
print(f"  bench sha1 {bsha}  shader sha1 {ssha}")
print(f"  float 8352: {f1} {f2} (max {fmax:.2f}; quiet baseline 23.2)")
print(f"  half  8352: {h1} {h2} (min {hmin:.2f})")
print(f"  float 960: {f9}   half 960: {h9}")
print(f"  ratio min(half)/max(float) = {ratio:.3f} (ship >=1.7 i.e. >=39 tok/s)")
suspect = not (18.0 <= fmax <= 30.0)
if suspect:
    print("VERDICT: SUSPECT -- float baseline failed to reproduce quiet 23.2; rejected")
    raise SystemExit(3)
elif ratio==ratio and ratio>=1.7 and hmin>=39.0:
    print("VERDICT: SHIP -- q27_matmul_q4_mm_h default-ON (Q27_METAL_GEMM_HALF_Q4=1)")
else:
    print("VERDICT: PARK -- <1.7x; keep Q4 on float-staged kernel + honest table")
EOF
