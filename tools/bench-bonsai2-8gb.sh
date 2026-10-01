#!/usr/bin/env bash
# Bonsai 2 small-card bench: one file to paste back.
#
# Assumes install-bonsai2-8gb.sh ran (default ~/bonsai2). Boots each pack it
# finds (plain, and the MTP one if you installed --mtp), records what the
# card is and how much context the server took, sends the same four greedy
# prompts my gates use (their replies must md5-match mine -- same kernels on
# any sm_86 card -- so a mismatch is itself a finding), then one ~5K-token
# prompt for the prefill rate. Everything lands in ~/bonsai2/bench-<date>.txt;
# paste that file back to me.
#
#   bash bench-bonsai2-8gb.sh            # ~4 minutes, nothing else on the GPU please
set -euo pipefail
DIR=${BONSAI2_DIR:-$HOME/bonsai2}; PORT=${PORT:-8090}
OUT=$DIR/bench-$(date +%Y%m%d-%H%M).txt
exec > >(tee "$OUT") 2>&1
echo "== bonsai2 small-card bench, $(date -u +%FT%TZ), $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,power.limit,clocks.max.sm,clocks.max.mem,pcie.link.gen.current,pcie.link.width.current,display_active --format=csv
echo "q27: $(cd "$DIR/q27" && git describe --tags --always)   cuda: $(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')   kernel: $(uname -r)"
echo "cpu: $(lscpu | sed -n 's/^Model name: *//p' | head -1)   ram: $(free -g | awk '/^Mem:/{print $2" GB"}')"

bench_pack() { # $1 pack file, $2 stack GB, $3 label, $4 expected md5s "short:.. cities:.. long:.. code:.."
  local pack=$1 stack=$2 label=$3 expect=$4 log=$DIR/bench-server-$3.log
  [ -f "$DIR/models/$pack" ] || { echo; echo "== $label: $pack not installed, skipped"; return 0; }
  echo; echo "== $label ($pack)"
  env Q27_FIXED_STACK_GB=$stack Q27_BATCH=0 Q27_PRINT_WSUM=1 "$DIR/q27/build/q27-server-12g" "$DIR/models/$pack" "$DIR/models/qwen38-27b-mtp.tok" \
    --slots 1 --host 127.0.0.1 --port $PORT --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0 >"$log" 2>&1 &
  local spid=$!
  for i in $(seq 1 180); do
    curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200 && break
    kill -0 $spid 2>/dev/null || { echo "   server died:"; tail -5 "$log"; return 0; }
    sleep 2
  done
  grep -E "wsum|vram: free|\[pool\] (paged|ctx|--ctx|cannot|sizing)|KV cache|clamping" "$log" | sed 's/^/   /'
  python3 - "$PORT" "$expect" "$DIR/bench-replies-$label.json" <<'PY'
import json, urllib.request, sys, time, hashlib
port, expect = sys.argv[1], dict(kv.split(":") for kv in sys.argv[2].split())
base = f"http://127.0.0.1:{port}"
def msg(prompt, n, think):
    b = {"model": "q27", "max_tokens": n, "stream": False, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    if think: b["thinking"] = {"type": "enabled", "budget_tokens": n - 16}
    req = urllib.request.Request(base + "/v1/messages", data=json.dumps(b).encode(),
                                 headers={"content-type": "application/json", "x-api-key": "local", "anthropic-version": "2023-06-01"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r: d = json.load(r)
    text = "".join((x.get("thinking", "") if x.get("type") == "thinking" else x.get("text", "")) for x in d["content"])
    return text, d["usage"]["output_tokens"], time.time() - t0
P = {"short": ("Reply with the single word ok.", 32, False),
     "cities": ("What is the capital of France, and name two other French cities? Answer in one sentence.", 200, True),
     "long": ("Write a 300-word explanation of how a hash table handles collisions, with a short Python example.", 700, True),
     "code": ("Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), with docstrings and a small pytest test file. Explain the data structure choice briefly first.", 1500, True)}
print("   prompt   out_tok   wall_s   md5 (vs mine)")
replies = {}
for name, (p, n, th) in P.items():
    text, ntok, wall = msg(p, n, th)
    replies[name] = text
    h = hashlib.md5(text.encode()).hexdigest()
    verdict = "MATCH" if expect.get(name) == h else f"DIFFERS (mine {expect.get(name, '?')[:8]})"
    print(f"   {name:7s}  {ntok:6d}   {wall:6.1f}   {h[:8]} {verdict}")
json.dump(replies, open(sys.argv[3], "w"), indent=1)
print(f"   replies saved to {sys.argv[3]} (paste a DIFFERS one back and I can find the position)")
# prefill: one long prompt, one output token, no thinking
para = ("The hash table keeps an array of buckets and a hash function that maps each key to one of them; "
        "when two keys land in the same bucket the table either chains them in a list or probes for the next free slot, "
        "and it resizes once the load factor crosses a threshold so lookups stay close to constant time. ")
text, ntok, wall = msg(para * 80 + "\n\nReply with the single word ok.", 8, False)
print(f"   prefill probe: {wall:.1f} s wall (see the pf= line below)")
PY
  echo "   server [req] lines: prompt=tokens pf_ms=prefill dec=tokens dec_ms rounds tps"
  grep "\[req\]" "$log" | sed -E 's/.*prompt=([0-9]+).*pf=([0-9]+) pf_ms=([0-9]+) dec=([0-9]+) dec_ms=([0-9]+).*rounds=([0-9]+) tps=([0-9.]+).*/   prompt=\1 pf_ms=\3 dec=\4 dec_ms=\5 rounds=\6 tps=\7/'
  grep "\[req\]" "$log" | tail -1 | sed -E 's/.*pf=([0-9]+) pf_ms=([0-9]+).*/\1 \2/' | awk '{ if ($2 > 0) printf "   prefill: %d tokens in %d ms = %.0f tok/s\n", $1, $2, $1 * 1000 / $2 }'
  kill $spid 2>/dev/null || true; wait $spid 2>/dev/null || true
  sleep 3
}

bench_pack bonsai2-27b-t3-slim.q27 0.6 plain \
  "short:fff73c83fd8995a1c944ef54152565b4 cities:9e103621ee7cf783068fbc211af5988d long:c29361414156a556f772bb51e6bb37b6 code:4228b9dd08b6b8675f33042d3bf04215"
bench_pack bonsai2-27b-t3-mtp-slim.q27 0.8 mtp \
  "short:fff73c83fd8995a1c944ef54152565b4 cities:0cd763b306054c9afc0fcdeabf425b54 long:d163fc3c7a5b0d6f7911fb625d79d2b3 code:d8c77c7ed3bdda30d34b8343f6c756a3"

echo; echo "== done -- paste $OUT back"
