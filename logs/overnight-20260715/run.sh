#!/bin/bash
# Overnight vendor-stack batch — 2026-07-15, 24GB M4 (daily driver, user-authorized)
# Batch A: binary-tier Phase 0A (docs/plans/2026-07-15-binary-tier.md)
# Batch B: sibling-drafter probe Phase 0 (docs/plans/2026-07-15-sibling-drafter-probe.md)
# All model loads strictly serialized. Runs in their fork stack only — no q27 engine work.
set -uo pipefail
Q27=~/projects/q27
FORK=~/prism-fork
LOG=$Q27/logs/overnight-20260715
MODELS=$Q27/models
T2=$MODELS/ternary-bonsai-27b/Ternary-Bonsai-27B-Q2_0.gguf
B1=$MODELS/binary-bonsai-27b/Bonsai-27B-Q1_0.gguf
DRAFT=$MODELS/ternary-bonsai-1.7b/Ternary-Bonsai-1.7B-Q2_0.gguf
DSPARK=$MODELS/binary-bonsai-27b/Bonsai-27B-dspark-Q4_1.gguf
mkdir -p "$LOG" "$MODELS/binary-bonsai-27b" "$MODELS/ternary-bonsai-1.7b" "$Q27/data"

step() { echo; echo "===== [$(date '+%H:%M:%S')] $1 ====="; }

# ---------- 0. Fork binaries (wiped from /tmp on reboot; pinned release) ----------
step "0. fork binaries"
if [ ! -x "$FORK/llama-bench" ]; then
  mkdir -p "$FORK" && cd "$FORK"
  curl -L -C - -o fork.tar.gz \
    "https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b9591-62061f9/llama-prism-b9591-62061f9-bin-macos-arm64.tar.gz" \
    && tar xzf fork.tar.gz --strip-components=1 2>/dev/null || tar xzf fork.tar.gz
  # binaries may land in a subdir; flatten if needed
  [ -x "$FORK/llama-bench" ] || { d=$(find "$FORK" -name llama-bench -maxdepth 3 | head -1); [ -n "$d" ] && cp "$(dirname "$d")"/llama-* "$FORK/" 2>/dev/null; }
fi
export DYLD_LIBRARY_PATH="$FORK:${DYLD_LIBRARY_PATH:-}"
ls "$FORK" | head -30
"$FORK/llama-bench" --help >/dev/null 2>&1 || { echo "FATAL: fork llama-bench not runnable"; exit 1; }

# ---------- 1. Downloads + checksums ----------
step "1. downloads"
dl() { [ -s "$2" ] || curl -L -C - -o "$2" "https://huggingface.co/$1"; }
dl "prism-ml/Bonsai-27B-gguf/resolve/main/Bonsai-27B-Q1_0.gguf"           "$B1"
dl "prism-ml/Ternary-Bonsai-1.7B-gguf/resolve/main/Ternary-Bonsai-1.7B-Q2_0.gguf" "$DRAFT"
dl "prism-ml/Bonsai-27B-gguf/resolve/main/Bonsai-27B-dspark-Q4_1.gguf"    "$DSPARK"
[ -s "$Q27/data/wikitext-2-raw/wiki.test.raw" ] || {
  curl -L -o "$Q27/data/wikitext-2-raw-v1.zip" "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
  cd "$Q27/data" && unzip -o wikitext-2-raw-v1.zip; }
WIKI=$(find "$Q27/data" -name "wiki.test.raw" | head -1)
( cd "$MODELS/binary-bonsai-27b" && md5 -q Bonsai-27B-Q1_0.gguf Bonsai-27B-dspark-Q4_1.gguf > CHECKSUMS.md5.new && paste <(cat CHECKSUMS.md5.new) <(echo -e "Bonsai-27B-Q1_0.gguf\nBonsai-27B-dspark-Q4_1.gguf") > CHECKSUMS.md5 && rm CHECKSUMS.md5.new )
( cd "$MODELS/ternary-bonsai-1.7b" && echo "$(md5 -q Ternary-Bonsai-1.7B-Q2_0.gguf)  Ternary-Bonsai-1.7B-Q2_0.gguf" > CHECKSUMS.md5 )
ls -la "$MODELS/binary-bonsai-27b" "$MODELS/ternary-bonsai-1.7b"

# ---------- 2. T2 baseline re-anchor (matched-session thermal) ----------
step "2. T2 27B llama-bench baseline (vs recorded 8.41 +/- 1.36 tg128)"
"$FORK/llama-bench" -m "$T2" 2>&1 | tee "$LOG/t2-bench.log"

# ---------- 3. B1 llama-bench ----------
step "3. B1 27B llama-bench (projection: ~2x T2 if bandwidth-bound in their stack)"
"$FORK/llama-bench" -m "$B1" 2>&1 | tee "$LOG/b1-bench.log"

# ---------- 4. B1 wikitext-2 PPL, ternary-spike protocol (ctx-512, 24 chunks) ----------
# T2 recorded 10.04 +/- 0.36 on this exact protocol; official tier 4.90 (0-2k bucket, our harness, indicative)
step "4. B1 perplexity (compare: T2=10.04)"
"$FORK/llama-perplexity" -m "$B1" -f "$WIKI" -c 512 --chunks 24 2>&1 | tee "$LOG/b1-ppl.log"

# ---------- 5. Drafter probe: 1.7B solo bench ----------
step "5. Ternary-1.7B solo llama-bench (expect ~80-100 tok/s class)"
"$FORK/llama-bench" -m "$DRAFT" 2>&1 | tee "$LOG/draft17-bench.log"

# ---------- 6. Speculative sweeps (vocab gate is implicit: llama-speculative refuses on mismatch) ----------
step "6. speculative sweeps"
SPEC="$FORK/llama-speculative"
[ -x "$SPEC" ] || SPEC=""
PROMPT_AGENTIC='You are a coding agent. Task: in the file utils.py below, fix the bug in merge_intervals and return ONLY a JSON object {"path": "utils.py", "patch": "<unified diff>"}.\n\ndef merge_intervals(intervals):\n    intervals.sort()\n    out = []\n    for s, e in intervals:\n        if out and s < out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out\n\n# bug: touching intervals like [1,2],[2,3] should merge'
PROMPT_PROSE='Write a detailed, factual overview of how tidal forces shape planetary ring systems, in about 600 words.'
if [ -n "$SPEC" ]; then
  for TGT_NAME in t2 b1; do
    TGT=$T2; [ $TGT_NAME = b1 ] && TGT=$B1
    for D in 4 8; do
      for PN in agentic prose; do
        P="$PROMPT_AGENTIC"; [ $PN = prose ] && P="$PROMPT_PROSE"
        step "6.$TGT_NAME draft-max=$D prompt=$PN"
        "$SPEC" -m "$TGT" -md "$DRAFT" --spec-draft-n-max $D -n 256 --top-k 1 --temp 0 \
          -p "$P" 2>&1 | tee "$LOG/spec-$TGT_NAME-d$D-$PN.log" | tail -25
      done
    done
  done
else
  echo "llama-speculative not in fork release; falling back to llama-server -md timing" | tee "$LOG/spec-missing.log"
  ls "$FORK" >> "$LOG/spec-missing.log"
fi

# ---------- 7. B1 behavioral probes (ternary-spike categories, scripted greedy; prompts are reconstructions) ----------
step "7. B1 behavioral probes via llama-server"
"$FORK/llama-server" -m "$B1" --port 18808 --temp 0 >"$LOG/b1-server.log" 2>&1 &
SRV=$!
for i in $(seq 1 120); do curl -s localhost:18808/health | grep -q ok && break; sleep 2; done
probe() { # name, prompt
  curl -s localhost:18808/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$2")}],\"temperature\":0,\"top_k\":1,\"max_tokens\":1024}" \
    > "$LOG/probe-$1.json"; echo "--- probe $1 done"; }
probe json 'Return ONLY a JSON object (no prose, no code fences) with keys: name (string), founded (integer year), languages (array of exactly 3 strings), for the city of Zurich.'
probe constraints 'List exactly 5 hiking essentials. Constraints: numbered list; each item exactly 3 words; no item starts with "A"; include one item containing the word "map"; all lowercase; no punctuation at line ends; final line must read: done.'
probe codeedit 'Fix this function so touching intervals such as [1,2],[2,3] merge, and return only the corrected python code: def merge_intervals(intervals): intervals.sort(); out=[];\n for s,e in intervals:\n  if out and s<out[-1][1]: out[-1][1]=max(out[-1][1],e)\n  else: out.append([s,e])\n return out'
# native tool call
curl -s localhost:18808/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"What is the weather in Taipei right now?"}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["c","f"]}},"required":["city"]}}}],
  "temperature":0,"max_tokens":512}' > "$LOG/probe-toolcall.json"; echo "--- probe toolcall done"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
python3 - <<'EOF'
import json,glob,os
for f in sorted(glob.glob(os.path.expanduser("~/projects/q27/logs/overnight-20260715/probe-*.json"))):
    try:
        d=json.load(open(f)); m=d["choices"][0]["message"]
        print(f"{os.path.basename(f)}: tool_calls={bool(m.get('tool_calls'))} len={len(m.get('content') or '')}")
    except Exception as e: print(f"{os.path.basename(f)}: PARSE-FAIL {e}")
EOF

# ---------- 8. DSpark force-enable hunt (timeboxed by design: flags or nothing) ----------
step "8. DSpark flag hunt"
for b in llama-server llama-speculative llama-cli llama-bench; do
  [ -x "$FORK/$b" ] && { echo "--- $b:"; "$FORK/$b" --help 2>&1 | grep -i -E "dspark|dflash|block.?draft|confidence" ; }
done | tee "$LOG/dspark-flags.log"
if grep -qi dspark "$LOG/dspark-flags.log"; then
  step "8b. DSpark attempt (t2 target)"
  FLAG=$(grep -oiE -- '--[a-z-]*dspark[a-z-]*' "$LOG/dspark-flags.log" | head -1)
  "$FORK/llama-speculative" -m "$T2" -md "$DSPARK" $FLAG -n 256 --top-k 1 --temp 0 \
    -p "$PROMPT_AGENTIC" 2>&1 | tee "$LOG/spec-dspark-t2.log" | tail -25 || true
else
  echo "No dspark flag exposed in fork help text — recorded, moving on (probe doc step 6 contingency)."
fi

step "DONE $(date)"
