#!/bin/bash
# Live gates for the Metal server agentic-parity round
# (docs/plans/2026-07-17-metal-agentic-parity.md). Runs against an already-
# running q27-metal-server; greedy decode; contention-tolerant (no timing).
#   usage: tools/agentic_parity_gate.sh [base-url]   (default http://127.0.0.1:8213)
set -u
BASE="${1:-http://127.0.0.1:8213}"
FAIL=0
note() { printf '%s\n' "$*"; }
pass() { note "PASS: $*"; }
fail() { note "FAIL: $*"; FAIL=1; }

TOOLS_ANTH='[{"name":"get_weather","description":"Get current weather for a city","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]'
TOOLS_OAI='[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
ASK='What is the weather in Paris right now? Use the get_weather tool.'

jget() { python3 -c "
import sys,json
d=json.load(sys.stdin)
try: print(eval(sys.argv[1],{},{'d':d}))
except Exception as e: print('JGET-ERR:'+str(e))
" "$1"; }

# ---- G2: count_tokens exists and equals /v1/messages usage.input_tokens ----
CT_BODY="{\"model\":\"q27-metal\",\"max_tokens\":400,\"tools\":$TOOLS_ANTH,\"messages\":[{\"role\":\"user\",\"content\":\"$ASK\"}]}"
CT=$(curl -s -H "Content-Type: application/json" --max-time 60 "$BASE/v1/messages/count_tokens" -d "$CT_BODY" | jget "d['input_tokens']")
case "$CT" in (''|*[!0-9]*) fail "G2 count_tokens not a positive int: '$CT'";; (*) note "G2 count_tokens=$CT";; esac

# ---- G3: Anthropic tool round, non-streaming ----
R3=$(curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/messages" -d "$CT_BODY")
IN_TOK=$(printf '%s' "$R3" | jget "d['usage']['input_tokens']")
SR=$(printf '%s' "$R3" | jget "d['stop_reason']")
TU_NAME=$(printf '%s' "$R3" | jget "[b for b in d['content'] if b['type']=='tool_use'][0]['name']")
TU_CITY=$(printf '%s' "$R3" | jget "[b for b in d['content'] if b['type']=='tool_use'][0]['input'].get('city','')")
TU_ID=$(printf '%s' "$R3" | jget "[b for b in d['content'] if b['type']=='tool_use'][0]['id']")
TU_INPUT=$(printf '%s' "$R3" | jget "__import__('json').dumps([b for b in d['content'] if b['type']=='tool_use'][0]['input'])")
[ "$CT" = "$IN_TOK" ] && pass "G2 count_tokens == usage.input_tokens ($CT)" || fail "G2 mismatch: count=$CT usage=$IN_TOK"
[ "$TU_NAME" = "get_weather" ] && pass "G3 tool_use block name=get_weather" || { fail "G3 no/wrong tool_use: '$TU_NAME'"; note "$R3"; }
[ -n "$TU_CITY" ] && [ "${TU_CITY#JGET-ERR}" = "$TU_CITY" ] && pass "G3 input.city='$TU_CITY'" || fail "G3 input.city missing: '$TU_CITY'"
[ "$SR" = "tool_use" ] && pass "G3 stop_reason=tool_use" || fail "G3 stop_reason='$SR'"

# ---- G3s: Anthropic tool round, streaming ----
S3=$(curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/messages" -d "${CT_BODY%\}},\"stream\":true}")
printf '%s' "$S3" | grep -q '"type":"tool_use"' && pass "G3s content_block_start(tool_use) event" || fail "G3s no tool_use block in stream"
printf '%s' "$S3" | grep -q '"input_json_delta"' && pass "G3s input_json_delta event" || fail "G3s no input_json_delta in stream"
printf '%s' "$S3" | grep -q '"stop_reason":"tool_use"' && pass "G3s stop_reason=tool_use" || fail "G3s stop_reason wrong"

# ---- G4: Anthropic round 2 with tool_result history ----
R4_BODY="{\"model\":\"q27-metal\",\"max_tokens\":400,\"tools\":$TOOLS_ANTH,\"messages\":[
 {\"role\":\"user\",\"content\":\"$ASK\"},
 {\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"$TU_ID\",\"name\":\"get_weather\",\"input\":$TU_INPUT}]},
 {\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"$TU_ID\",\"content\":\"14 degrees C, sunny, light wind\"}]}]}"
R4=$(curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/messages" -d "$R4_BODY")
R4_TEXT=$(printf '%s' "$R4" | jget "' '.join(b['text'] for b in d['content'] if b['type']=='text')")
R4_SR=$(printf '%s' "$R4" | jget "d['stop_reason']")
[ -n "$R4_TEXT" ] && [ "${R4_TEXT#JGET-ERR}" = "$R4_TEXT" ] && pass "G4 round-2 text: ${R4_TEXT:0:80}" || { fail "G4 no text after tool_result"; note "$R4"; }
[ "$R4_SR" = "end_turn" ] && pass "G4 stop_reason=end_turn" || fail "G4 stop_reason='$R4_SR'"

# ---- G5: OpenAI chat tool round ----
C_BODY="{\"model\":\"q27-metal\",\"max_tokens\":400,\"tools\":$TOOLS_OAI,\"messages\":[{\"role\":\"user\",\"content\":\"$ASK\"}]}"
R5=$(curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/chat/completions" -d "$C_BODY")
C_NAME=$(printf '%s' "$R5" | jget "d['choices'][0]['message']['tool_calls'][0]['function']['name']")
C_ARGS=$(printf '%s' "$R5" | jget "d['choices'][0]['message']['tool_calls'][0]['function']['arguments']")
C_ID=$(printf '%s' "$R5" | jget "d['choices'][0]['message']['tool_calls'][0]['id']")
C_FIN=$(printf '%s' "$R5" | jget "d['choices'][0]['finish_reason']")
[ "$C_NAME" = "get_weather" ] && pass "G5 message.tool_calls name=get_weather" || { fail "G5 no/wrong tool_calls: '$C_NAME'"; note "$R5"; }
printf '%s' "$C_ARGS" | python3 -c "import sys,json; json.loads(sys.stdin.read())" 2>/dev/null && pass "G5 arguments is JSON-encoded string" || fail "G5 arguments not parseable: '$C_ARGS'"
[ "$C_FIN" = "tool_calls" ] && pass "G5 finish_reason=tool_calls" || fail "G5 finish_reason='$C_FIN'"

S5=$(curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/chat/completions" -d "${C_BODY%\}},\"stream\":true}")
# Full wire-shape check on the streamed tool_calls chunk (codex P3): index,
# id, type, name, and arguments as a JSON-encoded string that decodes.
S5_OK=$(printf '%s' "$S5" | python3 -c "
import sys,json
ok='NO'
for line in sys.stdin:
    if not line.startswith('data: ') or line.strip()=='data: [DONE]': continue
    d=json.loads(line[6:])
    tcs=d.get('choices',[{}])[0].get('delta',{}).get('tool_calls')
    if not tcs: continue
    t=tcs[0]
    if (t.get('index')==0 and t.get('id','').startswith('call_') and
        t.get('type')=='function' and t['function']['name']=='get_weather' and
        isinstance(json.loads(t['function']['arguments']),dict)): ok='YES'
print(ok)
")
[ "$S5_OK" = "YES" ] && pass "G5s delta.tool_calls chunk (full wire shape)" || fail "G5s tool_calls chunk missing/misshapen"
printf '%s' "$S5" | grep -q '"finish_reason":"tool_calls"' && pass "G5s finish_reason=tool_calls" || fail "G5s finish_reason wrong"

R5B_BODY="{\"model\":\"q27-metal\",\"max_tokens\":400,\"tools\":$TOOLS_OAI,\"messages\":[
 {\"role\":\"user\",\"content\":\"$ASK\"},
 {\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"$C_ID\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\": \\\"Paris\\\"}\"}}]},
 {\"role\":\"tool\",\"tool_call_id\":\"$C_ID\",\"content\":\"14 degrees C, sunny, light wind\"}]}"
R5B=$(curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/chat/completions" -d "$R5B_BODY")
R5B_TEXT=$(printf '%s' "$R5B" | jget "d['choices'][0]['message']['content']")
[ -n "$R5B_TEXT" ] && [ "$R5B_TEXT" != "None" ] && [ "${R5B_TEXT#JGET-ERR}" = "$R5B_TEXT" ] && pass "G5 round-2 content: ${R5B_TEXT:0:80}" || { fail "G5 round-2 empty"; note "$R5B"; }

# ---- G6: overflow contract ("prompt is too long") ----
G6_FILE=$(mktemp)
python3 -c "
import json,sys
body={'model':'q27-metal','max_tokens':16,'messages':[{'role':'user','content':'word '*140000}]}
open(sys.argv[1],'w').write(json.dumps(body))
" "$G6_FILE"
G6=$(curl -s -H "Content-Type: application/json" --max-time 120 -w '\n%{http_code}' "$BASE/v1/messages" --data-binary @"$G6_FILE")
G6_CODE=$(printf '%s' "$G6" | tail -1)
printf '%s' "$G6" | head -1 | grep -q "prompt is too long" && [ "$G6_CODE" = "400" ] \
  && pass "G6 400 + 'prompt is too long'" || { fail "G6 overflow shape wrong (code=$G6_CODE)"; note "$(printf '%s' "$G6" | head -1 | head -c 300)"; }
rm -f "$G6_FILE"

# ---- G7: billing-header normalization -> full prefix hit ----
mk_g7() { python3 -c "
import json,sys
sys_txt='x-anthropic-billing-header: cc_version=2.1.100; cc_entrypoint=cli; cch=$1;You are a helpful assistant working in a code repository. '+('Context filler sentence. '*120)
body={'model':'q27-metal','max_tokens':8,'system':sys_txt,'messages':[{'role':'user','content':'Say OK.'}]}
print(json.dumps(body))
"; }
G7A=$(mk_g7 a5145 | curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/messages" --data-binary @-)
G7B=$(mk_g7 b7777 | curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/messages" --data-binary @-)
G7_IN=$(printf '%s' "$G7B" | jget "d['usage']['input_tokens']")
G7_HIT=$(printf '%s' "$G7B" | jget "d['q27_prefix_hit']")
[ "$G7_HIT" = "$G7_IN" ] && [ -n "$G7_HIT" ] && pass "G7 cch-varied prompt full prefix hit ($G7_HIT/$G7_IN)" \
  || fail "G7 prefix hit $G7_HIT of $G7_IN (normalization broken?)"

note "----"
[ $FAIL -eq 0 ] && note "agentic parity gates: ALL PASS" || note "agentic parity gates: FAILURES"
exit $FAIL
