#!/bin/bash
# Live gates for the /v1/responses full parity port + error-class split
# (docs/plans/2026-07-17-responses-parity-residue.md). Runs against an
# already-running q27-metal-server; greedy; contention-tolerant (no timing).
#   usage: tools/responses_parity_gate.sh [base-url]   (default http://127.0.0.1:8213)
set -u
BASE="${1:-http://127.0.0.1:8213}"
FAIL=0
note() { printf '%s\n' "$*"; }
pass() { note "PASS: $*"; }
fail() { note "FAIL: $*"; FAIL=1; }

TOOLS_RESP='[{"type":"function","name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]'
ASK='What is the weather in Paris right now? Use the get_weather tool.'

jget() { python3 -c "
import sys,json
d=json.load(sys.stdin)
try: print(eval(sys.argv[1],{},{'d':d}))
except Exception as e: print('JGET-ERR:'+str(e))
" "$1"; }

# ---- G8a: non-stream weather-tool request -> function_call item ----
A_BODY="{\"model\":\"q27-metal\",\"max_output_tokens\":400,\"tools\":$TOOLS_RESP,\"input\":\"$ASK\"}"
RA=$(curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/responses" -d "$A_BODY")
FC_NAME=$(printf '%s' "$RA" | jget "[i for i in d['output'] if i['type']=='function_call'][0]['name']")
FC_ARGS=$(printf '%s' "$RA" | jget "[i for i in d['output'] if i['type']=='function_call'][0]['arguments']")
FC_ARGS_ISSTR=$(printf '%s' "$RA" | jget "isinstance([i for i in d['output'] if i['type']=='function_call'][0]['arguments'],str)")
FC_CITY=$(printf '%s' "$RA" | jget "__import__('json').loads([i for i in d['output'] if i['type']=='function_call'][0]['arguments']).get('city','')")
[ "$FC_NAME" = "get_weather" ] && pass "G8a function_call item name=get_weather" || { fail "G8a no/wrong function_call: '$FC_NAME'"; note "$RA"; }
[ "$FC_ARGS_ISSTR" = "True" ] && pass "G8a arguments is a JSON-encoded string" || fail "G8a arguments not a string: $FC_ARGS_ISSTR"
[ -n "$FC_CITY" ] && [ "${FC_CITY#JGET-ERR}" = "$FC_CITY" ] && pass "G8a arguments.city='$FC_CITY'" || fail "G8a arguments.city missing: '$FC_CITY'"

# ---- G8b(i): stream text leg — codex 0.143 item lifecycle ordering ----
B_TEXT_BODY="{\"model\":\"q27-metal\",\"max_output_tokens\":64,\"input\":\"Say hello in one short sentence.\",\"stream\":true}"
SB=$(curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/responses" -d "$B_TEXT_BODY")
printf '%s' "$SB" | python3 -c "
import sys,json
e=[]
for line in sys.stdin:
    if not line.startswith('data: '): continue
    try: e.append(json.loads(line[6:]))
    except Exception: pass
def find(pred):
    return next((i for i,x in enumerate(e) if pred(x)),-1)
a=find(lambda x:x.get('type')=='response.output_item.added' and x.get('item',{}).get('type')=='message')
if a<0: print('ORDER-BAD:no-message-added'); raise SystemExit
idx=e[a].get('output_index'); iid=e[a].get('item',{}).get('id')
p=find(lambda x:x.get('type')=='response.content_part.added' and x.get('output_index')==idx and x.get('item_id')==iid)
d=find(lambda x:x.get('type')=='response.output_text.delta' and x.get('output_index')==idx and x.get('item_id')==iid)
z=find(lambda x:x.get('type')=='response.output_item.done' and x.get('output_index')==idx and x.get('item',{}).get('id')==iid)
c=find(lambda x:x.get('type')=='response.completed')
ok=a<p<d<z<c
print('ORDER-OK' if ok else 'ORDER-BAD:%s'%[(x.get('type'),x.get('output_index')) for x in e[:16]])
" | grep -q "ORDER-OK" && pass "G8b(i) same message item added+part precede delta and done" || { fail "G8b(i) message lifecycle ordering wrong"; printf '%s' "$SB" | head -20; }
printf '%s' "$SB" | grep -q '"response.completed"' && pass "G8b(i) response.completed terminator" || fail "G8b(i) no response.completed"

# Reasoning-producing leg: every reasoning done must pair with its own added.
B_REASON_BODY='{"model":"q27-metal","max_output_tokens":256,"input":"Think step by step about 17 times 19, then give the answer.","stream":true}'
SR=$(curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/responses" -d "$B_REASON_BODY")
printf '%s' "$SR" | python3 -c "
import sys,json
e=[]
for line in sys.stdin:
    if not line.startswith('data: '): continue
    try: e.append(json.loads(line[6:]))
    except Exception: pass
adds={(x.get('output_index'),x.get('item',{}).get('id')):i for i,x in enumerate(e)
      if x.get('type')=='response.output_item.added' and x.get('item',{}).get('type')=='reasoning'}
dones=[((x.get('output_index'),x.get('item',{}).get('id')),i) for i,x in enumerate(e)
       if x.get('type')=='response.output_item.done' and x.get('item',{}).get('type')=='reasoning']
ok=bool(dones) and all(k in adds and adds[k]<i for k,i in dones)
print('REASON-OK' if ok else 'REASON-BAD:adds=%r dones=%r'%(adds,dones))
" | grep -q "REASON-OK" && pass "G8b(i-r) reasoning items have paired added-before-done lifecycle" || { fail "G8b(i-r) reasoning lifecycle missing/mispaired"; printf '%s' "$SR" | head -24; }

# ---- G8b(ii): stream tool leg — done carries function_call; completed carries output+usage ----
B_TOOL_BODY="{\"model\":\"q27-metal\",\"max_output_tokens\":400,\"tools\":$TOOLS_RESP,\"input\":\"$ASK\",\"stream\":true}"
ST=$(curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/responses" -d "$B_TOOL_BODY")
DONE_FC=$(printf '%s' "$ST" | python3 -c "
import sys,json
e=[]
for line in sys.stdin:
    if not line.startswith('data: '): continue
    try: e.append(json.loads(line[6:]))
    except Exception: pass
adds={(x.get('output_index'),x.get('item',{}).get('id')) for x in e
      if x.get('type')=='response.output_item.added'}
dones=[x for x in e if x.get('type')=='response.output_item.done']
paired=all((x.get('output_index'),x.get('item',{}).get('id')) in adds for x in dones)
fc=next((x.get('item',{}).get('name','') for x in dones
         if x.get('item',{}).get('type')=='function_call'),'')
print(('OK:' if paired else 'UNPAIRED:')+fc)
")
[ "$DONE_FC" = "OK:get_weather" ] && pass "G8b(ii) paired added/done carries function_call" || { fail "G8b(ii) missing/unpaired function_call lifecycle: '$DONE_FC'"; printf '%s' "$ST" | tail -12; }
COMP_OK=$(printf '%s' "$ST" | python3 -c "
import sys,json
for line in sys.stdin:
    if not line.startswith('data: '): continue
    try: d=json.loads(line[6:])
    except Exception: continue
    if d.get('type')=='response.completed':
        r=d.get('response',{})
        ok = isinstance(r.get('output'),list) and len(r['output'])>0 and isinstance(r.get('usage'),dict) and 'input_tokens' in r['usage']
        print('OK' if ok else 'BAD'); break
")
[ "$COMP_OK" = "OK" ] && pass "G8b(ii) response.completed carries output items + usage" || fail "G8b(ii) response.completed missing output/usage"

# ---- G8c: round-trip — function_call_output in input is consumed, no spurious new call ----
C_BODY="{\"model\":\"q27-metal\",\"max_output_tokens\":400,\"tools\":$TOOLS_RESP,\"input\":[
 {\"type\":\"message\",\"role\":\"user\",\"content\":\"$ASK\"},
 {\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"},
 {\"type\":\"function_call_output\",\"call_id\":\"call_1\",\"output\":\"14 degrees C, sunny, light wind\"}]}"
RC=$(curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/responses" -d "$C_BODY")
C_TEXT=$(printf '%s' "$RC" | jget "' '.join(c['text'] for i in d['output'] if i['type']=='message' for c in i['content'] if c['type']=='output_text')")
C_FCALLS=$(printf '%s' "$RC" | jget "len([i for i in d['output'] if i['type']=='function_call'])")
case "$C_TEXT" in *14*|*sunny*|*Sunny*) pass "G8c round-trip text references the tool result: ${C_TEXT:0:80}";; *) { fail "G8c answer ignores tool result: '$C_TEXT'"; note "$RC"; };; esac
[ "$C_FCALLS" = "0" ] && pass "G8c no spurious new function_call" || fail "G8c spurious new call(s): $C_FCALLS"

# ---- G8d: custom freeform tool -> custom_tool_call item with bare-string input ----
D_BODY='{"model":"q27-metal","max_output_tokens":400,"tools":[{"type":"custom","name":"apply_patch","description":"Apply a code patch"}],"input":"Use the apply_patch tool to rename variable x to y in main.cpp."}'
RD=$(curl -s -H "Content-Type: application/json" --max-time 300 "$BASE/v1/responses" -d "$D_BODY")
D_NAME=$(printf '%s' "$RD" | jget "[i for i in d['output'] if i['type']=='custom_tool_call'][0]['name']")
D_ISSTR=$(printf '%s' "$RD" | jget "isinstance([i for i in d['output'] if i['type']=='custom_tool_call'][0]['input'],str)")
[ "$D_NAME" = "apply_patch" ] && pass "G8d custom_tool_call item name=apply_patch" || { fail "G8d no custom_tool_call: '$D_NAME'"; note "$RD"; }
[ "$D_ISSTR" = "True" ] && pass "G8d custom_tool_call input is a bare string" || fail "G8d input not a bare string: $D_ISSTR"

# ---- G9 controls: the split must not widen the 400 class ----
CODE_BADJSON=$(curl -s -o /dev/null -w '%{http_code}' -H "Content-Type: application/json" --max-time 30 "$BASE/v1/responses" --data-binary '{invalid json')
[ "$CODE_BADJSON" = "400" ] && pass "G9 invalid JSON -> 400" || fail "G9 invalid JSON -> $CODE_BADJSON"
CODE_EMPTY=$(curl -s -o /dev/null -w '%{http_code}' -H "Content-Type: application/json" --max-time 30 "$BASE/v1/responses" -d '{"model":"q27-metal","max_output_tokens":16}')
[ "$CODE_EMPTY" = "400" ] && pass "G9 missing input -> 400" || fail "G9 missing input -> $CODE_EMPTY"
BIG=$(python3 -c "print('lorem '*150000)")
R9=$(curl -s -H "Content-Type: application/json" --max-time 120 "$BASE/v1/responses" -d "{\"model\":\"q27-metal\",\"max_output_tokens\":16,\"input\":\"$BIG\"}")
CODE_BIG=$(printf '%s' "$R9" | jget "d['error']['code']")
[ "$CODE_BIG" = "context_length_exceeded" ] && pass "G9 oversize prompt -> 400 context_length_exceeded" || { fail "G9 oversize -> '$CODE_BIG'"; printf '%s' "$R9" | head -3; }

echo "----"
[ "$FAIL" = "0" ] && note "responses parity gates: ALL PASS" || note "responses parity gates: FAILURES"
exit $FAIL
