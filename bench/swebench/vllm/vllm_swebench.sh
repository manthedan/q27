#!/usr/bin/env bash
# Wrap swebench_run.sh vllm with vLLM /metrics snapshots so we get decode t/s
# (vLLM isn't a systemd unit, so the runner's journal harvest yields nothing).
set -u
SP=/mnt/ai/projects/q27/scratchpad
snap(){ curl -s -m 5 http://127.0.0.1:8080/metrics | grep -E '^vllm:(generation_tokens_total|inter_token_latency_seconds_sum|inter_token_latency_seconds_count|spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total)\{'; }
snap > "$SP/vllm_m0.txt"
bash "$SP/swebench_run.sh" vllm
snap > "$SP/vllm_m1.txt"
echo ""; echo "=== vLLM DECODE (from /metrics delta) ==="
python3 - "$SP/vllm_m0.txt" "$SP/vllm_m1.txt" "$SP/swebench_results.vllm.jsonl" <<'PY'
import sys,re,json
def load(f):
    d={}
    for ln in open(f):
        m=re.match(r'(vllm:\w+)\{[^}]*\}\s+([0-9.eE+]+)',ln)
        if m: d[m.group(1)]=float(m.group(2))
    return d
a=load(sys.argv[1]); b=load(sys.argv[2])
g=b.get('vllm:generation_tokens_total',0)-a.get('vllm:generation_tokens_total',0)
its=b.get('vllm:inter_token_latency_seconds_sum',0)-a.get('vllm:inter_token_latency_seconds_sum',0)
itc=b.get('vllm:inter_token_latency_seconds_count',0)-a.get('vllm:inter_token_latency_seconds_count',0)
dr=b.get('vllm:spec_decode_num_draft_tokens_total',0)-a.get('vllm:spec_decode_num_draft_tokens_total',0)
ac=b.get('vllm:spec_decode_num_accepted_tokens_total',0)-a.get('vllm:spec_decode_num_accepted_tokens_total',0)
rows=[json.loads(l) for l in open(sys.argv[3])]
done=len(rows); ne=sum(r['nonempty'] for r in rows); gh=sum(r['gold_hit'] for r in rows); wall=sum(r['wall_s'] for r in rows)
print(f"  instances: {done}  nonempty-diff: {ne}/{done}  edited-gold-file: {gh}/{done}")
print(f"  total wall: {wall}s ({wall/done:.0f}s/inst avg)" if done else "  none")
print(f"  decode: {g/its if its else 0:.1f} t/s  (gen_tokens={int(g)}, decode_sec={its:.1f})")
print(f"  MTP draft acceptance: {(ac/dr*100) if dr else 0:.0f}% ({int(ac)}/{int(dr)})")
PY
echo "=== END (vllm) ==="
