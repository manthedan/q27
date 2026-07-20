#!/bin/bash
# Census capability spot-check — one arm's generation pass
# (docs/plans/2026-07-17-census-capability-spotcheck.md).
#
#   usage: run_arm.sh <arm-name> <base-url> <local-model-path> [outdir]
#   e.g.:  run_arm.sh t2-base http://127.0.0.1:8213 models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
#
# COORDINATION.md applies: gen_runner is a model consumer — run ONLY inside
# a coordinated GPU slot against a server whose owner expects the traffic.
# The caller boots one server per pack (one model resident at a time) and
# runs this once per arm. Requests are serial and greedy (gen_runner sends
# no temperature; the server's missing-temperature default is 0.0), so an
# arm's outputs are deterministic for its pack + binary. The arm manifest,
# local full-file MD5, and the server's resident-mmap SHA1/filename are checked
# before the first request; a provenance sidecar is required by verdict tooling.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ $# -ge 3 ] || { echo "usage: $0 ARM BASE_URL LOCAL_MODEL_PATH [OUTDIR]" >&2; exit 2; }
ARM=$1; URL=$2; MODEL=$3; OUT=${4:-logs/eval-census}
python3 - "$URL" <<'PY' || { echo "run_arm: BASE_URL must be a credential-free loopback HTTP origin" >&2; exit 2; }
import ipaddress,sys,urllib.parse
u=urllib.parse.urlsplit(sys.argv[1])
if u.scheme != "http" or u.username is not None or u.password is not None or \
   u.path not in ("", "/") or u.query or u.fragment or u.hostname is None:
    raise SystemExit(1)
try: ip=ipaddress.ip_address(u.hostname)
except ValueError: raise SystemExit(1)  # no DNS/rebinding: literal IP only
if not ip.is_loopback: raise SystemExit(1)
PY
# urllib and curl must connect directly even under an ambient proxy config.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
MANIFEST_ROW=$(awk -F '\t' -v arm="$ARM" '$1==arm {print $2"\t"$3"\t"$4"\t"$5}' tools/eval/arms.tsv)
[ -n "$MANIFEST_ROW" ] || { echo "run_arm: unknown arm '$ARM'" >&2; exit 2; }
IFS=$'\t' read -r MANIFEST_NAME MANIFEST_MD5 MANIFEST_SHA1 MANIFEST_BYTES <<<"$MANIFEST_ROW"
[ -f "$MODEL" ] || { echo "run_arm: model not found: $MODEL" >&2; exit 2; }
ACTUAL_NAME=$(basename "$MODEL")
if command -v md5 >/dev/null 2>&1; then ACTUAL_MD5=$(md5 -q "$MODEL");
else ACTUAL_MD5=$(md5sum "$MODEL" | awk '{print $1}'); fi
if stat -f %z "$MODEL" >/dev/null 2>&1; then ACTUAL_BYTES=$(stat -f %z "$MODEL");
else ACTUAL_BYTES=$(stat -c %s "$MODEL"); fi
[ "$ACTUAL_NAME" = "$MANIFEST_NAME" ] && [ "$ACTUAL_MD5" = "$MANIFEST_MD5" ] &&
[ "$ACTUAL_BYTES" = "$MANIFEST_BYTES" ] || {
    echo "run_arm: artifact mismatch for $ARM: got $ACTUAL_NAME $ACTUAL_MD5 $ACTUAL_BYTES" >&2
    exit 2
}
HEALTH=$(curl --noproxy '*' -fsS "$URL/health?identity=1") || { echo "run_arm: server health failed" >&2; exit 2; }
HEALTH_ROW=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("model","")+"\t"+d.get("artifact_sha1","")+"\t"+d.get("boot_id","")+"\t"+json.dumps(d.get("runtime"),sort_keys=True,separators=(",",":")))' "$HEALTH")
IFS=$'\t' read -r HEALTH_MODEL HEALTH_SHA1 HEALTH_BOOT HEALTH_RUNTIME <<<"$HEALTH_ROW"
[ "$HEALTH_MODEL" = "$MANIFEST_NAME" ] && [ "$HEALTH_SHA1" = "$MANIFEST_SHA1" ] &&
[ -n "$HEALTH_BOOT" ] && [ -n "$HEALTH_RUNTIME" ] && [ "$HEALTH_RUNTIME" != "null" ] || {
    echo "run_arm: resident identity '$HEALTH_MODEL' '$HEALTH_SHA1' or runtime identity does not match frozen '$MANIFEST_NAME' '$MANIFEST_SHA1'" >&2
    exit 2
}
mkdir -p "$OUT"
LOCAL_HOST_ID=$(python3 - <<'PY'
import hashlib,os,secrets
root=os.path.expanduser("~/.q27"); os.makedirs(root,mode=0o700,exist_ok=True)
path=os.path.join(root,"eval-host-id")
try:
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"w") as f: f.write(secrets.token_hex(32)+"\n")
except FileExistsError:
    os.chmod(path,0o600)
with open(path,"rb") as f: print(hashlib.sha256(f.read()).hexdigest())
PY
)
PROVENANCE_RUNTIME=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["eval_host_id"]=sys.argv[2]; print(json.dumps(d,sort_keys=True,separators=(",",":")))' "$HEALTH_RUNTIME" "$LOCAL_HOST_ID")
RUN_ID=$(python3 -c 'import uuid; print(uuid.uuid4().hex)')
TMP=$(mktemp -d "$OUT/.${ARM}.run.XXXXXX")
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT
python3 - "$TMP/$ARM.provenance.json" "$ARM" "$MANIFEST_NAME" "$MANIFEST_MD5" "$MANIFEST_SHA1" "$MANIFEST_BYTES" "$PROVENANCE_RUNTIME" "$HEALTH_BOOT" "$RUN_ID" <<'PY'
import json,sys
path,arm,name,md5,sha1,nbytes,runtime,server_boot,run_id=sys.argv[1:]
with open(path,"w") as f:
    json.dump({"arm":arm,"model":name,"md5":md5,"resident_sha1":sha1,
               "bytes":int(nbytes),"runtime":json.loads(runtime),
               "server_boot":server_boot,"run_id":run_id},f,sort_keys=True)
    f.write("\n")
PY
# max_tokens must budget thinking + answer, not answer alone (the 2026-07-19
# thinking-budget artifact: at 1024 the think block exhausts the budget and
# no text block is emitted). 4096 covers the observed T2 thinking; override
# with Q27_EVAL_MAX_TOKENS. The scorer drops any row that still truncates.
MAX_TOKENS=${Q27_EVAL_MAX_TOKENS:-4096}
for mode in choice numeric freeform; do
    python3 tools/eval/gen_runner.py "tools/eval/prompts/$mode.jsonl" \
        --out "$TMP/$ARM.$mode.jsonl" --base-url "$URL" \
        --api anthropic --max-tokens "$MAX_TOKENS" --tag "$ARM-$RUN_ID-$mode"
done
FINAL_HEALTH=$(curl --noproxy '*' -fsS "$URL/health?identity=1") || { echo "run_arm: final server health failed" >&2; exit 2; }
FINAL_ROW=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("model","")+"\t"+d.get("artifact_sha1","")+"\t"+d.get("boot_id","")+"\t"+json.dumps(d.get("runtime"),sort_keys=True,separators=(",",":")))' "$FINAL_HEALTH")
[ "$FINAL_ROW" = "$HEALTH_ROW" ] || {
    echo "run_arm: server/model/runtime changed during run $RUN_ID; refusing publication" >&2
    exit 2
}
# Publish provenance LAST. If interrupted between renames, the old sidecar's
# run_id cannot validate any newly moved mode, so mixed-run output fails closed.
for mode in choice numeric freeform; do
    mv -f "$TMP/$ARM.$mode.jsonl" "$OUT/$ARM.$mode.jsonl"
done
mv -f "$TMP/$ARM.provenance.json" "$OUT/$ARM.provenance.json"
echo "run_arm: $ARM complete run=$RUN_ID ($MANIFEST_NAME md5=$MANIFEST_MD5 resident_sha1=$MANIFEST_SHA1) -> $OUT/$ARM.{choice,numeric,freeform}.jsonl"
