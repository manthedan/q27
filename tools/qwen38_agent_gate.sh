#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
model="${Q27_GATE_MODEL:-}"
tokenizer="${Q27_GATE_TOK:-}"
pack="${Q27_GATE_PACK:-q38}"
[[ "$model" = /* && -f "$model" ]] || {
  echo "Q27_GATE_MODEL must be an absolute Qwen3.8 .q27 path" >&2; exit 2; }
[[ "$tokenizer" = /* && -f "$tokenizer" ]] || {
  echo "Q27_GATE_TOK must be an absolute Qwen3.8 .tok path" >&2; exit 2; }

# shellcheck source=../packaging/lib/q27_bench_lib.sh
. "$root/packaging/lib/q27_bench_lib.sh"
artifact_file="$(q27_registry_field "$pack" 4)" || {
  echo "Q27_GATE_PACK '$pack' is not in packaging/models.tsv" >&2; exit 2; }
tokenizer_file="$(q27_registry_field "$pack" 16)"
profile="$(q27_registry_field "$pack" 13)"
[[ "$profile" = qwen38-thinking-v1 ]] || {
  echo "Q27_GATE_PACK '$pack' does not use qwen38-thinking-v1" >&2; exit 2; }
[[ -n "$tokenizer_file" ]] || tokenizer_file="${tokenizer##*/}"

# Stage beside the pack's real file (same volume, symlinks resolved): a hard
# link or clone, never a copy of the 15.7 GB pack onto another volume.
model="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$model")"
home="$(mktemp -d "$(dirname "$model")/.q27-qwen38-agent-home.XXXXXX")"
cleanup() { rm -rf "$home"; }
trap cleanup EXIT
mkdir -p "$home/$pack"
# The pack: link or clone only. The tokenizer (7 MB) may live on another
# volume: link, clone, or copy.
ln "$model" "$home/$pack/$artifact_file" 2>/dev/null || cp -c "$model" "$home/$pack/$artifact_file"
ln "$tokenizer" "$home/$pack/$tokenizer_file" 2>/dev/null || cp -c "$tokenizer" "$home/$pack/$tokenizer_file" 2>/dev/null ||
  cp "$tokenizer" "$home/$pack/$tokenizer_file"

Q27_HOME="$home" "$root/tools/packaged_agent_driver.sh" \
  "$root/packaging/bin/q27" "$pack"
echo "Qwen3.8 native-agent wrapper gate: PASS ($pack)"
