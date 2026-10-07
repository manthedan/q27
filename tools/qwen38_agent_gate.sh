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

home="$(mktemp -d "${TMPDIR:-/tmp}/q27-qwen38-agent-home.XXXXXX")"
cleanup() { rm -rf "$home"; }
trap cleanup EXIT
mkdir -p "$home/$pack"
stage_file() {
  local src="$1" dst="$2"
  ln "$src" "$dst" 2>/dev/null || cp "$src" "$dst"
}
stage_file "$model" "$home/$pack/$artifact_file"
stage_file "$tokenizer" "$home/$pack/$tokenizer_file"

Q27_HOME="$home" "$root/tools/packaged_agent_driver.sh" \
  "$root/packaging/bin/q27" "$pack"
echo "Qwen3.8 native-agent wrapper gate: PASS ($pack)"
