#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${MODEL:-}}"
TOKENIZER="${2:-${TOKENIZER:-}}"
BIN="$(dirname "$0")/../build/q27-metal"
CANON_TIER="${CANON_TIER:-q4s}"
CANON_IDS="760,6511,314,9338,369"

if [[ -z "$MODEL" || -z "$TOKENIZER" ]]; then
  echo "usage: $0 model.q27 tokenizer.tok" >&2
  exit 2
fi
if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN; run make build/q27-metal" >&2
  exit 2
fi

if [[ -z "${CANON_ARCH:-}" ]]; then
  if [[ "$(uname -s)" != Darwin ]]; then
    echo "CANON_ARCH is required outside macOS" >&2
    exit 2
  fi
  chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  case "$chip" in
    "Apple M4"|"Apple M4 Pro") CANON_ARCH=metal-m4 ;;
    *)
      echo "no auto-selected Metal canonical for chip '$chip'; set CANON_ARCH after deriving one" >&2
      exit 2
      ;;
  esac
fi

# shellcheck source=canonical_md5.sh
source "$(dirname "$0")/canonical_md5.sh"
if [[ -z "${CANON_MODEL:-}" ]]; then
  if ! CANON_MODEL="$(canonical_model_for_path "$MODEL")"; then
    if [[ -n "${CANON_MD5:-}" ]]; then
      CANON_MODEL=custom
    else
      echo "cannot infer checkpoint from '$MODEL'; set CANON_MODEL explicitly" >&2
      exit 2
    fi
  fi
fi
if [[ -z "${CANON_MD5:-}" ]]; then
  if ! CANON_MD5="$(canonical_md5_for "$CANON_ARCH" "$CANON_MODEL" "$CANON_TIER")"; then
    echo "no published canonical for architecture=$CANON_ARCH model=$CANON_MODEL tier=$CANON_TIER" >&2
    echo "derive it with a same-device differential before publishing a digest" >&2
    exit 2
  fi
fi

hash_file() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | cut -d' ' -f1
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    echo "md5 or md5sum is required" >&2
    return 2
  fi
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "== gate 1: $CANON_ARCH/$CANON_MODEL/$CANON_TIER 128-token canonical"
"$BIN" "$MODEL" "$TOKENIZER" --tokens "$CANON_IDS" -n 128 --ctx 256 --mtp 4 \
  --dump-token-ids "$tmp/mtp.ids" >/dev/null
md5="$(hash_file "$tmp/mtp.ids")"
if [[ "$md5" != "$CANON_MD5" ]]; then
  echo "FAIL canonical changed: got $md5 want $CANON_MD5" >&2
  exit 1
fi
echo "  OK md5=$md5"

echo "== gate 2: speculative prefix equals plain greedy"
"$BIN" "$MODEL" "$TOKENIZER" --tokens "$CANON_IDS" -n 16 --ctx 64 \
  --dump-token-ids "$tmp/plain.ids" >/dev/null
read -r -a mtp_ids < "$tmp/mtp.ids"
: > "$tmp/mtp-prefix.ids"
for ((i=0; i<16; i++)); do
  if ((i > 0)); then printf ' ' >> "$tmp/mtp-prefix.ids"; fi
  printf '%s' "${mtp_ids[$i]}" >> "$tmp/mtp-prefix.ids"
done
printf '\n' >> "$tmp/mtp-prefix.ids"
if ! cmp -s "$tmp/mtp-prefix.ids" "$tmp/plain.ids"; then
  echo "FAIL MTP and plain greedy trajectories differ" >&2
  exit 1
fi
echo "  OK first 16 token ids are byte-identical"

echo "METAL CANONICAL GATES: ALL PASS"
