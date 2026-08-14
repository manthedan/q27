#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=canonical_md5.sh
source "$(dirname "$0")/canonical_md5.sh"

expect_digest() {
  local arch="$1" model="$2" tier="$3" want="$4" got
  got="$(canonical_md5_for "$arch" "$model" "$tier")"
  if [[ "$got" != "$want" ]]; then
    echo "canonical mismatch for $arch/$model/$tier: got $got want $want" >&2
    exit 1
  fi
}

expect_missing() {
  if canonical_md5_for "$1" "$2" "$3" >/dev/null; then
    echo "unexpected canonical for $1/$2/$3" >&2
    exit 1
  fi
}

expect_model() {
  local path="$1" want="$2" got
  got="$(canonical_model_for_path "$path")"
  if [[ "$got" != "$want" ]]; then
    echo "model inference mismatch for $path: got $got want $want" >&2
    exit 1
  fi
}

expect_digest 5090 qwen36 q4s f64e7c02252ca4c40cea62db662205e0
expect_digest sm_120 Qwen3.8-27B-MTP q5f-v1 10e654eb9c9f2aeb47ea8e003aa77030
expect_digest apple-m4 qwen36-27b-mtp q4s-v1 f301095522174bdb99f75ec840ad1389
expect_digest metal-m4 qwen38-27b-mtp q4s b1a4a2802150081507a9b7cf8bad7a73
expect_missing sm86 qwen38-27b-mtp default
expect_model /models/qwen36-27b-mtp-q4s.q27 qwen36-27b-mtp
expect_model /models/qwen38-27b-mtp.q27 qwen38-27b-mtp
if canonical_model_for_path /models/custom-finetune.q27 >/dev/null; then
  echo "unknown checkpoint filename was inferred" >&2
  exit 1
fi

echo "checkpoint-aware canonical registry: PASS"
