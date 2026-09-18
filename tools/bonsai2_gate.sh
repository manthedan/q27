#!/usr/bin/env bash
# Model-backed parity gate: reference and q27 never reside concurrently.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PRISM_DIR:?set PRISM_DIR to the built pinned Prism llama.cpp checkout}"
[[ "$(git -C "$PRISM_DIR" rev-parse HEAD)" == 1a07bfa5f4144274c8f1c9963821dd9d9a51854b ]]
MODELS="${Q27_BONSAI2_DIR:-$ROOT/models/bonsai2}"
OUT="${1:-$ROOT/logs/bonsai2-gate-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
make -C "$ROOT" -j2 build/bonsai2-metal-probe build/tokenize_to_bin
make -C "$ROOT" -B build/bonsai2-prism-probe PRISM_DIR="$PRISM_DIR"
python3 - "$OUT" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1])
texts={
 'prose': 'The capital of France is Paris. A train leaves Paris at noon and travels east. Explain how to calculate its average speed when the distance and elapsed time are known.\n' * 3,
 'code': 'def total(values):\n    """Return the sum, including zero for an empty list."""\n    answer = 0\n    for value in values:\n        answer += value\n    return answer\n\nassert total([]) == 0\nassert total([1, -2, 3]) == 2\n' * 2,
 'tools_unicode': '<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nRead café.txt and summarize it. 日本語: こんにちは。 Ελληνικά: γειά. Emoji: 🌲.\n<|im_end|>\n<|im_start|>assistant\n<think>\nI should inspect the file first.\n</think>\n<tool_call>\n<function=read>\n<parameter=path>\ncafé.txt\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n<|im_start|>user\n<tool_response>\nThe experiment measured 27 tokens per second.\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n'
}
for name,text in texts.items():(out/(name+'.txt')).write_text(text)
PY
export Q27_METAL_SOURCE="$ROOT/src/metal/q27_kernels.metal"
for case in prose code tools_unicode; do
  "$ROOT/build/tokenize_to_bin" "$MODELS/bonsai2.tok" "$OUT/$case.txt" "$OUT/$case.u32"
  "$ROOT/build/bonsai2-prism-probe" "$MODELS/Ternary-Bonsai-2-27B-PQ2_0.gguf" \
    "$OUT/$case.u32" "$OUT/$case.reference.f32" > "$OUT/$case.reference.log" 2>&1
  "$ROOT/build/bonsai2-metal-probe" "$MODELS/bonsai2-t2.q27" \
    "$OUT/$case.u32" "$OUT/$case.metal.f32" > "$OUT/$case.metal.log" 2>&1
  python3 "$ROOT/tools/bonsai2_compare.py" "$OUT/$case.reference.f32" "$OUT/$case.metal.f32" \
    --output "$OUT/$case.json"
done
printf 'Bonsai 2 parity PASS; evidence: %s\n' "$OUT"
