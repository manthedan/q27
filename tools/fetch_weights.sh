#!/bin/zsh
# Fetch the Ternary-Bonsai-27B weights from PrismML's Hugging Face repo
# (Apache-2.0, verified 2026-07-16) and repack them locally into the q27
# T2_G128 artifact. The repack is a lossless, deterministic container
# transform, so q27 never redistributes weights — this script IS the
# distribution story (packaging/README.md §Weights).
#
# Every step is pinned and verified:
#   revision pinned to a commit sha (upstream re-uploads cannot drift in),
#   source GGUF sha256 checked against HF's LFS oid,
#   repacked artifact sha256 checked against the published q27 digest.
set -u
cd "$(dirname "$0")/.."

REPO="prism-ml/Ternary-Bonsai-27B-gguf"
REV="20e435f518bd5b882795954aba81e80a91894321"
FILE="Ternary-Bonsai-27B-Q2_0.gguf"
SRC_SHA="868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757"
OUT_SHA="25392b471d2e5c55798c2ea77d8b53fbbd1f720318e5ff23bd8c49e5d378e549"
DEST="models/ternary-bonsai-27b"
ARTIFACT="$DEST/ternary-bonsai-27b-t2.q27"
TOK="$DEST/ternary-bonsai-27b.tok"

python3 -c "import numpy, gguf" 2>/dev/null ||
    { echo "fetch: the repack needs python3 with numpy and gguf:"; \
      echo "  python3 -m pip install numpy gguf"; exit 1; }

mkdir -p "$DEST"

# The fast path needs BOTH files healthy — a verified artifact with a
# missing/empty tokenizer (e.g. a prior run that died during export)
# must fall through and repair (codex P2).
if [ -f "$ARTIFACT" ] && [ -s "$TOK" ] &&
   [ "$(shasum -a 256 "$ARTIFACT" | cut -d' ' -f1)" = "$OUT_SHA" ]; then
    echo "fetch: $ARTIFACT already present and verified ($OUT_SHA)"
    exit 0
fi

GGUF="$DEST/$FILE"
if [ ! -f "$GGUF" ] || [ "$(shasum -a 256 "$GGUF" | cut -d' ' -f1)" != "$SRC_SHA" ]; then
    # Two attempts: resume first (cheap for interrupted downloads), then a
    # clean restart — resuming onto corrupt bytes can never converge and
    # previously wedged until a manual delete (codex P2).
    for attempt in resume clean; do
        [ "$attempt" = "clean" ] && { echo "fetch: retrying with a clean download"; rm -f "$GGUF"; }
        echo "fetch: downloading $FILE (7.2 GB) from $REPO @ ${REV:0:12} ($attempt)"
        # A transport failure preserves the partial file for the next run's
        # resume; only a COMPLETED download with a wrong checksum triggers
        # the clean restart (codex P2 — transient errors must not discard
        # gigabytes of good partial data).
        curl -L --fail --continue-at - \
            "https://huggingface.co/$REPO/resolve/$REV/$FILE" -o "$GGUF" ||
            { echo "fetch: download interrupted — re-run to resume from the partial file"; exit 1; }
        [ "$(shasum -a 256 "$GGUF" | cut -d' ' -f1)" = "$SRC_SHA" ] && break
        [ "$attempt" = "clean" ] &&
            { echo "fetch: source GGUF sha256 MISMATCH after clean download — aborting"; exit 1; }
    done
fi
echo "fetch: source verified ($SRC_SHA)"

python3 tools/repack.py "$GGUF" "$ARTIFACT" ||
    { echo "fetch: repack failed"; exit 1; }
[ "$(shasum -a 256 "$ARTIFACT" | cut -d' ' -f1)" = "$OUT_SHA" ] ||
    { echo "fetch: repacked artifact sha256 MISMATCH — the repack is deterministic," \
           "so this means a code or source change; do not use the artifact"; exit 1; }
echo "fetch: artifact verified ($OUT_SHA)"

python3 tools/export_tokenizer.py "$GGUF" "$TOK" ||
    { echo "fetch: tokenizer export failed"; exit 1; }
echo "fetch: tokenizer exported to $TOK (sha256 $(shasum -a 256 "$TOK" | cut -d' ' -f1))"
echo "fetch: done. The source GGUF (7.2 GB) is kept at $GGUF; delete it to save disk."
echo "  q27-metal $ARTIFACT $TOK --prompt 'hello' -n 32"
