#!/usr/bin/env bash
# Hermetic contracts for tools/canonical_md5.sh: upstream's two-argument
# canonical_md5_for ARCH TIER keeps its Qwen3.6 answers; the optional third
# argument selects the Qwen3.8 checkpoint; artifact names infer the family;
# unpublished triples fail instead of borrowing another model's digest.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=canonical_md5.sh
source "$here/canonical_md5.sh"

fail=0
expect() { # description expected actual
  if [ "$2" != "$3" ]; then echo "FAIL: $1: want '$2', got '$3'"; fail=1; fi
}
expect_none() { # description command...
  local d=$1; shift
  if out=$("$@" 2>/dev/null); then echo "FAIL: $d: want no digest, got '$out'"; fail=1; fi
}

# Two-argument form: Qwen3.6, unchanged.
expect "metal q4s (2-arg)" f301095522174bdb99f75ec840ad1389 "$(canonical_md5_for metal-m4 q4s)"
expect "sm86 default alias (2-arg)" 6894254e3b1a184ee3802771ddd59c2b "$(canonical_md5_for 3090 vanilla)"
expect "sm120 q6f (2-arg)" 2a4d22eafcde63e962bf2408605fe502 "$(canonical_md5_for 5090 q6f-v1)"
expect "explicit qwen36 == default" "$(canonical_md5_for m4 q4s)" "$(canonical_md5_for m4 q4s qwen36)"

# Three-argument form: Qwen3.8.
expect "metal c-small" 909b162556fbc0ef776e40cc6c70677b "$(canonical_md5_for metal-m4 c-small qwen38-27b-mtp)"
expect "metal c-small via registry tier name" 909b162556fbc0ef776e40cc6c70677b "$(canonical_md5_for apple-m4 q38-c-small-v1 qwen38)"
expect "metal q38 q4s" b1a4a2802150081507a9b7cf8bad7a73 "$(canonical_md5_for m4 q4s Qwen3.8-27B-MTP)"
expect "sm120 q38 q6k" 067d81464ed4573b9e52841a503e111e "$(canonical_md5_for sm120 q6k qwen38)"

# No cross-model borrowing; unknown triples fail.
expect_none "qwen38 q4s must not fall back to qwen36" test "$(canonical_md5_for m4 q4s qwen38)" = f301095522174bdb99f75ec840ad1389
expect_none "c-small has no qwen36 canonical" canonical_md5_for metal-m4 c-small
expect_none "unknown model" canonical_md5_for metal-m4 q4s qwen99
expect_none "unknown arch" canonical_md5_for sm75 default

# Family from artifact paths.
expect "infer qwen38" qwen38-27b-mtp "$(canonical_model_for_path /x/models/qwen38-27b-mtp-c-small.q27)"
expect "infer qwen36" qwen36-27b-mtp "$(canonical_model_for_path qwen36-27b-mtp-q4s.q27)"
expect_none "bonsai has no canonical family" canonical_model_for_path models/bonsai2/bonsai2-27b-t2-slim.q27

# The Metal gate selects the Qwen3.8 digest from the artifact name, and names
# model+tier when a triple is unpublished: it stops at the lookup, before
# running build/q27-metal (built by the make target on macOS).
fake="$(mktemp -d "${TMPDIR:-/tmp}/q27-canon-test.XXXXXX")"
trap 'rm -rf "$fake"' EXIT

: > "$fake/qwen38-27b-mtp-c-small.q27"; : > "$fake/t.tok"
if [ -x "$here/../build/q27-metal" ]; then
  out=$(env -u CANON_MD5 -u CANON_MODEL CANON_ARCH=metal-m4 CANON_TIER=q5f \
        "$here/metal_canonical_gate.sh" "$fake/qwen38-27b-mtp-c-small.q27" "$fake/t.tok" 2>&1 || true)
  case "$out" in
    *"model=qwen38-27b-mtp tier=q5f"*) ;;
    *) echo "FAIL: gate did not name the inferred qwen38 model: $out"; fail=1 ;;
  esac
else
  echo "note: no build/q27-metal (non-macOS): Metal gate leg not exercised"
fi

[ "$fail" = 0 ] && echo "canonical md5 contracts: PASS" || { echo "canonical md5 contracts: FAIL"; exit 1; }
