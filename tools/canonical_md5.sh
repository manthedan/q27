#!/usr/bin/env bash

# Return the published canonical digest for one exact architecture/tier pair,
# optionally per checkpoint: canonical_md5_for ARCH TIER [MODEL]. MODEL
# defaults to qwen36-27b-mtp, so upstream's two-argument callers
# (sampling_gate.sh, anchor_check.sh, metal_canonical_gate.sh) are unchanged.
# CUDA entries hash the `generated:` line used by sampling_gate.sh. Metal
# entries hash the space-delimited token-id line emitted by --dump-token-ids.

# Checkpoint family from an artifact path (qwen36-27b-mtp / qwen38-27b-mtp).
canonical_model_for_path() {
  local name="${1##*/}"
  case "$name" in
    qwen36-27b-mtp*|Qwen3.6-27B-MTP*) printf '%s\n' qwen36-27b-mtp ;;
    qwen38-27b-mtp*|Qwen3.8-27B-MTP*) printf '%s\n' qwen38-27b-mtp ;;
    *) return 1 ;;
  esac
}

canonical_md5_for() {
  local arch="$1" tier="$2" model="${3:-qwen36-27b-mtp}"

  case "$arch" in
    sm120|sm_120|blackwell|5090) arch=sm120 ;;
    sm86|sm_86|ampere|3090|a40) arch=sm86 ;;
    metal-m4|apple-m4|m4) arch=metal-m4 ;;
  esac
  case "$tier" in
    default|vanilla|qwen36-27b-mtp) tier=default ;;
    q4s|q4s-v1) tier=q4s ;;
    q5f|q5f-v1) tier=q5f ;;
    q6|q6-v1) tier=q6 ;;
    q6f|q6f-v1) tier=q6f ;;
    q6k|q6k-v1) tier=q6k ;;
    c-small|q38-c-small-v1) tier=c-small ;;
  esac
  case "$model" in
    qwen36|qwen3.6|qwen36-27b-mtp|Qwen3.6-27B-MTP) model=qwen36-27b-mtp ;;
    qwen38|qwen3.8|qwen38-27b-mtp|Qwen3.8-27B-MTP) model=qwen38-27b-mtp ;;
  esac

  # sm86:default was derived on an RTX 3090 (82 SMs) and independently matched
  # the A40 result (84 SMs) byte-for-byte, establishing SM-count independence
  # within sm_86. A non-Blackwell `6894254e...` result therefore reproduces its
  # own architecture's canonical rather than failing to reproduce Blackwell.
  # Unlisted pairs require a same-device upstream/candidate differential.
  # metal-m4:q4s was re-derived after the serving stack promoted the measured
  # reciprocal-multiply activation quantization and revised shader arithmetic.
  # The new trajectory was reproduced byte-for-byte on a base Apple M4 and an
  # independent 24 GB Apple M4 Pro using the same q4s artifact (artifact MD5
  # 7e5454e0c0ded717136ad3e42634ba25), preserving the shared family canonical.

  # Qwen3.8 (from the metal-v0.7.0 branch): metal-m4 q4s was reproduced twice
  # on a 24 GB base Apple M4 (split-source artifact MD5
  # bd2eca11d7aefdec00cb58b0d8e8eb8e); the same differential produced c-small
  # (artifact MD5 e43dc81606d28825bfe129b56401220d) canonical 909b1625... twice.
  # Those were derived on the v0.7.0 branch engine; the revival engine must
  # reproduce them (tools/qwen38_laptop_gate.sh) before a release relies on it.
  case "$arch:$model:$tier" in
    sm120:qwen36-27b-mtp:default) printf '%s\n' a2982c5197c627551b27d76a0a94b220 ;;
    sm120:qwen36-27b-mtp:q4s)     printf '%s\n' f64e7c02252ca4c40cea62db662205e0 ;;
    sm120:qwen36-27b-mtp:q5f)     printf '%s\n' 683f7f4450ca4c60837abdb603ee3237 ;;
    sm120:qwen36-27b-mtp:q6f)     printf '%s\n' 2a4d22eafcde63e962bf2408605fe502 ;;
    sm86:qwen36-27b-mtp:default)  printf '%s\n' 6894254e3b1a184ee3802771ddd59c2b ;;
    metal-m4:qwen36-27b-mtp:q4s)  printf '%s\n' f301095522174bdb99f75ec840ad1389 ;;

    sm120:qwen38-27b-mtp:q4s)     printf '%s\n' a710b3a4de0f13da0b008d948c425735 ;;
    sm120:qwen38-27b-mtp:default) printf '%s\n' 98e7da0df4ae81511566ad1ce31b719a ;;
    sm120:qwen38-27b-mtp:q5f)     printf '%s\n' 10e654eb9c9f2aeb47ea8e003aa77030 ;;
    sm120:qwen38-27b-mtp:q6)      printf '%s\n' 067d81464ed4573b9e52841a503e111e ;;
    sm120:qwen38-27b-mtp:q6f)     printf '%s\n' d0c05c0c723df6208ae1e080be9c6fe5 ;;
    sm120:qwen38-27b-mtp:q6k)     printf '%s\n' 067d81464ed4573b9e52841a503e111e ;;
    metal-m4:qwen38-27b-mtp:q4s)  printf '%s\n' b1a4a2802150081507a9b7cf8bad7a73 ;;
    metal-m4:qwen38-27b-mtp:c-small) printf '%s\n' 909b162556fbc0ef776e40cc6c70677b ;;
    *) return 1 ;;
  esac
}

# Published sampled-seed anchors. The EXACT command is part of the anchor;
# every flag below is load-bearing. In particular `--spec` selects a
# different-but-valid sampled trajectory: omitting it produced a plausible
# WRONG md5 at v0.5.0 gating and again on 2026-08-18 (recorded as "anchor
# broken"; it never was -- see BUILDLOG 2026-08-18 (k)).
#   build/q27 <model> --tokens "760,6511,314,9338,369" --ctx 2048 --spec \
#     -n 64 --temp 0.7 --top-p 0.95 --seed 42 | grep '^generated:' | md5sum
sampled_md5_for() {
  local arch="$1" tier="$2"

  case "$arch" in
    sm120|sm_120|blackwell|5090) arch=sm120 ;;
    sm86|sm_86|ampere|3090|a40) arch=sm86 ;;
    metal-m4|apple-m4|m4) arch=metal-m4 ;;
  esac
  case "$tier" in
    default|vanilla|qwen36-27b-mtp) tier=default ;;
    q4s|q4s-v1) tier=q4s ;;
    q5f|q5f-v1) tier=q5f ;;
    q6f|q6f-v1) tier=q6f ;;
  esac

  case "$arch:$tier" in
    sm120:q4s) printf '%s\n' 900031e9b86df8f52493e6c1f4040c2e ;;
    *) return 1 ;;
  esac
}
