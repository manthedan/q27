#!/usr/bin/env bash

# Return the published canonical digest for one exact
# architecture/checkpoint/tier triple. CUDA entries hash the `generated:` line
# used by sampling_gate.sh. Metal entries hash the space-delimited token-id line
# emitted by --dump-token-ids.
canonical_model_for_path() {
  local name="${1##*/}"
  case "$name" in
    qwen36-27b-mtp*|Qwen3.6-27B-MTP*) printf '%s\n' qwen36-27b-mtp ;;
    qwen38-27b-mtp*|Qwen3.8-27B-MTP*) printf '%s\n' qwen38-27b-mtp ;;
    *) return 1 ;;
  esac
}

canonical_md5_for() {
  local arch="$1" model="$2" tier="$3"

  case "$arch" in
    sm120|sm_120|blackwell|5090) arch=sm120 ;;
    sm86|sm_86|ampere|3090|a40) arch=sm86 ;;
    metal-m4|apple-m4|m4) arch=metal-m4 ;;
  esac
  case "$model" in
    qwen36|qwen3.6|qwen36-27b-mtp|Qwen3.6-27B-MTP) model=qwen36-27b-mtp ;;
    qwen38|qwen3.8|qwen38-27b-mtp|Qwen3.8-27B-MTP) model=qwen38-27b-mtp ;;
  esac
  case "$tier" in
    default|vanilla) tier=default ;;
    q4s|q4s-v1) tier=q4s ;;
    q5f|q5f-v1) tier=q5f ;;
    q6|q6-v1) tier=q6 ;;
    q6f|q6f-v1) tier=q6f ;;
    q6k|q6k-v1) tier=q6k ;;
  esac

  # sm86:qwen36-27b-mtp:default was derived on an RTX 3090 (82 SMs) and
  # independently matched the A40 result (84 SMs), establishing SM-count
  # independence within sm_86. Unlisted triples require a same-device
  # upstream/candidate differential.
  #
  # metal-m4:qwen36-27b-mtp:q4s was reproduced byte-for-byte on a base Apple
  # M4 and an independent 24 GB Apple M4 Pro using artifact MD5
  # 7e5454e0c0ded717136ad3e42634ba25.
  # metal-m4:qwen38-27b-mtp:q4s was reproduced twice on a 24 GB base Apple M4
  # using split-source artifact MD5 bd2eca11d7aefdec00cb58b0d8e8eb8e.

  case "$arch:$model:$tier" in
    sm120:qwen36-27b-mtp:default) printf '%s\n' a2982c5197c627551b27d76a0a94b220 ;;
    sm120:qwen36-27b-mtp:q4s)    printf '%s\n' f64e7c02252ca4c40cea62db662205e0 ;;
    sm120:qwen36-27b-mtp:q5f)    printf '%s\n' 683f7f4450ca4c60837abdb603ee3237 ;;
    sm120:qwen36-27b-mtp:q6f)    printf '%s\n' 2a4d22eafcde63e962bf2408605fe502 ;;
    sm86:qwen36-27b-mtp:default)  printf '%s\n' 6894254e3b1a184ee3802771ddd59c2b ;;
    metal-m4:qwen36-27b-mtp:q4s) printf '%s\n' f301095522174bdb99f75ec840ad1389 ;;

    sm120:qwen38-27b-mtp:q4s)    printf '%s\n' a710b3a4de0f13da0b008d948c425735 ;;
    sm120:qwen38-27b-mtp:default) printf '%s\n' 98e7da0df4ae81511566ad1ce31b719a ;;
    sm120:qwen38-27b-mtp:q5f)    printf '%s\n' 10e654eb9c9f2aeb47ea8e003aa77030 ;;
    sm120:qwen38-27b-mtp:q6)     printf '%s\n' 067d81464ed4573b9e52841a503e111e ;;
    sm120:qwen38-27b-mtp:q6f)    printf '%s\n' d0c05c0c723df6208ae1e080be9c6fe5 ;;
    sm120:qwen38-27b-mtp:q6k)    printf '%s\n' 067d81464ed4573b9e52841a503e111e ;;
    metal-m4:qwen38-27b-mtp:q4s) printf '%s\n' b1a4a2802150081507a9b7cf8bad7a73 ;;
    *) return 1 ;;
  esac
}
