#!/usr/bin/env python3
"""Artifact-backed negative contracts for q38-c-small-v1 Metal routing."""

import json
import os
import struct
import subprocess
import sys
import tempfile

MAGIC = 0x46373251
Q4_G64 = 3
Q8_G128 = 2


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def table(path):
    entries = {}
    with open(path, "rb") as stream:
        magic, version, count, meta_size = struct.unpack("<IIII", stream.read(16))
        require(magic == MAGIC and version == 1, "not a supported q27 artifact")
        meta_raw = stream.read(meta_size)
        metadata = json.loads(meta_raw)
        for _ in range(count):
            name_size, = struct.unpack("<H", stream.read(2))
            name = stream.read(name_size).decode()
            dtype_offset = stream.tell()
            dtype, rank = struct.unpack("<BB", stream.read(2))
            stream.seek(rank * 8 + 32, os.SEEK_CUR)
            entries[name] = (dtype, dtype_offset)
    return metadata, meta_raw, entries


def clone(source, destination):
    # APFS clone: these negative fixtures must not duplicate a 15.7 GB model.
    subprocess.run(["cp", "-c", source, destination], check=True)


def reject(binary, model, tokenizer, expected):
    result = subprocess.run(
        [binary, model, tokenizer, "--validate-only"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    output = result.stdout + result.stderr
    require(result.returncode != 0, f"mutated artifact was accepted: {model}")
    require(expected in output, f"missing rejection {expected!r}: {output}")


def main():
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} Q27_METAL MODEL TOKENIZER")
    binary, source, tokenizer = map(os.path.abspath, sys.argv[1:])
    metadata, meta_raw, entries = table(source)
    require(metadata.get("quant_policy") == "q38-c-small-v1", str(metadata))
    require(metadata.get("q4_head") is True, "c-small artifact lacks q4_head")
    require(metadata.get("q8_extra") == r"^blk\.[0-9]+\.attn_output\.weight$",
            "c-small artifact has the wrong q8_extra recipe")

    mutations = (
        ("blk.3.attn_output.weight", Q8_G128, Q4_G64),
        ("blk.0.ffn_gate.weight", Q4_G64, Q8_G128),
        ("blk.64.attn_q.weight", Q8_G128, Q4_G64),
        ("output.weight", Q4_G64, Q8_G128),
    )
    with tempfile.TemporaryDirectory(prefix="q27-q38-policy.") as work:
        for index, (name, old, new) in enumerate(mutations):
            require(name in entries, f"missing required tensor {name}")
            actual, offset = entries[name]
            require(actual == old,
                    f"unexpected source dtype for {name}: {actual}, want {old}")
            candidate = os.path.join(work, f"dtype-{index}.q27")
            clone(source, candidate)
            with open(candidate, "r+b", buffering=0) as stream:
                stream.seek(offset)
                stream.write(bytes([new]))
            reject(binary, candidate, tokenizer, name)

        # Recipe metadata is part of the contract even if all table dtypes are
        # otherwise valid.
        needle = b"attn_output"
        require(meta_raw.count(needle) == 1, "q8 recipe marker is not unique")
        candidate = os.path.join(work, "metadata.q27")
        clone(source, candidate)
        header = meta_raw.replace(needle, b"attn_outpuX")
        with open(candidate, "r+b", buffering=0) as stream:
            stream.seek(16)
            stream.write(header)
        reject(binary, candidate, tokenizer, "recipe metadata mismatch")

        # A misspelled policy must not fall through to the permissive legacy
        # Q4/Q8 route used by older Qwen3.6 artifacts.
        needle = b"q38-c-small-v1"
        require(meta_raw.count(needle) == 1, "quant policy marker is not unique")
        candidate = os.path.join(work, "policy.q27")
        clone(source, candidate)
        header = meta_raw.replace(needle, b"q38-c-smalX-v1")
        with open(candidate, "r+b", buffering=0) as stream:
            stream.seek(16)
            stream.write(header)
        reject(binary, candidate, tokenizer, "unsupported Qwen3.8 quant policy")

    print("Qwen3.8 c-small Metal policy negative contracts: PASS")


if __name__ == "__main__":
    main()
