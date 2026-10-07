#!/usr/bin/env python3
"""Artifact-backed negative contracts for q38-c-small-v1 Metal routing.

Each fixture is an APFS clone of the real pack with one targeted edit, and must
be rejected by the ENGINE's policy validation (not by the loader): dtype
fixtures rewrite the tensor's payload sizes to match the new dtype, so they
pass Model::open() and reach validate_architecture()'s
"required tensor mismatch: NAME"; recipe/policy fixtures edit the metadata.
Ported from the metal-v0.7.0 branch and adapted to this engine's messages.
"""

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


def payload_sizes(dtype, shape):
    rows, cols = shape[0], shape[1] if len(shape) > 1 else 1
    if dtype == Q8_G128:
        return rows * cols, rows * (cols // 128) * 2
    if dtype == Q4_G64:
        return rows * cols // 2, rows * (cols // 64) * 2
    raise ValueError(f"unsupported dtype {dtype}")


def table(path):
    """name -> (dtype, dtype_offset, shape, data_offset, scales_offset); offsets
    of the payload fields are absolute file offsets once data_base is known."""
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
            shape = list(struct.unpack(f"<{rank}Q", stream.read(rank * 8)))
            doff, dsize, soff, ssize = struct.unpack("<QQQQ", stream.read(32))
            entries[name] = dict(dtype=dtype, dtype_offset=dtype_offset, shape=shape,
                                 doff=doff, soff=soff)
        header_end = stream.tell()
    data_base = (header_end + 255) // 256 * 256
    return metadata, meta_raw, entries, data_base


def clone(source, destination, check_space):
    # Fixtures must be clones (copy-on-write), never 15.7 GB copies: create
    # them beside the source (same volume) and verify the first one did not
    # consume real space.
    before = os.statvfs(os.path.dirname(destination))
    subprocess.run(["cp", "-c", source, destination], check=True)
    if check_space:
        after = os.statvfs(os.path.dirname(destination))
        used = (before.f_bavail - after.f_bavail) * before.f_frsize
        require(used < (1 << 30),
                f"cp -c did not clone (used {used >> 20} MiB); put the pack on an APFS volume")


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
    # realpath: fixtures are cloned beside the real file (a symlink's directory
    # may be on another volume, where cp -c would fully copy 15.7 GB).
    binary, source, tokenizer = map(os.path.realpath, sys.argv[1:])
    metadata, meta_raw, entries, data_base = table(source)
    file_size = os.path.getsize(source)
    require(metadata.get("quant_policy") == "q38-c-small-v1", str(metadata))
    require(metadata.get("q4_head") is True, "c-small artifact lacks q4_head")
    require(metadata.get("q8_extra") == r"^blk\.[0-9]+\.attn_output\.weight$",
            "c-small artifact has the wrong q8_extra recipe")

    # (tensor, packed dtype, mutated dtype, engine check that must reject it):
    # projections go through validate_architecture's matrix(), the head
    # through require().
    mutations = (
        ("blk.3.attn_output.weight", Q8_G128, Q4_G64, "required matrix mismatch: "),
        ("blk.0.ffn_gate.weight", Q4_G64, Q8_G128, "required matrix mismatch: "),
        ("blk.64.attn_q.weight", Q8_G128, Q4_G64, "required matrix mismatch: "),
        ("output.weight", Q4_G64, Q8_G128, "required tensor mismatch: "),
    )
    first = True
    with tempfile.TemporaryDirectory(prefix=".q27-q38-policy.", dir=os.path.dirname(source)) as work:
        for index, (name, old, new, check) in enumerate(mutations):
            require(name in entries, f"missing required tensor {name}")
            e = entries[name]
            require(e["dtype"] == old, f"unexpected source dtype for {name}: {e['dtype']}, want {old}")
            dsize, ssize = payload_sizes(new, e["shape"])
            candidate = os.path.join(work, f"dtype-{index}.q27")
            clone(source, candidate, first)
            first = False
            # A grown payload (Q4 -> Q8) may run past EOF if the tensor is last
            # in the file: extend the clone sparsely so the fixture still passes
            # the loader's range checks and reaches the policy validator.
            need = max(data_base + e["doff"] + dsize, data_base + e["soff"] + ssize)
            if need > file_size:
                os.truncate(candidate, need)
            with open(candidate, "r+b", buffering=0) as stream:
                stream.seek(e["dtype_offset"])
                stream.write(bytes([new]))
                rank = len(e["shape"])
                stream.seek(e["dtype_offset"] + 2 + rank * 8 + 8)
                stream.write(struct.pack("<Q", dsize))
                stream.seek(e["dtype_offset"] + 2 + rank * 8 + 24)
                stream.write(struct.pack("<Q", ssize))
            reject(binary, candidate, tokenizer, check + name)
            os.remove(candidate)

        # Recipe metadata is part of the contract even if all table dtypes are
        # otherwise valid.
        needle = b"attn_output"
        require(meta_raw.count(needle) == 1, "q8 recipe marker is not unique")
        candidate = os.path.join(work, "metadata.q27")
        clone(source, candidate, first)
        first = False
        with open(candidate, "r+b", buffering=0) as stream:
            stream.seek(16)
            stream.write(meta_raw.replace(needle, b"attn_outpuX"))
        reject(binary, candidate, tokenizer, "q8_extra does not match quantization policy")
        os.remove(candidate)

        # A misspelled policy must not fall through to the permissive legacy
        # Q4/Q8 route used by older Qwen3.6 artifacts.
        needle = b"q38-c-small-v1"
        require(meta_raw.count(needle) == 1, "quant policy marker is not unique")
        candidate = os.path.join(work, "policy.q27")
        clone(source, candidate, first)
        with open(candidate, "r+b", buffering=0) as stream:
            stream.seek(16)
            stream.write(meta_raw.replace(needle, b"q38-c-smalX-v1"))
        reject(binary, candidate, tokenizer, "unsupported quantization policy")

    print("Qwen3.8 c-small Metal policy negative contracts: PASS (6 fixtures)")


if __name__ == "__main__":
    main()
