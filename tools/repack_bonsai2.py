#!/usr/bin/env python3
"""Lossless, bounded-memory Bonsai 2 PQ2_0 -> q27 conversion.

python3 tools/repack_bonsai2.py SOURCE.gguf OUTPUT.q27
python3 tools/export_tokenizer.py SOURCE.gguf OUTPUT.tok

The policy is deliberately narrow: the released rotated Qwen3.8 text stack,
not arbitrary private GGUF codecs. BF16 GDN gates widen exactly to F32.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np
from prism_gguf import GGUFReader
from repack import (ALIGN, MAGIC, VERSION, DTYPE_F32, DTYPE_T2,
                    QWEN35_BASE_SPECS, _require_exact_specs,
                    _require_qwen38_metadata, _normalized_metadata,
                    repack_t2, to_f32)

POLICY = "bonsai2-t2-hadamard-v1"
PROFILE = "bonsai2-qwen38-v1"
PREFIX = "prism.hadamard."
ROTATION_FIELDS = {PREFIX + key for key in (
    "version", "block_size", "transform", "axis", "sign_mode", "sign_widths",
    "sign_values", "weight_names", "inverse_weight_names", "gdn_v_grouped")}
SPECS = {name: ("PQ2_0" if dtype == "BF16" and
               not name.endswith((".ssm_alpha.weight", ".ssm_beta.weight"))
               else dtype, shape)
         for name, (dtype, shape) in QWEN35_BASE_SPECS.items()}


def validate_rotation(meta, rotated_names):
    """Pin the released rotation ABI, including all and only affected tensors."""
    if {k for k in meta if k.startswith(PREFIX)} != ROTATION_FIELDS:
        raise ValueError("incomplete or unknown prism.hadamard metadata")
    for key, value in (("version", 1), ("block_size", 1024),
                       ("transform", "normalized-sylvester-walsh-hadamard"),
                       ("axis", "input-last-dimension"), ("sign_mode", "explicit"),
                       ("gdn_v_grouped", True)):
        actual = meta[PREFIX + key]
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"unsupported {PREFIX + key}: {actual!r}")
    names = meta[PREFIX + "weight_names"]
    if (not isinstance(names, list) or any(type(n) is not str for n in names)
            or len(names) != len(set(names)) or set(names) != set(rotated_names)):
        raise ValueError("Hadamard weight_names must cover every PQ2_0 projection exactly once")
    if meta[PREFIX + "inverse_weight_names"] != ["token_embd.weight"]:
        raise ValueError("Hadamard inverse must be token_embd.weight only")
    widths = meta[PREFIX + "sign_widths"]
    if (not isinstance(widths, list) or any(type(w) is not int for w in widths)
            or sorted(widths) != [5120, 6144, 17408]):
        raise ValueError("Hadamard sign_widths must be 5120/6144/17408, without duplicates")
    signs = meta[PREFIX + "sign_values"]
    if (not isinstance(signs, list) or len(signs) != sum(widths)
            or any(type(v) is not int or v not in (-1, 1) for v in signs)):
        raise ValueError("Hadamard sign_values must contain one +/-1 vector per width")


def source_metadata(reader):
    meta = {k: _normalized_metadata(v.contents()) for k, v in reader.fields.items()
            if k.startswith(("qwen35.", "general.", PREFIX))}
    if meta.get("general.architecture") != "qwen35" or type(meta.get("qwen35.block_count")) is not int or meta["qwen35.block_count"] != 64:
        raise ValueError("Bonsai 2 requires the 64-block qwen35 text architecture")
    if meta.get("qwen35.nextn_predict_layers", 0) != 0:
        raise ValueError("Bonsai 2 has no MTP layer")
    _require_qwen38_metadata(reader, "Bonsai 2")
    tensors = {t.name: t for t in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise ValueError("duplicate source tensor names")
    _require_exact_specs("Bonsai 2", tensors, SPECS)
    rotated = {n for n, (dtype, _) in SPECS.items() if dtype == "PQ2_0" and n != "token_embd.weight"}
    validate_rotation(meta, rotated)
    meta.update(q27_version=VERSION, quant_policy=POLICY, group_q4=64, group_q8=128,
                nibble_order="even=low", group_t2=128,
                t2_codes="0=-1,1=0,2=+1;3 forbidden", t2_slot_order="seq-lsb-first")
    meta["q27.model_profile"] = PROFILE
    meta["attn_layers"] = list(range(3, 64, 4))
    meta["ssm_layers"] = [i for i in range(64) if i % 4 != 3]
    template = reader.fields.get("tokenizer.chat_template")
    if template is None or not isinstance(template.contents(), str):
        raise ValueError("Bonsai 2 source must carry its chat template")
    meta["source.chat_template_sha256"] = hashlib.sha256(template.contents().encode()).hexdigest()
    return meta


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def convert(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    reader = GGUFReader(source)
    meta = source_metadata(reader)
    meta["source.gguf_sha256"] = sha256(source)
    entries = []
    # Spool one tensor at a time. Do not retain 7 GB of output blobs alongside
    # the source mmap on the very small Macs this converter exists to serve.
    with tempfile.TemporaryFile(dir=destination.parent) as data:
        for t in reader.tensors:
            shape = tuple(reversed([int(d) for d in t.shape]))
            if t.tensor_type.name == "PQ2_0":
                values, scales, _ = repack_t2(t)
                if not np.isfinite(np.frombuffer(scales, dtype="<f2")).all():
                    raise ValueError(f"nonfinite source scales: {t.name}")
                dtype = DTYPE_T2
            else:
                floats = to_f32(t)
                if not np.isfinite(floats).all():
                    raise ValueError(f"nonfinite source tensor: {t.name}")
                values, scales, dtype = floats.astype("<f4").tobytes(), b"", DTYPE_F32
                del floats
            doff = data.tell()
            data.write(values)
            data.seek((data.tell() + ALIGN - 1) // ALIGN * ALIGN)
            soff = data.tell() if scales else 0
            data.write(scales)
            data.seek((data.tell() + ALIGN - 1) // ALIGN * ALIGN)
            entries.append((t.name, dtype, shape, doff, len(values), soff, len(scales)))
            del values, scales
        # Atomic publication: validation/conversion failures never replace an
        # existing artifact or leave a plausible partial output at its path.
        fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as out:
                metadata = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
                out.write(struct.pack("<IIII", MAGIC, VERSION, len(entries), len(metadata)))
                out.write(metadata)
                for name, dtype, shape, doff, dsize, soff, ssize in entries:
                    encoded = name.encode()
                    out.write(struct.pack("<H", len(encoded)) + encoded)
                    out.write(struct.pack("<BB", dtype, len(shape)))
                    out.write(struct.pack("<" + "Q" * len(shape), *shape))
                    out.write(struct.pack("<QQQQ", doff, dsize, soff, ssize))
                out.write(b"\0" * (-out.tell() % ALIGN))
                data.seek(0)
                shutil.copyfileobj(data, out, length=8 << 20)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(f"Bonsai 2: {len(entries)} tensors, exact PQ2_0 + BF16->F32, {destination.stat().st_size} bytes")
    print(f"sha256 {sha256(destination)}  {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    convert(args.source, args.destination)
