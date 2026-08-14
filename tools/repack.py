#!/usr/bin/env python3
"""Repack BF16 GGUF weights into the q27 v1 format (see docs/FORMAT.md).

Usage:
  repack.py input.gguf output.q27 [--mtp mtp.gguf] [--only REGEX] [--report N]

--mtp joins the companion MTP GGUF emitted by current llama.cpp conversions;
the primary GGUF supplies blocks 0..63 and the companion supplies block 64.
--only limits to tensors matching REGEX (smoke tests).
--report prints the N worst tensors by relative RMSE after quantization.

Ternary source packs (PrismML fork "Q2_0", ggml type 42) are detected
automatically and repacked losslessly to T2_G128 (quant_policy bonsai-t2-v1);
see docs/FORMAT.md and docs/metal/plans/2026-07-14-ternary-tier.md for the encoding.
Binary source packs (fork "Q1_0", ggml type 41) likewise repack losslessly
to B1_G128 (quant_policy bonsai-b1-v1); see
docs/metal/plans/2026-07-15-binary-tier.md.
"""
import argparse
import json
import re
import struct
import sys
import time

import numpy as np
import gguf.constants as _ggc
from gguf import GGUFReader

# The PrismML fork's types are absent from mainline gguf-py; forge the enum
# members so GGUFReader can parse their packs. (block_size, type_size) per
# ggml/src/ggml-common.h at tag prism-b9591-62061f9.
def _forge_fork_type(name, value, blck, tsize):
    if value in _ggc.GGMLQuantizationType._value2member_map_:
        return _ggc.GGMLQuantizationType(value)
    m = int.__new__(_ggc.GGMLQuantizationType, value)
    m._name_, m._value_ = name, value
    _ggc.GGMLQuantizationType._member_map_[name] = m
    _ggc.GGMLQuantizationType._value2member_map_[value] = m
    _ggc.GGML_QUANT_SIZES[m] = (blck, tsize)
    return m

_forge_fork_type("Q2_0", 42, 128, 34)
_forge_fork_type("Q1_0", 41, 128, 18)

MAGIC = 0x46373251  # "Q27F" LE
VERSION = 1
ALIGN = 256

# dtype 5 is reserved for the parked T3_G128 (never emitted; see FORMAT.md).
DTYPE_F32, DTYPE_F16, DTYPE_Q8, DTYPE_Q4, DTYPE_T2 = 0, 1, 2, 3, 4
DTYPE_B1 = 6
DTYPE_NAMES = {DTYPE_F32: "F32", DTYPE_F16: "F16", DTYPE_Q8: "Q8_G128", DTYPE_Q4: "Q4_G64",
               DTYPE_T2: "T2_G128", DTYPE_B1: "B1_G128"}
GROUP_Q4, GROUP_Q8, GROUP_T2, GROUP_B1 = 64, 128, 128, 128


Q8_EXTRA = None  # set from --q8 (v1.4 sensitivity experiments)
Q4_HEAD = False  # set from --q4-head (q4s tier: single Q4 lm_head)


def policy(name: str) -> int:
    if (name.endswith("_norm.weight") or name.endswith("norm.weight")
            or name.endswith(".ssm_a") or name.endswith(".ssm_dt.bias")
            or "ssm_conv1d" in name):
        return DTYPE_F32
    if "ssm_alpha" in name or "ssm_beta" in name:
        return DTYPE_F16
    if name == "output_q4.weight":
        return DTYPE_Q4  # v1.3: extra Q4 copy of the lm_head for MTP DRAFT passes only
    if name == "output.weight" and Q4_HEAD:
        return DTYPE_Q4  # q4s: the ONE lm_head, Q4 -- draft/verify/plain all read it
    if name in ("token_embd.weight", "output.weight") or name.startswith("blk.64."):
        return DTYPE_Q8
    if re.match(r"blk\.\d+\.attn_(k|v)\.weight$", name):
        return DTYPE_Q8  # KV projections: worst Q4 RMSE + errors persist in KV cache; ~84 MB total
    if Q8_EXTRA and name.endswith(".weight") and Q8_EXTRA.search(name):
        return DTYPE_Q8  # v1.4: PPL-sensitive tensors promoted per experiment
    if name.endswith(".weight"):
        return DTYPE_Q4
    return DTYPE_F32  # biases and anything unrecognized stay f32


def to_f32(t) -> np.ndarray:
    """GGUF tensor -> f32 numpy array, row-major with contiguous axis last."""
    tt = t.tensor_type.name
    raw = np.asarray(t.data)
    if tt == "F32":
        arr = raw.view(np.float32)
    elif tt == "F16":
        arr = raw.view(np.float16).astype(np.float32)
    elif tt == "BF16":
        u16 = raw.view(np.uint16).astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    else:
        raise ValueError(f"{t.name}: unsupported source type {tt} (need BF16/F16/F32 input)")
    shape = tuple(reversed([int(d) for d in t.shape]))  # ne[0] is innermost
    return arr.reshape(shape)


def quant_q4(w: np.ndarray):
    rows, cols = (1, w.shape[0]) if w.ndim == 1 else (int(np.prod(w.shape[:-1])), w.shape[-1])
    assert cols % GROUP_Q4 == 0, f"cols {cols} not divisible by {GROUP_Q4}"
    g = w.reshape(rows, cols // GROUP_Q4, GROUP_Q4)
    scale = np.abs(g).max(axis=2) / 7.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.rint(g / scale[..., None]), -8, 7).astype(np.int8) + 8
    q = q.reshape(rows, cols).astype(np.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    deq = ((q.reshape(rows, cols // GROUP_Q4, GROUP_Q4).astype(np.float32) - 8)
           * scale[..., None]).reshape(rows, cols)
    return packed.tobytes(), scale.astype(np.float16).tobytes(), deq


def quant_q8(w: np.ndarray):
    rows, cols = (1, w.shape[0]) if w.ndim == 1 else (int(np.prod(w.shape[:-1])), w.shape[-1])
    assert cols % GROUP_Q8 == 0, f"cols {cols} not divisible by {GROUP_Q8}"
    g = w.reshape(rows, cols // GROUP_Q8, GROUP_Q8)
    scale = np.abs(g).max(axis=2) / 127.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.rint(g / scale[..., None]), -127, 127).astype(np.int8)
    deq = (q.astype(np.float32) * scale[..., None]).reshape(rows, cols)
    return q.tobytes(), scale.astype(np.float16).tobytes(), deq


def repack_t2(t):
    """Fork Q2_0 tensor -> (data, scales, zero_frac). Lossless byte-copy.

    Source blocks are {fp16 d; uint8 qs[32]} x (n/128); codes are sequential
    LSB-first 2-bit fields, code c decodes to (c-1)*d. T2_G128 keeps the code
    bytes verbatim and splits scales into the usual contiguous fp16 blob, so
    the round-trip is exact by construction — still verified below.
    Hard-fails on code 3 (+2): the Bonsai packs must be strictly ternary.
    """
    shape = tuple(reversed([int(d) for d in t.shape]))  # ne[0] innermost -> last
    rows, cols = int(np.prod(shape[:-1])), shape[-1]
    if cols % GROUP_T2 != 0:  # hard contract, must survive python -O (codex P2)
        raise ValueError(f"{t.name}: cols {cols} not divisible by {GROUP_T2}")
    nblocks = rows * cols // GROUP_T2
    blocks = np.asarray(t.data).reshape(nblocks, 34)
    scales = blocks[:, :2].copy()                       # fp16 LE bytes, [rows, cols/128]
    qs = np.ascontiguousarray(blocks[:, 2:])            # [nblocks, 32] code bytes

    n_zero = 0
    for shift in (0, 2, 4, 6):
        c = (qs >> shift) & 3
        if np.any(c == 3):
            raise ValueError(f"{t.name}: code 3 (+2) present — pack is not strictly ternary; "
                             f"T2_G128 cannot represent it losslessly as ternary")
        n_zero += int(np.count_nonzero(c == 1))
    zero_frac = n_zero / (rows * cols)

    # Round-trip gate: dequantize the GGUF blocks per the fork's reference
    # (dequantize_row_q2_0) and our (data, scales) blobs per FORMAT.md, compare
    # bit-exact, chunked by rows to bound memory on token_embd.
    d_f32 = scales.view(np.float16).astype(np.float32).reshape(rows, cols // GROUP_T2)
    qs_rows = qs.reshape(rows, cols // 4)
    step = max(1, (1 << 25) // cols)  # ~128 MB f32 per chunk
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        blk = blocks.reshape(rows, cols // GROUP_T2, 34)[r0:r1]
        codes_g = np.stack([(blk[..., 2:] >> s) & 3 for s in (0, 2, 4, 6)],
                           axis=-1).reshape(r1 - r0, cols)
        deq_gguf = ((codes_g.astype(np.float32) - 1.0)
                    * np.repeat(blk[..., :2].copy().view(np.float16).astype(np.float32)
                                .reshape(r1 - r0, cols // GROUP_T2), GROUP_T2, axis=1))
        q = qs_rows[r0:r1]
        codes_o = np.stack([(q >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(r1 - r0, cols)
        deq_ours = ((codes_o.astype(np.float32) - 1.0)
                    * np.repeat(d_f32[r0:r1], GROUP_T2, axis=1))
        if not np.array_equal(deq_gguf, deq_ours):
            raise ValueError(f"{t.name}: T2 round-trip mismatch in rows {r0}:{r1}")

    return qs.tobytes(), scales.tobytes(), zero_frac


def repack_b1(t):
    """Fork Q1_0 tensor -> (data, scales). Lossless byte-copy (B1_G128, dtype 6).

    Source blocks are {fp16 d; uint8 qs[16]} x (n/128); bit j of a group lives
    at qs[j/8] bit (j%8) — sequential LSB-first like type 42 — and decodes to
    (2b-1)*d (dequantize_row_q1_0 at tag prism-b9591-62061f9). B1_G128 keeps
    the code bytes verbatim and splits scales into the contiguous fp16 blob,
    exactly the binary-tier plan's Phase-1 layout. Round-trip verified below.
    """
    shape = tuple(reversed([int(d) for d in t.shape]))  # ne[0] innermost -> last
    rows, cols = int(np.prod(shape[:-1])), shape[-1]
    if cols % GROUP_B1 != 0:  # hard contract, must survive python -O (codex P2)
        raise ValueError(f"{t.name}: cols {cols} not divisible by {GROUP_B1}")
    nblocks = rows * cols // GROUP_B1
    blocks = np.asarray(t.data).reshape(nblocks, 18)
    scales = blocks[:, :2].copy()                       # fp16 LE bytes
    qs = np.ascontiguousarray(blocks[:, 2:])            # [nblocks, 16] code bytes

    d_f32 = scales.view(np.float16).astype(np.float32).reshape(rows, cols // GROUP_B1)
    qs_rows = qs.reshape(rows, cols // 8)
    step = max(1, (1 << 25) // cols)  # ~128 MB f32 per chunk
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        blk = blocks.reshape(rows, cols // GROUP_B1, 18)[r0:r1]
        bits_g = np.unpackbits(blk[..., 2:], axis=-1,
                               bitorder="little").reshape(r1 - r0, cols)
        deq_gguf = ((bits_g.astype(np.float32) * 2.0 - 1.0)
                    * np.repeat(blk[..., :2].copy().view(np.float16).astype(np.float32)
                                .reshape(r1 - r0, cols // GROUP_B1), GROUP_B1, axis=1))
        bits_o = np.unpackbits(qs_rows[r0:r1], axis=-1,
                               bitorder="little").reshape(r1 - r0, cols)
        deq_ours = ((bits_o.astype(np.float32) * 2.0 - 1.0)
                    * np.repeat(d_f32[r0:r1], GROUP_B1, axis=1))
        if not np.array_equal(deq_gguf, deq_ours):
            raise ValueError(f"{t.name}: B1 round-trip mismatch in rows {r0}:{r1}")

    return qs.tobytes(), scales.tobytes()

def _field_value(reader, name):
    field = reader.fields.get(name)
    if field is None:
        return None
    value = field.contents()
    if isinstance(value, bytes):
        return value.decode()
    return value


def _tensor_signature(tensor):
    return tensor.tensor_type.name, tuple(int(d) for d in tensor.shape)

def _tensor_data_equal(left, right):
    left_bytes = np.asarray(left.data).view(np.uint8).reshape(-1)
    right_bytes = np.asarray(right.data).view(np.uint8).reshape(-1)
    if left_bytes.size != right_bytes.size:
        return False
    chunk = 64 * 1024 * 1024
    return all(np.array_equal(left_bytes[off:off + chunk], right_bytes[off:off + chunk])
               for off in range(0, left_bytes.size, chunk))


def merge_mtp_tensors(primary, companion):
    """Join llama.cpp's --no-mtp and --mtp GGUF views without duplicates."""
    if _field_value(primary, "general.architecture") != "qwen35":
        raise ValueError("--mtp requires a qwen35 primary GGUF")
    if _field_value(companion, "general.architecture") != "qwen35":
        raise ValueError("--mtp companion is not a qwen35 GGUF")
    primary_name = _field_value(primary, "general.name")
    companion_name = _field_value(companion, "general.name")
    if primary_name and companion_name and primary_name != companion_name:
        raise ValueError(f"--mtp checkpoint mismatch: {primary_name!r} != {companion_name!r}")
    if _field_value(primary, "qwen35.block_count") != 64:
        raise ValueError("--mtp primary must contain the 64 base blocks")
    if (_field_value(companion, "qwen35.block_count") != 65
            or _field_value(companion, "qwen35.nextn_predict_layers") != 1):
        raise ValueError("--mtp companion must describe one MTP layer (block_count=65)")

    primary_by_name = {t.name: t for t in primary.tensors}
    if len(primary_by_name) != len(primary.tensors):
        raise ValueError("primary GGUF contains duplicate tensor names")
    companion_by_name = {t.name: t for t in companion.tensors}
    if len(companion_by_name) != len(companion.tensors):
        raise ValueError("MTP GGUF contains duplicate tensor names")

    allowed_shared = {"token_embd.weight", "output_norm.weight", "output.weight"}
    shared = set(primary_by_name) & set(companion_by_name)
    unexpected = shared - allowed_shared
    if unexpected:
        raise ValueError("--mtp companion overlaps primary tensors: "
                         + ", ".join(sorted(unexpected)))
    for name in shared:
        if _tensor_signature(primary_by_name[name]) != _tensor_signature(companion_by_name[name]):
            raise ValueError(f"--mtp shared tensor shape/type mismatch: {name}")
        if not _tensor_data_equal(primary_by_name[name], companion_by_name[name]):
            raise ValueError(f"--mtp shared tensor contents differ: {name}")

    mtp_only = [t for t in companion.tensors if t.name not in primary_by_name]
    if not mtp_only or any(not t.name.startswith("blk.64.") for t in mtp_only):
        raise ValueError("--mtp companion must add only blk.64 tensors")
    return list(primary.tensors) + mtp_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--mtp", default=None,
                    help="companion BF16 MTP GGUF (llama.cpp --mtp output)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--report", type=int, default=15)
    ap.add_argument("--q8", default=None,
                    help="extra tensor-name regex forced to Q8_G128 (v1.4 policy experiments)")
    ap.add_argument("--tag", default=None,
                    help="quant_policy meta override (e.g. q6-v1 for the 6-bit tier)")
    ap.add_argument("--q4-head", action="store_true",
                    help="emit output.weight at Q4_G64 and skip the output_q4.weight copy "
                         "(q4s tier; engine falls back to output.weight for drafts)")
    args = ap.parse_args()
    global Q8_EXTRA, Q4_HEAD
    if args.q8:
        Q8_EXTRA = re.compile(args.q8)
    Q4_HEAD = args.q4_head

    t0 = time.time()
    r = GGUFReader(args.input)
    mtp_reader = GGUFReader(args.mtp) if args.mtp else None
    source_tensors = merge_mtp_tensors(r, mtp_reader) if mtp_reader else list(r.tensors)
    ternary = any(t.tensor_type.name == "Q2_0" for t in source_tensors)
    binary = any(t.tensor_type.name == "Q1_0" for t in source_tensors)
    if mtp_reader and (ternary or binary):
        raise ValueError("--mtp is only supported for BF16/F16/F32 source GGUFs")
    if binary and ternary:
        raise ValueError("pack unexpectedly contains both binary (Q1_0) and ternary (Q2_0) tensors")
    if (not mtp_reader and not ternary and not binary
            and _field_value(r, "general.architecture") == "qwen35"
            and _field_value(r, "qwen35.block_count") == 64):
        raise ValueError("base-only qwen35 GGUF: supply its companion with --mtp")

    meta = {"q27_version": VERSION,
            "quant_policy": args.tag or ("bonsai-t2-v1" if ternary
                                         else "bonsai-b1-v1" if binary
                                         else "v1.4" if args.q8 else "v1.3"),
            "group_q4": GROUP_Q4, "group_q8": GROUP_Q8, "nibble_order": "even=low"}
    if ternary:
        meta["group_t2"] = GROUP_T2
        meta["t2_codes"] = "0=-1,1=0,2=+1;3 forbidden"
        meta["t2_slot_order"] = "seq-lsb-first"
    if binary:
        # Verbatim fork Q1_0 encoding; see the repack_b1 docstring and the
        # binary-tier plan.
        meta["group_b1"] = GROUP_B1
        meta["b1_codes"] = "1=+d,0=-d"
        meta["b1_bit_order"] = "seq-lsb-first"
    if args.q8:
        meta["q8_extra"] = args.q8
    if args.q4_head:
        meta["q4_head"] = True
    metadata_readers = (r, mtp_reader) if mtp_reader else (r,)
    for metadata_reader in metadata_readers:
        for f in metadata_reader.fields.values():
            if f.name.startswith(("qwen35.", "general.architecture", "general.name")):
                try:
                    v = f.contents()
                    if isinstance(v, bytes):
                        v = v.decode()
                    meta[f.name] = v
                except Exception:
                    pass
    # layer map
    attn_layers, ssm_layers = set(), set()
    for t in source_tensors:
        if t.name.startswith("blk."):
            n = int(t.name.split(".")[1])
            leaf = t.name.split(".", 2)[2]
            if leaf.startswith("attn_q."):
                attn_layers.add(n)
            if leaf.startswith("ssm_out"):
                ssm_layers.add(n)
    meta["attn_layers"] = sorted(attn_layers)
    meta["ssm_layers"] = sorted(ssm_layers)

    only = re.compile(args.only) if args.only else None
    entries, blobs = [], []
    errors = []
    offset = 0
    n_bytes_in = n_bytes_out = 0

    extra = []
    for t in source_tensors:
        # MTP draft head copy: only for non-quantized-head packs and pack
        # types that carry MTP (no MTP in ternary/binary packs).
        if t.name == "output.weight" and not args.q4_head \
                and not ternary and not binary:
            extra.append(("output_q4.weight", t))
    class _Alias:
        def __init__(self, name, t):
            self.name, self.tensor_type, self.data, self.shape = name, t.tensor_type, t.data, t.shape
    tensor_iter = source_tensors + [_Alias(n, t) for n, t in extra]
    zero_fracs = []
    for t in tensor_iter:
        if only and not only.search(t.name):
            continue
        verbatim = None  # (dtype, repack_fn) for lossless byte-copy source types
        if t.tensor_type.name == "Q2_0":
            verbatim = (DTYPE_T2, repack_t2)
        elif binary and t.tensor_type.name == "Q1_0":
            verbatim = (DTYPE_B1, repack_b1)
        if verbatim is not None:
            vdt, fn = verbatim
            shape = tuple(reversed([int(d) for d in t.shape]))
            out = fn(t)
            if vdt == DTYPE_T2:
                data, scales, zero_frac = out
                zero_fracs.append((zero_frac, int(np.prod(shape)), t.name))
            else:
                data, scales = out
            n_bytes_in += int(np.prod(shape)) * 4
            n_bytes_out += len(data) + len(scales)
            errors.append((0.0, t.name, DTYPE_NAMES[vdt]))  # lossless, gate-verified
            data_off = offset
            offset = (offset + len(data) + ALIGN - 1) // ALIGN * ALIGN
            scale_off = offset
            offset = (offset + len(scales) + ALIGN - 1) // ALIGN * ALIGN
            entries.append((t.name, vdt, shape, data_off, len(data), scale_off, len(scales)))
            blobs.append((data_off, data))
            blobs.append((scale_off, scales))
            continue
        if ternary and t.tensor_type.name != "F32":
            raise ValueError(f"{t.name}: unexpected source type {t.tensor_type.name} in a "
                             f"ternary pack (expected Q2_0 or F32 only)")
        if binary and t.tensor_type.name != "F32":
            raise ValueError(f"{t.name}: unexpected source type {t.tensor_type.name} in a "
                             f"binary pack (expected Q1_0 or F32 only)")
        w = to_f32(t)
        n_bytes_in += w.nbytes
        dt = policy(t.name)
        if dt == DTYPE_Q4 and w.shape[-1] % GROUP_Q4 != 0:
            dt = DTYPE_F16  # fallback, shouldn't happen on this model
        if dt == DTYPE_Q8 and w.shape[-1] % GROUP_Q8 != 0:
            dt = DTYPE_F16

        scales = b""
        if dt == DTYPE_F32:
            data = w.astype(np.float32).tobytes()
            deq = w
        elif dt == DTYPE_F16:
            data = w.astype(np.float16).tobytes()
            deq = w.astype(np.float16).astype(np.float32)
        elif dt == DTYPE_Q8:
            data, scales, deq = quant_q8(w)
        else:
            data, scales, deq = quant_q4(w)

        denom = float(np.sqrt(np.mean(w.astype(np.float64) ** 2))) or 1e-12
        rel_rmse = float(np.sqrt(np.mean((w - deq.reshape(w.shape)).astype(np.float64) ** 2))) / denom
        errors.append((rel_rmse, t.name, DTYPE_NAMES[dt]))

        data_off = offset
        offset += len(data)
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        scale_off = offset if scales else 0
        offset += len(scales)
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        n_bytes_out += len(data) + len(scales)

        entries.append((t.name, dt, w.shape, data_off, len(data), scale_off, len(scales)))
        blobs.append((data_off, data))
        if scales:
            blobs.append((scale_off, scales))
        del w, deq

    meta_b = json.dumps(meta).encode()
    with open(args.output, "wb") as f:
        f.write(struct.pack("<IIII", MAGIC, VERSION, len(entries), len(meta_b)))
        f.write(meta_b)
        for name, dt, shape, doff, dsize, soff, ssize in entries:
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<BB", dt, len(shape)))
            for d in shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<QQQQ", doff, dsize, soff, ssize))
        table_end = f.tell()
        pad = (table_end + ALIGN - 1) // ALIGN * ALIGN - table_end
        f.write(b"\0" * pad)
        base = f.tell()
        for off, blob in blobs:
            f.seek(base + off)
            f.write(blob)

    dt_s = time.time() - t0
    print(f"repacked {len(entries)} tensors: {n_bytes_in/1e9:.2f} GB f32-equiv -> "
          f"{n_bytes_out/1e9:.2f} GB in {dt_s:.0f}s -> {args.output}")
    errors.sort(reverse=True)
    print(f"\nworst {args.report} tensors by relative RMSE:")
    for rmse, name, dtn in errors[:args.report]:
        print(f"  {rmse:.4f}  {dtn:8s} {name}")

    if zero_fracs:
        total = sum(n for _, n, _ in zero_fracs)
        mean_zero = sum(z * n for z, n, _ in zero_fracs) / total
        zero_fracs.sort()
        print(f"\nT2 slot verification passed on {len(zero_fracs)} tensors "
              f"({total/1e9:.2f} B ternary weights, {mean_zero:.1%} zeros overall)")
        print(f"  least sparse: {zero_fracs[0][0]:.1%} {zero_fracs[0][2]}")
        print(f"  most sparse:  {zero_fracs[-1][0]:.1%} {zero_fracs[-1][2]}")


if __name__ == "__main__":
    main()
