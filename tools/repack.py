#!/usr/bin/env python3
"""Repack a BF16 GGUF into the q27 v1 format (see docs/FORMAT.md).

Usage:
  repack.py input.gguf output.q27 [--only REGEX] [--report N]

--only limits to tensors matching REGEX (smoke tests).
--report prints the N worst tensors by relative RMSE after quantization.

Ternary source packs (PrismML fork "Q2_0", ggml type 42) are detected
automatically and repacked losslessly to T2_G128 (quant_policy bonsai-t2-v1);
see docs/FORMAT.md and docs/plans/2026-07-14-ternary-tier.md for the encoding.
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

DTYPE_F32, DTYPE_F16, DTYPE_Q8, DTYPE_Q4, DTYPE_T2 = 0, 1, 2, 3, 4
DTYPE_NAMES = {DTYPE_F32: "F32", DTYPE_F16: "F16", DTYPE_Q8: "Q8_G128", DTYPE_Q4: "Q4_G64",
               DTYPE_T2: "T2_G128"}
GROUP_Q4, GROUP_Q8, GROUP_T2 = 64, 128, 128


Q8_EXTRA = None  # set from --q8 (v1.4 sensitivity experiments)


def policy(name: str) -> int:
    if (name.endswith("_norm.weight") or name.endswith("norm.weight")
            or name.endswith(".ssm_a") or name.endswith(".ssm_dt.bias")
            or "ssm_conv1d" in name):
        return DTYPE_F32
    if "ssm_alpha" in name or "ssm_beta" in name:
        return DTYPE_F16
    if name == "output_q4.weight":
        return DTYPE_Q4  # v1.3: extra Q4 copy of the lm_head for MTP DRAFT passes only
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
    assert cols % GROUP_T2 == 0, f"{t.name}: cols {cols} not divisible by {GROUP_T2}"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--only", default=None)
    ap.add_argument("--report", type=int, default=15)
    ap.add_argument("--q8", default=None,
                    help="extra tensor-name regex forced to Q8_G128 (v1.4 policy experiments)")
    ap.add_argument("--tag", default=None,
                    help="quant_policy meta override (e.g. q6-v1 for the 6-bit tier)")
    args = ap.parse_args()
    global Q8_EXTRA
    if args.q8:
        Q8_EXTRA = re.compile(args.q8)

    t0 = time.time()
    r = GGUFReader(args.input)
    ternary = any(t.tensor_type.name == "Q2_0" for t in r.tensors)

    meta = {"q27_version": VERSION,
            "quant_policy": args.tag or ("bonsai-t2-v1" if ternary
                                         else "v1.4" if args.q8 else "v1.3"),
            "group_q4": GROUP_Q4, "group_q8": GROUP_Q8, "nibble_order": "even=low"}
    if ternary:
        meta["group_t2"] = GROUP_T2
        meta["t2_codes"] = "0=-1,1=0,2=+1;3 forbidden"
        meta["t2_slot_order"] = "seq-lsb-first"
    if args.q8:
        meta["q8_extra"] = args.q8
    for f in r.fields.values():
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
    for t in r.tensors:
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
    for t in r.tensors:
        if t.name == "output.weight" and not ternary:
            extra.append(("output_q4.weight", t))  # MTP draft head copy; no MTP in ternary packs
    class _Alias:
        def __init__(self, name, t):
            self.name, self.tensor_type, self.data, self.shape = name, t.tensor_type, t.data, t.shape
    tensor_iter = list(r.tensors) + [_Alias(n, t) for n, t in extra]
    zero_fracs = []
    for t in tensor_iter:
        if only and not only.search(t.name):
            continue
        if t.tensor_type.name == "Q2_0":
            shape = tuple(reversed([int(d) for d in t.shape]))
            data, scales, zero_frac = repack_t2(t)
            zero_fracs.append((zero_frac, int(np.prod(shape)), t.name))
            n_bytes_in += int(np.prod(shape)) * 4
            n_bytes_out += len(data) + len(scales)
            errors.append((0.0, t.name, DTYPE_NAMES[DTYPE_T2]))  # lossless, gate-verified
            data_off = offset
            offset = (offset + len(data) + ALIGN - 1) // ALIGN * ALIGN
            scale_off = offset
            offset = (offset + len(scales) + ALIGN - 1) // ALIGN * ALIGN
            entries.append((t.name, DTYPE_T2, shape, data_off, len(data), scale_off, len(scales)))
            blobs.append((data_off, data))
            blobs.append((scale_off, scales))
            continue
        if ternary and t.tensor_type.name != "F32":
            raise ValueError(f"{t.name}: unexpected source type {t.tensor_type.name} in a "
                             f"ternary pack (expected Q2_0 or F32 only)")
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
