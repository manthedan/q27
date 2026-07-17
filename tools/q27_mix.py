#!/usr/bin/env python3
"""Assemble a mixed-tier q27 pack from two same-model q27 containers.

Mixed weight-tier census (docs/plans/2026-07-17-mixed-tier-census.md):
every tensor comes from the BASE pack except those matching --take, which
come from the DONOR pack. The packs are architecture-compatible Bonsai
siblings; this is a pure byte-level reassembly — no quantization runs.

Usage:
  q27_mix.py base.q27 donor.q27 output.q27 --take REGEX [--dry-run]

The output carries quant_policy bonsai-mixed-v1 and BOTH tiers' layout
meta (the engine's validate_architecture demands the declarations for
every dtype that can appear). Tensor compatibility is validated: the two
packs must agree on the full name set and per-tensor shapes.
"""
import argparse
import json
import os
import re
import struct
import sys

MAGIC = 0x46373251
VERSION = 1
ALIGN = 256
DTYPE_NAMES = {0: "F32", 1: "F16", 2: "Q8_G128", 3: "Q4_G64", 4: "T2_G128",
               6: "B1_G128", 7: "Q4_1_G32"}


def read_pack(path):
    with open(path, "rb") as f:
        magic, version, n_tensors, meta_len = struct.unpack("<IIII", f.read(16))
        if magic != MAGIC or version != VERSION:
            sys.exit(f"{path}: not a q27 v1 container")
        meta = json.loads(f.read(meta_len))
        table = []
        for _ in range(n_tensors):
            (name_len,) = struct.unpack("<H", f.read(2))
            name = f.read(name_len).decode()
            dtype, n_dims = struct.unpack("<BB", f.read(2))
            shape = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
            data_off, data_size, scale_off, scale_size = struct.unpack("<QQQQ", f.read(32))
            table.append(dict(name=name, dtype=dtype, shape=shape,
                              data_off=data_off, data_size=data_size,
                              scale_off=scale_off, scale_size=scale_size))
        # Data section starts at the 256-aligned table end (loader.cpp /
        # repack.py convention). Blob reads are deferred; record the base.
        data_start = align(f.tell())
    return dict(path=path, meta=meta, table=table, data_start=data_start)


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("donor")
    ap.add_argument("output")
    ap.add_argument("--take", required=True,
                    help="regex (search) of tensor names taken from the donor")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the selection and byte delta, write nothing")
    args = ap.parse_args()

    output_real = os.path.realpath(args.output)
    for source in (args.base, args.donor):
        if os.path.realpath(source) == output_real:
            sys.exit("mix: output must not overwrite either source pack")
        if os.path.exists(args.output) and os.path.samefile(source, args.output):
            sys.exit("mix: output must not alias either source pack")

    base = read_pack(args.base)
    donor = read_pack(args.donor)
    take = re.compile(args.take)

    policies = {base["meta"].get("quant_policy"), donor["meta"].get("quant_policy")}
    expected_policies = {"bonsai-b1-v1", "bonsai-t2-v1"}
    if policies != expected_policies:
        sys.exit(f"mix: expected one B1 and one T2 pack, got {sorted(str(p) for p in policies)}")

    base_by_name = {t["name"]: t for t in base["table"]}
    donor_by_name = {t["name"]: t for t in donor["table"]}
    if len(base_by_name) != len(base["table"]) or len(donor_by_name) != len(donor["table"]):
        sys.exit("mix: duplicate tensor name in source pack")
    if set(base_by_name) != set(donor_by_name):
        only_b = sorted(set(base_by_name) - set(donor_by_name))[:5]
        only_d = sorted(set(donor_by_name) - set(base_by_name))[:5]
        sys.exit(f"tensor name sets differ (base-only {only_b}, donor-only {only_d})")
    for name, bt in base_by_name.items():
        dt = donor_by_name[name]
        if bt["shape"] != dt["shape"]:
            sys.exit(f"shape mismatch for {name}: {bt['shape']} vs {dt['shape']}")
        if bt["dtype"] != dt["dtype"] and {bt["dtype"], dt["dtype"]} != {4, 6}:
            sys.exit(f"mix: non-Bonsai dtype mismatch for {name}: "
                     f"{DTYPE_NAMES.get(bt['dtype'], bt['dtype'])} vs "
                     f"{DTYPE_NAMES.get(dt['dtype'], dt['dtype'])}")

    taken, kept, delta = [], 0, 0
    entries = []
    for bt in base["table"]:  # base table order is preserved in the output
        dt = donor_by_name[bt["name"]]
        if take.search(bt["name"]):
            # Same-dtype tensors are not necessarily byte-identical between
            # the separately trained Bonsai siblings. --take means donor
            # bytes, not merely a dtype flip.
            entries.append((donor, dt))
            taken.append(bt["name"])
            delta += (dt["data_size"] + dt["scale_size"]) - (bt["data_size"] + bt["scale_size"])
        else:
            entries.append((base, bt))
            kept += 1

    print(f"mix: {len(taken)} tensors from donor, {kept} from base, "
          f"byte delta {delta / 1e6:+.1f} MB")
    for name in taken[:10]:
        print(f"  take {name}")
    if len(taken) > 10:
        print(f"  ... and {len(taken) - 10} more")
    if not taken:
        sys.exit("mix: --take matched no tensors; refusing a vacuous pack")
    if args.dry_run:
        return

    meta = dict(base["meta"])
    # Copy each tier's layout declaration from the pack that actually owns
    # that encoding. This also supports a T2 base with a B1 donor without
    # trusting unrelated/default keys that the other pack may carry.
    by_policy = {p["meta"]["quant_policy"]: p for p in (base, donor)}
    for policy, keys in (
            ("bonsai-t2-v1", ("group_t2", "t2_codes", "t2_slot_order")),
            ("bonsai-b1-v1", ("group_b1", "b1_codes", "b1_bit_order"))):
        source_meta = by_policy[policy]["meta"]
        missing = [key for key in keys if key not in source_meta]
        if missing:
            sys.exit(f"mix: {policy} pack lacks layout metadata: {', '.join(missing)}")
        for key in keys:
            meta[key] = source_meta[key]
    meta["quant_policy"] = "bonsai-mixed-v1"
    meta["mixed_base"] = base["meta"].get("quant_policy", "?")
    meta["mixed_donor"] = donor["meta"].get("quant_policy", "?")
    meta["mixed_take"] = args.take
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode()

    # Lay out the output: recompute offsets in table order, blobs aligned.
    out_entries = []
    cursor = 0
    for src, t in entries:
        e = dict(t)
        e["_src"] = src
        e["_src_data_off"], e["_src_scale_off"] = t["data_off"], t["scale_off"]
        e["data_off"] = cursor
        cursor = align(cursor + t["data_size"])
        if t["scale_size"]:
            e["scale_off"] = cursor
            cursor = align(cursor + t["scale_size"])
        else:
            e["scale_off"] = 0
        out_entries.append(e)

    with open(args.output, "wb") as out:
        out.write(struct.pack("<IIII", MAGIC, VERSION, len(out_entries), len(meta_bytes)))
        out.write(meta_bytes)
        for e in out_entries:
            name = e["name"].encode()
            out.write(struct.pack("<H", len(name)))
            out.write(name)
            out.write(struct.pack("<BB", e["dtype"], len(e["shape"])))
            out.write(struct.pack(f"<{len(e['shape'])}Q", *e["shape"]))
            out.write(struct.pack("<QQQQ", e["data_off"], e["data_size"],
                                  e["scale_off"], e["scale_size"]))
        table_end = out.tell()
        out.write(b"\0" * (align(table_end) - table_end))
        data_start = out.tell()
        sources = {}
        for e in out_entries:
            src = e["_src"]
            f = sources.setdefault(src["path"], open(src["path"], "rb"))
            for src_off, out_off, size in (
                    (e["_src_data_off"], e["data_off"], e["data_size"]),
                    (e["_src_scale_off"], e["scale_off"], e["scale_size"])):
                if not size:
                    continue
                f.seek(src["data_start"] + src_off)
                out.seek(data_start + out_off)
                remaining = size
                while remaining:
                    chunk = f.read(min(remaining, 64 << 20))
                    if not chunk:
                        raise RuntimeError(f"unexpected EOF reading {src['path']} for {e['name']}")
                    out.write(chunk)
                    remaining -= len(chunk)
        for f in sources.values():
            f.close()
    print(f"mix: wrote {args.output}")


if __name__ == "__main__":
    main()
