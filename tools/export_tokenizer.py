#!/usr/bin/env python3
"""Export tokenizer data from the GGUF into a compact q27.tok file.

Format (little-endian):
  magic u32 'Q27T', version u32=1
  n_tokens u32, bos u32, eos u32
  n_tokens x { len u16, bytes }        # token strings in GPT-2 byte-encoded space
  n_tokens x { type u8 }               # 1=normal, 3=control(special), others as in gguf
  n_merges u32
  n_merges x { len u16, bytes }        # merge lines "left right"
"""
import struct
import sys

import gguf.constants as _ggc
from gguf import GGUFReader


# The PrismML fork's quant types are absent from mainline gguf-py; forge the
# enum members so GGUFReader can parse their packs (GGUFReader eagerly builds
# every tensor even though we only read tokenizer metadata). Mirrors
# tools/repack.py's _forge_fork_type; (block_size, type_size) per
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


def main():
    src, dst = sys.argv[1], sys.argv[2]
    r = GGUFReader(src)
    f = {x.name: x for x in r.fields.values()}
    toks = f["tokenizer.ggml.tokens"].contents()
    types = f["tokenizer.ggml.token_type"].contents()
    merges = f["tokenizer.ggml.merges"].contents()
    bos = f["tokenizer.ggml.bos_token_id"].contents()
    eos = f["tokenizer.ggml.eos_token_id"].contents()

    with open(dst, "wb") as o:
        o.write(struct.pack("<IIIII", 0x54373251, 1, len(toks), bos, eos))
        for t in toks:
            b = t.encode("utf-8")
            o.write(struct.pack("<H", len(b)))
            o.write(b)
        o.write(bytes(int(x) & 0xFF for x in types))
        o.write(struct.pack("<I", len(merges)))
        for m in merges:
            b = m.encode("utf-8")
            o.write(struct.pack("<H", len(b)))
            o.write(b)
    print(f"exported {len(toks)} tokens, {len(merges)} merges, bos={bos}, eos={eos} -> {dst}")

if __name__ == "__main__":
    main()
