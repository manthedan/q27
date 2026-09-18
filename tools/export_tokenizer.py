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

from prism_gguf import GGUFReader

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
