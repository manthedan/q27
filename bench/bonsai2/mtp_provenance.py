#!/usr/bin/env python3
"""Does the HF-named Bonsai 2 MTP checkpoint sit in the same row layout as the
Qwen3.8 MTP block the q27 packs carry (llama.cpp conversion, no q/k permute)?
The head was fine-tuned from the byte-identical Qwen3.8 donor at a small LR,
so per-row correlation against the q27 pack's Q8 blk.64 tensors should be
high everywhere if the mapping is right, and a permuted q/k would show as
rows correlating with the WRONG rows. Norms should be near-identical."""
import json, struct, sys
import numpy as np
import torch
from safetensors import safe_open

PACK = "/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27"
ST = "/mnt/ai/models/bonsai2-27b-mtp-bf16/model_mtp.safetensors"
MAP = {"mtp.fc.weight": "nextn.eh_proj.weight", "mtp.pre_fc_norm_embedding.weight": "nextn.enorm.weight",
       "mtp.pre_fc_norm_hidden.weight": "nextn.hnorm.weight", "mtp.norm.weight": "nextn.shared_head_norm.weight",
       "mtp.layers.0.input_layernorm.weight": "attn_norm.weight",
       "mtp.layers.0.post_attention_layernorm.weight": "post_attention_norm.weight",
       "mtp.layers.0.self_attn.q_norm.weight": "attn_q_norm.weight", "mtp.layers.0.self_attn.k_norm.weight": "attn_k_norm.weight",
       "mtp.layers.0.self_attn.q_proj.weight": "attn_q.weight", "mtp.layers.0.self_attn.k_proj.weight": "attn_k.weight",
       "mtp.layers.0.self_attn.v_proj.weight": "attn_v.weight", "mtp.layers.0.self_attn.o_proj.weight": "attn_output.weight",
       "mtp.layers.0.mlp.gate_proj.weight": "ffn_gate.weight", "mtp.layers.0.mlp.up_proj.weight": "ffn_up.weight",
       "mtp.layers.0.mlp.down_proj.weight": "ffn_down.weight"}

def read_pack_table(path):
    f = open(path, "rb")
    magic, ver, nt, ml = struct.unpack("<IIII", f.read(16))
    assert magic == 0x46373251 and ver == 1, (hex(magic), ver)
    f.read(ml)
    table = {}
    for _ in range(nt):
        (nl,) = struct.unpack("<H", f.read(2)); name = f.read(nl).decode()
        dt, nd = struct.unpack("<BB", f.read(2))
        shape = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        doff, dsz, soff, ssz = struct.unpack("<QQQQ", f.read(32))
        table[name] = (dt, shape, doff, dsz, soff, ssz)
    data_start = f.tell()
    data_start = (data_start + 255) // 256 * 256
    return f, table, data_start

def load_pack_tensor(f, table, base, name):
    dt, shape, doff, dsz, soff, ssz = table[name]
    f.seek(base + doff); raw = np.frombuffer(f.read(dsz), dtype=np.uint8)
    if dt == 0:
        return raw.view(np.float32).reshape(shape)
    if dt == 2:  # Q8_G128
        rows, cols = shape
        f.seek(base + soff); sc = np.frombuffer(f.read(ssz), dtype=np.float16).astype(np.float32).reshape(rows, cols // 128)
        q = raw.view(np.int8).reshape(rows, cols // 128, 128).astype(np.float32)
        return (q * sc[:, :, None]).reshape(rows, cols)
    raise ValueError(f"{name}: dtype {dt} not handled")

f, table, base = read_pack_table(PACK)
print("pack blk.64 tensors:", sum(1 for n in table if n.startswith("blk.64.")))
with safe_open(ST, "pt") as st:
    for hf, leaf in MAP.items():
        name = "blk.64." + leaf
        a = st.get_tensor(hf).to(torch.float32).numpy()
        b = load_pack_tensor(f, table, base, name)
        if a.shape != b.shape:
            print(f"  {name:40s} SHAPE MISMATCH {a.shape} vs {b.shape}"); continue
        if a.ndim == 1:
            d = np.abs(a - b).max(); print(f"  {name:40s} norm max|d| {d:.4f}  corr {np.corrcoef(a, b)[0,1]:.6f}")
            continue
        # per-row correlation (same row index) vs the best-matching row for a sample of rows
        idx = np.linspace(0, a.shape[0] - 1, 24).astype(int)
        same = [np.corrcoef(a[i], b[i])[0, 1] for i in idx]
        # 'permuted?' probe: correlate sample rows against a block of neighbouring rows
        best_other = []
        for i in idx[:8]:
            lo, hi = max(0, i - 256), min(a.shape[0], i + 256)
            c = (b[lo:hi] @ a[i]) / (np.linalg.norm(b[lo:hi], axis=1) * np.linalg.norm(a[i]) + 1e-9)
            j = lo + int(np.argmax(c)); best_other.append((int(i), j, float(c.max())))
        mism = sum(1 for (i, j, _) in best_other if i != j)
        print(f"  {name:40s} same-row corr mean {np.mean(same):.4f} min {np.min(same):.4f} | best-row != same-row: {mism}/8")
