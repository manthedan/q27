#!/usr/bin/env python3
"""Model-free Bonsai 2 contracts for upstream's converter (tools/repack.py):
rotation metadata, PQ2_0 -> T2/T3 bytes, and the forged Prism type ids.
Run directly or with python -O."""
from types import SimpleNamespace
import unittest
import numpy as np

import gguf.constants as ggc
import repack


class Field:
    def __init__(self, name, value): self.name, self.value = name, value
    def contents(self): return self.value


def reader(**overrides):
    h = {"version": 1, "block_size": 1024,
         "transform": "normalized-sylvester-walsh-hadamard",
         "axis": "input-last-dimension", "sign_mode": "explicit",
         "sign_widths": [5120, 6144, 17408], "sign_values": [1, -1] * 14336,
         "weight_names": ["blk.0.ffn_down.weight"],
         "inverse_weight_names": ["token_embd.weight"], "gdn_v_grouped": True}
    h.update(overrides)
    fields = {"prism.hadamard." + k: Field("prism.hadamard." + k, v)
              for k, v in h.items() if v is not None}
    fields["general.name"] = Field("general.name", "Hf")
    return SimpleNamespace(fields=fields)


def pq2_tensor(codes, scales):
    """codes uint8 [rows, cols] in {0,1,2} (= trit+1), scales fp16 [rows*cols/128]."""
    rows, cols = codes.shape
    packed = (codes[:, 0::4] | codes[:, 1::4] << 2 | codes[:, 2::4] << 4 |
              codes[:, 3::4] << 6).reshape(rows * cols // 128, 32)
    blocks = np.concatenate([scales.view(np.uint8).reshape(-1, 2), packed], axis=1)
    return SimpleNamespace(name="fixture", shape=[cols, rows], data=blocks,
                           tensor_type=SimpleNamespace(name="PQ2_0")), packed


class Contracts(unittest.TestCase):
    def test_hadamard_meta(self):
        out = repack.bonsai2_hadamard_meta(reader())
        self.assertEqual(out["sign_widths"], [5120, 6144, 17408])
        self.assertEqual(out["inverse_weight_names"], ["token_embd.weight"])
        self.assertIs(out["gdn_v_grouped"], True)
        for key in ("version", "block_size", "transform", "axis", "sign_mode",
                    "sign_widths", "sign_values", "weight_names"):
            with self.assertRaises(ValueError, msg=key):
                repack.bonsai2_hadamard_meta(reader(**{key: None}))
        for key, value in {"version": 2, "transform": "other", "axis": "rows",
                           "sign_mode": "identity", "sign_values": [1] * 100,
                           "sign_widths": [5120, 6144, 17000]}.items():
            with self.assertRaises(ValueError, msg=key):
                repack.bonsai2_hadamard_meta(reader(**{key: value}))

    def test_pq2_to_t2_bytes(self):
        rng = np.random.default_rng(123)
        codes = rng.integers(0, 3, (3, 256), dtype=np.uint8)
        scales = np.array([0.125, 0.25, 0.5, 1, 2, 4], dtype="<f2")
        t, packed = pq2_tensor(codes, scales)
        data, out_scales, zero = repack.repack_bonsai2_t2(t)
        self.assertEqual(data, packed.tobytes())
        self.assertEqual(out_scales, scales.tobytes())
        self.assertEqual(zero, float(np.mean(codes == 1)))
        t.data = t.data.copy(); t.data[0, 2] |= 3
        with self.assertRaises(ValueError):
            repack.repack_bonsai2_t2(t)

    def test_pq2_to_t3_bytes(self):
        rng = np.random.default_rng(7)
        codes = rng.integers(0, 3, (2, 256), dtype=np.uint8)
        scales = np.array([0.5, 1, 2, 4], dtype="<f2")
        t, _ = pq2_tensor(codes, scales)
        data, out_scales, _ = repack.repack_bonsai2_t3(t)
        self.assertEqual(out_scales, scales.tobytes())
        raw = np.frombuffer(data, dtype=np.uint8).reshape(2, 2, 26)
        self.assertLessEqual(int(raw.max()), 242)
        # Independent decode per FORMAT.md T3_G128: c0 least significant,
        # byte 25 holds columns 125..127 and two code-1 pads.
        digits = np.stack([(raw.astype(np.uint16) // 3 ** k) % 3 for k in range(5)], axis=-1)
        slots = digits.reshape(2, 2, 130)
        np.testing.assert_array_equal(slots[..., :128].reshape(2, 256), codes)
        np.testing.assert_array_equal(slots[..., 128:], 1)

    def test_gate_tensors_are_f16(self):
        # Upstream stores the BF16 GDN gates as F16 (the Q-tier choice); the
        # Metal validator pins the same dtype.
        for name in ("blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"):
            self.assertEqual(repack.policy(name), repack.DTYPE_F16)

    def test_forged_types(self):
        for name, value, size in (("PQ2_0", 142, (128, 34)), ("PTQ1_0", 143, (128, 28))):
            member = ggc.GGMLQuantizationType(value)
            self.assertEqual(member.name, name)
            self.assertEqual(ggc.GGML_QUANT_SIZES[member], size)


if __name__ == "__main__":
    unittest.main()
