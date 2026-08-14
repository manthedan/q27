#!/usr/bin/env python3
"""Contract checks for joining llama.cpp base and MTP GGUF views."""

import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("q27_repack", Path(__file__).with_name("repack.py"))
repack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repack)


class Field:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def contents(self):
        return self.value


class TensorType:
    name = "BF16"


class Tensor:
    def __init__(self, name, shape=(5120,), data=b"\0"):
        self.name = name
        self.shape = shape
        self.tensor_type = TensorType()
        self.data = repack.np.frombuffer(data, dtype=repack.np.uint8)


class Reader:
    def __init__(self, name, block_count, tensors, nextn=None):
        values = {
            "general.architecture": "qwen35",
            "general.name": name,
            "qwen35.block_count": block_count,
        }
        if nextn is not None:
            values["qwen35.nextn_predict_layers"] = nextn
        self.fields = {key: Field(key, value) for key, value in values.items()}
        self.tensors = tensors


def expect_error(message, primary, companion):
    try:
        repack.merge_mtp_tensors(primary, companion)
    except ValueError as error:
        assert message in str(error), error
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


shared = [Tensor("token_embd.weight"), Tensor("output_norm.weight"), Tensor("output.weight")]
primary = Reader("Qwen3.8-27B", 64, shared + [Tensor("blk.0.attn_q.weight")])
companion = Reader("Qwen3.8-27B", 65, shared + [Tensor("blk.64.attn_q.weight")], nextn=1)
merged = repack.merge_mtp_tensors(primary, companion)
assert [tensor.name for tensor in merged] == [
    "token_embd.weight",
    "output_norm.weight",
    "output.weight",
    "blk.0.attn_q.weight",
    "blk.64.attn_q.weight",
]

expect_error(
    "checkpoint mismatch",
    primary,
    Reader("Qwen3.6-27B", 65, shared + [Tensor("blk.64.attn_q.weight")], nextn=1),
)
expect_error(
    "overlaps primary tensors",
    primary,
    Reader("Qwen3.8-27B", 65, shared + [Tensor("blk.0.attn_q.weight")], nextn=1),
)
expect_error(
    "shape/type mismatch",
    primary,
    Reader(
        "Qwen3.8-27B",
        65,
        [Tensor("output.weight", shape=(4096,)), Tensor("blk.64.attn_q.weight")],
        nextn=1,
    ),
)
expect_error(
    "contents differ",
    primary,
    Reader(
        "Qwen3.8-27B",
        65,
        [Tensor("output.weight", data=b"\1"), Tensor("blk.64.attn_q.weight")],
        nextn=1,
    ),
)

print("split MTP GGUF merge contracts: PASS")
