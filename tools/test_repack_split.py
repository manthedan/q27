#!/usr/bin/env python3
"""Helper-level contracts for joining standalone Qwen3.8 base and MTP GGUFs."""

import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "q27_repack", Path(__file__).with_name("repack.py"))
repack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repack)


class Field:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def contents(self):
        return self.value


class TensorType:
    def __init__(self, name):
        self.name = name


class Tensor:
    def __init__(self, name, shape=None, data=b"\0", dtype=None):
        self.name = name
        expected = (repack.QWEN35_BASE_SPECS.get(name) or
                    repack.QWEN35_MTP_SPECS.get(name))
        expected_type, expected_shape = expected or ("BF16", (5120,))
        q27_shape = shape or expected_shape
        self.shape = tuple(reversed(q27_shape))
        self.tensor_type = TensorType(dtype or expected_type)
        self.data = repack.np.frombuffer(data, dtype=repack.np.uint8)


class Reader:
    def __init__(self, name, block_count, tensors, nextn=None, arch="qwen35",
                 metadata=None, complete_metadata=True):
        architecture = (dict(repack.QWEN35_REQUIRED_METADATA)
                        if complete_metadata else {})
        architecture.update(metadata or {})
        values = {
            "general.architecture": arch,
            "general.name": name,
            "qwen35.block_count": block_count,
            **architecture,
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


def tensors_for(names):
    return [Tensor(name) for name in sorted(names)]


def companion_tensors(*, output=None, omit_shared=(), extras=()):
    shared = {
        "token_embd.weight": Tensor("token_embd.weight"),
        "output_norm.weight": Tensor("output_norm.weight"),
        "output.weight": output or Tensor("output.weight"),
    }
    return ([tensor for name, tensor in shared.items() if name not in omit_shared]
            + tensors_for(repack.QWEN35_MTP_TENSORS) + list(extras))


def main():
    assert len(repack.QWEN35_BASE_SPECS) == 851
    assert len(repack.QWEN35_MTP_SPECS) == 15
    assert repack.QWEN35_BASE_SPECS["token_embd.weight"] == (
        "BF16", (248320, 5120))
    assert repack.QWEN35_BASE_SPECS["blk.3.attn_q.weight"] == (
        "BF16", (12288, 5120))
    assert repack.QWEN35_MTP_SPECS["blk.64.nextn.eh_proj.weight"] == (
        "BF16", (5120, 10240))

    primary = Reader("Qwen3.8-27B", 64,
                     tensors_for(repack.QWEN35_BASE_TENSORS),
                     metadata={"qwen35.embedding_length": 5120})
    companion = Reader("Qwen3.8-27B", 65, companion_tensors(), nextn=1,
                       metadata={"qwen35.embedding_length": 5120})
    merged = repack.merge_mtp_tensors(primary, companion)
    assert len(merged) == 866
    assert {tensor.name for tensor in merged} == (
        repack.QWEN35_BASE_TENSORS | repack.QWEN35_MTP_TENSORS)

    expect_error(
        "qwen35 primary",
        Reader("Qwen3.8-27B", 64, tensors_for(repack.QWEN35_BASE_TENSORS),
               arch="llama"), companion)
    expect_error(
        "missing general.name",
        Reader(None, 64, tensors_for(repack.QWEN35_BASE_TENSORS),
               metadata={"qwen35.embedding_length": 5120}), companion)
    expect_error(
        "missing architecture metadata",
        Reader("Qwen3.8-27B", 64, tensors_for(repack.QWEN35_BASE_TENSORS),
               complete_metadata=False), companion)
    expect_error(
        "checkpoint mismatch",
        primary,
        Reader("Qwen3.6-27B", 65, companion_tensors(), nextn=1,
               metadata={"qwen35.embedding_length": 5120}),
    )
    expect_error(
        "architecture metadata mismatch",
        primary,
        Reader("Qwen3.8-27B", 65, companion_tensors(), nextn=1,
               metadata={"qwen35.embedding_length": 4096}),
    )
    expect_error(
        "architecture metadata mismatch",
        primary,
        Reader("Qwen3.8-27B", 65, companion_tensors(), nextn=1,
               metadata={"qwen35.embedding_length": 5120.0}),
    )
    # newer llama.cpp GGUFs: per-layer recurrent_layers, 64 + a trailing MTP 0
    iv = repack.QWEN35_REQUIRED_METADATA["qwen35.full_attention_interval"]
    rl64 = [0 if (i + 1) % iv == 0 else 1 for i in range(64)]
    rl_key = "qwen35.attention.recurrent_layers"
    def rl_pair(p_rl, c_rl):
        pm = {"qwen35.embedding_length": 5120}
        cm = {"qwen35.embedding_length": 5120}
        if p_rl is not None: pm[rl_key] = p_rl
        if c_rl is not None: cm[rl_key] = c_rl
        return (Reader("Qwen3.8-27B", 64, tensors_for(repack.QWEN35_BASE_TENSORS), metadata=pm),
                Reader("Qwen3.8-27B", 65, companion_tensors(), nextn=1, metadata=cm))
    assert len(repack.merge_mtp_tensors(*rl_pair(rl64, rl64 + [0]))) == 866
    expect_error("present in one view only", *rl_pair(rl64, None))
    expect_error("recurrent_layers", *rl_pair(rl64, rl64))          # no MTP entry
    expect_error("recurrent_layers", *rl_pair(rl64, rl64 + [1]))    # MTP marked recurrent
    expect_error("recurrent_layers", *rl_pair([1] * 64, [1] * 64 + [0]))  # contradicts the interval
    expect_error(
        "primary tensor manifest mismatch",
        Reader("Qwen3.8-27B", 64,
               tensors_for(repack.QWEN35_BASE_TENSORS - {"blk.0.ssm_out.weight"}),
               metadata={"qwen35.embedding_length": 5120}),
        companion,
    )
    malformed_primary = tensors_for(repack.QWEN35_BASE_TENSORS)
    malformed_primary = [
        Tensor(tensor.name, shape=(1,))
        if tensor.name == "blk.0.ssm_out.weight" else tensor
        for tensor in malformed_primary
    ]
    expect_error(
        "primary tensor spec mismatch",
        Reader("Qwen3.8-27B", 64, malformed_primary,
               metadata={"qwen35.embedding_length": 5120}),
        companion,
    )
    expect_error(
        "overlaps primary tensors",
        primary,
        Reader("Qwen3.8-27B", 65,
               companion_tensors(extras=[Tensor("blk.0.attn_norm.weight")]), nextn=1,
               metadata={"qwen35.embedding_length": 5120}),
    )
    expect_error(
        "missing shared tensors",
        primary,
        Reader("Qwen3.8-27B", 65,
               companion_tensors(omit_shared={"token_embd.weight"}), nextn=1,
               metadata={"qwen35.embedding_length": 5120}),
    )
    expect_error(
        "shape/type mismatch",
        primary,
        Reader("Qwen3.8-27B", 65,
               companion_tensors(output=Tensor("output.weight", shape=(4096,))), nextn=1,
               metadata={"qwen35.embedding_length": 5120}),
    )
    expect_error(
        "contents differ",
        primary,
        Reader("Qwen3.8-27B", 65,
               companion_tensors(output=Tensor("output.weight", data=b"\1")), nextn=1,
               metadata={"qwen35.embedding_length": 5120}),
    )
    expect_error(
        "MTP tensor manifest mismatch",
        primary,
        Reader("Qwen3.8-27B", 65,
               companion_tensors(extras=[Tensor("blk.64.unexpected.weight")]), nextn=1,
               metadata={"qwen35.embedding_length": 5120}),
    )

    print("split MTP GGUF merge contracts: PASS")


if __name__ == "__main__":
    main()
