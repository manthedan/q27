"""Register the pinned Prism GGUF codecs without aliasing incompatible layouts.

Legacy q27 inputs use Prism's old type 42/g128 (not today's Q2_0/g64).
Use tools/requirements-repack-test.txt; a codec collision fails closed.
PQ2_0's private id 142 is from Prism llama.cpp 1a07bfa5f414.
"""
import gguf.constants as _ggc
from gguf import GGUFReader


def register_type(name, value, block, size):
    enum = _ggc.GGMLQuantizationType
    if value in enum._value2member_map_:
        member = enum(value)
        if member.name != name or _ggc.GGML_QUANT_SIZES.get(member) != (block, size):
            raise ValueError(f"incompatible gguf codec {value}: expected {name} "
                             f"g{block}/{size} bytes; use requirements-repack-test.txt")
        return member
    member = int.__new__(enum, value)
    member._name_, member._value_ = name, value
    enum._member_map_[name] = member
    enum._value2member_map_[value] = member
    _ggc.GGML_QUANT_SIZES[member] = (block, size)
    return member


register_type("Q2_0", 42, 128, 34)
register_type("Q1_0", 41, 128, 18)
register_type("PQ2_0", 142, 128, 34)
# Parsing is permitted so converters can give a useful explicit rejection.
register_type("PTQ1_0", 143, 128, 28)
