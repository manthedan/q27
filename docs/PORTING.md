# Porting q27 to a new Qwen checkpoint

q27 is not a general engine. It implements one architecture, and most of that
architecture is fixed at compile time. A new checkpoint either matches those
constants or it does not, and you can tell which from its `config.json` before
downloading a single shard.

This file is the checklist for that decision: what has to match, what is
already data-driven, what a mismatch actually costs, and how to run the check
against a Hub config while the weights are still embargoed.

The reference point throughout is `Qwen/Qwen3.6-27B`, the checkpoint the
engine was built against.

## What is fixed at compile time

**There are TWO constant tables and a port has to edit both.** The CUDA engine
declares them at `src/engine.cuh:38-46`; `MetalEngine` declares its own private
copy at `src/metal/metal_engine.h`. They agree exactly today, all fourteen
values, and nothing enforces that they keep agreeing: each is checked against
the artifact independently, so a port that updates one and forgets the other
gets a working backend and a backend that refuses to load, not a compile error.

Every constant maps to one field of the Hub config's `text_config` subtree. If
all of them match, a new checkpoint is a repack job (`tools/repack.py`,
container spec in `docs/FORMAT.md`) and not an engine job.

| constant | value | `text_config` field |
|---|--:|---|
| `N_LAYER` | 64 | `num_hidden_layers` |
| `N_EMBD` | 5120 | `hidden_size` |
| `N_FFN` | 17408 | `intermediate_size` |
| `N_HEAD` | 24 | `num_attention_heads` |
| `N_KV` | 4 | `num_key_value_heads` |
| `HEAD_DIM` | 256 | `head_dim` |
| `N_ROT` | 64 | `head_dim` x `partial_rotary_factor` (0.25) |
| `FREQ_BASE` | 1e7 | `rope_parameters.rope_theta` |
| `EPS` | 1e-6 | `rms_norm_eps` |
| `GDN_HEADS` | 48 | `linear_num_value_heads` |
| `GDN_DIM` | 128 | `linear_value_head_dim` |
| `GDN_V` | 6144 | `linear_num_value_heads` x `linear_value_head_dim` |
| `GDN_CH` | 10240 | (`linear_num_key_heads` x 2 + `linear_num_value_heads`) x 128 |
| `VOCAB` | 248320 | `vocab_size` |

`GDN_CH` is the packed q/k/v channel width the causal conv runs over: q and k
each take `linear_num_key_heads` (16) heads and v takes
`linear_num_value_heads` (48), all at dim 128, so (16 + 16 + 48) x 128 =
10240. The conv ring is allocated at `3 * GDN_CH` (`engine.cu:996`), which is
`linear_conv_kernel_dim - 1` history slots; a kernel dim other than 4 changes
that factor.

`N_ROT` is the only derived one. The model uses partial rotary embedding, so
q27 rotates the first 64 of each 256-wide head and passes the rest through
(`rope_neox_partial`, `engine.cuh:1076`). The config also carries mrope
(`mrope_interleaved`, `mrope_section [11, 11, 10]`, summing to the 32 rotary
pairs), which only bites for multimodal position ids. q27 implements the text
stack, so the sections collapse to ordinary rope and are not read.

Metal declares two names CUDA does not. `GDN_QK_HEADS` (16) is
`linear_num_key_heads` stated outright, where the CUDA side only carries it
folded into `GDN_CH`, so a change to that field is a one-line edit on Metal and
an arithmetic one on CUDA. `CHUNK_MAX` is a prefill batching parameter, not
architecture, and no config field corresponds to it.

**Both engines now refuse a mismatch rather than running at the wrong shape.**
`validate_arch()` (`engine.cuh`, called before the weight upload) and
`validate_architecture()` (`src/metal/metal_engine.cpp`, called before
allocation) each check the artifact metadata against their own table and abort
naming the offending key. So the failure mode for a wrong checkpoint is a clear
message at startup, not silently wrong numbers. Neither validator can tell you
a checkpoint will work; they only tell you it will not.

The Metal validator is the stricter of the two, because it is a q4s-only engine
slice: it additionally requires `quant_policy == "q4s-v1"`, the full tensor
name/dtype/shape table, and the attention layout discussed below. The CUDA
validator is deliberately tier-agnostic, since `quant_policy`, `q4_head` and
`q8_extra` differ across the seven published tiers and say nothing about the
graph.

## What is already data-driven

Do not add these to the checklist. They flow through without a code change.

**Which layers are full attention, on the CUDA path.** `tools/repack.py:136-145`
derives `attn_layers` by inspecting tensor names, writes it into the `.q27`
metadata, and `engine.cuh:684-691` reads it back into `attn_layer[]`. Every
consumer asks `is_attn_layer(il)` (`engine.cuh:2453`) rather than computing an
interval. A checkpoint with a different `full_attention_interval`, or an
irregular layout that no interval describes, needs no CUDA engine edit -- the
KV allocation and the layer dispatch both loop over the flag.

**The Metal engine does NOT share this property.** `MetalEngine` computes the
layout instead of reading it: `attention_layer(l)` returns `l % 4 == 3` and
`gdn_slot()` is derived from the same rule. That is correct for every current
checkpoint and it fails closed rather than silently -- `validate_architecture()`
asserts `qwen35.full_attention_interval == 4` and walks the whole `attn_layers`
array against the generated expectation, so a different layout throws at
construction. But it means a checkpoint that moves the interval **loads on CUDA
and refuses on Metal**, and porting it means editing the Metal engine even
though the CUDA constant table above is untouched. Add the interval back to the
checklist the moment you care about the Metal backend.

**Weight quantization tier.** The repack recipe is per-tensor and lives
outside the engine (`docs/FORMAT.md`).

## Constraints that are not single constants

A checkpoint can match the table above and still not run.

- **Decode `cols` must be a multiple of `VG_KB` = 256.** `vgemm.cuh:69`
  documents it and `vgemm.cu:337` aborts rather than corrupt. The current
  shapes are 5120 (20x), 6144 (24x) and 17408 (68x).
- **The KV formats tile 128-element groups.** `QK_TURBO3` and `QK_TURBO5` are
  both 128 (`turbo3.cuh:22`, `turbo5.cuh:33`), and the WHT group size is the
  block size. A `head_dim` that is not a multiple of 128 breaks turbo3,
  turbo5 and their fp8 siblings, not just one of them.
- **The attention kernels are built at `HEAD_DIM` 256.** fd2, fdmma and H16
  all assume it in their tiling.
- **The MTP head is one layer.** `mtp_num_hidden_layers` is 1, and the draft
  ladder (`D_MAX_MTP`, `docs/plans/2026-07-13-mtp-draft-head.md`) is built on
  that shape. A checkpoint with no MTP head still decodes, but loses the
  speculative path that most of q27's margin comes from.

## Triage

**Cheap -- edit the constant in BOTH tables, repack, re-gate.** `N_LAYER`,
`VOCAB`, and `N_EMBD` / `N_FFN` changes that stay multiples of 256. Re-derive
the canonical md5 afterwards; it is checkpoint-specific by construction, so a
new checkpoint gets a new canonical rather than failing the old one
(`tools/sampling_gate.sh` header). Cheap does not mean one edit: the CUDA and
Metal tables are separate declarations and the Metal one is only exercised on
macOS, so a CUDA-only change looks complete from Linux and fails on the first
Mac that loads the artifact.

**Expensive.** `head_dim` off 256 (attention kernels plus both KV formats),
`linear_value_head_dim` off 128 (the GDN block in `blocks.cu`), or a
`linear_conv_kernel_dim` change (conv ring sizing).

**A different engine.** MoE. There is no expert routing anywhere in q27, and
adding it is not a port.

## Checking a config before the weights land

The generation's config usually appears with the repo, so both of these are
runnable the moment a Hub page exists, before any weights download.

**`tools/check_checkpoint.py` runs this whole checklist.** It reads the
constants out of `src/engine.cuh` and `src/metal/metal_engine.h` rather than
restating them, so it cannot drift from the engines, and it cross-checks the
two tables against each other on every run:

```
tools/check_checkpoint.py Qwen/Qwen3.8-27B     # Hub repo id, or a local config.json
tools/check_checkpoint.py --tables-only        # just CUDA vs Metal
```

Exit 0 is a repack job, 1 is an engine change with the offending rows named, 2
is a config it could not read. It also reports the shape constraints below and
flags a mixture of experts explicitly. A pass means the constants agree; it
does not promise the checkpoint behaves the same.

The snippet below is the other half: it diffs any two checkpoints against each
other, which is what tells you what a generation actually changed. It collapses
`layer_types` to a count so the 64-entry list does not bury the real diffs:

```python
import json, urllib.request
def get(m):
    return json.load(urllib.request.urlopen(
        f"https://huggingface.co/{m}/raw/main/config.json", timeout=30))
def flat(d, p=""):
    o = {}
    for k, v in d.items():
        key = p + k
        if isinstance(v, dict):
            o.update(flat(v, key + "."))
        elif isinstance(v, list) and k == "layer_types":
            o[key] = f"<{len(v)}: {v.count('full_attention')} full / {v.count('linear_attention')} linear>"
        else:
            o[key] = v
    return o
a, b = flat(get("Qwen/Qwen3.6-27B")), flat(get("Qwen/<new>"))
for k in sorted(set(a) | set(b)):
    if a.get(k, "--") != b.get(k, "--"):
        print(f"{k:<46} {str(a.get(k,'--'))[:27]:<28} {b.get(k,'--')}")
```

## Recon log

### Qwen3.8-27B release

The released `Qwen/Qwen3.8-27B` checkpoint is a repack, not an engine port.
`tools/check_checkpoint.py Qwen/Qwen3.8-27B` passes every CUDA and Metal
constant plus the non-constant shape gates. The config still declares
`Qwen3_5ForConditionalGeneration` / `qwen3_5`; compared with
`Qwen/Qwen3.6-27B`, the only flattened config difference is the recorded
`transformers_version`.

The Hugging Face safetensor indexes have the same 1,199 tensor names and the
same aggregate byte count. The released GGUF layout is different, however:
`ggml-org/Qwen3.8-27B-GGUF` publishes the 64 base blocks and the MTP block as
separate standalone files rather than GGUF split shards. The base BF16 has 851
tensors and `qwen35.block_count=64`; the MTP BF16 has 18 tensors and
`qwen35.block_count=65`. Three MTP-file tensors duplicate the base embedding,
normalization, and output head; the other 15 are `blk.64.*`.

`tools/repack.py --mtp` joins those views, rejects mixed checkpoints and
unexpected overlaps, keeps the primary copy of the three shared tensors, and
takes the MTP architecture metadata so the output validates as 65 blocks:

```sh
tools/repack.py Qwen3.8-27B-BF16.gguf qwen38-27b-mtp-q4s.q27 \
  --mtp mtp-Qwen3.8-27B-BF16.gguf --q4-head --tag q4s-v1
tools/export_tokenizer.py Qwen3.8-27B-BF16.gguf qwen38-27b-mtp.tok
```

The vision encoder remains out of scope. It is published as a separate
`mmproj` GGUF and does not change the text graph q27 implements. Qwen3.8 adds
chat-template controls such as `reasoning_effort`; the token inventory and
ChatML control IDs used by q27 remain compatible, while those optional template
features are not exposed by q27's hand-written prompt renderer.

No runtime or kernel source changes are required. The remaining work for any
new Qwen3.8 artifact is checkpoint-specific: repack it, export its tokenizer,
derive a new canonical, and run the normal numerical and serving gates rather
than reusing Qwen3.6 quality claims.
