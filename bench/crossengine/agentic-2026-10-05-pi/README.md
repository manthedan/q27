# Qwen3.8-27B-pi on the agentic campaign (2026-10-05)

[bytkim/Qwen3.8-27B-pi](https://huggingface.co/bytkim/Qwen3.8-27B-pi) is a
Qwen3.8-27B fine-tune (SFT on successful Pi agent-harness coding sessions,
then GRPO with a reasoning-efficiency reward; Apache-2.0). Repacked from its
BF16 GGUF plus its own MTP GGUF with the base default recipe
(`--q8 '(attn_output)\.' --tag v2.0 --name "Qwen38 27b Pi"`), 16.996 GB,
wsum `e18697210b412d64` on two loads. Tokenizer byte-identical to base
(`.tok` md5 bb95b3ca); chat template byte-identical.

Three legs of `campaign.sh` (a copy of ../agentic-2026-09-07's on the
q27-master tree, master 44b6b20 build), same afternoon, 12 pinned SWE-bench
instances under Claude Code, medium effort, KV fp8, fresh prefix-cache root
per leg, `Q27_SEED=random` on the seed legs:

| leg | model | drafter | gold | turns/inst | think K/inst | out tok/inst | wall s/inst | agg t/s | tok/round | reuse |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| q27seed | base 3.8 default | DFlash2 Q8 | 11/12 | 21.8 | 32.8 | 12250 | 75 | 226.8 | 4.14 | 96.4% |
| **piseed** | Pi | DFlash2 Q8 (base-trained) | 11/12 | 18.7 | 22.3 | 8593 | **53** | 242.2 | 4.36 | 96.3% |
| piladr | Pi | Pi's own MTP head, ladder | 11/12 | 22.5 | 24.0 | 9756 | 73 | 187.9 | 3.56 | 95.7% |

Reading: Pi lands the same gold count with 30% less wall per instance on
the production drafter: 30% fewer output tokens (8.6K vs 12.3K), 32% less
thinking (22K vs 33K chars), 14% fewer turns, and 7% faster decode. The DFlash2
drafter was trained on base Qwen3.8 and still accepts MORE on Pi (4.36 vs
4.14 tokens/round), probably because shorter, more decisive text is easier
to draft. Pi's own MTP head on the ladder is the slower path (3.56
tok/round, 188 t/s; CLI salad prompt 3.52 vs base's 3.03), same ordering as
every 3.8 leg: DFlash2 wins single-slot.

Scope: one sampled trial per instance per leg; per-instance turns swing a
lot between legs (xarray-4094 75/59/49, pylint-4970 25/51/38), so the
12-instance means carry the result, not any row. gold = the patch touched a
gold file (cheap proxy, no test run). Effort medium (Pi's card says medium
matches base xhigh on its own harness; not tested here).

Files: `results.<leg>.jsonl`, `<leg>.log`, `swebench_<leg>.journal`,
`turns_cmp.txt` (per-instance table, ../agentic-2026-09-09-echo/turns_cmp.py
with this dir first in its search path). Request bodies:
/mnt/ai/data/reqbody/2026-10-05-pi.* (session content, out of the repo).
