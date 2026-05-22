# Foundation

The Phase 0 work that everything else builds on: the **base model** we
chose, the **evaluation harness** every method runs through, and the
**baseline numbers** that define what "better" means for the methods to come.

Holding all three constant across phases is what makes the cross-method
comparison fair — methods can't choose their own eval set, their own model
size, or their own definition of "good."

## Pages in this section

| Page | What it covers |
|------|----------------|
| [Base model](#base-model) | Why Qwen2.5-0.5B-Instruct (model size, license, instruction-tuned starting point) |
| [Eval harness](eval-harness.md) | The four tasks (MMLU / GSM8K / TruthfulQA / IFEval), the lm-eval-harness wrapper, and how metrics are persisted |
| [Baseline](baseline.md) | Phase 0 numbers on the un-tuned base model — the bar every method has to clear |

## Base model

**Qwen2.5-0.5B-Instruct** ([HF Hub](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct))
— a 494 M-parameter instruction-tuned decoder-only LM from Alibaba (Sep 2024).

### Why this model (not a 7B, not Llama, not Mistral)

| Constraint | Implication | Why Qwen2.5-0.5B-Instruct satisfies it |
|------------|-------------|----------------------------------------|
| **Free-tier compute** (Kaggle T4 16 GB) | Model must fit in ~16 GB *with* 4-bit base + LoRA + gradient checkpointing + batch=4 + optimizer states | 0.5 B base is ~1 GB in 4-bit; total run footprint ~6–8 GB |
| **Apache 2.0 license** | Adapters can be pushed to HF Hub publicly, results can be shared | Qwen2.5 family is Apache 2.0 |
| **Instruction-tuned starting point** | Phase 1 SFT and Phase 2 DPO improve an already-following model, not a base completion model — that's the realistic post-training setting | The `-Instruct` variant is already SFT'd by Alibaba |
| **Strong modern baseline** | If the base is too weak, every method "wins" trivially and the comparison is meaningless | Qwen2.5-0.5B-Instruct already scores ~47% on MMLU and ~35% on IFEval — solid headroom for SFT/DPO to improve on without trivializing the floor |
| **Small enough to iterate** | Phase 1 + 2 + 3 all need to fit in weekend-scale time budgets | Full SFT run finishes in 15 min on a T4 |

### Why not bigger / not different

- **Qwen2.5-1.5B-Instruct** is the [stretch base](https://github.com/agaonker/post-training-lab/blob/main/PROJECT.md#5-compute-plan)
  per PROJECT.md if Phase 1 lands cleanly. Same family means the comparison
  scales cleanly without a new license / tokenizer / template story.
- **Llama-3.2-1B-Instruct** would also have worked. Qwen was picked because
  its tokenizer and chat template are well-documented and TRL's defaults
  handle them out of the box.
- **TinyLlama / Pythia 1B** were considered and rejected — neither is
  instruction-tuned out of the box, which would have forced a Phase 0.5 of
  doing the instruction-tune ourselves before we could compare *post*-training
  methods.

### Pinning

`configs/base.yaml` currently sets `revision: null` (latest HEAD on HF Hub).
Per PROJECT.md, once Phase 1 results are locked in, this should be pinned to
the SHA seen at training time so subsequent reruns are bit-identical. That
hasn't happened yet because we're still iterating on the SFT recipe.
