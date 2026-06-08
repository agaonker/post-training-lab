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
| [Base model](#base-model) | Why pretrained `Qwen/Qwen2.5-0.5B` (model size, license, pretrained-not-aligned) |
| [Eval harness](eval-harness.md) | The four tasks (MMLU / GSM8K / TruthfulQA / IFEval), the lm-eval-harness wrapper, and how metrics are persisted |
| [Baseline](baseline.md) | Phase 0 numbers on the un-tuned base model — the bar every method has to clear |

## Base model

**`Qwen/Qwen2.5-0.5B`** ([HF Hub](https://huggingface.co/Qwen/Qwen2.5-0.5B))
— a 494 M-parameter **pretrained** decoder-only LM from Alibaba (Sep 2024).
**Not** the `-Instruct` variant — see the rationale below.

### Why this model (not a 7B, not Llama, not Mistral)

| Constraint | Implication | Why `Qwen/Qwen2.5-0.5B` satisfies it |
|------------|-------------|--------------------------------------|
| **Free-tier compute** (Kaggle T4 16 GB) | Model must fit in ~16 GB *with* 4-bit base + LoRA + gradient checkpointing + batch=4 + optimizer states | 0.5 B base is ~1 GB in 4-bit; total run footprint ~6–8 GB |
| **Apache 2.0 license** | Adapters can be pushed to HF Hub publicly, results can be shared | Qwen2.5 family is Apache 2.0 |
| **Pretrained, not aligned** | SFT, DPO, PPO, GRPO each need *room to actually move* — re-SFT-ing already-aligned weights produces flat or regressing evals | The pretrained variant has not been instruction-tuned by Alibaba; each method has clear headroom |
| **Comparable to known SFT recipe** | Qwen's own `-Instruct` is essentially this base + SFT — gives a natural ceiling for what supervised tuning recovers | The `-Instruct` numbers sit in `metrics.json` as a reference; method results are compared *to the pretrained base* but *against the Instruct ceiling* |
| **Small enough to iterate** | Phase 1 + 2 + 3 all need to fit in weekend-scale time budgets | Full SFT run finishes in ~30 min on a Modal L4 |

### Why we switched away from `-Instruct`

The first SFT attempt used `Qwen2.5-0.5B-Instruct` as the base. That regressed
uniformly on every eval (−1 to −2.6pp). Re-SFT-ing already-aligned weights had
nothing to do; the comparison-repo lever needs an *un*aligned starting point so
each method can demonstrate its own contribution. See the
[README Learnings section](https://github.com/agaonker/post-training-lab#learnings)
and [`experiments/002`](https://github.com/agaonker/post-training-lab/blob/main/experiments/002_sft_qwen05b.md)
for the full story.

### Why not bigger / not different

- **Qwen2.5-1.5B** is the stretch base per PROJECT.md §7 if the Phase 0–6
  comparison lands cleanly. Same family means the comparison scales without a
  new license / tokenizer / template story.
- **Llama-3.2-1B** would also have worked. Qwen was picked because its
  tokenizer and chat template are well-documented and TRL's defaults handle
  them — with one patch — out of the box.
- **TinyLlama / Pythia 1B** were considered. They satisfy the "pretrained, not
  aligned" criterion but lack the strong public Instruct comparator that Qwen
  ships, so we'd lose the "vs Qwen's own SFT recipe" ceiling.

### Tokenizer routing

Pretrained `Qwen2.5-0.5B` has `pad == eos == <|endoftext|>` (an SFT
supervision footgun — TRL masks pad in the labels, so the model never sees
eos in supervision) and a chat template without `{% generation %}` markers
(required for TRL's assistant-only loss mask). Both are fixed by loading the
**`-Instruct` tokenizer** instead — the vocab is byte-identical, but
`pad=<|endoftext|>`, `eos=<|im_end|>` (correctly distinct), and the chat
template is patched at load time to inject `{% generation %}` markers. See
`src/atlas/models/base.py:patch_chat_template_for_assistant_mask`.

### Pinning

`configs/base.yaml` currently sets `revision: null` (latest HEAD on HF Hub).
Per PROJECT.md, once the Phase 0–2 results are locked in for the final
write-up, this should be pinned to the SHA seen at eval time so subsequent
reruns are bit-identical.
