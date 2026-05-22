# Supervised Fine-Tuning (SFT)

Phase 1 of the project. Train the base Qwen2.5-0.5B-Instruct to follow
instructions in a chat format by showing it ~5,000 high-quality user/assistant
conversations from UltraChat-200k and minimizing next-token cross-entropy.

Everything on the [LoRA / QLoRA](../training/lora.md) page applies — same r=16 adapters,
same 4-bit NF4 base, same target modules. This page only documents what's
**different** for SFT: the loss, the data, the training hparams, and the
success criterion.

## What SFT is

Given a dataset of prompt/response pairs (or full conversations), SFT
fine-tunes a language model by **maximum likelihood** on the responses:

\[
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{i} \log p_\theta(y_i \mid y_{<i}, x)
\]

where \(x\) is the prompt context and \(y_1, \ldots, y_T\) is the assistant
response (or, in the conversational case, the full rendered conversation
through the model's chat template). It's the same causal-LM objective the
model was pretrained on — just on a much smaller, much more curated
distribution of *instruction-following* text.

There's no single canonical "SFT paper" — the technique predates the LLM era
— but the modern post-training playbook traces to **InstructGPT** (Ouyang et
al. 2022), which used SFT as Step 1 before RLHF.

!!! note "Reference"
    **Training language models to follow instructions with human feedback** — Ouyang, Wu, Jiang, et al. (2022).
    [arxiv:2203.02155](https://arxiv.org/abs/2203.02155). The InstructGPT paper. SFT is "Step 1" in §3.2.

Why we run SFT first (and as a baseline for every later method):

- **Without SFT**, the base model continues text. With SFT, it follows
  instructions in a chat format. That's a prerequisite for DPO / PPO / GRPO to
  even be doing the right kind of comparison — those methods refine an
  already-instruct-able model.
- **With SFT alone**, we have a meaningful "is this preference-learning method
  actually helping?" baseline for Phase 2+. PROJECT.md §6 sets the win-rate-vs-SFT
  bar as the success criterion for DPO.

## Dataset: UltraChat-200k

We use [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k),
the filtered/deduped version of the original UltraChat corpus that the Zephyr
project popularized.

| Field | Value | Why |
|------|-------|-----|
| Split | `train_sft` | UltraChat-200k has four splits; `train_sft` is the SFT-formatted one. `train_gen` is reserved for generation-style training (unused here). |
| Slice | **5,000 rows** | PROJECT.md §4.1 starting point. Small enough that one Kaggle T4 finishes in ~15 min; large enough for SFT to move IFEval. |
| Seed | **42** | Shuffled with `cfg.seed` so the same YAML produces the same training slice. |
| Columns kept | `messages` only | Each row is a list of `{"role": "user" | "assistant", "content": str}` dicts. Other columns dropped at load time. |

UltraChat is also the parent corpus of **UltraFeedback-binarized**, which
Phase 2 (DPO) will use — so the **data lineage stays clean** across phases.
DPO won't be learning preferences on a totally different distribution from
what SFT taught.

!!! note "Paper"
    **Enhancing Chat Language Models by Scaling High-quality Instructional Conversations** — Ding, Chen, Xu, Qin, Liu, Hu, Sun, Zhou (2023).
    [arxiv:2305.14233](https://arxiv.org/abs/2305.14233). The original UltraChat paper.

### Chat template and loss masking

TRL's `SFTTrainer` applies the model's chat template (the Qwen2.5 ChatML
variant — `<|im_start|>` / `<|im_end|>` turn markers) to every conversation at
training time. After templating, the loss is standard next-token cross-entropy
over the rendered token sequence.

The current config does **not** set `assistant_only_loss: true`, so the model
sees gradient signal from both user *and* assistant turns. For UltraChat —
where user turns are short prompts and assistant turns are long, careful
responses — the practical difference vs assistant-only masking is small. If
Phase 1 results suggest the model is over-imitating user-style text, flipping
this on is a one-line YAML change in `cfg.train`.

## Our config

### Hyperparameters

From [`configs/sft_qwen05b.yaml`][sft-yaml] — these are the **SFT-specific**
knobs. Everything else (LoRA rank, quantization, target modules) is inherited
from `base.yaml` and covered on the [LoRA / QLoRA](../training/lora.md) page.

| Knob | Value | Why |
|------|-------|-----|
| `learning_rate` | **2.0e-4** | QLoRA paper recommendation for r=16. Higher than full fine-tuning would tolerate because LoRA's low-rank update bounds per-step weight change. |
| `num_train_epochs` | **1** | One pass over 5k rows. Phase 1 goal is "make IFEval move past base," not "saturate." More epochs risks overfitting to UltraChat style. |
| `per_device_train_batch_size` | **4** | T4 memory ceiling with 0.5B + 4-bit + LoRA + gradient checkpointing. |
| `gradient_accumulation_steps` | **4** | Effective batch = 16. Standard for instruction-tuning at this scale. |
| `warmup_ratio` | **0.03** | 3% of total steps. Short warmup because the run is short (~80 steps total) — longer warmup costs progress. |
| `gradient_checkpointing` | `true` | Required to fit batch=4 inside 16 GB. Trades ~30% compute for ~50% memory. |
| `logging_steps` | **10** | Roughly every ~12% of training — enough resolution for the loss curve to be readable without spamming W&B. |
| `save_strategy` | `"no"` | Single checkpoint saved at the end via `trainer.save_model`. No mid-run checkpoints for an ~80-step run — they'd be noise. |
| `report_to` | `wandb` | If `WANDB_API_KEY` is set, run logs go to the project. The Kaggle notebook auto-disables this when the secret isn't configured. |

### Step count math

```
5,000 samples / (batch_size × grad_accum) = 5,000 / (4 × 4) = 312.5 → ~80 steps per epoch
× 1 epoch                                                              = ~80 total steps
```

The smoke run (`make sft-smoke`) caps this at 50 steps with `--max-steps 50`
to catch wiring failures in ~5 minutes before the full run.

## Output

On success, the trained LoRA adapter is pushed to HF Hub at
`agaonker/atlas-sft-qwen05b-v1` (the `cfg.output.hub_repo` in
[`sft_qwen05b.yaml`][sft-yaml]). The base model is never modified; only the
~2.1 M LoRA params are saved.

Subsequent phases reload the base model and **attach this adapter** as their
starting point — DPO refines on top of these SFT weights, not on the raw
pretrained Qwen.

## Success criterion

From PROJECT.md §6:

> *SFT model beats base on IFEval (instruction-following) by a clear margin.*

Concretely, with the Phase 0 baseline numbers in hand:

- **IFEval prompt-strict > 0.25** (base: ~0.14)
- **IFEval inst-strict > 0.40** (base: ~0.27)

These thresholds are set to be **clear**, not just statistically significant —
SFT should be obviously better. If the trained model doesn't clear them, that's
a data or training-recipe problem, not a measurement-noise problem.

MMLU and GSM8K aren't expected to move much (both are knowledge / reasoning
benchmarks, and SFT on conversation data doesn't add either). TruthfulQA can
go *down* slightly — instruction-tuned models sometimes get less hedged in
their false-statement detection.

## Decisions worth understanding

### Why 5,000 samples (not the full 200k)

Two reasons:

1. **Compute budget.** The full UltraChat-200k at one epoch on a T4 takes
   ~12 hours of QLoRA training. 5k finishes in ~15 minutes. The marginal
   benefit beyond ~5k for a 0.5B model on instruction-following is small —
   the model's *capacity* is the bottleneck, not data quantity.
2. **Comparison fairness.** Phase 2 DPO uses ~5k preference pairs from
   UltraFeedback-binarized. Holding the data budget roughly equal across
   methods means we're comparing methods, not "SFT with 40× more data."

### Why UltraChat over OpenHermes / Tulu / Alpaca

UltraChat is **multi-turn** and **diverse-topic**. OpenHermes is
single-turn-heavy; Alpaca is short / synthetic-feeling. The Zephyr project's
ablations showed UltraChat-200k produces stronger chat models in this size
range. And — most importantly — it's the data source UltraFeedback-binarized
is derived from, so Phase 2 DPO can refine on the same distribution.

### Why one epoch

LoRA fine-tunes overfit quickly. Two epochs on 5k UltraChat with r=16 already
shows mild memorization on the train set. One epoch is the conservative
default; PROJECT.md treats it as the starting point with permission to scale
up if Phase 1 underperforms its IFEval bar.

### Why no assistant-only loss masking

The current YAML doesn't set `assistant_only_loss: true`. Modern SFT recipes
(Zephyr, OpenHermes) typically *do* mask user turns from the loss so the
model only learns to generate assistant text, not user text. For UltraChat
specifically — where assistant turns are 10–20× longer than user turns — the
practical difference is small. We're holding this as a tuning lever for
Phase 1.5 if results disappoint; flipping it on is one YAML line.

## References

- Ouyang, L., et al. (2022). **Training language models to follow instructions with human feedback** (InstructGPT). [arxiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Ding, N., et al. (2023). **Enhancing Chat Language Models by Scaling High-quality Instructional Conversations** (UltraChat). [arxiv:2305.14233](https://arxiv.org/abs/2305.14233)
- Tunstall, L., et al. (2023). **Zephyr: Direct Distillation of LM Alignment** — the recipe UltraChat-200k was packaged for. [arxiv:2310.16944](https://arxiv.org/abs/2310.16944)
- TRL `SFTTrainer` reference: [huggingface.co/docs/trl/sft_trainer](https://huggingface.co/docs/trl/main/sft_trainer)
- The training entrypoint in this repo: [`src/atlas/train/sft.py`][sft-py]

[sft-yaml]: https://github.com/agaonker/post-training-lab/blob/main/configs/sft_qwen05b.yaml
[sft-py]: https://github.com/agaonker/post-training-lab/blob/main/src/atlas/train/sft.py
