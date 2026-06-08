# Results

The cross-method comparison this project is built around. One row per training
run; every row uses the same base model, the same eval harness, and the same
JSON schema. Apples-to-apples by construction.

Live source: [`results/metrics.json`](https://github.com/agaonker/post-training-lab/blob/main/results/metrics.json).
The table below is **updated manually** when each phase lands — the workflow
that auto-renders it from `metrics.json` is a Phase 6 nicety, not yet built.

## Current state

| Method | Phase | config_hash | MMLU | GSM8K (strict) | TruthfulQA | IFEval prompt-strict | IFEval inst-strict |
|---|---|---|---:|---:|---:|---:|---:|
| **base** (pretrained `Qwen/Qwen2.5-0.5B`) | 0 | `fde0720e` | 0.4813 | 0.3389 | 0.3988 | 0.1238 | 0.2278 |
| **sft_v2** (UltraChat-200k, 5k, QLoRA r=16, 1 epoch) | 1 | `b133712d` | 0.4713 | 0.3450 | 0.3893 | 0.1201 | 0.2398 |
| **dpo_v1** (UltraFeedback, 5k pairs, β=0.1, 1 epoch) | 2 | `d53fd258` | **0.4802** | **0.3495** | **0.3958** | **0.1275** | **0.2422** |
| _rm_v1_ | 3 | planned | — | — | — | — | — |
| _ppo_v1_ | 3 | planned | — | — | — | — | — |
| _grpo_v1_ | 4 | planned | — | — | — | — | — |
| _kto_v1 or orpo_v1_ | 5 | planned | — | — | — | — | — |

### Deltas

| Metric | sft_v2 − base | dpo_v1 − sft_v2 | dpo_v1 − base |
|---|---:|---:|---:|
| MMLU | −1.00pp | **+0.89pp** | −0.11pp |
| GSM8K strict | +0.61pp | **+0.45pp** | +1.06pp |
| TruthfulQA | −0.95pp | **+0.65pp** | −0.30pp |
| **IFEval prompt-strict** | −0.37pp (flat) | **+0.74pp** | **+0.37pp** |
| IFEval inst-strict | +1.20pp | +0.24pp | +1.44pp |

**The headline finding:** SFT on a 0.5B with 5k UltraChat rows was *flat* on
the headline IFEval prompt-strict metric. DPO on top of that SFT moved every
lm-eval metric in the right direction and put `dpo_v1` slightly above the
pretrained base on IFEval prompt-strict — the first policy in this lab to
clear that bar. Full Phase 1 story in
[`experiments/002_sft_qwen05b.md`](https://github.com/agaonker/post-training-lab/blob/main/experiments/002_sft_qwen05b.md);
Phase 2 in
[`experiments/003_dpo_qwen05b.md`](https://github.com/agaonker/post-training-lab/blob/main/experiments/003_dpo_qwen05b.md).

## What's tracked beyond this table

| Signal | Where it lives | Purpose |
|--------|----------------|---------|
| Pairwise judge win rate vs SFT | `results/judge/*.json` (Phase 6) | DPO's *formal* success criterion per PROJECT.md §6. Not yet measured. |
| KL divergence from reference | W&B run logs | PPO sanity — verifies the policy isn't drifting wildly from SFT. |
| Reward model accuracy on held-out pairs | `results/rm/*.json` (Phase 3+) | Reward model isn't load-bearing if it can't distinguish preferred from rejected. |
| Per-task `stderr` from lm-eval | `metrics.json` (dropped from flattened keys, present in raw) | When two runs differ by less than `2 * stderr`, the difference isn't meaningful. |

## How to interpret a comparison

A method "winning" is **not** "highest IFEval." It's:

1. **Clear IFEval improvement over base** (the floor everyone has to clear).
2. **No catastrophic regression** on MMLU or TruthfulQA. If a method bumps
   IFEval by 5 points but drops MMLU by 8, that's not a win — it's a forgetting
   pathology.
3. **Reproducible.** Same `config_hash`, same data revision, same seed →
   same metrics within noise. PROJECT.md treats irreproducible deltas as
   nonexistent.

## How the comparison stays fair

Five things are held constant across every row:

- **Base model**: pretrained `Qwen/Qwen2.5-0.5B`, same revision.
- **LoRA setup**: r=16, α=32, same target modules ([LoRA / QLoRA page](training/lora.md)).
- **Quantization**: 4-bit NF4 + double quant for SFT (Phase 1); bf16 for
  DPO (Phase 2 disables quant to allow `merge_and_unload` of the SFT adapter).
- **Eval tasks**: the four in [Eval harness](foundation/eval-harness.md), same `num_fewshot` and `limit`.
- **Random seed**: 42 across data shuffling, weight init, eval sampling.

Only the **method-specific** knobs (loss, dataset format, RL hparams) vary
per row. That's the load-bearing fairness invariant of the project.
