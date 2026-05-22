# Results

The cross-method comparison this project is built around. One row per training
run; every row uses the same base model, the same eval harness, and the same
JSON schema. Apples-to-apples by construction.

Live source: [`results/metrics.json`](https://github.com/agaonker/post-training-lab/blob/main/results/metrics.json).
The table below is **updated manually** when each phase lands — the workflow
that auto-renders it from `metrics.json` is a Phase 5 nicety, not yet built.

## Current state

| Method | Phase | MMLU | GSM8K (strict) | TruthfulQA | IFEval prompt-strict | IFEval inst-strict |
|--------|-------|------|----------------|------------|----------------------|--------------------|
| **base** | 0 | 0.474 | 0.265 | 0.418 | **0.201** | **0.347** |
| _sft_v1_ | 1 | _running_ | _running_ | _running_ | _running_ | _running_ |
| _dpo_v1_ | 2 | planned | planned | planned | planned | planned |
| _ppo_v1_ | 4 | planned | planned | planned | planned | planned |
| _grpo_v1_ | 5 | planned | planned | planned | planned | planned |

**Bold** column = Phase 1 success criterion (see [Baseline](foundation/baseline.md#phase-1-target)).
Italic rows = not yet evaluated.

## What's tracked beyond this table

| Signal | Where it lives | Purpose |
|--------|----------------|---------|
| Pairwise judge win rate vs SFT | `results/judge/*.json` (Phase 2+) | DPO's success criterion. Not a single number per run, but a win-rate over ~200 held-out prompts. |
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

- **Base model**: Qwen2.5-0.5B-Instruct, same revision.
- **LoRA setup**: r=16, α=32, same target modules ([LoRA / QLoRA page](training/lora.md)).
- **Quantization**: 4-bit NF4 + double quant.
- **Eval tasks**: the four in [Eval harness](foundation/eval-harness.md), same `num_fewshot` and `limit`.
- **Random seed**: 42 across data shuffling, weight init, eval sampling.

Only the **method-specific** knobs (loss, dataset format, RL hparams) vary
per row. That's the load-bearing fairness invariant of the project.
