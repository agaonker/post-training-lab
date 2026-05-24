# post-training-lab

A controlled, reproducible comparison of modern LLM post-training methods — SFT, DPO, KTO,
full RLHF (reward model + PPO), and GRPO/RLVR — applied to the same small base model
(Qwen2.5-0.5B-Instruct) with the same evaluation harness.

📖 **Docs:** [agaonker.github.io/post-training-lab](https://agaonker.github.io/post-training-lab/) — method explainers, paper links, and hyperparameter reasoning.

See [`PROJECT.md`](PROJECT.md) for the full charter and phase plan.

> **Status:** Phase 0 (scaffolding) in progress. This README is expanded with the headline
> results table and writeup links as the project progresses.

## Quickstart

```bash
uv sync --extra dev --extra eval
```

## Results

All runs: `config_hash 6af9a640`, vLLM/bf16, Modal L4.
IFEval = `prompt_level_strict_acc,none`. GSM8K = `exact_match,strict-match`.

| Method | MMLU | GSM8K | TruthfulQA | IFEval (prompt strict) | Judge win-rate vs SFT |
|--------|------|-------|------------|------------------------|-----------------------|
| base | 0.4732 | 0.3404 | 0.4190 | 0.1885 | — |
| sft_v1 (UltraChat-200k, 5k steps) | 0.4595 | 0.3207 | 0.4073 | 0.1719 | — |

**Phase 1 finding:** SFT regressed on all metrics vs base, including IFEval (−1.7pp prompt-strict).
Likely cause: UltraChat chat-template drift and/or task-distribution mismatch. See [`experiments/002_sft_qwen05b.md`](experiments/002_sft_qwen05b.md).

## Writeups

One per phase, added as each phase completes — see [`writeups/`](writeups/).
