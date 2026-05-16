# post-training-lab

A controlled, reproducible comparison of modern LLM post-training methods — SFT, DPO, KTO,
full RLHF (reward model + PPO), and GRPO/RLVR — applied to the same small base model
(Qwen2.5-0.5B-Instruct) with the same evaluation harness.

See [`PROJECT.md`](PROJECT.md) for the full charter and phase plan.

> **Status:** Phase 0 (scaffolding) in progress. This README is expanded with the headline
> results table and writeup links as the project progresses.

## Quickstart

```bash
uv sync --extra dev --extra eval
```

## Results

_Populated from `results/metrics.json` once the baseline eval and training runs land._

| Method | MMLU | GSM8K | TruthfulQA | IFEval | Judge win-rate vs SFT |
|--------|------|-------|------------|--------|-----------------------|
| _base_ | — | — | — | — | — |

## Writeups

One per phase, added as each phase completes — see [`writeups/`](writeups/).
