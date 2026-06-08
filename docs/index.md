# post-training-lab

A controlled, reproducible comparison of LLM post-training methods —
**SFT, DPO, KTO, RLHF (reward model + PPO), and GRPO** — applied to the same
small base model (`Qwen/Qwen2.5-0.5B`, pretrained — *not* the `-Instruct`
variant) with the same evaluation harness.

The **comparison** is the contribution, not any single method. Same LoRA rank,
same quantization, same evaluation tasks, same JSON schema — apples-to-apples
by construction.

## Where to start

- **[Results](results.md)** — current numbers for base / sft_v2 / dpo_v1.
- **[Methods](methods/index.md)** — explainers for each post-training method,
  including paper links and the exact hyperparameters this repo uses.
- **[LoRA / QLoRA](training/lora.md)** — the parameter-efficient setup shared
  across every phase.

## Status

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Scaffolding + baseline eval | done |
| 1 | SFT on UltraChat-200k | done (sft_v2) |
| 2 | DPO on UltraFeedback-binarized | done (dpo_v1) |
| 3 | Reward model + PPO | next |
| 4 | GRPO / RLVR | planned |
| 5 | KTO or ORPO | planned |
| 6 | LLM-judge comparison + writeup | planned |

See [PROJECT.md](https://github.com/agaonker/post-training-lab/blob/main/PROJECT.md)
for the full charter and per-phase plan.
