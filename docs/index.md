# post-training-lab

A controlled, reproducible comparison of LLM post-training methods —
**SFT, DPO, KTO, RLHF (reward model + PPO), and GRPO** — applied to the same
small base model (Qwen2.5-0.5B-Instruct) with the same evaluation harness.

The **comparison** is the contribution, not any single method. Same LoRA rank,
same quantization, same evaluation tasks, same JSON schema — apples-to-apples
by construction.

## Where to start

- **[Methods](methods/index.md)** — explainers for each post-training method,
  including paper links and the exact hyperparameters this repo uses.
- **[LoRA / QLoRA](training/lora.md)** — the parameter-efficient setup shared
  across every phase.

## Status

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Scaffolding + baseline eval | in progress |
| 1 | SFT on UltraChat-200k | in progress |
| 2 | DPO | planned |
| 3 | KTO / ORPO | planned |
| 4 | Reward model + PPO | planned |
| 5 | GRPO / RLVR | planned |

See [PROJECT.md](https://github.com/agaonker/post-training-lab/blob/main/PROJECT.md)
for the full charter and per-phase plan.
