# Methods

One page per post-training method, structured the same way: paper link, what
the method is, what this repo's specific config does, and the reasoning behind
each hyperparameter.

| Method | Phase | Page | Status |
|--------|-------|------|--------|
| LoRA / QLoRA | all | [LoRA / QLoRA](lora.md) | written |
| SFT | 1 | _coming_ | in progress |
| DPO | 2 | _coming_ | planned |
| KTO / ORPO | 3 | _coming_ | planned |
| Reward model + PPO | 4 | _coming_ | planned |
| GRPO / RLVR | 5 | _coming_ | planned |

LoRA / QLoRA is shared across every phase — same rank, same target modules,
same quantization. Holding that constant is the load-bearing assumption behind
the cross-method comparison; per-method pages will only document the bits that
differ (loss, dataset format, RL-specific knobs).
