# Methods

One page per post-training method, structured the same way: paper link, what
the method is, what this repo's specific config does, and the reasoning behind
each hyperparameter.

| Method | Phase | Page | Status |
|--------|-------|------|--------|
| LoRA / QLoRA | all | [LoRA / QLoRA](lora.md) | written |
| SFT | 1 | [SFT](sft.md) | written |
| DPO | 2 | _coming_ | planned |
| KTO / ORPO | 3 | _coming_ | planned |
| Reward model + PPO | 4 | _coming_ | planned |
| GRPO / RLVR | 5 | _coming_ | planned |

LoRA / QLoRA is shared across every phase — same rank, same target modules,
same quantization. Holding that constant is the load-bearing assumption behind
the cross-method comparison; per-method pages only document the bits that
differ (loss, dataset format, RL-specific knobs).

## How to read the pages

Every method page follows the same shape:

1. **What it is** — the math and the canonical paper.
2. **Dataset / inputs** — what the method consumes and why we picked it.
3. **Our config** — hyperparameters specific to this method (the shared LoRA
   block stays on the [LoRA / QLoRA](lora.md) page).
4. **Success criterion** — the bar this phase has to clear, per PROJECT.md §6.
5. **Decisions worth understanding** — non-obvious choices and when to revisit
   them.
