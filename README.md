# post-training-lab

Controlled comparison of LLM post-training methods (SFT, DPO, KTO, RLHF/PPO, GRPO) on
pretrained `Qwen/Qwen2.5-0.5B` with one eval harness. Scope, phase plan, datasets, compute
budget, anti-goals — see [`PROJECT.md`](PROJECT.md).

📖 **Docs:** [agaonker.github.io/post-training-lab](https://agaonker.github.io/post-training-lab/)

> **Status:** Phase 2 (DPO) closed. Phase 3 (Reward Model + PPO) next. Three methods rows in
> the comparison table below; each phase adds one. The README is updated as each phase lands.

## Quickstart

```bash
uv sync --extra dev --extra eval
```

## Results

All runs: vLLM/bf16, Modal L4. IFEval = `prompt_level_strict_acc,none`. GSM8K =
`exact_match,strict-match`. Each row carries its own `config_hash` — same fingerprint
means apples-to-apples; different fingerprints mean the model / tokenizer / adapter
config differs (deliberately).

| Method | config_hash | MMLU | GSM8K | TruthfulQA | IFEval (prompt strict) | Judge win-rate vs SFT |
|---|---|---:|---:|---:|---:|---:|
| **base** (pretrained Qwen2.5-0.5B) | `fde0720e` | 0.4813 | 0.3389 | 0.3988 | 0.1238 | — |
| **sft_v2** (UltraChat-200k, 5k, QLoRA r=16, 1 epoch) | `b133712d` | 0.4713 | 0.3450 | 0.3893 | 0.1201 | — |
| **dpo_v1** (UltraFeedback, 5k pairs, β=0.1, 1 epoch) | `d53fd258` | **0.4802** | **0.3495** | **0.3958** | **0.1275** | *Phase 6* |

### Deltas

| Metric | sft_v2 − base | dpo_v1 − sft_v2 | dpo_v1 − base |
|---|---:|---:|---:|
| MMLU | −1.00pp | **+0.89pp** | −0.11pp |
| GSM8K strict | +0.61pp | **+0.45pp** | +1.06pp |
| TruthfulQA | −0.95pp | **+0.65pp** | −0.30pp |
| **IFEval prompt-strict** | −0.37pp (flat) | **+0.74pp** | **+0.37pp** |
| IFEval inst-strict | +1.20pp | +0.24pp | +1.44pp |

**The headline finding so far:** SFT on a 0.5B with 5k UltraChat rows was *flat* on the
hardest IFEval metric. DPO on top of that SFT moved every lm-eval metric in the right
direction and put `dpo_v1` slightly above the pretrained base on IFEval prompt-strict —
the first policy in this lab to clear that bar. The Phase 2 *formal* success criterion
(LLM-judge pairwise win-rate) is gated on Phase 6 and not yet measured. See
[`experiments/003_dpo_qwen05b.md`](experiments/003_dpo_qwen05b.md).

## Learnings

[`LESSONS.md`](LESSONS.md) — terse one-liners with dates, grouped by category
(base model, tokenizer, training framework, HF Hub, wandb, process).
