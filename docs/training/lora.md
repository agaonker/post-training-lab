# LoRA and QLoRA

This page covers the parameter-efficient fine-tuning method this repo uses for
**every** post-training phase: **QLoRA** (LoRA + 4-bit base). The `lora:` block
in [`configs/base.yaml`][base-yaml] is **held constant** across SFT, DPO, KTO,
the reward model, PPO, and GRPO — the comparison across methods is the
contribution, so adapter capacity (rank, target modules, dropout) doesn't vary.

The `quant:` block has **one method-specific exception**: DPO (Phase 2)
disables 4-bit because it merges the SFT adapter into the base via
`merge_and_unload` before attaching the DPO LoRA — and `merge_and_unload`
needs full-precision weights. The merge-then-DPO recipe lives in
[`src/atlas/train/dpo.py`](https://github.com/agaonker/post-training-lab/blob/main/src/atlas/train/dpo.py);
the Phase 2 numbers are in
[`experiments/003`](https://github.com/agaonker/post-training-lab/blob/main/experiments/003_dpo_qwen05b.md).

## What LoRA is

**LoRA** (Low-Rank Adaptation) freezes the pretrained weights
\(W \in \mathbb{R}^{d \times k}\) and learns a low-rank update:

\[
W' = W + \Delta W, \qquad \Delta W = B A, \qquad A \in \mathbb{R}^{r \times k}, \quad B \in \mathbb{R}^{d \times r}
\]

with \(r \ll \min(d, k)\). Only \(A\) and \(B\) are trained; the forward pass
adds the term \(\frac{\alpha}{r} \cdot B A x\) to the frozen layer's output.
Hu et al. (2021) showed this matches full fine-tuning quality at a fraction
of the trainable-parameter count, because gradient updates during fine-tuning
empirically have low intrinsic rank.

!!! note "Paper"
    **LoRA: Low-Rank Adaptation of Large Language Models** — Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen (2021).
    [arxiv:2106.09685](https://arxiv.org/abs/2106.09685)

Why we use it here: the frozen base model can stay in 4-bit (see QLoRA below),
so training-time memory for a 0.5B base + LoRA adapters is roughly **1 GB** on
a Kaggle T4 instead of ~10 GB for full fine-tuning in bf16. The same memory
ratio holds at larger scales.

## What QLoRA adds

**QLoRA** (Dettmers et al. 2023) is LoRA with three quantization-side changes
that don't hurt quality:

1. **4-bit base weights** — the frozen \(W\) is stored in 4 bits per parameter
   using a custom datatype called **NF4** (NormalFloat-4). NF4's code-points
   are placed at the quantiles of a \(\mathcal{N}(0,1)\) distribution rather
   than evenly spaced, which is information-theoretically optimal when weights
   are normally distributed (LLM weights empirically are).
2. **Double quantization** — the quantization constants themselves are quantized
   again, saving another ~0.4 bits per parameter at no quality cost.
3. **Paged optimizer states** — uses NVIDIA's unified-memory paging for the
   optimizer to avoid OOM on long contexts. TRL's `SFTTrainer` enables this
   transparently when `bitsandbytes` is available.

The 4-bit weights are dequantized on-the-fly to a higher-precision **compute
dtype** (bf16 on Ampere+; fp16 on Turing/Pascal — see
[compute dtype](#compute-dtype-on-t4-p100)) for the matmul, then discarded.
Backprop flows through the LoRA matrices \(A\) and \(B\) in the compute dtype;
the frozen base never has gradients.

!!! note "Paper"
    **QLoRA: Efficient Finetuning of Quantized LLMs** — Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023).
    [arxiv:2305.14314](https://arxiv.org/abs/2305.14314)

## Our config

These values are in [`configs/base.yaml`][base-yaml] and inherited by every
experiment. Where the paper recommends a value, we use it; where the choice
is repo-specific (compute budget, model size, comparison constraint), the
reasoning is called out.

### LoRA

| Knob | Value | Source | Why |
|------|-------|--------|-----|
| `r` (rank) | **16** | repo-specific | QLoRA paper swept r ∈ {8, 16, 64}; 16 is the smallest that didn't underperform on instruction-following for sub-1B models. r=8 underfits IFEval; r=64 is wasted capacity for a 0.5B base + 5k UltraChat. |
| `alpha` | **32** | paper default | Sets the effective update scale \(\alpha/r = 2.0\). The "α = 2r" rule comes from QLoRA's empirics — it keeps update magnitude consistent across rank choices, so changing `r` doesn't require re-tuning the learning rate. |
| `dropout` | **0.05** | paper default | Low end of QLoRA's 0.05–0.1 range because 5k UltraChat is a small-data regime and we don't want to over-regularize. |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | paper default | "Attention + MLP" — every linear projection in a Qwen2 decoder block. QLoRA Table 4 showed this beats attention-only ("q,v" classic LoRA) by ~3 pts on MMLU, and the cost is small because all layers are still 4-bit underneath. |
| `bias` | `"none"` | paper default (hard-coded in [`adapters.py`][adapters]) | Don't train bias terms; saves params for no measurable quality gain on SFT. |

Trainable parameter count for Qwen2.5-0.5B with this config: roughly
**2.1 M of 494 M total** (≈0.4%).

### Quantization

| Knob | Value | Source | Why |
|------|-------|--------|-----|
| `load_in_4bit` | `true` (SFT/PPO/GRPO/KTO); `false` for DPO | repo-wide with one override | Even though the 0.5B base fits without quantization, keeping 4-bit on means the code path is identical when we scale to larger bases later. The DPO YAML overrides to `false` because the merge-then-DPO recipe (fuse `sft_v2` into the base before attaching the DPO LoRA) requires full-precision weights. |
| `bnb_4bit_quant_type` | `nf4` | paper default | NormalFloat-4 outperforms `fp4` by ~1 pt on the QLoRA benchmark — "free quality" with no extra cost. |
| `bnb_4bit_compute_dtype` | `bfloat16` | paper default (Ampere+) | On free Kaggle T4 / Colab P100, the [Kaggle notebook][kaggle-notebook] auto-patches this to `float16` because Turing/Pascal emulate bf16 in software at ~half speed. |
| `double_quant` | `true` | paper default | Free 0.4 bits/param savings. Always on. |

### Training hyperparameters by phase

The LoRA block (rank, target modules, dropout) is constant across phases.
Method-specific training knobs vary by phase and live in their own configs.

**Phase 1 / SFT** ([`configs/sft_qwen05b.yaml`][sft-yaml]):

| Knob | Value | Why |
|------|-------|-----|
| `learning_rate` | **2.0e-4** | QLoRA paper recommendation for LoRA adapters with r=16–64. Higher than full fine-tuning would tolerate because LoRA's low-rank update bounds the effective per-step weight change. |
| `per_device_train_batch_size` | **4** | Memory ceiling with 0.5B + 4-bit + LoRA + gradient checkpointing. |
| `gradient_accumulation_steps` | **4** | Effective batch = 16. Standard for instruction-tuning at this scale. |
| `gradient_checkpointing` | `true` | Required to fit batch=4 inside 16 GB. Trades ~30% compute for ~50% memory. |
| `warmup_ratio` | **0.03** | 3% of steps. Standard for short fine-tunes; longer warmup costs progress when total steps is ~313. |
| `num_train_epochs` | **1** | One pass over the 5k slice. |
| `n_samples` | **5000** | Per PROJECT.md §4.1. Small enough to iterate, large enough for SFT to move IFEval. |
| `assistant_only_loss` | `true` | Mask user/system turns out of the loss. TRL 1.4 defaults this to `False` — the lack of this flag in Phase 1 v1 was a real bug; see [`SFT` method page](../methods/sft.md#chat-template-and-loss-masking). |

**Phase 2 / DPO** ([`configs/dpo_qwen05b.yaml`][dpo-yaml]):

| Knob | Value | Why |
|------|-------|-----|
| `learning_rate` | **5.0e-6** | 1–2 orders of magnitude lower than SFT's 2e-4 because the DPO loss surface is steeper than cross-entropy (grad norm ≈ 25× at init). HF alignment-handbook range. |
| `beta` | **0.1** | Mitchell's recommendation for HH-style preference data; TRL's default. Conservative starting point. |
| `loss_type` | `sigmoid` | Canonical DPO. TRL also exposes `ipo`, `hinge`, `kto_pair`. |
| `per_device_train_batch_size` | **2** | DPO duplicates the forward (policy + ref) so smaller per-device batch than SFT; `gradient_accumulation_steps: 8` keeps effective batch = 16. |
| `max_length` | **1024** | UltraFeedback prompts + responses can be long; cap to bound memory. TRL 1.4 dropped `max_prompt_length`; `max_length` alone now governs the combined sequence. |
| `n_samples` | **5000** | Matches SFT's row count for an apples-to-apples compute envelope. |

## Decisions worth understanding

### Why r=16 specifically (not 8 or 32)

The QLoRA paper (Table 9) swept r ∈ {8, 16, 32, 64, 256} on a 7B base. Below
r=16 the paper saw underfitting on harder reasoning tasks; above r=64 was
wasted compute. For a 0.5B base the "underfit boundary" shifts lower — but
r=16 is the **conservative** choice that still has headroom.

Phase 1 (`sft_v2`) came in flat on IFEval prompt-strict (`−0.37pp` vs
pretrained base) — exactly the "obvious capacity ceiling" signal that would
normally argue for bumping to r=32. But Phase 2 DPO on top of `sft_v2`
cleared the bar (+0.74pp), suggesting the bottleneck was *the method on 5k
rows*, not *adapter capacity*. r=32 stays parked as a Phase 7 stretch
experiment rather than the next move.

### Why the same `lora:` block across all methods

The PROJECT.md anti-goal: *"the comparison [across methods] is the
contribution."* Tuning LoRA rank per method would let SFT use r=32 and DPO use
r=8, then we'd be comparing "best-tuned SFT" to "median DPO" — which proves
nothing about the methods themselves. So the LoRA capacity is held constant;
only the **method-specific** knobs (loss, dataset format, RL-specific hparams)
vary per experiment.

### When *not* to use this config

- **Different base model size**: a 7B base would want r=8 (relatively smaller)
  for the same per-method budget; a 100 M base might want r=32 to have enough
  capacity.
- **Continued pretraining** (not fine-tuning): use full fine-tuning, not LoRA.
- **Catastrophic-forgetting-sensitive domains** (e.g. preserving code while
  teaching math): LoRA helps here precisely because the base is frozen; keep
  the config.

## Compute dtype on T4 / P100

bf16 doesn't have hardware support on Turing (T4) or Pascal (P100). The math
still *works* — bnb dequantizes to bf16 and the GPU emulates it — but
throughput drops by roughly 50%. The [Kaggle notebook][kaggle-notebook]
detects T4/P100 and rewrites `dtype: bfloat16` → `dtype: float16` in both
[`configs/base.yaml`][base-yaml] entries before training. fp16 has native
hardware support on every NVIDIA GPU back to Volta, so this is a strict win
for free-tier training. On Ampere (A100, RTX 30-series), L4, or Hopper
(H100), leave the default.

## References

- Hu, E. J., et al. (2021). **LoRA: Low-Rank Adaptation of Large Language Models.** [arxiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers, T., et al. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs.** [arxiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Hugging Face PEFT: [huggingface.co/docs/peft](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- bitsandbytes 4-bit: [github.com/bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

[base-yaml]: https://github.com/agaonker/post-training-lab/blob/main/configs/base.yaml
[sft-yaml]: https://github.com/agaonker/post-training-lab/blob/main/configs/sft_qwen05b.yaml
[dpo-yaml]: https://github.com/agaonker/post-training-lab/blob/main/configs/dpo_qwen05b.yaml
[adapters]: https://github.com/agaonker/post-training-lab/blob/main/src/atlas/models/adapters.py
[kaggle-notebook]: https://github.com/agaonker/post-training-lab/blob/main/notebooks/kaggle/run_sft_kaggle.ipynb
