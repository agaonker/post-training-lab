"""LoRA / QLoRA helpers — thin wrappers over ``peft`` that consume ``cfg.lora``.

Kept deliberately small: ``make_lora_config`` is the single bridge from the
project's YAML schema to ``peft.LoraConfig``. The actual attachment to a base
model happens inside TRL's trainers via their ``peft_config=`` argument, so
no separate attach helper is exposed for the training path.
"""

from __future__ import annotations

from peft import LoraConfig

from atlas.utils.config import Config


def make_lora_config(cfg: Config, task_type: str = "CAUSAL_LM") -> LoraConfig:
    """Build a ``peft.LoraConfig`` from ``cfg.lora``.

    Args:
        cfg: resolved Config; ``cfg.lora`` (r / alpha / dropout / target_modules)
            is read.
        task_type: PEFT task family; defaults to ``"CAUSAL_LM"`` which is right
            for SFT/DPO/PPO/GRPO/KTO on a decoder-only LM. The reward-model phase
            overrides this to ``"SEQ_CLS"``.
    """
    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type=task_type,
    )
