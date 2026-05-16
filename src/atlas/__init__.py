"""atlas — post-training-lab.

A controlled comparison of LLM post-training methods (SFT, DPO, KTO, RLHF/PPO,
GRPO) on one small base model with one shared evaluation harness.

Subpackages:
    data    dataset loading and prep, shared across experiments
    models  base model + tokenizer loaders, LoRA/QLoRA helpers
    train   one module per training method
    eval    the shared eval harness, custom evals, LLM judge, comparison
    utils   config, logging, checkpointing
    cloud   Modal entrypoints for the paid GPU phases
"""

__version__ = "0.1.0"
