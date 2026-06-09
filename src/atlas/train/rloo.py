"""RLOO training entrypoint for Phase 3B.

Wires the resolved ``Config`` to TRL's ``RLOOTrainer`` — REINFORCE Leave-One-Out,
the canonical replacement for PPO in TRL 1.4 (PPOTrainer was deprecated upstream).

The setup uses three model instances on the GPU:

1. **Policy** — pretrained base + sft_v2 (merged in via ``merge_and_unload``)
   + a fresh trainable LoRA. The trainable LoRA is what RLOO updates; the
   reference policy is the same merged model with the new LoRA disabled
   (TRL handles this automatically when ``peft_config`` is set).
2. **Reward model** — same pretrained base loaded separately as a
   SequenceClassification head + 4-bit quant, with the Phase 3A rm_v1 adapter
   attached. Passed to ``RLOOTrainer`` via ``reward_funcs``.
3. **Reference** — implicit; same as policy with the trainable LoRA disabled.

Why RLOO instead of PPO: TRL 1.4 doesn't export PPOTrainer anymore. The field
moved (PROJECT.md §8). RLOO is the simpler successor — REINFORCE with
``num_generations`` siblings per prompt acting as each other's baseline. No
value head, no separate critic.

``cfg.train`` is the same permissive passthrough Phases 1-2 use — every key
in the YAML's ``train:`` block flows straight to ``RLOOConfig``, including
RLOO-specific knobs like ``beta`` (KL), ``num_generations``, ``epsilon``
(clip), ``temperature``, ``reward_clip_range``.

trl + peft + transformers are imported lazily inside the functions that need
them. Test collection and the CLI's ``--help`` work without trl present.
"""

from __future__ import annotations

import argparse
import json
import os

# Same single-GPU pin as SFT/DPO/RM. RLOO has 3 model instances on the GPU
# (policy, ref-via-LoRA-toggle, reward) so memory matters; DataParallel
# replication would multiply that.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from atlas.data.preference_data import load_ultrafeedback_prompts
from atlas.models.adapters import make_lora_config
from atlas.models.base import load_base_model_and_tokenizer
from atlas.utils.config import Config, load_config, set_global_seed

SUPPORTED_DATASETS = {
    "HuggingFaceH4/ultrafeedback_binarized": load_ultrafeedback_prompts,
}


def _build_train_config(cfg: Config, max_steps_override: int | None) -> Any:
    """Construct TRL's ``RLOOConfig`` from ``cfg.train.model_dump()``.

    Mirrors the SFT/DPO/RM pattern. RLOO-specific knobs (beta, num_generations,
    epsilon, temperature, reward_clip_range, mask_truncated_completions) flow
    through TrainCfg's ``extra="allow"`` passthrough.
    """
    from trl import RLOOConfig

    train_kwargs: dict[str, Any] = cfg.train.model_dump() if cfg.train else {}
    if max_steps_override is not None:
        train_kwargs["max_steps"] = max_steps_override
    return RLOOConfig(**train_kwargs)


def _load_dataset(cfg: Config) -> Any:
    """Dispatch on ``cfg.dataset.name``. Only UltraFeedback is wired for Phase 3B."""
    if cfg.dataset is None:
        raise ValueError("cfg.dataset is required for RLOO; add a `dataset:` block to your YAML.")
    loader = SUPPORTED_DATASETS.get(cfg.dataset.name)
    if loader is None:
        raise ValueError(
            f"Unsupported RLOO dataset {cfg.dataset.name!r}. Phase 3B only wires "
            f"these: {sorted(SUPPORTED_DATASETS)}."
        )
    return loader(
        n_samples=cfg.dataset.n_samples,
        seed=cfg.seed,
        revision=cfg.dataset.revision,
    )


def _preflight_hub_access(repo_id: str) -> None:
    """Same fail-fast preflight contract as SFT/DPO/RM."""
    api = HfApi()
    try:
        api.whoami()
    except HfHubHTTPError as e:
        raise RuntimeError(
            "HF preflight failed: whoami() rejected. Set "
            "HUGGING_FACE_HUB_TOKEN (with Write scope) or pass --no-push-to-hub "
            f"to skip the upload. Original error: {e}"
        ) from e
    try:
        api.create_repo(repo_id, exist_ok=True)
    except HfHubHTTPError as e:
        raise RuntimeError(
            f"HF preflight failed: token cannot write to {repo_id!r}. "
            "Regenerate with Write scope at https://huggingface.co/settings/tokens, "
            "fix the namespace in cfg.output.hub_repo, or pass --no-push-to-hub. "
            f"Original error: {e}"
        ) from e


def _build_policy_model(cfg: Config) -> Any:
    """Load base + fuse the SFT warm-start adapter, returning a merged HF model.

    Same merge-then-X recipe as DPO: load base in full precision, attach
    sft_v2, ``merge_and_unload``. ``RLOOTrainer`` then attaches a fresh LoRA
    via ``peft_config`` and disables it for the ref forward.
    """
    from peft import PeftModel

    if cfg.model.sft_adapter is None:
        raise ValueError(
            "cfg.model.sft_adapter is required for RLOO — Phase 3B starts from "
            "a Phase 1 SFT adapter. Set model.sft_adapter in your YAML."
        )

    base, _ = load_base_model_and_tokenizer(cfg)
    del _  # tokenizer comes from a separate call

    peft_kwargs: dict[str, Any] = {}
    if cfg.model.sft_adapter_revision is not None:
        peft_kwargs["revision"] = cfg.model.sft_adapter_revision
    policy = PeftModel.from_pretrained(base, cfg.model.sft_adapter, **peft_kwargs)
    return policy.merge_and_unload()


def _build_reward_model(cfg: Config, tokenizer: Any) -> Any:
    """Load ``base + rm_v1`` as a separate SequenceClassification model.

    This is a second copy of the base on the GPU, in 4-bit to save memory.
    The RM adapter is attached as a PEFT model and *not* merged — the
    classification head's adapter weights stay live during scoring.

    ``tokenizer`` is needed only to wire ``pad_token_id`` onto the RM's model
    config — SequenceClassification heads can't batch-forward without it
    ("Cannot handle batch sizes > 1 if no padding token is defined").
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification

    from atlas.models.base import _build_quant_config

    if cfg.model.rm_adapter is None:
        raise ValueError(
            "cfg.model.rm_adapter is required for RLOO — Phase 3B uses the "
            "Phase 3A Reward Model to score rollouts. Set model.rm_adapter."
        )

    # Build a temporary cfg view that re-enables 4-bit for the RM only —
    # cfg.quant.load_in_4bit is False (because the *policy* merge needs full
    # precision). The reward model doesn't need merging, so we can quantize it.
    rm_quant_cfg = type(cfg.quant)(**{**cfg.quant.model_dump(), "load_in_4bit": True})
    rm_cfg_view = type(cfg)(**{**cfg.model_dump(), "quant": rm_quant_cfg.model_dump()})
    rm_quant = _build_quant_config(rm_cfg_view)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        cfg.model.dtype
    ]

    base_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": {"": 0} if torch.cuda.is_available() else "auto",
        "num_labels": 1,
    }
    if cfg.model.revision:
        base_kwargs["revision"] = cfg.model.revision
    if rm_quant is not None:
        base_kwargs["quantization_config"] = rm_quant
    rm_base = AutoModelForSequenceClassification.from_pretrained(cfg.model.name, **base_kwargs)

    # SequenceClassification forward needs pad_token_id to handle variable
    # length sequences in a batch. The base.config doesn't ship it on the
    # pretrained Qwen2.5-0.5B side, and PeftModel inherits the base's config.
    if rm_base.config.pad_token_id is None:
        rm_base.config.pad_token_id = tokenizer.pad_token_id

    rm_peft_kwargs: dict[str, Any] = {}
    if cfg.model.rm_adapter_revision is not None:
        rm_peft_kwargs["revision"] = cfg.model.rm_adapter_revision
    rm = PeftModel.from_pretrained(rm_base, cfg.model.rm_adapter, **rm_peft_kwargs)
    rm.eval()
    return rm


def run_rloo(
    cfg: Config,
    *,
    max_steps_override: int | None = None,
    push_to_hub: bool = True,
) -> dict[str, Any]:
    """End-to-end RLOO run. Returns a summary dict for logging."""
    from trl import RLOOTrainer

    set_global_seed(cfg.seed)

    hub_repo: str | None = cfg.output.hub_repo if (push_to_hub and cfg.output is not None) else None
    if hub_repo is not None:
        _preflight_hub_access(hub_repo)

    train_ds = _load_dataset(cfg)
    _, tokenizer = load_base_model_and_tokenizer(cfg)
    policy = _build_policy_model(cfg)
    reward_model = _build_reward_model(cfg, tokenizer)
    lora_config = make_lora_config(cfg)
    train_config = _build_train_config(cfg, max_steps_override)

    trainer = RLOOTrainer(
        model=policy,
        reward_funcs=[reward_model],
        args=train_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        # RM and policy share the byte-identical Instruct vocab + the same
        # patched chat_template, so the reward processing class is the same
        # tokenizer instance. TRL still needs it passed explicitly — without
        # it, the per-reward apply_chat_template call NoneType-errors.
        reward_processing_classes=[tokenizer],
        peft_config=lora_config,
    )

    result = trainer.train()
    trainer.save_model(train_config.output_dir)
    tokenizer.save_pretrained(train_config.output_dir)

    pushed_to: str | None = None
    push_error: str | None = None
    if hub_repo is not None:
        try:
            HfApi().upload_folder(
                folder_path=train_config.output_dir,
                repo_id=hub_repo,
                repo_type="model",
                commit_message=f"RLOO adapter (config_hash={cfg.config_hash})",
            )
            pushed_to = hub_repo
        except Exception as e:  # noqa: BLE001
            push_error = f"{type(e).__name__}: {e}"
            abs_path = Path(train_config.output_dir).resolve()
            print(
                f"\n[hub upload failed; adapter is safe at {abs_path}]\n"
                f"Original error: {push_error}\n",
                flush=True,
            )

    return {
        "config_hash": cfg.config_hash,
        "output_dir": train_config.output_dir,
        "hub_repo": pushed_to,
        "push_error": push_error,
        "training_loss": float(result.training_loss) if result.training_loss is not None else None,
        "global_step": result.global_step,
        "n_train_samples": len(train_ds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RLOO adapter from an experiment YAML.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override cfg.train.max_steps; useful for a 50-step smoke run.",
    )
    parser.add_argument(
        "--no-push-to-hub",
        action="store_true",
        help="Skip the final HF Hub push even if cfg.output.hub_repo is set.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = run_rloo(
        cfg,
        max_steps_override=args.max_steps,
        push_to_hub=not args.no_push_to_hub,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
