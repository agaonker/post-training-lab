"""Reward Model training entrypoint for Phase 3A.

Wires the resolved ``Config`` to TRL's ``RewardTrainer``. The RM is a
sequence-classification head on top of the pretrained ``Qwen2.5-0.5B`` base:
it scores ``r(prompt + completion)`` so that chosen > rejected on the
UltraFeedback-binarized pairs. Phase 3B (PPO) loads this adapter and uses
``r(·)`` to score policy rollouts.

Two recipe choices worth flagging:

1. **No SFT warm-start.** Unlike DPO, the RM is a classifier — it doesn't
   need to be a good generator. Starting from the raw pretrained base + a
   fresh regression head is the canonical RM recipe (Stiennon et al. 2020,
   InstructGPT §3.3).
2. **LoRA + 4-bit base.** Same QLoRA setup as SFT, with ``task_type="SEQ_CLS"``
   on the LoRA config (overriding the ``CAUSAL_LM`` default from
   ``atlas.models.adapters.make_lora_config``). The base stays in 4-bit; PPO
   in Phase 3B loads this adapter directly with no merging.

``cfg.train`` is the same permissive passthrough Phases 1-2 use — every key
in the YAML's ``train:`` block flows straight to ``RewardConfig`` (TRL),
including ``max_length`` and any other RM-specific knobs.

trl + peft are imported lazily inside the functions that need them. Test
collection and the CLI's ``--help`` work without trl present.
"""

from __future__ import annotations

import argparse
import json
import os

# Same single-GPU pin as SFT/DPO. The RM is small (0.5B + LoRA + head) and
# fits one card; multi-GPU complicates LoRA replication unnecessarily.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from atlas.data.preference_data import load_ultrafeedback_prefs
from atlas.models.adapters import make_lora_config
from atlas.utils.config import Config, load_config, set_global_seed

SUPPORTED_DATASETS = {
    "HuggingFaceH4/ultrafeedback_binarized": load_ultrafeedback_prefs,
}

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _build_train_config(cfg: Config, max_steps_override: int | None) -> Any:
    """Construct TRL's ``RewardConfig`` from ``cfg.train.model_dump()``.

    Mirrors ``atlas.train.sft._build_train_config`` and the DPO equivalent.
    """
    from trl import RewardConfig

    train_kwargs: dict[str, Any] = cfg.train.model_dump() if cfg.train else {}
    if max_steps_override is not None:
        train_kwargs["max_steps"] = max_steps_override
    return RewardConfig(**train_kwargs)


def _load_dataset(cfg: Config) -> Any:
    """Dispatch on ``cfg.dataset.name``. Only UltraFeedback is wired for Phase 3A."""
    if cfg.dataset is None:
        raise ValueError("cfg.dataset is required for RM; add a `dataset:` block to your YAML.")
    loader = SUPPORTED_DATASETS.get(cfg.dataset.name)
    if loader is None:
        raise ValueError(
            f"Unsupported RM dataset {cfg.dataset.name!r}. Phase 3A only wires "
            f"these: {sorted(SUPPORTED_DATASETS)}."
        )
    return loader(
        n_samples=cfg.dataset.n_samples,
        seed=cfg.seed,
        revision=cfg.dataset.revision,
    )


def _preflight_hub_access(repo_id: str) -> None:
    """Same fail-fast preflight contract as the SFT/DPO paths.

    A read-only token surfaces as 403 in ~200ms instead of after an hour of L4
    compute. See LESSONS.md for the lesson that earned this preflight.
    """
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


def _build_rm_model_and_tokenizer(cfg: Config) -> tuple[Any, Any]:
    """Load ``AutoModelForSequenceClassification`` with the same 4-bit quant +
    tokenizer override as SFT — but with ``num_labels=1`` (regression head).

    This deliberately doesn't reuse ``atlas.models.base.load_base_model_and_tokenizer``:
    that helper loads ``AutoModelForCausalLM``, which is the wrong head for an
    RM. The quant config + tokenizer routing logic is duplicated minimally here
    to keep the SFT helper focused on causal LM loading.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    from atlas.models.base import (
        _build_quant_config,
        patch_chat_template_for_assistant_mask,
    )

    if cfg.model.dtype not in _DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype {cfg.model.dtype!r}; expected one of {sorted(_DTYPE_MAP)}"
        )
    dtype = _DTYPE_MAP[cfg.model.dtype]
    quant = _build_quant_config(cfg)

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": {"": 0} if torch.cuda.is_available() else "auto",
        "num_labels": 1,  # regression head: scalar reward per sequence
    }
    if cfg.model.revision:
        model_kwargs["revision"] = cfg.model.revision
    if quant is not None:
        model_kwargs["quantization_config"] = quant
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model.name, **model_kwargs)

    tokenizer_repo = cfg.model.tokenizer_name or cfg.model.name
    tok_kwargs: dict[str, Any] = {}
    if cfg.model.revision and tokenizer_repo == cfg.model.name:
        tok_kwargs["revision"] = cfg.model.revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    patch_chat_template_for_assistant_mask(tokenizer)

    # transformers' SequenceClassification heads need the model's pad_token_id
    # explicitly set so they can mask attention correctly during the reward
    # forward. Without this, the head reads garbage from pad positions.
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def run_rm(
    cfg: Config,
    *,
    max_steps_override: int | None = None,
    push_to_hub: bool = True,
) -> dict[str, Any]:
    """End-to-end RM run. Returns a summary dict for logging."""
    from trl import RewardTrainer

    set_global_seed(cfg.seed)

    hub_repo: str | None = cfg.output.hub_repo if (push_to_hub and cfg.output is not None) else None
    if hub_repo is not None:
        _preflight_hub_access(hub_repo)

    train_ds = _load_dataset(cfg)
    model, tokenizer = _build_rm_model_and_tokenizer(cfg)
    # Override the LoRA task type to SEQ_CLS so PEFT wires the adapter onto
    # the classification head correctly. The default (CAUSAL_LM) targets the
    # LM head, which RM doesn't have.
    lora_config = make_lora_config(cfg, task_type="SEQ_CLS")
    train_config = _build_train_config(cfg, max_steps_override)

    trainer = RewardTrainer(
        model=model,
        args=train_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
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
                commit_message=f"RM adapter (config_hash={cfg.config_hash})",
            )
            pushed_to = hub_repo
        except Exception as e:  # noqa: BLE001 — preserve run summary on push fail
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
    parser = argparse.ArgumentParser(
        description="Train a Reward Model adapter from an experiment YAML."
    )
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
    summary = run_rm(
        cfg,
        max_steps_override=args.max_steps,
        push_to_hub=not args.no_push_to_hub,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
