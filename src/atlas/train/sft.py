"""SFT training entrypoint for Phase 1.

Wires the resolved ``Config`` to TRL's ``SFTTrainer``: dataset from
``atlas.data.sft_data``, base model + tokenizer from ``atlas.models.base``,
``LoraConfig`` from ``atlas.models.adapters``. ``cfg.train`` is a permissive
passthrough — every key in the YAML's ``train:`` block flows straight to
``SFTConfig`` (TRL), so the schema survives TRL minor bumps and we never
re-declare its knob surface here.

trl is imported lazily inside the functions that need it. Test collection
and the CLI's ``--help`` work without trl present.
"""

from __future__ import annotations

import argparse
import json
import os

# Hide all but cuda:0 before any module triggers CUDA init. On multi-GPU
# hosts (Kaggle T4x2) HF Trainer auto-wraps the model in DataParallel,
# which silently corrupts bnb's 4-bit quant state when replicating to the
# second GPU and crashes inside cuBLAS. A 0.5B model fits on one card; we
# only need DDP for the bigger phases. Users wanting multi-GPU can export
# CUDA_VISIBLE_DEVICES themselves — setdefault leaves an explicit setting
# alone.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from pathlib import Path
from typing import Any

from atlas.data.sft_data import load_ultrachat_sft
from atlas.models.adapters import make_lora_config
from atlas.models.base import load_base_model_and_tokenizer
from atlas.utils.config import Config, load_config, set_global_seed

SUPPORTED_DATASETS = {"HuggingFaceH4/ultrachat_200k": load_ultrachat_sft}


def _build_train_config(cfg: Config, max_steps_override: int | None) -> Any:
    """Construct TRL's ``SFTConfig`` from ``cfg.train.model_dump()``.

    The passthrough is the whole point: any field a user puts in the YAML's
    ``train:`` block becomes a kwarg to ``SFTConfig``. ``max_steps_override``
    is layered on top for smoke-run convenience.
    """
    from trl import SFTConfig

    train_kwargs: dict[str, Any] = cfg.train.model_dump() if cfg.train else {}
    if max_steps_override is not None:
        train_kwargs["max_steps"] = max_steps_override
    return SFTConfig(**train_kwargs)


def _load_dataset(cfg: Config) -> Any:
    """Dispatch on ``cfg.dataset.name``. Only UltraChat is wired for Phase 1."""
    if cfg.dataset is None:
        raise ValueError(
            "cfg.dataset is required for SFT; add a `dataset:` block to your YAML."
        )
    loader = SUPPORTED_DATASETS.get(cfg.dataset.name)
    if loader is None:
        raise ValueError(
            f"Unsupported SFT dataset {cfg.dataset.name!r}. Phase 1 only wires "
            f"these: {sorted(SUPPORTED_DATASETS)}."
        )
    return loader(
        n_samples=cfg.dataset.n_samples,
        seed=cfg.seed,
        revision=cfg.dataset.revision,
    )


def _preflight_hub_access(repo_id: str) -> None:
    """Validate the HF token can write to ``repo_id`` before training starts.

    Without this, a read-only or wrong-namespace token surfaces as a 403 only
    after ``trainer.train()`` completes — costing hours of compute. ``whoami``
    proves the token is valid; ``create_repo(..., exist_ok=True)`` is a no-op
    on an existing repo but still requires write scope on the namespace.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

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


def run_sft(
    cfg: Config,
    *,
    max_steps_override: int | None = None,
    push_to_hub: bool = True,
) -> dict[str, Any]:
    """End-to-end SFT run. Returns a summary dict for logging / metrics.json."""
    from trl import SFTTrainer

    set_global_seed(cfg.seed)

    hub_repo: str | None = (
        cfg.output.hub_repo if (push_to_hub and cfg.output is not None) else None
    )
    if hub_repo is not None:
        # Fail in ~200ms instead of 3 hours if the token can't push.
        _preflight_hub_access(hub_repo)

    train_ds = _load_dataset(cfg)
    model, tokenizer = load_base_model_and_tokenizer(cfg)
    lora_config = make_lora_config(cfg)
    train_config = _build_train_config(cfg, max_steps_override)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=train_config,
        peft_config=lora_config,
    )

    result = trainer.train()
    trainer.save_model(train_config.output_dir)

    pushed_to: str | None = None
    push_error: str | None = None
    if hub_repo is not None:
        try:
            trainer.push_to_hub(
                repo_id=hub_repo,
                commit_message=f"SFT adapter (config_hash={cfg.config_hash})",
            )
            pushed_to = hub_repo
        except Exception as e:  # noqa: BLE001 — preserve run summary on push fail
            push_error = f"{type(e).__name__}: {e}"
            abs_path = Path(train_config.output_dir).resolve()
            print(
                f"\n[push_to_hub failed; adapter is safe at {abs_path}]\n"
                f"Recover with:\n"
                f"  from huggingface_hub import HfApi\n"
                f'  HfApi().upload_folder(folder_path="{abs_path}", '
                f'repo_id="{hub_repo}", repo_type="model")\n'
                f"Original error: {push_error}\n",
                flush=True,
            )

    return {
        "config_hash": cfg.config_hash,
        "output_dir": train_config.output_dir,
        "hub_repo": pushed_to,
        "push_error": push_error,
        "training_loss": float(result.training_loss)
        if result.training_loss is not None
        else None,
        "global_step": result.global_step,
        "n_train_samples": len(train_ds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an SFT adapter from an experiment YAML."
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
    summary = run_sft(
        cfg,
        max_steps_override=args.max_steps,
        push_to_hub=not args.no_push_to_hub,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
