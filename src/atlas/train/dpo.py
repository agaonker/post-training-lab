"""DPO training entrypoint for Phase 2.

Wires the resolved ``Config`` to TRL's ``DPOTrainer``. The policy starting point
is Phase 1's SFT adapter, *merged* into the base — so the trainable LoRA sits
on top of a fully aligned starting point, and the reference policy (which TRL
computes by disabling the trainable LoRA) is exactly that merged sft_v2 model.
This is the canonical merge-then-DPO recipe; it avoids the multi-adapter
juggling that the "stacked LoRAs" alternative requires (and that varies across
TRL minor versions).

``cfg.train`` is the same permissive passthrough Phase 1 uses — every key in
the YAML's ``train:`` block flows straight to ``DPOConfig`` (TRL), including
DPO-specific knobs like ``beta``, ``loss_type``, ``max_length``.

trl + peft are imported lazily inside the functions that need them. Test
collection and the CLI's ``--help`` work without trl present.
"""

from __future__ import annotations

import argparse
import json
import os

# Hide all but cuda:0 before any module triggers CUDA init. Same rationale as
# atlas.train.sft — multi-GPU hosts (Kaggle T4x2) wrap the model in
# DataParallel after sharding, which breaks LoRA replication. A 0.5B base + two
# LoRAs (sft_v2 to merge in, plus DPO trainable) fits on one card; we'll add
# DDP only when the comparison scales up.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from atlas.data.preference_data import load_ultrafeedback_prefs
from atlas.models.adapters import make_lora_config
from atlas.models.base import load_base_model_and_tokenizer
from atlas.utils.config import Config, load_config, set_global_seed

SUPPORTED_DATASETS = {
    "HuggingFaceH4/ultrafeedback_binarized": load_ultrafeedback_prefs,
}


def _build_train_config(cfg: Config, max_steps_override: int | None) -> Any:
    """Construct TRL's ``DPOConfig`` from ``cfg.train.model_dump()``.

    Mirrors ``atlas.train.sft._build_train_config``: any field a user puts in
    the YAML's ``train:`` block becomes a kwarg to ``DPOConfig``, including
    DPO-specific knobs (``beta``, ``loss_type``, ``max_length``, etc.).
    """
    from trl import DPOConfig

    train_kwargs: dict[str, Any] = cfg.train.model_dump() if cfg.train else {}
    if max_steps_override is not None:
        train_kwargs["max_steps"] = max_steps_override
    return DPOConfig(**train_kwargs)


def _load_dataset(cfg: Config) -> Any:
    """Dispatch on ``cfg.dataset.name``. Only UltraFeedback is wired for Phase 2."""
    if cfg.dataset is None:
        raise ValueError("cfg.dataset is required for DPO; add a `dataset:` block to your YAML.")
    loader = SUPPORTED_DATASETS.get(cfg.dataset.name)
    if loader is None:
        raise ValueError(
            f"Unsupported DPO dataset {cfg.dataset.name!r}. Phase 2 only wires "
            f"these: {sorted(SUPPORTED_DATASETS)}."
        )
    return loader(
        n_samples=cfg.dataset.n_samples,
        seed=cfg.seed,
        revision=cfg.dataset.revision,
    )


def _preflight_hub_access(repo_id: str) -> None:
    """Validate the HF token can write to ``repo_id`` before training starts.

    Same fail-fast contract as ``atlas.train.sft._preflight_hub_access``: a
    read-only or wrong-namespace token surfaces as a 403 in ~200ms instead of
    only after ``trainer.train()`` completes (~hour of L4 compute on Phase 2's
    5k pairs). The lesson from the sft_v2 run.
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


def _build_policy_model(cfg: Config, tokenizer: Any) -> Any:
    """Load base + fuse the SFT warm-start adapter, returning a merged HF model.

    Steps:
    1. Load base in full precision (no 4-bit). ``cfg.quant.load_in_4bit`` should
       be ``false`` in the DPO YAML — merge_and_unload doesn't support 4-bit.
       A 0.5B model in bf16 is ~1 GB; the L4's 22 GiB easily fits policy + ref
       + optimizer state.
    2. Attach the SFT adapter via ``PeftModel.from_pretrained``.
    3. ``merge_and_unload`` fuses the LoRA weights into the base, returning a
       regular HF ``AutoModelForCausalLM``. This is the canonical
       "merge-then-DPO" recipe — it avoids the stacked-LoRA juggling that
       TRL's multi-adapter path requires.
    4. The fresh DPO LoRA is then attached by ``DPOTrainer`` via ``peft_config``
       in :func:`run_dpo`. TRL's ref-model handling computes the reference
       policy by disabling that fresh LoRA — so ref == base + sft_v2, exactly
       as we want.
    """
    from peft import PeftModel

    if cfg.model.sft_adapter is None:
        raise ValueError(
            "cfg.model.sft_adapter is required for DPO — Phase 2 trains on top "
            "of a Phase 1 SFT adapter. Set model.sft_adapter in your YAML."
        )

    base, _ = load_base_model_and_tokenizer(cfg)  # tokenizer comes from outside
    del _  # we already have the tokenizer; load_base returns a fresh one

    peft_kwargs: dict[str, Any] = {}
    if cfg.model.sft_adapter_revision is not None:
        peft_kwargs["revision"] = cfg.model.sft_adapter_revision
    policy = PeftModel.from_pretrained(base, cfg.model.sft_adapter, **peft_kwargs)
    # Fuse SFT into the base weights so DPO's fresh LoRA starts from a clean
    # AutoModelForCausalLM (no nested PeftModel state to confuse TRL).
    merged = policy.merge_and_unload()
    # Make sure the tokenizer's pad/eos config makes it onto the merged model's
    # generation_config — load_base_model_and_tokenizer doesn't touch the
    # generation config, but transformers Trainer/DPOTrainer will read it.
    if tokenizer.pad_token_id is not None:
        merged.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        merged.config.eos_token_id = tokenizer.eos_token_id
    return merged


def run_dpo(
    cfg: Config,
    *,
    max_steps_override: int | None = None,
    push_to_hub: bool = True,
) -> dict[str, Any]:
    """End-to-end DPO run. Returns a summary dict for logging."""
    from trl import DPOTrainer

    set_global_seed(cfg.seed)

    hub_repo: str | None = cfg.output.hub_repo if (push_to_hub and cfg.output is not None) else None
    if hub_repo is not None:
        _preflight_hub_access(hub_repo)

    train_ds = _load_dataset(cfg)
    # Load the tokenizer once; the policy model loader internally calls
    # load_base_model_and_tokenizer too but throws away its tokenizer (we want
    # one shared instance that we can then save explicitly after train).
    _, tokenizer = load_base_model_and_tokenizer(cfg)
    model = _build_policy_model(cfg, tokenizer)
    lora_config = make_lora_config(cfg)
    train_config = _build_train_config(cfg, max_steps_override)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL builds ref by disabling the peft_config LoRA
        args=train_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    result = trainer.train()
    trainer.save_model(train_config.output_dir)
    # Same explicit tokenizer save as the SFT path — the patched chat_template
    # and the Instruct eos config need to travel with the adapter to HF Hub,
    # so eval can load the right tokenizer alongside the adapter.
    tokenizer.save_pretrained(train_config.output_dir)

    pushed_to: str | None = None
    push_error: str | None = None
    if hub_repo is not None:
        try:
            HfApi().upload_folder(
                folder_path=train_config.output_dir,
                repo_id=hub_repo,
                repo_type="model",
                commit_message=f"DPO adapter (config_hash={cfg.config_hash})",
            )
            pushed_to = hub_repo
        except Exception as e:  # noqa: BLE001 — preserve run summary on push fail
            push_error = f"{type(e).__name__}: {e}"
            abs_path = Path(train_config.output_dir).resolve()
            print(
                f"\n[hub upload failed; adapter is safe at {abs_path}]\n"
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
        "training_loss": float(result.training_loss) if result.training_loss is not None else None,
        "global_step": result.global_step,
        "n_train_samples": len(train_ds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DPO adapter from an experiment YAML.")
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
    summary = run_dpo(
        cfg,
        max_steps_override=args.max_steps,
        push_to_hub=not args.no_push_to_hub,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
