"""Modal entrypoint for DPO — the scripted counterpart to ``sft_modal.py``,
adapted for Phase 2.

Each ``modal run`` is a fresh container that provisions a GPU, runs the
*existing* ``atlas.train.dpo.run_dpo`` against ``configs/dpo_qwen05b.yaml``,
pushes the trained DPO adapter to HF Hub, and exits.

Structure mirrors ``sft_modal.py`` line-for-line:
- Image installs deps from pyproject.toml (trl/peft/transformers/datasets all
  in default deps; no extras needed for DPO).
- ``WANDB_MODE=disabled`` in the image env so a missing ``WANDB_API_KEY``
  secret doesn't crash ``trainer.train()`` in ``on_train_begin`` (LESSONS.md).
- Local ``atlas`` source + ``configs/`` are mounted, so edits run immediately.
- HF cache shared with the SFT runs via the ``atlas-hf-cache`` Volume — both
  base model and UltraFeedback dataset are reused warm.
- ``hf-token`` secret with WRITE scope for the adapter push.

One-time setup (same as SFT):
    pip install modal && modal token new
    modal secret create hf-token HUGGING_FACE_HUB_TOKEN=hf_xxx   # WRITE scope

Usage:
    make dpo-modal-smoke    # 50 steps, no Hub push, ~5 min, ~$0.30
    make dpo-modal          # full run, pushes to cfg.output.hub_repo

Raw entrypoints (the make targets wrap these):
    modal run src/atlas/cloud/dpo_modal.py::smoke
    modal run src/atlas/cloud/dpo_modal.py::main
    modal run src/atlas/cloud/dpo_modal.py::main --max-steps 50 --no-push-to-hub
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

# Same sys.path guard pattern as eval_modal.py / sft_modal.py — modal CLI loads
# this file from the laptop's Python (not this project's .venv), so atlas may
# not be importable. In the Modal container, /root/atlas is already mounted.
_parents = Path(__file__).resolve().parents  # local: <repo>/src/atlas/cloud/dpo_modal.py
if len(_parents) > 2 and (_parents[2] / "atlas").is_dir() and str(_parents[2]) not in sys.path:
    sys.path.insert(0, str(_parents[2]))  # <repo>/src

# bf16-native GPU default. L4 22GB easily fits a 0.5B in bf16 + DPO LoRA + ref
# (which TRL builds by disabling the LoRA — zero extra memory). Override with
# ATLAS_DPO_GPU.
GPU = os.environ.get("ATLAS_DPO_GPU", "L4")

app = modal.App("atlas-dpo")

# Default-extras only — DPO doesn't need lm-eval or vllm (those live in
# eval_modal.py's image). trl / peft / transformers / datasets / accelerate /
# bitsandbytes are all in pyproject's main dependencies.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    # Disable wandb at the image level so a missing WANDB_API_KEY doesn't crash
    # trainer.train() in on_train_begin. To enable: create a `wandb` Modal
    # secret, attach it on the function below, and drop this env override.
    .env({"WANDB_MODE": "disabled"})
    .add_local_dir("src/atlas", remote_path="/root/atlas")
    .add_local_dir("configs", remote_path="/root/configs")
)

# Shared HF download cache with SFT + eval. Phase 2 fetches both the base
# model (already warm from Phase 1) and UltraFeedback-binarized (new).
hf_cache = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    # Generous: cold model + dataset download + full 5k-pair run. Smoke (50
    # steps) finishes in ~5 min; full run is ~30-60 min on L4.
    timeout=60 * 60 * 2,
    volumes={"/root/.cache/huggingface": hf_cache},
    # hf-token must have WRITE scope on the namespace in cfg.output.hub_repo —
    # the DPO preflight will fail-fast in ~200ms if it doesn't (LESSONS.md).
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_dpo_remote(
    max_steps: int | None = None,
    push_to_hub: bool = True,
) -> dict:
    """Run DPO on a Modal GPU; return the summary dict from ``atlas.train.dpo.run_dpo``."""
    from atlas.train.dpo import run_dpo
    from atlas.utils.config import load_config

    # HfApi reads HF_TOKEN; the hf-token secret exposes HUGGING_FACE_HUB_TOKEN.
    os.environ.setdefault("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))

    cfg = load_config("/root/configs/dpo_qwen05b.yaml")
    return run_dpo(
        cfg,
        max_steps_override=max_steps,
        push_to_hub=push_to_hub,
    )


def _print_summary(summary: dict) -> None:
    """Pretty-print the run_dpo summary on the local entrypoint side."""
    print("\n--- DPO run summary ---")
    print(json.dumps(summary, indent=2))
    if summary.get("push_error"):
        print(
            "\n[!] Hub push failed; the adapter trained successfully but the "
            "upload didn't land. Recover via HfApi().upload_folder on the "
            "persisted output_dir (Modal containers are ephemeral by default — "
            "if you need a re-push, re-run with the same config)."
        )
    elif summary.get("hub_repo"):
        print(f"\nAdapter pushed to: https://huggingface.co/{summary['hub_repo']}")


@app.local_entrypoint()
def smoke() -> None:
    """50-step wiring probe (~5 min, ~$0.30). No Hub push.

    Proves the DPO pipeline trains end-to-end on Modal:
    - SFT warm-start adapter loads + merges cleanly
    - Fresh DPO LoRA attaches on top
    - β / loss_type / max_length / max_prompt_length flow through to DPOConfig
    - Loss doesn't NaN; ref model is implicitly frozen (TRL disables the LoRA)

    Watch: DPO loss should start near −log(0.5) ≈ 0.693 and decrease *slowly*.
    If it crashes to ~0 in 50 steps, β is wrong or the reference policy isn't
    actually frozen — that's the canonical "you have a DPO bug" signal.
    """
    _print_summary(run_dpo_remote.remote(max_steps=50, push_to_hub=False))


@app.local_entrypoint()
def main(max_steps: int | None = None, no_push_to_hub: bool = False) -> None:
    """Full DPO run per ``configs/dpo_qwen05b.yaml``. Pushes the trained
    adapter to ``cfg.output.hub_repo`` unless ``--no-push-to-hub`` is passed.
    """
    _print_summary(run_dpo_remote.remote(max_steps=max_steps, push_to_hub=not no_push_to_hub))
