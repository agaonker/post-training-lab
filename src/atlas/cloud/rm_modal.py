"""Modal entrypoint for Reward Model training — Phase 3A.

Each ``modal run`` is a fresh container that provisions a GPU, runs
``atlas.train.reward_model.run_rm`` against ``configs/rm_qwen05b.yaml``, pushes
the trained RM adapter to HF Hub, and exits. Mirrors ``sft_modal.py`` and
``dpo_modal.py`` line-for-line — same image, same secrets, same volume.

The trained adapter is the input to Phase 3B (PPO): the PPO trainer loads
``base + rm-adapter`` to score rollouts, alongside ``base + sft-adapter`` as
the policy.

One-time setup (shared with SFT/DPO):
    pip install modal && modal token new
    modal secret create hf-token HUGGING_FACE_HUB_TOKEN=hf_xxx   # WRITE scope

Usage:
    make rm-modal-smoke    # 50 steps, no Hub push, ~5 min, ~$0.30
    make rm-modal          # full run, pushes to cfg.output.hub_repo

Raw entrypoints (the make targets wrap these):
    modal run src/atlas/cloud/rm_modal.py::smoke
    modal run src/atlas/cloud/rm_modal.py::main
    modal run src/atlas/cloud/rm_modal.py::main --max-steps 50 --no-push-to-hub
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

# Same sys.path guard pattern as eval_modal.py / sft_modal.py / dpo_modal.py —
# the modal CLI loads this file from the laptop's Python (not the project's
# .venv), so atlas may not be importable yet. In the container atlas is
# already at /root/atlas, so the depth + layout check makes this a no-op
# remotely.
_parents = Path(__file__).resolve().parents  # local: <repo>/src/atlas/cloud/rm_modal.py
if len(_parents) > 2 and (_parents[2] / "atlas").is_dir() and str(_parents[2]) not in sys.path:
    sys.path.insert(0, str(_parents[2]))  # <repo>/src

# bf16-native GPU default. L4 22GB easily fits a 0.5B 4-bit + LoRA + the
# regression head. Override with ATLAS_RM_GPU.
GPU = os.environ.get("ATLAS_RM_GPU", "L4")

app = modal.App("atlas-rm")

# Default-extras only — RM doesn't need lm-eval or vllm. trl/peft/transformers/
# datasets/accelerate/bitsandbytes are all in pyproject's main dependencies.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    # Disable wandb at the image level so a missing WANDB_API_KEY doesn't
    # crash trainer.train() in on_train_begin (LESSONS.md). To enable: create
    # a `wandb` Modal secret, attach it on the function below, and drop this
    # env override.
    .env({"WANDB_MODE": "disabled"})
    .add_local_dir("src/atlas", remote_path="/root/atlas")
    .add_local_dir("configs", remote_path="/root/configs")
)

# Shared HF download cache with SFT + DPO + eval — base model and
# UltraFeedback are reused warm.
hf_cache = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    # Generous: covers cold model + dataset download + the full 5k-pair RM
    # run. Smoke (50 steps) finishes in ~5 min; full run is ~30-45 min on L4.
    timeout=60 * 60 * 2,
    volumes={"/root/.cache/huggingface": hf_cache},
    # hf-token must have WRITE scope on the namespace in cfg.output.hub_repo —
    # the RM preflight will fail-fast in ~200ms if it doesn't (LESSONS.md).
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_rm_remote(
    max_steps: int | None = None,
    push_to_hub: bool = True,
) -> dict:
    """Run RM training on a Modal GPU; return the summary dict from
    ``atlas.train.reward_model.run_rm``."""
    from atlas.train.reward_model import run_rm
    from atlas.utils.config import load_config

    # HfApi reads HF_TOKEN; the hf-token secret exposes HUGGING_FACE_HUB_TOKEN.
    os.environ.setdefault("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))

    cfg = load_config("/root/configs/rm_qwen05b.yaml")
    return run_rm(
        cfg,
        max_steps_override=max_steps,
        push_to_hub=push_to_hub,
    )


def _print_summary(summary: dict) -> None:
    """Pretty-print the run_rm summary on the local entrypoint side."""
    print("\n--- RM run summary ---")
    print(json.dumps(summary, indent=2))
    if summary.get("push_error"):
        print(
            "\n[!] Hub push failed; the adapter trained successfully but the "
            "upload didn't land. Modal containers are ephemeral by default — "
            "if you need a re-push, re-run with the same config."
        )
    elif summary.get("hub_repo"):
        print(f"\nAdapter pushed to: https://huggingface.co/{summary['hub_repo']}")


@app.local_entrypoint()
def smoke() -> None:
    """50-step wiring probe (~5 min, ~$0.30). No Hub push.

    Proves the RM pipeline trains end-to-end on Modal: the SequenceClassification
    head loads, the regression head sees chosen/rejected gradient signal, the
    LoRA SEQ_CLS task_type wires correctly to the model, and the loss doesn't
    NaN. RM loss should decrease from ~0.693 (random) toward 0 as the
    classifier separates chosen from rejected.
    """
    _print_summary(run_rm_remote.remote(max_steps=50, push_to_hub=False))


@app.local_entrypoint()
def main(max_steps: int | None = None, no_push_to_hub: bool = False) -> None:
    """Full RM run per ``configs/rm_qwen05b.yaml``. Pushes the trained adapter
    to ``cfg.output.hub_repo`` unless ``--no-push-to-hub`` is passed.
    """
    _print_summary(run_rm_remote.remote(max_steps=max_steps, push_to_hub=not no_push_to_hub))
