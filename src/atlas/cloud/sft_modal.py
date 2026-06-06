"""Modal entrypoint for SFT — the scripted counterpart to
``notebooks/modal/run_sft_modal.ipynb``.

Each ``modal run`` is a fresh container that provisions a GPU, runs the *existing*
``atlas.train.sft.run_sft`` against ``configs/sft_qwen05b.yaml``, pushes the trained
adapter to HF Hub, and exits. The notebook stays as the exploratory entrypoint; this
module is the scripted, reproducible version once the wiring is known to work.

Why the structure mirrors ``eval_modal.py``:
- Image installs deps from ``pyproject.toml`` (single source of truth — trl, peft,
  transformers, datasets, accelerate, bitsandbytes all live there). No optional extras
  needed for the training path; SFT does not pull lm-eval or vllm.
- Local ``atlas`` source + ``configs/`` are mounted, so edits run immediately with no
  git push.
- WANDB_MODE is forced to ``disabled`` at the image level — wandb is nice-to-have for
  loss curves, but a missing ``WANDB_API_KEY`` secret crashes ``trainer.train()`` on
  ``on_train_begin``. Re-enable by creating a ``wandb`` Modal secret + attaching it
  below + dropping the env override.
- HF cache (model weights + dataset) lives in the same ``atlas-hf-cache`` Volume the
  eval uses, so a warm SFT skips the ~1 GB Qwen + UltraChat downloads.
- The trained adapter is *only* persisted via ``HfApi.upload_folder`` inside
  ``run_sft`` — there is deliberately no container-side checkpoint volume.

One-time setup on your laptop:
    pip install modal && modal token new
    modal secret create hf-token HUGGING_FACE_HUB_TOKEN=hf_xxx   # WRITE scope

Usage (run from the repo root — the image mounts src/atlas + configs from cwd):
    make sft-modal-smoke    # 50 steps, no Hub push, ~5 min, ~$0.30  — wiring probe
    make sft-modal          # full run per configs/sft_qwen05b.yaml, pushes to Hub

Raw entrypoints (the make targets wrap these):
    modal run src/atlas/cloud/sft_modal.py::smoke
    modal run src/atlas/cloud/sft_modal.py::main
    modal run src/atlas/cloud/sft_modal.py::main --max-steps 50 --no-push-to-hub
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

# Same sys.path guard pattern as eval_modal.py — the modal CLI loads this file from your
# laptop's Python (not this project's .venv), so atlas may not be importable yet. In the
# Modal container this file is /root/sft_modal.py and atlas is already at /root/atlas, so
# the depth + layout check makes it a no-op remotely.
_parents = Path(__file__).resolve().parents  # local: <repo>/src/atlas/cloud/sft_modal.py
if len(_parents) > 2 and (_parents[2] / "atlas").is_dir() and str(_parents[2]) not in sys.path:
    sys.path.insert(0, str(_parents[2]))  # <repo>/src

# bf16-native GPU default. L4 22GB easily fits a 0.5B + LoRA + 4-bit quant; bump to A10G
# (or A100) if you scale the base model up. Override with ATLAS_SFT_GPU.
GPU = os.environ.get("ATLAS_SFT_GPU", "L4")

app = modal.App("atlas-sft")

# Default-extras only — SFT doesn't need lm-eval or vllm. trl/peft/transformers/datasets/
# accelerate/bitsandbytes are all in pyproject's main dependencies.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    # Disable wandb at the image level so a missing WANDB_API_KEY doesn't crash
    # trainer.train() in on_train_begin. To enable: create a `wandb` Modal secret,
    # attach it on the function below, and drop this env override.
    .env({"WANDB_MODE": "disabled"})
    .add_local_dir("src/atlas", remote_path="/root/atlas")
    .add_local_dir("configs", remote_path="/root/configs")
)

# Reuse the HF download cache from the eval module — Qwen2.5-0.5B + UltraChat-200k are
# shared between SFT and eval, so a warm SFT skips a re-download.
hf_cache = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    # Generous: covers a cold model download + the full 5k-sample run. Smoke (50 steps)
    # finishes in ~5 min; full run is ~30-60 min on L4.
    timeout=60 * 60 * 2,
    volumes={"/root/.cache/huggingface": hf_cache},
    # hf-token must have WRITE scope on the namespace named in cfg.output.hub_repo —
    # SFT preflights this before training to fail-fast on a read-only token.
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_sft_remote(
    max_steps: int | None = None,
    push_to_hub: bool = True,
) -> dict:
    """Run SFT on a Modal GPU; return the summary dict from ``atlas.train.sft.run_sft``."""
    from atlas.train.sft import run_sft
    from atlas.utils.config import load_config

    # HfApi reads HF_TOKEN; the hf-token secret exposes HUGGING_FACE_HUB_TOKEN.
    os.environ.setdefault("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))

    cfg = load_config("/root/configs/sft_qwen05b.yaml")
    return run_sft(
        cfg,
        max_steps_override=max_steps,
        push_to_hub=push_to_hub,
    )


def _print_summary(summary: dict) -> None:
    """Pretty-print the run_sft summary on the local entrypoint side."""
    print("\n--- SFT run summary ---")
    print(json.dumps(summary, indent=2))
    if summary.get("push_error"):
        print(
            "\n[!] Hub push failed; the adapter still trained successfully — recover via "
            "HfApi().upload_folder on Modal's persisted output_dir if Modal mode keeps "
            "containers (default: ephemeral, no recovery)."
        )
    elif summary.get("hub_repo"):
        print(f"\nAdapter pushed to: https://huggingface.co/{summary['hub_repo']}")


@app.local_entrypoint()
def smoke() -> None:
    """50-step wiring probe (~5 min, ~$0.30). No Hub push — proves the pipeline trains
    end-to-end on Modal (4-bit quant + LoRA + patched chat_template + assistant_only_loss)
    before committing to the full ~30-60 min, $1-2 run.
    """
    _print_summary(run_sft_remote.remote(max_steps=50, push_to_hub=False))


@app.local_entrypoint()
def main(max_steps: int | None = None, no_push_to_hub: bool = False) -> None:
    """Full SFT run per ``configs/sft_qwen05b.yaml``. Pushes the trained adapter to
    ``cfg.output.hub_repo`` unless ``--no-push-to-hub`` is passed.
    """
    _print_summary(run_sft_remote.remote(max_steps=max_steps, push_to_hub=not no_push_to_hub))
