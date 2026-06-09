"""Modal entrypoint for RLOO training — Phase 3B.

Mirrors ``sft_modal.py`` / ``dpo_modal.py`` / ``rm_modal.py`` line-for-line:
same image (default pyproject deps, no vllm — RLOO can use HF generation for
the smoke), same hf-cache volume, same hf-token secret, same WANDB_MODE=disabled.

What's different: RLOO has three model instances on the GPU (policy + ref
via LoRA toggle + reward model), so memory matters more than for SFT/DPO/RM.
The reward model loads in 4-bit; the policy loads in bf16 because
``merge_and_unload`` of sft_v2 needs full precision.

If the L4 22GB ends up tight, the right escape hatches are:
- Reduce ``per_device_train_batch_size`` (default 2)
- Reduce ``num_generations`` (default 4)
- Reduce ``max_length`` (default 1024)

PPO upstream is deprecated in TRL 1.4 — RLOO is the replacement. See
PROJECT.md §6 Phase 3 and LESSONS.md.

One-time setup (shared with SFT/DPO/RM):
    pip install modal && modal token new
    modal secret create hf-token HUGGING_FACE_HUB_TOKEN=hf_xxx   # WRITE scope

Usage:
    make rloo-modal-smoke    # 50 steps, no Hub push, ~5-10 min, ~$0.30
    make rloo-modal          # full run, pushes to cfg.output.hub_repo
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

# sys.path guard — same pattern as the other cloud entrypoints.
_parents = Path(__file__).resolve().parents
if len(_parents) > 2 and (_parents[2] / "atlas").is_dir() and str(_parents[2]) not in sys.path:
    sys.path.insert(0, str(_parents[2]))

# RLOO holds three models on the GPU. L4 22GB fits a 0.5B policy (bf16) +
# RM (4-bit) + rollout KV cache. Bump to A10G if scaling the base.
GPU = os.environ.get("ATLAS_RLOO_GPU", "L4")

app = modal.App("atlas-rloo")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .env({"WANDB_MODE": "disabled"})
    .add_local_dir("src/atlas", remote_path="/root/atlas")
    .add_local_dir("configs", remote_path="/root/configs")
)

hf_cache = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    # Generous: covers cold model + dataset load + the full 5k-prompt run.
    # Smoke (50 steps) is ~5-10 min (longer than DPO because of rollouts);
    # full run measured ~3.2h on L4 (RLOO with num_generations=4 +
    # max_completion_length=512 is rollout-bound, much slower per step than
    # SFT/DPO/RM). 4h covers it with cushion. vLLM rollouts (use_vllm=True
    # + vllm_mode="colocate") would cut this ~5-10x; deferred to a future
    # optimization.
    timeout=60 * 60 * 4,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_rloo_remote(
    max_steps: int | None = None,
    push_to_hub: bool = True,
) -> dict:
    """Run RLOO on a Modal GPU; return the summary dict from
    ``atlas.train.rloo.run_rloo``."""
    from atlas.train.rloo import run_rloo
    from atlas.utils.config import load_config

    os.environ.setdefault("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))

    cfg = load_config("/root/configs/rloo_qwen05b.yaml")
    return run_rloo(
        cfg,
        max_steps_override=max_steps,
        push_to_hub=push_to_hub,
    )


def _print_summary(summary: dict) -> None:
    print("\n--- RLOO run summary ---")
    print(json.dumps(summary, indent=2))
    if summary.get("push_error"):
        print(
            "\n[!] Hub push failed; the adapter trained but the upload didn't "
            "land. Modal containers are ephemeral — re-run with the same config."
        )
    elif summary.get("hub_repo"):
        print(f"\nAdapter pushed to: https://huggingface.co/{summary['hub_repo']}")


@app.local_entrypoint()
def smoke() -> None:
    """50-step wiring probe (~5-10 min, ~$0.30). No Hub push.

    Proves the RLOO pipeline trains end-to-end on Modal: the three-model
    setup (policy merged with sft_v2 + new LoRA, frozen ref via LoRA toggle,
    4-bit RM) loads, rollouts generate, rewards score, REINFORCE updates the
    policy, KL penalty against the ref keeps the policy near sft_v2.

    Watch:
    - Mean reward should *increase* over the smoke (otherwise the policy
      isn't responding to the gradient).
    - KL should stay bounded — TRL prints it every logging_steps.
    - If KL blows up rapidly, ``beta`` (KL coefficient) is too small.
    """
    _print_summary(run_rloo_remote.remote(max_steps=50, push_to_hub=False))


@app.local_entrypoint()
def main(max_steps: int | None = None, no_push_to_hub: bool = False) -> None:
    """Full RLOO run per ``configs/rloo_qwen05b.yaml``. Pushes to
    ``cfg.output.hub_repo`` unless ``--no-push-to-hub`` is passed."""
    _print_summary(run_rloo_remote.remote(max_steps=max_steps, push_to_hub=not no_push_to_hub))
