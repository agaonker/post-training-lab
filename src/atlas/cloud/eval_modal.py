"""Modal entrypoint for the vLLM eval — the scripted, reproducible counterpart to
``notebooks/modal/run_sft_modal.ipynb``.

Each ``modal run`` is a fresh container that provisions a GPU, runs the *existing*
``atlas.eval.harness.run_eval``, and exits — so vLLM's GPU pinning / kernel-restart
pain (the reason this moved out of notebooks) simply doesn't apply. The harness is
backend-agnostic; this module only chooses the runner and forces ``backend="vllm"``.

Why the structure is what it is:
- The image installs the project's deps straight from ``pyproject.toml`` (single source
  of truth, no duplicated dep list) and adds **vllm** only here — vllm has no macOS
  build and stays out of pyproject by project convention (see CLAUDE.md).
- The local ``atlas`` source + ``configs/`` are mounted, so edits run immediately with
  no git push. (Swap to a ``git+https://...@branch`` install for a fully pinned image.)
- Results: the GPU function returns the run ``entry`` dict; the local entrypoint appends
  it to the repo's ``results/metrics.json`` on your laptop. Only the HF *download* cache
  is a Volume (warm reruns) — there is deliberately no result/checkpoint cache.

One-time setup on your laptop:
    pip install modal && modal token new
    # reuse the secret the SFT notebook already uses (exposes HUGGING_FACE_HUB_TOKEN):
    modal secret create hf-token HUGGING_FACE_HUB_TOKEN=hf_xxx

Usage (run from the repo root — the image reads pyproject.toml + configs/ locally):
    # cheap de-risk probe (validates Modal + vLLM wiring without a full run):
    modal run src/atlas/cloud/eval_modal.py --name base --method none --limit 50
    # the real comparison — base + sft back-to-back, clean two-row metrics.json:
    modal run src/atlas/cloud/eval_modal.py::suite --fresh
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# bf16-native GPU by default (Ada). A10G / A100 / L4 are all native bf16; a T4/P100
# would force float16 and reopen the dtype-consistency issue. Override with ATLAS_EVAL_GPU.
GPU = os.environ.get("ATLAS_EVAL_GPU", "L4")

app = modal.App("atlas-eval")

# Deps from pyproject (incl. the [eval] extra) + vllm (Modal-image-only). vllm is left
# unpinned for the first build; pin it to the resolved version afterward for reproducibility.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["eval"])
    .pip_install("vllm")  # TODO: pin to the version the first green build resolves
    .add_local_python_source("atlas")
    .add_local_dir("configs", remote_path="/root/configs")
)

# Persist the HF download cache (model weights + datasets) so reruns skip the ~1 GB
# Qwen + MMLU parquet downloads. This is a download cache only — not request/result caching.
hf_cache = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,  # generous: covers a cold model download + a full 4-task run
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("hf-token")],  # exposes HUGGING_FACE_HUB_TOKEN
)
def run_eval_remote(
    name: str, method: str, adapter: str | None = None, limit: int | None = None
) -> dict:
    """Run the eval on a Modal GPU with the vLLM backend; return the metrics entry."""
    from atlas.eval.harness import run_eval
    from atlas.utils.config import load_config

    # snapshot_download reads HF_TOKEN; the hf-token secret exposes HUGGING_FACE_HUB_TOKEN.
    os.environ.setdefault("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))

    cfg = load_config("/root/configs/baseline.yaml")
    cfg.eval.backend = "vllm"  # excluded from config_hash → fingerprint unchanged
    return run_eval(
        cfg,
        name=name,
        method=method,
        adapter=adapter,
        metrics_path=Path("/tmp/metrics.json"),  # throwaway in-container; we use the return
        limit_override=limit,
        config_path=Path("configs/baseline.yaml"),
    )


def _append_local(entry: dict) -> None:
    """Append a returned entry to the repo's results/metrics.json (runs on your laptop)."""
    from atlas.eval.harness import append_run

    append_run(entry, Path("results/metrics.json"))
    print(f"\n--- {entry['name']} ({entry['method']}) [{entry['config_hash']}] ---")
    for k, v in entry["metrics"].items():
        if "sample_len" not in k:
            print(f"  {k:50s} {v:.4f}")


def _reset_metrics() -> None:
    """Start results/metrics.json fresh (used by --fresh for a clean re-baseline)."""
    from atlas.eval.harness import SCHEMA_VERSION

    path = Path("results/metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "runs": []}, indent=2) + "\n")
    print("Reset results/metrics.json to empty.")


@app.local_entrypoint()
def main(
    name: str = "base",
    method: str = "none",
    adapter: str | None = None,
    limit: int | None = None,
) -> None:
    """Single eval run (mirrors the CLI). Use --limit 50 for the cheap wiring probe."""
    _append_local(run_eval_remote.remote(name, method, adapter, limit))


@app.local_entrypoint()
def suite(
    adapter: str = "agaonker/atlas-sft-qwen05b-v1",
    limit: int | None = None,
    fresh: bool = False,
) -> None:
    """Run base + sft_v1 back-to-back so they share image/backend/dtype → clean comparison.

    Sequential so the HF-cache Volume warms once (base downloads, sft reuses). Pass --fresh
    to reset metrics.json first (the re-baseline: yields exactly the two clean rows).
    """
    if fresh:
        _reset_metrics()
    _append_local(run_eval_remote.remote("base", "none", None, limit))
    _append_local(run_eval_remote.remote("sft_v1", "sft", adapter, limit))
