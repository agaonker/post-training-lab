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

Usage (run from the repo root — the image reads pyproject.toml + configs/ locally).
A tiered loop, cheapest first — only pay for what a change can actually break:
    make eval-modal-check    # local, seconds, $0: imports/entrypoints/config preflight
    make eval-modal-probe    # ~1-2 min: one task, limit 5 — proves the vLLM path end-to-end
    make eval-modal          # ~10 min: the real base + sft comparison (full task list)

    # interactive container in this exact image (vllm + atlas + GPU) — iterate without a
    # fresh `modal run` per attempt; best for image-level issues (imports, the nvcc class):
    make eval-modal-shell    # then e.g.  python -c "import vllm, atlas; print('ok')"

Raw entrypoints (the make targets wrap these):
    modal run src/atlas/cloud/eval_modal.py::probe --task gsm8k --limit 5
    modal run src/atlas/cloud/eval_modal.py::main --name base --method none --limit 50
    modal run src/atlas/cloud/eval_modal.py::suite --fresh
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

# The `modal` CLI loads this file as a standalone module in your laptop's Python — which
# isn't this project's .venv and may not have `atlas` installed (modal is a separate CLI by
# design; see CLAUDE.md). The local entrypoints below import atlas.eval.harness to append
# results, so put the repo's src/ on sys.path. In the Modal container this file is /root/
# eval_modal.py (parents = [/root, /], no [2]) and atlas is already mounted at /root/atlas —
# so guard on depth AND on the src/atlas layout existing, making this a no-op remotely.
_parents = Path(__file__).resolve().parents  # local: <repo>/src/atlas/cloud/eval_modal.py
if len(_parents) > 2 and (_parents[2] / "atlas").is_dir() and str(_parents[2]) not in sys.path:
    sys.path.insert(0, str(_parents[2]))  # <repo>/src

# bf16-native GPU by default (Ada). A10G / A100 / L4 are all native bf16; a T4/P100
# would force float16 and reopen the dtype-consistency issue. Override with ATLAS_EVAL_GPU.
GPU = os.environ.get("ATLAS_EVAL_GPU", "L4")

app = modal.App("atlas-eval")

# Deps from pyproject (incl. the [eval] extra) + vllm (Modal-image-only). vllm is pinned to
# the version the first green build resolved, so the image is reproducible across rebuilds.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["eval"])
    .pip_install("vllm==0.21.0")  # pinned: first green build on L4 (CUDA runtime, no nvcc)
    # Force vLLM's native Torch top-k/top-p sampler. The default auto-picks FlashInfer,
    # which JIT-compiles its sampling kernel at warmup and needs nvcc + the CUDA toolkit
    # (/usr/local/cuda) — absent from debian_slim, which carries only the CUDA runtime.
    # Irrelevant for eval anyway: MMLU is loglikelihood (no sampling), IFEval is greedy.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    # Mount the source by path rather than add_local_python_source("atlas"): the latter
    # imports `atlas` in the *local* Python running the modal CLI to locate it, but modal
    # is a standalone laptop CLI (not in this project's .venv), where atlas isn't installed.
    # /root is on the container PYTHONPATH, so /root/atlas is importable as `atlas`.
    .add_local_dir("src/atlas", remote_path="/root/atlas")
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
    # Retire the container after one eval. `suite` calls this twice (base, then sft_v1);
    # without this Modal reuses the warm container, where the first run's vLLM EngineCore
    # subprocess still pins the GPU (~20/22 GiB) — so the second init fails the
    # gpu_memory_utilization check (OOM at startup). A fresh container = a clean GPU per run.
    single_use_containers=True,
)
def run_eval_remote(
    name: str,
    method: str,
    adapter: str | None = None,
    limit: int | None = None,
    only_tasks: list[str] | None = None,
    tokenizer: str | None = None,
) -> dict:
    """Run the eval on a Modal GPU with the vLLM backend; return the metrics entry.

    ``only_tasks`` trims ``eval.tasks`` to the named subset (the fast ``probe`` path uses it
    to skip MMLU's 57-subject fan-out). None = the full task list from the config.

    ``tokenizer`` overrides ``cfg.model.tokenizer_name`` at runtime — set this to the
    adapter repo id when evaling an SFT adapter whose tokenizer was modified at train
    time (patched chat_template, eos override). Without it, lm-eval falls back to the
    base's tokenizer and generation tasks won't terminate cleanly. Changes config_hash.
    """
    from atlas.eval.harness import run_eval
    from atlas.utils.config import compute_config_hash, load_config

    # snapshot_download reads HF_TOKEN; the hf-token secret exposes HUGGING_FACE_HUB_TOKEN.
    os.environ.setdefault("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))

    cfg = load_config("/root/configs/baseline.yaml")
    cfg.eval.backend = "vllm"  # excluded from config_hash → fingerprint unchanged
    if tokenizer:
        cfg.model.tokenizer_name = tokenizer
        cfg.config_hash = compute_config_hash(cfg)  # included in hash → rehash
    if only_tasks:
        missing = [t for t in only_tasks if t not in cfg.eval.tasks]
        if missing:
            raise ValueError(f"--task {missing} not in config tasks {sorted(cfg.eval.tasks)}")
        cfg.eval.tasks = {k: v for k, v in cfg.eval.tasks.items() if k in only_tasks}
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
def probe(task: str = "gsm8k", limit: int = 5) -> None:
    """Fast wiring probe (~1-2 min): one task, tiny limit — exercises the whole vLLM path
    (model load → generate → metric extraction → local append) without MMLU's 57-subject
    fan-out. For *numbers*, not this: use ``main``/``suite``. Run ``eval-modal-check`` first.
    """
    _append_local(run_eval_remote.remote("probe", "none", None, limit, [task]))


@app.local_entrypoint()
def main(
    name: str = "base",
    method: str = "none",
    adapter: str | None = None,
    limit: int | None = None,
    tokenizer: str | None = None,
) -> None:
    """Single full-task-list eval run (mirrors the CLI). For the cheap wiring check use the
    ``probe`` entrypoint instead — ``--limit`` here still runs all 4 tasks (MMLU fans out to
    57 subjects × 4 choices, so even ``--limit 50`` is ~11k requests).

    Pass ``--tokenizer <adapter-repo>`` when evaling an SFT adapter whose tokenizer was
    modified during training (patched chat_template, eos override). Without it, generation
    tasks (GSM8K, IFEval) won't terminate cleanly. Changes config_hash."""
    _append_local(
        run_eval_remote.remote(name, method, adapter, limit, None, tokenizer)
    )


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
