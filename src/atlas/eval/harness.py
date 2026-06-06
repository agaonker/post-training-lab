"""Shared eval harness — wraps lm-eval-harness, emits results/metrics.json.

One entry point for every phase. Phase 0's baseline calls this with ``adapter=None``;
SFT, DPO, etc. call it with their HF Hub adapter id. Same metric extraction, same
JSON schema — apples-to-apples by construction.

lm-eval is imported lazily inside :func:`_build_lm` / :func:`_evaluate_one_task` so
importing this module (e.g. in CI without the ``[eval]`` extra) stays cheap and
tests can stub the heavy bits without ever loading model weights.

Incremental persistence: results are written to a ``.<run_name>.partial.json``
sibling of ``metrics.json`` after each task completes. A crash mid-run loses at
most the in-flight task; a subsequent invocation with the same ``--name`` resumes
from the partial and skips already-completed tasks. The partial is deleted once
the run is finalized into ``metrics.json``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atlas.utils.config import Config, load_config

SCHEMA_VERSION = 1
DEFAULT_METRICS_PATH = Path("results/metrics.json")


def _build_model_args(cfg: Config, adapter: str | None) -> str:
    """lm-eval ``model_args`` string — kept for diagnostics even though we pass an LM object."""
    parts = [f"pretrained={cfg.model.name}", f"dtype={cfg.model.dtype}"]
    if cfg.model.revision:
        parts.append(f"revision={cfg.model.revision}")
    if cfg.model.tokenizer_name:
        parts.append(f"tokenizer={cfg.model.tokenizer_name}")
    if adapter:
        parts.append(f"peft={adapter}")
    parts.append(f"backend={cfg.eval.backend}")
    if cfg.eval.backend == "vllm":
        v = cfg.eval.vllm
        parts.append(f"tensor_parallel_size={v.tensor_parallel_size}")
        parts.append(f"data_parallel_size={v.data_parallel_size}")
    return ",".join(parts)


def _build_lm(cfg: Config, adapter: str | None) -> Any:
    """Instantiate the lm-eval LM once; reused across every task call.

    Two backends, selected by ``cfg.eval.backend``: the default HF path (``HFLM``) and
    vLLM (CUDA-only, the Modal fast path). Both lm-eval imports stay *inside* this
    function so importing the harness in CI (no ``[eval]`` extra, no vllm) stays cheap
    and the function remains a monkeypatch seam for the offline tests.
    """
    if cfg.eval.backend == "vllm":
        return _build_vllm_lm(cfg, adapter)

    from lm_eval.models.huggingface import HFLM

    kwargs: dict[str, Any] = {"pretrained": cfg.model.name, "dtype": cfg.model.dtype}
    if cfg.model.revision:
        kwargs["revision"] = cfg.model.revision
    if cfg.model.tokenizer_name:
        kwargs["tokenizer"] = cfg.model.tokenizer_name
    if adapter:
        kwargs["peft"] = adapter
    return HFLM(**kwargs)


def _build_vllm_lm(cfg: Config, adapter: str | None) -> Any:
    """vLLM backend — fast continuous-batching generation on a CUDA GPU.

    A Hub adapter is first materialized to a local directory: vLLM's LoRA loader takes a
    local path (``lora_local_path``), not a Hub id, and derives ``enable_lora`` from it.
    ``gpu_memory_utilization`` rides through ``**kwargs`` into the vLLM engine.
    """
    from lm_eval.models.vllm_causallms import VLLM

    v = cfg.eval.vllm
    kwargs: dict[str, Any] = {
        "pretrained": cfg.model.name,
        "dtype": cfg.model.dtype,
        "seed": cfg.seed,
        "tensor_parallel_size": v.tensor_parallel_size,
        "data_parallel_size": v.data_parallel_size,
        "max_lora_rank": v.max_lora_rank,
        "gpu_memory_utilization": v.gpu_memory_utilization,
    }
    if cfg.model.revision:
        kwargs["revision"] = cfg.model.revision
    if cfg.model.tokenizer_name:
        kwargs["tokenizer"] = cfg.model.tokenizer_name
    if v.max_model_len is not None:
        kwargs["max_model_len"] = v.max_model_len
    if adapter:
        from huggingface_hub import snapshot_download

        kwargs["lora_local_path"] = snapshot_download(adapter)
    return VLLM(**kwargs)


def _evaluate_one_task(
    lm: Any,
    task_name: str,
    num_fewshot: int,
    batch_size: int | str,
    limit: int | None,
    random_seed: int,
) -> dict[str, Any]:
    """Single ``simple_evaluate`` call — one task at a time so per-task num_fewshot
    flows naturally and a failure in one task doesn't sink the rest."""
    from lm_eval import simple_evaluate

    result = simple_evaluate(
        model=lm,
        tasks=[task_name],
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        random_seed=random_seed,
    )
    return result.get("results", {}).get(task_name, {})


def _flatten_metrics(aggregated: dict[str, Any]) -> dict[str, float]:
    """Flatten lm-eval's ``{task: {metric: value}}`` to ``{task/metric: value}``.

    Drops stderr keys and non-numeric values (e.g. ``alias``). lm-eval metric names
    include the filter suffix after a comma (``acc,none``) — kept verbatim.
    """
    flat: dict[str, float] = {}
    for task_name, task_metrics in aggregated.get("results", {}).items():
        for metric_name, value in task_metrics.items():
            if isinstance(value, int | float) and "stderr" not in metric_name:
                flat[f"{task_name}/{metric_name}"] = float(value)
    return flat


def append_run(entry: dict[str, Any], metrics_path: Path = DEFAULT_METRICS_PATH) -> None:
    """Append one run entry to ``metrics.json``, creating the file (and parent dir) if absent."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        doc = json.loads(metrics_path.read_text())
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{metrics_path} has schema_version={doc.get('schema_version')}, "
                f"harness expects {SCHEMA_VERSION}"
            )
    else:
        doc = {"schema_version": SCHEMA_VERSION, "runs": []}
    doc["runs"].append(entry)
    metrics_path.write_text(json.dumps(doc, indent=2) + "\n")


def _partial_path(metrics_path: Path, run_name: str) -> Path:
    """Sibling ``.<run_name>.partial.json`` next to the canonical metrics file.

    Run-name-scoped so a smoke run and a baseline run on the same metrics path
    don't collide. Deleted once the run is finalized into ``metrics.json``.
    """
    return metrics_path.parent / f"{metrics_path.stem}.{run_name}.partial.json"


def _write_partial(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write: temp + rename, so a crash mid-write can't leave a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _load_partial(path: Path, expected_config_hash: str) -> dict[str, Any]:
    """Return partial payload if present and valid for this config_hash, else ``{}``.

    A hash mismatch means the config changed since the partial was written — refuse
    to resume rather than silently mix runs. The user can delete the partial to
    start fresh, or revert the config change.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    seen_hash = payload.get("config_hash")
    if seen_hash != expected_config_hash:
        raise ValueError(
            f"{path} was written for config_hash={seen_hash}, current config "
            f"is {expected_config_hash}. Delete the partial file to start fresh "
            f"or revert your config."
        )
    return payload


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_eval(
    cfg: Config,
    *,
    name: str,
    method: str,
    adapter: str | None = None,
    limit_override: int | None = None,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    config_path: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run every task in ``cfg.eval.tasks`` and append one entry to metrics.json.

    Persists per-task results to a sibling ``.<name>.partial.json`` file as the
    loop progresses, so a crash mid-run loses at most one task. With ``resume=True``
    (default), a re-invocation picks up where the previous run left off; pass
    ``resume=False`` to ignore any existing partial and start fresh.
    """
    partial_path = _partial_path(metrics_path, name)
    completed: dict[str, Any] = {}
    if resume:
        completed = _load_partial(partial_path, cfg.config_hash).get("results", {})

    aggregated: dict[str, Any] = {"results": dict(completed)}
    # Defer LM construction until we know there's a task to actually run — a
    # fully-resumed run pays no model-load cost.
    lm: Any = None

    for task_name, task_cfg in cfg.eval.tasks.items():
        if task_name in completed:
            continue
        if lm is None:
            lm = _build_lm(cfg, adapter)
        limit = limit_override if limit_override is not None else task_cfg.limit
        aggregated["results"][task_name] = _evaluate_one_task(
            lm,
            task_name,
            num_fewshot=task_cfg.num_fewshot,
            batch_size=cfg.eval.batch_size,
            limit=limit,
            random_seed=cfg.seed,
        )
        _write_partial(
            partial_path,
            {"config_hash": cfg.config_hash, "results": aggregated["results"]},
        )

    entry: dict[str, Any] = {
        "name": name,
        "method": method,
        "model": cfg.model.name,
        "adapter": adapter,
        "config_hash": cfg.config_hash,
        "config_path": str(config_path) if config_path else None,
        "model_args": _build_model_args(cfg, adapter),
        "timestamp": _utc_timestamp(),
        "metrics": _flatten_metrics(aggregated),
        "extras": {},
    }
    append_run(entry, metrics_path)
    # Only clear the partial after a successful append — if append raises (e.g.
    # schema_version mismatch), the partial stays around for manual recovery.
    if partial_path.exists():
        partial_path.unlink()
    return entry


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run lm-eval on an experiment config and append to metrics.json"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--name", required=True, help="Run name, e.g. 'base' or 'sft_v1'")
    parser.add_argument(
        "--method",
        required=True,
        help="Method label: none | sft | dpo | rm | ppo | grpo | kto | orpo",
    )
    parser.add_argument("--adapter", default=None, help="Optional HF Hub adapter repo id")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Override cfg.model.tokenizer_name. Set this to the adapter repo id when "
            "evaluating an SFT adapter that was trained on a tokenizer (different eos / "
            "chat_template) shipped with it, so lm-eval terminates generation correctly. "
            "Changes config_hash."
        ),
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["hf", "vllm"],
        help="Override eval.backend from the config (hf | vllm). vllm is CUDA-only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap samples per task (smoke runs); overrides per-task limits in YAML",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Where to write/append results JSON (default: results/metrics.json)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any .<name>.partial.json sibling and re-run every task from scratch",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg.eval.backend = args.backend  # excluded from config_hash, so no rehash needed
    if args.tokenizer:
        from atlas.utils.config import compute_config_hash

        cfg.model.tokenizer_name = args.tokenizer
        cfg.config_hash = compute_config_hash(cfg)
    entry = run_eval(
        cfg,
        name=args.name,
        method=args.method,
        adapter=args.adapter,
        limit_override=args.limit,
        metrics_path=args.metrics_path,
        config_path=args.config,
        resume=not args.no_resume,
    )
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
