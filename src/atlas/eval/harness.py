"""Shared eval harness — wraps lm-eval-harness, emits results/metrics.json.

One entry point for every phase. Phase 0's baseline calls this with ``adapter=None``;
SFT, DPO, etc. call it with their HF Hub adapter id. Same metric extraction, same
JSON schema — apples-to-apples by construction.

lm-eval is imported lazily inside :func:`_build_lm` / :func:`_evaluate_one_task` so
importing this module (e.g. in CI without the ``[eval]`` extra) stays cheap and
tests can stub the heavy bits without ever loading model weights.
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
    if adapter:
        parts.append(f"peft={adapter}")
    return ",".join(parts)


def _build_lm(cfg: Config, adapter: str | None) -> Any:
    """Instantiate lm-eval's HuggingFace LM once; reused across every task call.

    Re-importing per task call would reload the weights — wasteful even at 0.5B.
    """
    from lm_eval.models.huggingface import HFLM

    kwargs: dict[str, Any] = {"pretrained": cfg.model.name, "dtype": cfg.model.dtype}
    if cfg.model.revision:
        kwargs["revision"] = cfg.model.revision
    if adapter:
        kwargs["peft"] = adapter
    return HFLM(**kwargs)


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
            if isinstance(value, (int, float)) and "stderr" not in metric_name:
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


def _utc_timestamp() -> str:
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def run_eval(
    cfg: Config,
    *,
    name: str,
    method: str,
    adapter: str | None = None,
    limit_override: int | None = None,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run every task in ``cfg.eval.tasks`` and append one entry to metrics.json."""
    lm = _build_lm(cfg, adapter)
    aggregated: dict[str, Any] = {"results": {}}

    for task_name, task_cfg in cfg.eval.tasks.items():
        limit = limit_override if limit_override is not None else task_cfg.limit
        aggregated["results"][task_name] = _evaluate_one_task(
            lm,
            task_name,
            num_fewshot=task_cfg.num_fewshot,
            batch_size=cfg.eval.batch_size,
            limit=limit,
            random_seed=cfg.seed,
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    entry = run_eval(
        cfg,
        name=args.name,
        method=args.method,
        adapter=args.adapter,
        limit_override=args.limit,
        metrics_path=args.metrics_path,
        config_path=args.config,
    )
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
