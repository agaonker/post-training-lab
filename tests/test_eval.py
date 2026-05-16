"""Tests for atlas.eval.harness — pure helpers plus a fully-mocked end-to-end run.

lm-eval is never imported here: ``_build_lm`` and ``_evaluate_one_task`` are the
seams we monkeypatch, so CI runs without the ``[eval]`` extra and without ever
loading model weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas.eval import harness
from atlas.eval.harness import (
    SCHEMA_VERSION,
    _build_model_args,
    _flatten_metrics,
    append_run,
    run_eval,
)
from atlas.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


# --- pure helpers ---------------------------------------------------------------

def test_flatten_metrics_keeps_scalars_and_drops_stderr():
    aggregated = {
        "results": {
            "mmlu": {"acc,none": 0.45, "acc_stderr,none": 0.01, "alias": "mmlu"},
            "ifeval": {
                "prompt_level_strict_acc,none": 0.27,
                "prompt_level_strict_acc_stderr,none": 0.02,
            },
        }
    }
    flat = _flatten_metrics(aggregated)
    assert flat == {
        "mmlu/acc,none": 0.45,
        "ifeval/prompt_level_strict_acc,none": 0.27,
    }


def test_flatten_metrics_handles_missing_results_block():
    assert _flatten_metrics({}) == {}


def test_build_model_args_baseline_and_adapter():
    cfg = load_config(CONFIGS / "baseline.yaml")

    args = _build_model_args(cfg, adapter=None)
    assert "pretrained=Qwen/Qwen2.5-0.5B-Instruct" in args
    assert "dtype=bfloat16" in args
    assert "peft=" not in args

    args_adapter = _build_model_args(cfg, adapter="user/atlas-sft-v1")
    assert "peft=user/atlas-sft-v1" in args_adapter


# --- append_run / file I/O ------------------------------------------------------

def test_append_run_creates_file_with_schema(tmp_path: Path):
    path = tmp_path / "metrics.json"
    entry = {"name": "x", "method": "none", "metrics": {}}
    append_run(entry, path)
    doc = json.loads(path.read_text())
    assert doc == {"schema_version": SCHEMA_VERSION, "runs": [entry]}


def test_append_run_appends_to_existing_file(tmp_path: Path):
    path = tmp_path / "metrics.json"
    append_run({"name": "a"}, path)
    append_run({"name": "b"}, path)
    doc = json.loads(path.read_text())
    assert [r["name"] for r in doc["runs"]] == ["a", "b"]


def test_append_run_rejects_unknown_schema_version(tmp_path: Path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"schema_version": 99, "runs": []}))
    with pytest.raises(ValueError, match="schema_version"):
        append_run({"name": "x"}, path)


def test_append_run_creates_parent_directories(tmp_path: Path):
    path = tmp_path / "deep" / "nested" / "metrics.json"
    append_run({"name": "x"}, path)
    assert path.exists()


# --- run_eval end-to-end (lm-eval stubbed) --------------------------------------

def test_run_eval_writes_well_formed_entry(tmp_path: Path, monkeypatch):
    """Walks the full run_eval path with both lm-eval seams stubbed."""
    fake_results = {
        "mmlu": {"acc,none": 0.40, "acc_stderr,none": 0.01, "alias": "mmlu"},
        "gsm8k": {"strict-match,none": 0.30, "strict-match_stderr,none": 0.02},
        "truthfulqa_mc2": {"acc,none": 0.42},
        "ifeval": {"prompt_level_strict_acc,none": 0.25},
    }
    seen_tasks: list[str] = []

    monkeypatch.setattr(harness, "_build_lm", lambda cfg, adapter: object())

    def fake_eval(lm, task_name, *, num_fewshot, batch_size, limit, random_seed):
        seen_tasks.append(task_name)
        return fake_results[task_name]

    monkeypatch.setattr(harness, "_evaluate_one_task", fake_eval)

    cfg = load_config(CONFIGS / "baseline.yaml")
    metrics_path = tmp_path / "metrics.json"
    entry = run_eval(
        cfg,
        name="base",
        method="none",
        metrics_path=metrics_path,
        config_path=CONFIGS / "baseline.yaml",
    )

    # every task in the YAML got hit, exactly once
    assert set(seen_tasks) == set(cfg.eval.tasks.keys())
    assert len(seen_tasks) == len(cfg.eval.tasks)

    assert entry["name"] == "base"
    assert entry["method"] == "none"
    assert entry["adapter"] is None
    assert entry["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert entry["config_hash"] == cfg.config_hash
    assert entry["metrics"]["mmlu/acc,none"] == 0.40
    assert entry["metrics"]["ifeval/prompt_level_strict_acc,none"] == 0.25
    # stderr keys filtered out
    assert all("stderr" not in k for k in entry["metrics"])

    doc = json.loads(metrics_path.read_text())
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["runs"][0]["name"] == "base"


def test_run_eval_propagates_limit_override(tmp_path: Path, monkeypatch):
    seen_limits: list[int | None] = []

    monkeypatch.setattr(harness, "_build_lm", lambda cfg, adapter: object())

    def fake_eval(lm, task_name, *, num_fewshot, batch_size, limit, random_seed):
        seen_limits.append(limit)
        return {"acc,none": 0.0}

    monkeypatch.setattr(harness, "_evaluate_one_task", fake_eval)

    cfg = load_config(CONFIGS / "baseline.yaml")
    run_eval(
        cfg,
        name="smoke",
        method="none",
        limit_override=7,
        metrics_path=tmp_path / "metrics.json",
    )
    assert seen_limits == [7] * len(cfg.eval.tasks)
