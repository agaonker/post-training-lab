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
    _partial_path,
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


def test_build_model_args_records_backend():
    cfg = load_config(CONFIGS / "baseline.yaml")
    assert "backend=hf" in _build_model_args(cfg, adapter=None)

    cfg.eval.backend = "vllm"
    args = _build_model_args(cfg, adapter=None)
    assert "backend=vllm" in args
    assert "tensor_parallel_size=1" in args
    assert "data_parallel_size=1" in args


# --- backend selection in _build_lm (lm-eval faked, so this runs offline) -------


def _fake_lm_eval_submodule(monkeypatch, dotted: str, **attrs):
    """Inject a fake lm_eval.* module tree into sys.modules for the duration of a test."""
    import sys
    import types

    parts = dotted.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    leaf = types.ModuleType(dotted)
    for key, val in attrs.items():
        setattr(leaf, key, val)
    monkeypatch.setitem(sys.modules, dotted, leaf)
    return leaf


def test_build_lm_hf_passes_pretrained_dtype_and_peft(monkeypatch):
    _fake_lm_eval_submodule(
        monkeypatch, "lm_eval.models.huggingface", HFLM=lambda **kw: ("HFLM", kw)
    )
    cfg = load_config(CONFIGS / "baseline.yaml")  # backend defaults to hf

    tag, kw = harness._build_lm(cfg, adapter="user/atlas-sft-v1")
    assert tag == "HFLM"
    assert kw["pretrained"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert kw["dtype"] == "bfloat16"
    assert kw["peft"] == "user/atlas-sft-v1"


def test_build_lm_vllm_downloads_adapter_and_passes_knobs(monkeypatch):
    _fake_lm_eval_submodule(monkeypatch, "lm_eval.models.vllm_causallms", VLLM=lambda **kw: kw)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: f"/local/{repo_id}")

    cfg = load_config(CONFIGS / "baseline.yaml")
    cfg.eval.backend = "vllm"

    kw = harness._build_lm(cfg, adapter="user/atlas-sft-v1")
    assert kw["pretrained"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert kw["dtype"] == "bfloat16"
    assert kw["seed"] == cfg.seed
    assert kw["max_lora_rank"] == 16
    assert kw["gpu_memory_utilization"] == 0.6  # lowered from 0.9: see configs/base.yaml
    assert kw["lora_local_path"] == "/local/user/atlas-sft-v1"
    # max_model_len is null in the config → not forwarded
    assert "max_model_len" not in kw


def test_build_lm_vllm_omits_lora_without_adapter(monkeypatch):
    _fake_lm_eval_submodule(monkeypatch, "lm_eval.models.vllm_causallms", VLLM=lambda **kw: kw)
    cfg = load_config(CONFIGS / "baseline.yaml")
    cfg.eval.backend = "vllm"

    kw = harness._build_lm(cfg, adapter=None)
    assert "lora_local_path" not in kw


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


# --- incremental persistence + resume -------------------------------------------


def _stub_lm_and_eval(monkeypatch, fake_results: dict[str, dict], seen_tasks: list[str]):
    """Helper: stub out the two lm-eval seams so run_eval is fully offline."""
    monkeypatch.setattr(harness, "_build_lm", lambda cfg, adapter: object())

    def fake_eval(lm, task_name, *, num_fewshot, batch_size, limit, random_seed):
        seen_tasks.append(task_name)
        return fake_results[task_name]

    monkeypatch.setattr(harness, "_evaluate_one_task", fake_eval)


def test_partial_path_is_run_scoped(tmp_path: Path):
    """Two runs sharing a metrics_path get separate partial files (no collision)."""
    metrics = tmp_path / "metrics.json"
    assert _partial_path(metrics, "base") == tmp_path / "metrics.base.partial.json"
    assert _partial_path(metrics, "smoke") == tmp_path / "metrics.smoke.partial.json"
    assert _partial_path(metrics, "base") != _partial_path(metrics, "smoke")


def test_run_eval_writes_partial_after_each_task(tmp_path: Path, monkeypatch):
    """The partial file's results dict grows by one entry per completed task."""
    cfg = load_config(CONFIGS / "baseline.yaml")
    fake_results = {name: {"acc,none": 0.1} for name in cfg.eval.tasks}
    seen: list[str] = []
    snapshots: list[set[str]] = []

    monkeypatch.setattr(harness, "_build_lm", lambda cfg, adapter: object())

    metrics_path = tmp_path / "metrics.json"
    partial = _partial_path(metrics_path, "base")

    def fake_eval(lm, task_name, *, num_fewshot, batch_size, limit, random_seed):
        seen.append(task_name)
        # Snapshot the partial AFTER it gets written for this task. We capture it
        # via a side effect on the next call — see the wrapper below.
        return fake_results[task_name]

    monkeypatch.setattr(harness, "_evaluate_one_task", fake_eval)

    # Wrap _write_partial to record what the partial held at each write point.
    real_write = harness._write_partial

    def recording_write(path, payload):
        real_write(path, payload)
        snapshots.append(set(payload["results"].keys()))

    monkeypatch.setattr(harness, "_write_partial", recording_write)

    run_eval(cfg, name="base", method="none", metrics_path=metrics_path)

    expected_growth = [set(seen[: i + 1]) for i in range(len(seen))]
    assert snapshots == expected_growth, "partial should grow by one task per write"
    # Final run cleaned up the partial...
    assert not partial.exists()
    # ...and the canonical metrics.json has the finalized entry.
    assert metrics_path.exists()


def test_run_eval_resume_skips_completed_tasks(tmp_path: Path, monkeypatch):
    """A pre-existing partial with some tasks done makes run_eval skip them."""
    cfg = load_config(CONFIGS / "baseline.yaml")
    task_names = list(cfg.eval.tasks.keys())
    already_done = task_names[:2]
    remaining = task_names[2:]

    metrics_path = tmp_path / "metrics.json"
    partial = _partial_path(metrics_path, "base")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps(
            {
                "config_hash": cfg.config_hash,
                "results": {name: {"acc,none": 0.99} for name in already_done},
            }
        )
    )

    fake_results = {name: {"acc,none": 0.1} for name in cfg.eval.tasks}
    seen: list[str] = []
    _stub_lm_and_eval(monkeypatch, fake_results, seen)

    entry = run_eval(cfg, name="base", method="none", metrics_path=metrics_path)

    # Only the not-yet-done tasks got executed
    assert seen == remaining
    # Resumed-task results survive into the final entry unchanged
    for name in already_done:
        assert entry["metrics"][f"{name}/acc,none"] == 0.99
    # Fresh-run results are also in the entry
    for name in remaining:
        assert entry["metrics"][f"{name}/acc,none"] == 0.1
    # Partial cleaned up on success
    assert not partial.exists()


def test_run_eval_no_resume_ignores_partial(tmp_path: Path, monkeypatch):
    """resume=False forces a full re-run even if a valid partial exists."""
    cfg = load_config(CONFIGS / "baseline.yaml")
    task_names = list(cfg.eval.tasks.keys())

    metrics_path = tmp_path / "metrics.json"
    partial = _partial_path(metrics_path, "base")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps(
            {
                "config_hash": cfg.config_hash,
                "results": {name: {"acc,none": 0.99} for name in task_names[:2]},
            }
        )
    )

    fake_results = {name: {"acc,none": 0.1} for name in cfg.eval.tasks}
    seen: list[str] = []
    _stub_lm_and_eval(monkeypatch, fake_results, seen)

    entry = run_eval(cfg, name="base", method="none", metrics_path=metrics_path, resume=False)

    assert seen == task_names  # every task re-run
    for name in task_names:
        assert entry["metrics"][f"{name}/acc,none"] == 0.1  # fresh values
    assert not partial.exists()


def test_run_eval_rejects_partial_with_different_config_hash(tmp_path: Path, monkeypatch):
    """A config change since the partial was written must refuse to resume."""
    cfg = load_config(CONFIGS / "baseline.yaml")

    metrics_path = tmp_path / "metrics.json"
    partial = _partial_path(metrics_path, "base")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps({"config_hash": "deadbeef", "results": {"mmlu": {"acc,none": 0.5}}})
    )

    _stub_lm_and_eval(monkeypatch, {}, [])

    with pytest.raises(ValueError, match="config_hash"):
        run_eval(cfg, name="base", method="none", metrics_path=metrics_path)
    # Partial untouched so the user can inspect / delete it themselves.
    assert partial.exists()


def test_run_eval_partial_survives_when_append_fails(tmp_path: Path, monkeypatch):
    """If append_run raises (e.g. schema mismatch), the partial sticks around."""
    cfg = load_config(CONFIGS / "baseline.yaml")
    fake_results = {name: {"acc,none": 0.1} for name in cfg.eval.tasks}
    seen: list[str] = []
    _stub_lm_and_eval(monkeypatch, fake_results, seen)

    metrics_path = tmp_path / "metrics.json"
    # Pre-seed with a wrong schema_version so append_run will raise.
    metrics_path.write_text(json.dumps({"schema_version": 99, "runs": []}))

    partial = _partial_path(metrics_path, "base")
    with pytest.raises(ValueError, match="schema_version"):
        run_eval(cfg, name="base", method="none", metrics_path=metrics_path)

    # Every task ran (we got far enough to attempt the append)...
    assert seen == list(cfg.eval.tasks.keys())
    # ...and the partial was kept so the work isn't lost.
    assert partial.exists()
    payload = json.loads(partial.read_text())
    assert set(payload["results"].keys()) == set(cfg.eval.tasks.keys())
