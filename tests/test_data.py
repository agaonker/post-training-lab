"""Tests for the config system (atlas.utils.config).

Phase 0 covers config loading / merging / hashing. Dataset-loader tests are added to
this file in later phases as the loaders land.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.utils.config import (
    Config,
    compute_config_hash,
    load_config,
    merge_dicts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def test_merge_dicts_is_deep():
    base = {"model": {"name": "qwen", "dtype": "bfloat16"}, "seed": 42}
    override = {"model": {"dtype": "float16"}, "seed": 7}
    merged = merge_dicts(base, override)
    # nested override keeps the sibling key...
    assert merged["model"] == {"name": "qwen", "dtype": "float16"}
    # ...and top-level scalars are replaced
    assert merged["seed"] == 7
    # inputs are not mutated
    assert base["model"]["dtype"] == "bfloat16"


def test_merge_dicts_replaces_lists_wholesale():
    base = {"lora": {"target_modules": ["q_proj", "k_proj"]}}
    override = {"lora": {"target_modules": ["v_proj"]}}
    assert merge_dicts(base, override)["lora"]["target_modules"] == ["v_proj"]


def test_base_yaml_loads_and_validates():
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg.model.name == "Qwen/Qwen2.5-0.5B"
    assert cfg.lora.r == 16
    assert "ifeval" in cfg.eval.tasks
    assert cfg.eval.tasks["mmlu"].limit == 1000
    # no experiment-only blocks live in the shared base
    assert cfg.dataset is None and cfg.train is None


def test_load_config_merges_experiment_over_base(tmp_path):
    (tmp_path / "base.yaml").write_text("model:\n  name: qwen\n  dtype: bfloat16\nseed: 42\n")
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "model:\n  dtype: float16\ntrain:\n  output_dir: outputs/exp\n  learning_rate: 2.0e-4\n"
    )
    cfg = load_config(exp)
    assert cfg.model.name == "qwen"  # inherited from base, untouched
    assert cfg.model.dtype == "float16"  # overridden by the experiment
    assert cfg.train.output_dir == "outputs/exp"
    # TrainCfg is a permissive passthrough — unknown TRL knobs survive intact
    assert cfg.train.model_dump()["learning_rate"] == 2.0e-4


def test_config_rejects_unknown_top_level_key():
    with pytest.raises(ValidationError):
        Config.model_validate({"model": {"name": "qwen"}, "bogus": 1})


def test_config_hash_is_stable_and_sensitive():
    a = Config.model_validate({"model": {"name": "qwen"}})
    b = Config.model_validate({"model": {"name": "qwen"}})
    c = Config.model_validate({"model": {"name": "qwen"}, "seed": 7})
    ha, hb, hc = compute_config_hash(a), compute_config_hash(b), compute_config_hash(c)
    assert ha == hb  # identical configs hash identically
    assert ha != hc  # a single changed field changes the hash
    assert len(ha) == 8 and all(ch in "0123456789abcdef" for ch in ha)


def test_load_config_attaches_hash():
    cfg = load_config(CONFIGS / "base.yaml")
    assert len(cfg.config_hash) == 8
    # the hash field is excluded from serialization...
    assert "config_hash" not in cfg.model_dump()
    # ...so recomputing from the loaded config stays consistent
    assert compute_config_hash(cfg) == cfg.config_hash
