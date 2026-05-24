"""Tests for atlas.utils.config — focused on the config_hash fingerprint contract.

The hash must fingerprint *what is measured*, not *how fast it runs*: tuning runtime
knobs (batch_size, backend, the vllm block) must not move it, while result-determining
fields (dtype, num_fewshot, ...) must.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.utils.config import compute_config_hash, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def test_config_hash_excludes_runtime_eval_knobs():
    baseline = compute_config_hash(load_config(CONFIGS / "baseline.yaml"))

    cfg = load_config(CONFIGS / "baseline.yaml")
    cfg.eval.batch_size = 64
    cfg.eval.backend = "vllm"
    cfg.eval.vllm.gpu_memory_utilization = 0.5
    cfg.eval.vllm.data_parallel_size = 4
    cfg.eval.vllm.max_model_len = 8192
    assert compute_config_hash(cfg) == baseline


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: setattr(c.model, "dtype", "float16"),
        lambda c: setattr(c.eval.tasks["mmlu"], "num_fewshot", 0),
        lambda c: setattr(c.eval.tasks["gsm8k"], "limit", 100),
        lambda c: setattr(c, "seed", 7),
    ],
)
def test_config_hash_includes_semantic_fields(mutate):
    baseline = compute_config_hash(load_config(CONFIGS / "baseline.yaml"))
    cfg = load_config(CONFIGS / "baseline.yaml")
    mutate(cfg)
    assert compute_config_hash(cfg) != baseline


def test_vllm_block_parses_with_defaults():
    cfg = load_config(CONFIGS / "baseline.yaml")
    assert cfg.eval.backend == "hf"
    assert cfg.eval.vllm.max_lora_rank == 16
    assert cfg.eval.vllm.tensor_parallel_size == 1
    assert cfg.eval.vllm.max_model_len is None


def test_vllm_block_rejects_typoed_key():
    from pydantic import ValidationError

    from atlas.utils.config import Config

    with pytest.raises(ValidationError):
        Config.model_validate(
            {"model": {"name": "x"}, "eval": {"vllm": {"gpu_memory_utilizaton": 0.9}}}
        )
