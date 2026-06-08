"""Tests for atlas.train.dpo helpers — mirrors tests/test_sft_train.py.

The full ``run_dpo`` path needs TRL + a real model + the SFT warm-start adapter
on disk, so it's left for the 50-step smoke. These tests cover the helpers that
are easy to break and hard to debug mid-training: dataset dispatch, the
DPOConfig passthrough, the hub-write preflight, and the config canary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.train import dpo
from atlas.utils.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def _cfg(**overrides) -> Config:
    payload: dict = {"model": {"name": "Qwen/Qwen2.5-0.5B"}}
    payload.update(overrides)
    return Config.model_validate(payload)


# --- _load_dataset --------------------------------------------------------------


def test_load_dataset_raises_when_dataset_block_missing():
    """A DPO YAML without a dataset block must fail loudly, not start training on nothing."""
    cfg = _cfg()
    with pytest.raises(ValueError, match="cfg.dataset is required"):
        dpo._load_dataset(cfg)


def test_load_dataset_raises_on_unsupported_dataset_name():
    cfg = _cfg(dataset={"name": "Anthropic/hh-rlhf", "split": "train", "n_samples": 5})
    with pytest.raises(ValueError, match="Unsupported DPO dataset"):
        dpo._load_dataset(cfg)


def test_load_dataset_dispatches_to_ultrafeedback_loader(monkeypatch):
    """The supported dataset name routes through to atlas.data.preference_data."""
    seen: dict = {}

    def fake_loader(n_samples, seed, revision):
        seen["n_samples"] = n_samples
        seen["seed"] = seed
        seen["revision"] = revision
        return "FAKE_DS"

    monkeypatch.setitem(
        dpo.SUPPORTED_DATASETS, "HuggingFaceH4/ultrafeedback_binarized", fake_loader
    )
    cfg = _cfg(
        dataset={
            "name": "HuggingFaceH4/ultrafeedback_binarized",
            "split": "train_prefs",
            "n_samples": 1234,
            "revision": "abc",
        },
        seed=7,
    )
    assert dpo._load_dataset(cfg) == "FAKE_DS"
    assert seen == {"n_samples": 1234, "seed": 7, "revision": "abc"}


# --- _build_train_config --------------------------------------------------------


class _FakeDPOConfig:
    """Stand-in for trl.DPOConfig that just records its kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _patch_dpo_config(monkeypatch):
    """Make ``from trl import DPOConfig`` resolve to our fake."""
    import sys
    import types

    fake_trl = types.ModuleType("trl")
    fake_trl.DPOConfig = _FakeDPOConfig
    monkeypatch.setitem(sys.modules, "trl", fake_trl)


def test_build_train_config_passes_through_yaml_train_block(monkeypatch):
    """Every key in cfg.train flows verbatim into DPOConfig kwargs — including
    DPO-specific knobs like beta and loss_type."""
    _patch_dpo_config(monkeypatch)
    cfg = _cfg(
        train={
            "output_dir": "outputs/dpo_v1",
            "learning_rate": 5.0e-6,
            "beta": 0.1,
            "loss_type": "sigmoid",
            "max_length": 1024,
            "per_device_train_batch_size": 2,
        }
    )
    dc = dpo._build_train_config(cfg, max_steps_override=None)
    assert dc.kwargs["output_dir"] == "outputs/dpo_v1"
    assert dc.kwargs["learning_rate"] == 5.0e-6
    assert dc.kwargs["beta"] == 0.1
    assert dc.kwargs["loss_type"] == "sigmoid"
    assert dc.kwargs["max_length"] == 1024


def test_build_train_config_applies_max_steps_override(monkeypatch):
    """--max-steps 50 from the CLI wins over whatever the YAML says."""
    _patch_dpo_config(monkeypatch)
    cfg = _cfg(train={"output_dir": "outputs/x", "max_steps": 1000})
    dc = dpo._build_train_config(cfg, max_steps_override=50)
    assert dc.kwargs["max_steps"] == 50


def test_build_train_config_handles_missing_train_block(monkeypatch):
    """A YAML with no train: block produces a DPOConfig with TRL defaults."""
    _patch_dpo_config(monkeypatch)
    cfg = _cfg()
    dc = dpo._build_train_config(cfg, max_steps_override=None)
    assert dc.kwargs == {}


# --- _preflight_hub_access ------------------------------------------------------


class _FakeHfApi:
    def __init__(self, *, whoami_raises=None, create_repo_raises=None):
        self._whoami_raises = whoami_raises
        self._create_repo_raises = create_repo_raises
        self.calls: list[tuple[str, tuple, dict]] = []

    def whoami(self, *args, **kwargs):
        self.calls.append(("whoami", args, kwargs))
        if self._whoami_raises is not None:
            raise self._whoami_raises
        return {"name": "test-user"}

    def create_repo(self, *args, **kwargs):
        self.calls.append(("create_repo", args, kwargs))
        if self._create_repo_raises is not None:
            raise self._create_repo_raises


class _FakeHfHubHTTPError(Exception):
    pass


def _patch_hf_api(monkeypatch, fake_api):
    monkeypatch.setattr(dpo, "HfApi", lambda: fake_api)
    monkeypatch.setattr(dpo, "HfHubHTTPError", _FakeHfHubHTTPError)


def test_preflight_hub_access_passes_when_token_has_write_scope(monkeypatch):
    """Happy path: whoami succeeds, create_repo no-ops on existing repo."""
    fake = _FakeHfApi()
    _patch_hf_api(monkeypatch, fake)
    dpo._preflight_hub_access("agaonker/atlas-dpo-qwen05b-v1")
    assert [c[0] for c in fake.calls] == ["whoami", "create_repo"]
    assert fake.calls[1][2].get("exist_ok") is True


def test_preflight_hub_access_raises_on_whoami_failure(monkeypatch):
    fake = _FakeHfApi(whoami_raises=_FakeHfHubHTTPError("401 invalid token"))
    _patch_hf_api(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="whoami"):
        dpo._preflight_hub_access("agaonker/atlas-dpo-qwen05b-v1")


def test_preflight_hub_access_raises_on_create_repo_403(monkeypatch):
    """Same bug we already shipped a lesson for: a Read-only token preflight-403s
    in ~200ms (LESSONS.md), saving a full DPO training run."""
    fake = _FakeHfApi(create_repo_raises=_FakeHfHubHTTPError("403 Forbidden"))
    _patch_hf_api(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="cannot write"):
        dpo._preflight_hub_access("agaonker/atlas-dpo-qwen05b-v1")


# --- configs/dpo_qwen05b.yaml canary -------------------------------------------


def test_dpo_qwen05b_yaml_loads_and_merges():
    """End-to-end load of the Phase 2 YAML — catches typos that would crash mid-run."""
    cfg = load_config(CONFIGS / "dpo_qwen05b.yaml")
    # Inherited from base.yaml
    assert cfg.model.name == "Qwen/Qwen2.5-0.5B"
    assert cfg.lora.r == 16
    # Phase-2-specific overlay: tokenizer + SFT warm-start
    assert cfg.model.tokenizer_name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.model.sft_adapter == "agaonker/atlas-sft-qwen05b-v2"
    # 4-bit quant is deliberately disabled for DPO: merge_and_unload of the SFT
    # adapter needs full-precision weights.
    assert cfg.quant.load_in_4bit is False
    # Phase-2-specific blocks
    assert cfg.dataset is not None
    assert cfg.dataset.name == "HuggingFaceH4/ultrafeedback_binarized"
    assert cfg.dataset.n_samples == 5000
    assert cfg.train is not None
    td = cfg.train.model_dump()
    # YAML float: 5.0e-6 (parsed as float), not 5e-6 (string).
    assert td["learning_rate"] == 5.0e-6
    # DPO knobs flow through TrainCfg.extra="allow" to DPOConfig.
    assert td["beta"] == 0.1
    assert td["loss_type"] == "sigmoid"
    assert td["max_length"] == 1024
    # TRL 1.4 dropped max_prompt_length; max_length now governs the combined
    # prompt + completion sequence on its own.
    assert "max_prompt_length" not in td
    assert cfg.train.output_dir == "outputs/dpo_qwen05b_v1"
