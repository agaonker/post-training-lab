"""Tests for atlas.train.sft helpers.

The full ``run_sft`` path needs TRL + a real model, so it's left for the
50-step smoke. These tests cover the two pure helpers — dataset dispatch
and the SFTConfig passthrough — that are easy to break and hard to debug
mid-training.
"""

from __future__ import annotations

import pytest

from atlas.train import sft
from atlas.utils.config import Config


def _cfg(**overrides) -> Config:
    payload: dict = {"model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"}}
    payload.update(overrides)
    return Config.model_validate(payload)


# --- _load_dataset --------------------------------------------------------------

def test_load_dataset_raises_when_dataset_block_missing():
    """A YAML without a dataset block must fail loudly, not start training on nothing."""
    cfg = _cfg()
    with pytest.raises(ValueError, match="cfg.dataset is required"):
        sft._load_dataset(cfg)


def test_load_dataset_raises_on_unsupported_dataset_name():
    cfg = _cfg(
        dataset={"name": "teknium/OpenHermes-2.5", "split": "train", "n_samples": 5}
    )
    with pytest.raises(ValueError, match="Unsupported SFT dataset"):
        sft._load_dataset(cfg)


def test_load_dataset_dispatches_to_ultrachat_loader(monkeypatch):
    """The supported dataset name routes through to atlas.data.sft_data."""
    seen: dict = {}

    def fake_loader(n_samples, seed, revision):
        seen["n_samples"] = n_samples
        seen["seed"] = seed
        seen["revision"] = revision
        return "FAKE_DS"

    # Patch the entry registered in SUPPORTED_DATASETS, not the module-level
    # import, so dispatch goes through the actual table.
    monkeypatch.setitem(
        sft.SUPPORTED_DATASETS, "HuggingFaceH4/ultrachat_200k", fake_loader
    )
    cfg = _cfg(
        dataset={
            "name": "HuggingFaceH4/ultrachat_200k",
            "split": "train_sft",
            "n_samples": 1234,
            "revision": "abc",
        },
        seed=7,
    )
    assert sft._load_dataset(cfg) == "FAKE_DS"
    assert seen == {"n_samples": 1234, "seed": 7, "revision": "abc"}


# --- _build_train_config --------------------------------------------------------

class _FakeSFTConfig:
    """Stand-in for trl.SFTConfig that just records its kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _patch_sft_config(monkeypatch):
    """Make ``from trl import SFTConfig`` resolve to our fake."""
    import sys
    import types

    fake_trl = types.ModuleType("trl")
    fake_trl.SFTConfig = _FakeSFTConfig
    monkeypatch.setitem(sys.modules, "trl", fake_trl)


def test_build_train_config_passes_through_yaml_train_block(monkeypatch):
    """Every key in cfg.train flows verbatim into SFTConfig kwargs."""
    _patch_sft_config(monkeypatch)
    cfg = _cfg(
        train={
            "output_dir": "outputs/sft_v1",
            "learning_rate": 2.0e-4,
            "max_steps": 1000,
            "per_device_train_batch_size": 4,
        }
    )
    sc = sft._build_train_config(cfg, max_steps_override=None)
    assert sc.kwargs["output_dir"] == "outputs/sft_v1"
    assert sc.kwargs["learning_rate"] == 2.0e-4
    assert sc.kwargs["max_steps"] == 1000
    assert sc.kwargs["per_device_train_batch_size"] == 4


def test_build_train_config_applies_max_steps_override(monkeypatch):
    """--max-steps 50 from the CLI wins over whatever the YAML says."""
    _patch_sft_config(monkeypatch)
    cfg = _cfg(train={"output_dir": "outputs/x", "max_steps": 1000})
    sc = sft._build_train_config(cfg, max_steps_override=50)
    assert sc.kwargs["max_steps"] == 50


def test_build_train_config_handles_missing_train_block(monkeypatch):
    """A YAML with no train: block produces an SFTConfig with TRL defaults."""
    _patch_sft_config(monkeypatch)
    cfg = _cfg()
    sc = sft._build_train_config(cfg, max_steps_override=None)
    assert sc.kwargs == {}
