"""Tests for atlas.train.sft helpers.

The full ``run_sft`` path needs TRL + a real model, so it's left for the
50-step smoke. These tests cover the two pure helpers — dataset dispatch
and the SFTConfig passthrough — that are easy to break and hard to debug
mid-training.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.train import sft
from atlas.utils.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


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


# --- _preflight_hub_access -----------------------------------------------------

class _FakeHfApi:
    """Captures whoami/create_repo calls and lets tests inject failures."""

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
    """Make ``from huggingface_hub import HfApi`` and ``HfHubHTTPError`` resolve to fakes.

    The error class is module-level (not built inside this helper) so the class
    the fake raises is the same class ``_preflight_hub_access`` imports — calling
    this helper twice in a test would otherwise create two distinct classes and
    the except branch wouldn't match.
    """
    import sys
    import types

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = lambda: fake_api
    fake_errors = types.ModuleType("huggingface_hub.errors")
    fake_errors.HfHubHTTPError = _FakeHfHubHTTPError
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake_errors)


def test_preflight_hub_access_passes_when_token_has_write_scope(monkeypatch):
    """Happy path: whoami succeeds, create_repo no-ops on existing repo."""
    fake = _FakeHfApi()
    _patch_hf_api(monkeypatch, fake)
    sft._preflight_hub_access("agaonker/atlas-sft-qwen05b-v1")
    assert [c[0] for c in fake.calls] == ["whoami", "create_repo"]
    # exist_ok=True is the contract — without it we'd crash on re-runs.
    assert fake.calls[1][2].get("exist_ok") is True


def test_preflight_hub_access_raises_on_whoami_failure(monkeypatch):
    """A bad/missing token should fail fast, not after hours of training."""
    fake = _FakeHfApi(whoami_raises=_FakeHfHubHTTPError("401 invalid token"))
    _patch_hf_api(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="whoami"):
        sft._preflight_hub_access("agaonker/atlas-sft-qwen05b-v1")


def test_preflight_hub_access_raises_on_create_repo_403(monkeypatch):
    """The exact bug we got burned by: read token, 403 only after train()."""
    fake = _FakeHfApi(create_repo_raises=_FakeHfHubHTTPError("403 Forbidden"))
    _patch_hf_api(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="cannot write"):
        sft._preflight_hub_access("agaonker/atlas-sft-qwen05b-v1")


# --- configs/sft_qwen05b.yaml canary -------------------------------------------

def test_sft_qwen05b_yaml_loads_and_merges():
    """End-to-end load of the Phase 1 YAML — catches typos that would crash mid-run."""
    cfg = load_config(CONFIGS / "sft_qwen05b.yaml")
    # Inherited from base.yaml
    assert cfg.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.lora.r == 16
    assert cfg.quant.load_in_4bit is True
    # Phase-1-specific blocks
    assert cfg.dataset is not None
    assert cfg.dataset.name == "HuggingFaceH4/ultrachat_200k"
    assert cfg.dataset.n_samples == 5000
    assert cfg.train is not None
    # The YAML float must be 2.0e-4 (parsed as float), not 2e-4 (parsed as str).
    assert cfg.train.model_dump()["learning_rate"] == 2.0e-4
    assert cfg.train.output_dir == "outputs/sft_qwen05b_v1"
    # Reserved fields with `_` aren't passed through — sanity check the passthrough
    # only forwards what the user wrote.
    assert "gradient_checkpointing" in cfg.train.model_dump()
