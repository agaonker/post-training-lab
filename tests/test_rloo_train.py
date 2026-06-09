"""Tests for atlas.train.rloo helpers — mirrors test_sft_train.py /
test_dpo_train.py / test_rm_train.py.

The full ``run_rloo`` path needs TRL + three model loads + the sft_v2 and
rm_v1 adapters on disk, so it's left for the 50-step Modal smoke. These
tests cover the helpers that are easy to break and hard to debug mid-training.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.train import rloo
from atlas.utils.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def _cfg(**overrides) -> Config:
    payload: dict = {"model": {"name": "Qwen/Qwen2.5-0.5B"}}
    payload.update(overrides)
    return Config.model_validate(payload)


# --- _load_dataset --------------------------------------------------------------


def test_load_dataset_raises_when_dataset_block_missing():
    """A RLOO YAML without a dataset block must fail loudly."""
    cfg = _cfg()
    with pytest.raises(ValueError, match="cfg.dataset is required"):
        rloo._load_dataset(cfg)


def test_load_dataset_raises_on_unsupported_dataset_name():
    cfg = _cfg(dataset={"name": "Anthropic/hh-rlhf", "split": "train", "n_samples": 5})
    with pytest.raises(ValueError, match="Unsupported RLOO dataset"):
        rloo._load_dataset(cfg)


def test_load_dataset_dispatches_to_prompts_loader(monkeypatch):
    """The supported dataset name routes through to
    atlas.data.preference_data.load_ultrafeedback_prompts — RLOO consumes
    prompts only, not chosen/rejected."""
    seen: dict = {}

    def fake_loader(n_samples, seed, revision):
        seen["n_samples"] = n_samples
        seen["seed"] = seed
        seen["revision"] = revision
        return "FAKE_DS"

    monkeypatch.setitem(
        rloo.SUPPORTED_DATASETS, "HuggingFaceH4/ultrafeedback_binarized", fake_loader
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
    assert rloo._load_dataset(cfg) == "FAKE_DS"
    assert seen == {"n_samples": 1234, "seed": 7, "revision": "abc"}


# --- _build_train_config --------------------------------------------------------


class _FakeRLOOConfig:
    """Stand-in for trl.RLOOConfig that just records its kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _patch_rloo_config(monkeypatch):
    """Make ``from trl import RLOOConfig`` resolve to our fake."""
    import sys
    import types

    fake_trl = types.ModuleType("trl")
    fake_trl.RLOOConfig = _FakeRLOOConfig
    monkeypatch.setitem(sys.modules, "trl", fake_trl)


def test_build_train_config_passes_through_yaml_train_block(monkeypatch):
    """Every key in cfg.train flows verbatim into RLOOConfig kwargs —
    including RLOO-specific knobs like beta, num_generations, epsilon."""
    _patch_rloo_config(monkeypatch)
    cfg = _cfg(
        train={
            "output_dir": "outputs/rloo_v1",
            "learning_rate": 5.0e-6,
            "beta": 0.05,
            "num_generations": 4,
            "epsilon": 0.2,
            "temperature": 0.7,
            "max_completion_length": 512,
        }
    )
    rc = rloo._build_train_config(cfg, max_steps_override=None)
    assert rc.kwargs["output_dir"] == "outputs/rloo_v1"
    assert rc.kwargs["learning_rate"] == 5.0e-6
    assert rc.kwargs["beta"] == 0.05
    assert rc.kwargs["num_generations"] == 4
    assert rc.kwargs["epsilon"] == 0.2


def test_build_train_config_applies_max_steps_override(monkeypatch):
    """--max-steps 50 from the CLI wins over whatever the YAML says."""
    _patch_rloo_config(monkeypatch)
    cfg = _cfg(train={"output_dir": "outputs/x", "max_steps": 1000})
    rc = rloo._build_train_config(cfg, max_steps_override=50)
    assert rc.kwargs["max_steps"] == 50


def test_build_train_config_handles_missing_train_block(monkeypatch):
    """A YAML with no train: block produces a RLOOConfig with TRL defaults."""
    _patch_rloo_config(monkeypatch)
    cfg = _cfg()
    rc = rloo._build_train_config(cfg, max_steps_override=None)
    assert rc.kwargs == {}


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
    monkeypatch.setattr(rloo, "HfApi", lambda: fake_api)
    monkeypatch.setattr(rloo, "HfHubHTTPError", _FakeHfHubHTTPError)


def test_preflight_hub_access_passes_when_token_has_write_scope(monkeypatch):
    fake = _FakeHfApi()
    _patch_hf_api(monkeypatch, fake)
    rloo._preflight_hub_access("agaonker/atlas-rloo-qwen05b-v1")
    assert [c[0] for c in fake.calls] == ["whoami", "create_repo"]
    assert fake.calls[1][2].get("exist_ok") is True


def test_preflight_hub_access_raises_on_whoami_failure(monkeypatch):
    fake = _FakeHfApi(whoami_raises=_FakeHfHubHTTPError("401 invalid token"))
    _patch_hf_api(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="whoami"):
        rloo._preflight_hub_access("agaonker/atlas-rloo-qwen05b-v1")


def test_preflight_hub_access_raises_on_create_repo_403(monkeypatch):
    fake = _FakeHfApi(create_repo_raises=_FakeHfHubHTTPError("403 Forbidden"))
    _patch_hf_api(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="cannot write"):
        rloo._preflight_hub_access("agaonker/atlas-rloo-qwen05b-v1")


# --- configs/rloo_qwen05b.yaml canary -----------------------------------------


def test_rloo_qwen05b_yaml_loads_and_merges():
    """End-to-end load of the Phase 3B YAML — catches typos that would crash mid-run."""
    cfg = load_config(CONFIGS / "rloo_qwen05b.yaml")
    # Inherited from base.yaml
    assert cfg.model.name == "Qwen/Qwen2.5-0.5B"
    assert cfg.lora.r == 16
    # Phase-3B overlay: tokenizer + SFT warm-start (merged) + RM adapter
    assert cfg.model.tokenizer_name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.model.sft_adapter == "agaonker/atlas-sft-qwen05b-v2"
    assert cfg.model.rm_adapter == "agaonker/atlas-rm-qwen05b-v1"
    # 4-bit is deliberately off so sft_v2 can merge_and_unload (full
    # precision); the reward model is independently 4-bit'd in code.
    assert cfg.quant.load_in_4bit is False
    # Phase-3B blocks
    assert cfg.dataset is not None
    assert cfg.dataset.name == "HuggingFaceH4/ultrafeedback_binarized"
    assert cfg.dataset.n_samples == 5000
    assert cfg.train is not None
    td = cfg.train.model_dump()
    assert td["learning_rate"] == 5.0e-6
    # RLOO knobs flow through TrainCfg.extra="allow" to RLOOConfig.
    assert td["beta"] == 0.05
    assert td["num_generations"] == 4
    assert td["epsilon"] == 0.2
    assert td["temperature"] == 0.7
    # Currently False (gradient-flow workaround for sft_v2's weak eos
    # emission); restore to True after sft_v3 or a stronger SFT recipe.
    assert td["mask_truncated_completions"] is False
    # RLOO uses max_completion_length (rollout length), not max_length
    # (combined sequence) — TRL's RLOOConfig has the former, not the latter.
    assert td["max_completion_length"] == 512
    assert cfg.train.output_dir == "outputs/rloo_qwen05b_v1"
