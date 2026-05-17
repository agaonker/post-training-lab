"""Tests for atlas.models.base and atlas.models.adapters.

The model loader is exercised by stubbing the HuggingFace ``from_pretrained``
classmethods so unit tests never touch the network or load weights. A
``@pytest.mark.slow`` integration test would load the real Qwen base — kept
out of CI; run by hand against a GPU when the wiring is suspect.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atlas.models import adapters, base
from atlas.utils.config import Config


def _cfg(**overrides) -> Config:
    """Build a Config rooted at the project defaults with overrides applied."""
    payload: dict = {"model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"}}
    payload.update(overrides)
    return Config.model_validate(payload)


# --- make_lora_config -----------------------------------------------------------

def test_make_lora_config_propagates_yaml_defaults():
    """Defaults defined on LoraCfg flow through to the peft.LoraConfig."""
    cfg = _cfg()
    lc = adapters.make_lora_config(cfg)
    assert lc.r == 16
    assert lc.lora_alpha == 32
    assert lc.lora_dropout == 0.05
    assert lc.task_type == "CAUSAL_LM"
    target = set(lc.target_modules)
    assert "q_proj" in target and "down_proj" in target


def test_make_lora_config_overrides_apply():
    """Tweaking lora.r in the config is reflected in the peft LoraConfig."""
    cfg = _cfg(lora={"r": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["q_proj"]})
    lc = adapters.make_lora_config(cfg)
    assert lc.r == 8
    assert lc.lora_alpha == 16
    assert lc.lora_dropout == 0.1
    # peft normalizes the list to a set internally; compare as a set so the
    # test doesn't drift with their internal representation choice.
    assert set(lc.target_modules) == {"q_proj"}


def test_make_lora_config_supports_seq_cls_for_reward_model():
    """RM phase needs task_type=SEQ_CLS — we expose it as an arg, not a fork."""
    lc = adapters.make_lora_config(_cfg(), task_type="SEQ_CLS")
    assert lc.task_type == "SEQ_CLS"


# --- _resolve_dtype -------------------------------------------------------------

def test_resolve_dtype_maps_known_names():
    import torch as _torch

    assert base._resolve_dtype("bfloat16") == _torch.bfloat16
    assert base._resolve_dtype("float16") == _torch.float16
    assert base._resolve_dtype("float32") == _torch.float32


def test_resolve_dtype_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        base._resolve_dtype("int8")


# --- _build_quant_config --------------------------------------------------------

def test_quant_config_is_none_when_load_in_4bit_false(monkeypatch):
    cfg = _cfg(quant={"load_in_4bit": False})
    assert base._build_quant_config(cfg) is None


def test_quant_config_is_none_without_cuda(monkeypatch):
    """No CUDA → no bnb → no quant config, regardless of YAML."""
    monkeypatch.setattr(base.torch.cuda, "is_available", lambda: False)
    cfg = _cfg(quant={"load_in_4bit": True})
    assert base._build_quant_config(cfg) is None


# --- load_base_model_and_tokenizer ---------------------------------------------

class _FakeTokenizer:
    def __init__(self, has_pad: bool):
        self.eos_token = "<eos>"
        self.pad_token = "<pad>" if has_pad else None


def _patch_loaders(monkeypatch, *, has_pad: bool):
    """Stub HF auto-classes; record the kwargs they're called with."""
    seen: dict = {"model_kwargs": None, "tok_kwargs": None}

    def fake_model(name, **kwargs):
        seen["model_name"] = name
        seen["model_kwargs"] = kwargs
        return SimpleNamespace(name=name)

    def fake_tok(name, **kwargs):
        seen["tok_name"] = name
        seen["tok_kwargs"] = kwargs
        return _FakeTokenizer(has_pad=has_pad)

    monkeypatch.setattr(base.AutoModelForCausalLM, "from_pretrained", fake_model)
    monkeypatch.setattr(base.AutoTokenizer, "from_pretrained", fake_tok)
    monkeypatch.setattr(base.torch.cuda, "is_available", lambda: False)
    return seen


def test_load_base_sets_pad_token_to_eos_when_missing(monkeypatch):
    """Qwen2.5's tokenizer ships without pad; TRL needs one set."""
    _patch_loaders(monkeypatch, has_pad=False)
    cfg = _cfg()
    _model, tok = base.load_base_model_and_tokenizer(cfg)
    assert tok.pad_token == tok.eos_token


def test_load_base_preserves_existing_pad_token(monkeypatch):
    """If the tokenizer already has a pad token, don't clobber it."""
    _patch_loaders(monkeypatch, has_pad=True)
    cfg = _cfg()
    _model, tok = base.load_base_model_and_tokenizer(cfg)
    assert tok.pad_token == "<pad>"


def test_load_base_passes_revision_through(monkeypatch):
    """Pinned commit SHA flows to BOTH the model and tokenizer loaders."""
    seen = _patch_loaders(monkeypatch, has_pad=True)
    cfg = _cfg(model={"name": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "abc123"})
    base.load_base_model_and_tokenizer(cfg)
    assert seen["model_kwargs"]["revision"] == "abc123"
    assert seen["tok_kwargs"]["revision"] == "abc123"


def test_load_base_omits_quant_kwarg_on_non_cuda(monkeypatch):
    """On Mac / CPU, no quantization_config is passed even if YAML says 4-bit."""
    seen = _patch_loaders(monkeypatch, has_pad=True)
    cfg = _cfg(quant={"load_in_4bit": True})
    base.load_base_model_and_tokenizer(cfg)
    assert "quantization_config" not in seen["model_kwargs"]


def test_load_base_resolves_dtype(monkeypatch):
    """cfg.model.dtype string is turned into a real torch.dtype."""
    import torch as _torch

    seen = _patch_loaders(monkeypatch, has_pad=True)
    cfg = _cfg(model={"name": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "float16"})
    base.load_base_model_and_tokenizer(cfg)
    assert seen["model_kwargs"]["torch_dtype"] == _torch.float16
