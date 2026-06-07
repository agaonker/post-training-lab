"""Configuration system: pydantic models + YAML loading with layered defaults.

Every experiment is fully described by a YAML file in ``configs/``. ``configs/base.yaml``
holds shared defaults; each experiment config is deep-merged over it by :func:`load_config`.

The ``train`` block is a deliberate passthrough (``extra="allow"``): its keys map straight
onto a TRL ``*Config`` (e.g. ``SFTConfig(**cfg.train.model_dump())``), so we never have to
re-declare the full TRL knob surface here and the schema survives TRL minor bumps.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

BASE_CONFIG_NAME = "base.yaml"


# --- sub-models -----------------------------------------------------------------


class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    revision: str | None = None  # pin to a commit SHA for reproducibility; null = latest
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    # Optional override: load the tokenizer from a different repo than the model.
    # Used when the chosen base is the pretrained Qwen2.5-0.5B (eos == pad, no
    # {% generation %} markers in chat_template) but training needs the Instruct
    # variant's tokenizer config (correct eos, markers for assistant_only_loss).
    # The two share a byte-identical vocab. null = use ``name`` for both.
    tokenizer_name: str | None = None
    # Optional warm-start: an SFT adapter (HF Hub repo) to fuse into the base
    # before training. Used by Phase 2+ methods (DPO/PPO/GRPO/KTO/ORPO) so they
    # start from the SFT-aligned policy, not the raw pretrained base. null = no
    # warm-start (Phase 0 baseline and Phase 1 SFT).
    sft_adapter: str | None = None
    # Optional revision SHA pin for the sft_adapter — keeps the comparison
    # reproducible across any future re-trains of the anchor. null = latest.
    sft_adapter_revision: str | None = None


class QuantCfg(BaseModel):
    """4-bit QLoRA quantization. Only applied on CUDA — ignored on the Mac CPU path."""

    model_config = ConfigDict(extra="forbid")

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    double_quant: bool = True


class LoraCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class WandbCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str = "post-training-lab"
    entity: str | None = None
    mode: str = "online"  # online | offline | disabled


class TaskCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_fewshot: int = 0
    limit: int | None = None  # cap samples for speed; null = full task


class VllmCfg(BaseModel):
    """vLLM engine knobs — runtime/perf only, excluded from ``config_hash``.

    Used only when ``eval.backend == "vllm"`` (the Modal GPU path); ignored on the HF
    backend. vLLM does its own continuous batching, so ``eval.batch_size`` doesn't apply.
    """

    model_config = ConfigDict(extra="forbid")

    gpu_memory_utilization: float = 0.6  # see configs/base.yaml: 0.9 OOMs the loglikelihood spike
    max_model_len: int | None = None  # null = model default; must fit few-shot prompt + gen
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1  # the scale lever; >1 needs Ray + a multi-GPU container
    max_lora_rank: int = 16  # must be >= lora.r


class EvalCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: dict[str, TaskCfg] = Field(default_factory=dict)
    batch_size: int | str = "auto"
    backend: str = "hf"  # hf | vllm — vllm is the Modal-only fast path
    vllm: VllmCfg = Field(default_factory=VllmCfg)


class DatasetCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    split: str
    revision: str | None = None  # pin the dataset commit SHA
    config: str | None = None  # HF dataset config name (e.g. gsm8k's "main")
    n_samples: int | None = None  # null = use the whole split


class TrainCfg(BaseModel):
    """Permissive on purpose: every extra key flows straight to the TRL ``*Config``."""

    model_config = ConfigDict(extra="allow")

    output_dir: str = "outputs/run"


class OutputCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hub_repo: str | None = None  # <user>/atlas-<method>-qwen05b


class Config(BaseModel):
    # extra="forbid" catches typo'd YAML keys; protected_namespaces=() lets us
    # keep the natural field name `model` without a pydantic warning.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: ModelCfg
    seed: int = 42
    quant: QuantCfg = Field(default_factory=QuantCfg)
    lora: LoraCfg = Field(default_factory=LoraCfg)
    wandb: WandbCfg = Field(default_factory=WandbCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    dataset: DatasetCfg | None = None
    train: TrainCfg | None = None
    output: OutputCfg | None = None

    # Stable fingerprint of the resolved config, set by load_config. Excluded
    # from serialization so it can never feed back into its own hash.
    config_hash: str = Field(default="", exclude=True)


# --- loading --------------------------------------------------------------------


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` onto a copy of ``base``.

    Nested dicts merge recursively; every other value (including lists) is replaced
    wholesale. This lets an experiment config tweak a single nested key (e.g. ``lora.r``)
    without restating the rest of the block. Inputs are not mutated.
    """
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = merge_dicts(out[key], val)
        else:
            out[key] = val
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def load_config(path: str | Path) -> Config:
    """Load an experiment YAML, deep-merged over the sibling ``base.yaml``.

    Resolution order: ``base.yaml`` first, then the experiment file on top. Loading
    ``base.yaml`` itself just returns it (no self-merge). The resolved config's stable
    8-char hash is attached as ``cfg.config_hash``.
    """
    path = Path(path)
    raw = _read_yaml(path)

    base_path = path.parent / BASE_CONFIG_NAME
    if base_path.exists() and base_path.resolve() != path.resolve():
        raw = merge_dicts(_read_yaml(base_path), raw)

    cfg = Config.model_validate(raw)
    cfg.config_hash = compute_config_hash(cfg)
    return cfg


def compute_config_hash(cfg: Config) -> str:
    """Stable 8-char fingerprint of a resolved config.

    sha256 over the canonical (sorted-key) JSON of the config. Logged to W&B and the
    experiment markdown so any run is traceable to its exact settings.

    Runtime-only eval knobs (``batch_size``, ``backend``, the whole ``vllm`` block) are
    excluded: they change *how fast* a run goes, not *what is measured*, so tuning them
    must not flip the fingerprint, break partial-resume, or make a baseline look
    incomparable to the run it baselines.
    """
    payload = cfg.model_dump(
        mode="json",
        exclude={"eval": {"batch_size": True, "backend": True, "vllm": True}},
    )
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs via ``transformers.set_seed``.

    Imported lazily so ``import atlas.utils.config`` stays cheap for tests that only
    touch the config models.
    """
    from transformers import set_seed

    set_seed(seed)
