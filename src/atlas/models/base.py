"""Base model + tokenizer loader.

Single entry point for every phase. Pulls dtype, revision, and quantization
intent from the resolved ``Config``. The quantization branch (4-bit QLoRA via
``bitsandbytes``) is **CUDA-only** — on Mac/CPU we silently drop it so the same
YAML runs locally for a 50-step smoke and on Kaggle for the real thing. The
pyproject pins ``bitsandbytes`` to ``platform_system == 'Linux'`` for the same
reason; importing it eagerly here would break ``uv sync`` on macOS.

The tokenizer's ``pad_token`` is forced to ``eos_token`` when missing because
TRL's ``SFTTrainer`` requires a pad token; Qwen2.5's tokenizer ships without
one set, and the resulting error mid-training is opaque.

We also patch Qwen's chat_template to inject ``{% generation %}`` markers
around assistant content — required for TRL's ``assistant_only_loss=True`` to
build a non-empty mask. See :func:`patch_chat_template_for_assistant_mask`.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from atlas.utils.config import Config

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# Qwen2.5's chat_template wraps user / system / non-tool-calling assistant turns
# in a *single* combined branch — there's no place to inject the {% generation %}
# / {% endgeneration %} pair that TRL's ``assistant_only_loss=True`` needs. The
# patcher splits that combined branch in two: keep user/system in the original
# shape and lift assistant-no-tools into its own branch with markers around the
# content + closing <|im_end|>. (The leading <|im_start|>assistant\n stays
# outside the markers — it's the prompt the model is conditioned on, not the
# generation.) Tool-calling assistant turns and tool responses are untouched;
# they don't appear in UltraChat anyway.
_QWEN_COMBINED_BRANCH = (
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) '
    'or (message.role == "assistant" and not message.tool_calls) %}\n'
    "        {{- '<|im_start|>' + message.role + '\\n' + message.content + "
    "'<|im_end|>' + '\\n' }}"
)

_QWEN_SPLIT_BRANCH = (
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n'
    "        {{- '<|im_start|>' + message.role + '\\n' + message.content + "
    "'<|im_end|>' + '\\n' }}\n"
    '    {%- elif message.role == "assistant" and not message.tool_calls %}\n'
    "        {{- '<|im_start|>' + message.role + '\\n' }}"
    "{% generation %}{{- message.content + '<|im_end|>' }}{% endgeneration %}"
    "{{- '\\n' }}"
)


def patch_chat_template_for_assistant_mask(tokenizer: Any) -> bool:
    """Inject ``{% generation %}`` markers into Qwen2.5's chat_template so TRL's
    ``assistant_only_loss=True`` (and ``return_assistant_tokens_mask=True``)
    actually produce a non-empty mask.

    Idempotent: returns ``False`` if the template already has markers or doesn't
    match the expected Qwen2.5 shape; returns ``True`` if it patched something.

    Verified empirically against the Qwen/Qwen2.5-0.5B-Instruct tokenizer at the
    project's pinned transformers version. If Qwen reshapes their template, this
    helper will no-op rather than scramble — the no-op path raises downstream at
    SFT time when the audit shows mask == all zeros, so the failure is loud.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return False
    if "{% generation %}" in template:
        return False
    if _QWEN_COMBINED_BRANCH not in template:
        return False
    tokenizer.chat_template = template.replace(
        _QWEN_COMBINED_BRANCH, _QWEN_SPLIT_BRANCH, 1
    )
    return True


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


def _build_quant_config(cfg: Config) -> Any | None:
    """Return a ``BitsAndBytesConfig`` when 4-bit + CUDA + bnb available; else ``None``.

    Imported lazily because ``bitsandbytes`` is not installed on macOS (and
    importing it on a system that hasn't shipped a wheel for this Python/OS
    combo raises at import time, not at call time).
    """
    if not cfg.quant.load_in_4bit:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quant.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_resolve_dtype(cfg.quant.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=cfg.quant.double_quant,
    )


def load_base_model_and_tokenizer(
    cfg: Config,
    adapter: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the base model + tokenizer described by ``cfg.model``.

    Args:
        cfg: resolved Config; ``cfg.model`` (name/revision/dtype) and ``cfg.quant``
            (QLoRA settings) are honored.
        adapter: optional PEFT adapter id (HF Hub repo) to merge / attach.
            Reserved for the inference side; SFT training attaches its LoRA
            via the trainer itself (see ``atlas.models.adapters``).

    Returns:
        ``(model, tokenizer)``. The tokenizer's ``pad_token`` is guaranteed to
        be set (eos fallback) so TRL's trainers don't blow up.
    """
    dtype = _resolve_dtype(cfg.model.dtype)
    quant = _build_quant_config(cfg)

    model_kwargs: dict[str, Any] = {
        # `dtype=` is the post-transformers-4.56 spelling; `torch_dtype=` is the
        # deprecated alias and triggers a runtime warning starting in 4.56.
        "dtype": dtype,
        # Pin to cuda:0 instead of "auto" on multi-GPU hosts (e.g. Kaggle T4x2):
        # "auto" shards the model, then HF Trainer wraps it in DataParallel and
        # crashes with "parameters and buffers must be on cuda:0". A 0.5B model
        # + LoRA fits on one card. Fall back to "auto" off-GPU so tests / Mac
        # smoke runs still load onto CPU.
        "device_map": {"": 0} if torch.cuda.is_available() else "auto",
    }
    if cfg.model.revision:
        model_kwargs["revision"] = cfg.model.revision
    if quant is not None:
        model_kwargs["quantization_config"] = quant

    model: Any = AutoModelForCausalLM.from_pretrained(cfg.model.name, **model_kwargs)

    # Tokenizer may live in a different repo than the model. The pretrained
    # Qwen2.5-0.5B base ships eos == pad and a chat_template without
    # {% generation %} markers; loading the Instruct tokenizer (same vocab,
    # correct eos = <|im_end|>, markers present) lets SFT train cleanly on the
    # pretrained weights. Revision is only applied when tokenizer matches model.
    tokenizer_repo = cfg.model.tokenizer_name or cfg.model.name
    tok_kwargs: dict[str, Any] = {}
    if cfg.model.revision and tokenizer_repo == cfg.model.name:
        tok_kwargs["revision"] = cfg.model.revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    patch_chat_template_for_assistant_mask(tokenizer)

    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)

    return model, tokenizer
