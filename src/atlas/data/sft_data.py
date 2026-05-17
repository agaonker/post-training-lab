"""SFT dataset loader for UltraChat-200k.

Returns a HF Datasets object with a single ``messages`` column. TRL's
``SFTTrainer`` auto-applies the model's chat template at training time, so the
loader doesn't need to render text — it just shuffles, caps, and projects.

UltraChat-200k has four splits (``train_sft``, ``test_sft``, ``train_gen``,
``test_gen``); Phase 1 uses ``train_sft``. UltraFeedback-binarized (Phase 2 DPO
data) is derived from this same corpus, so the data-lineage story for the
comparison stays clean.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset

ULTRACHAT_REPO = "HuggingFaceH4/ultrachat_200k"
ULTRACHAT_SPLIT = "train_sft"


def load_ultrachat_sft(
    n_samples: int | None = 5000,
    seed: int = 42,
    revision: str | None = None,
) -> Dataset:
    """Load a shuffled, capped slice of UltraChat-200k for SFT.

    Args:
        n_samples: row cap; ``None`` uses the entire split (~207k rows).
        seed: shuffle seed. Pin in the experiment config so the same YAML
            reproduces the same training slice.
        revision: dataset commit SHA on HF Hub; ``None`` = latest. Pin once a
            config is locked in to avoid silent upstream drift.

    Returns:
        ``Dataset`` with a single ``messages`` column. Each row is a list of
        ``{"role": "user" | "assistant", "content": str}`` dicts — the format
        TRL's SFTTrainer expects when a chat template is in play.
    """
    ds = load_dataset(ULTRACHAT_REPO, split=ULTRACHAT_SPLIT, revision=revision)
    ds = ds.shuffle(seed=seed)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
    return ds.select_columns(["messages"])
