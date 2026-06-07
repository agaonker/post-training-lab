"""Preference dataset loader for UltraFeedback-binarized — Phase 2 DPO data.

Returns a HF ``Dataset`` with three columns — ``prompt``, ``chosen``, ``rejected``
— each a list of ``{"role", "content"}`` dicts. This is the *conversational*
form of TRL's preference schema (TRL also accepts a string form); the
conversational form lets TRL apply the model's chat template, which routes
through our patched ``{% generation %}``-marked template the same way SFT does
(see ``atlas.models.base.patch_chat_template_for_assistant_mask``).

UltraFeedback-binarized is derived from UltraChat-200k, so the preference data
is in-distribution for an SFT model trained on UltraChat-200k — the
"keeps prefs in-distribution for the policy" rationale (PROJECT.md §4.1).

Splits on the Hub: ``train_prefs`` (~62k) / ``test_prefs`` (~2k). Phase 2 uses
``train_prefs``.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset

ULTRAFEEDBACK_REPO = "HuggingFaceH4/ultrafeedback_binarized"
ULTRAFEEDBACK_SPLIT = "train_prefs"


def _project_to_conversational(row: dict) -> dict:
    """Project a raw UltraFeedback row to TRL's conversational preference schema.

    Raw row carries ``chosen`` / ``rejected`` as full multi-turn message lists
    that include the user prompt. TRL wants:
    - ``prompt``: just the user turn(s) (list of role/content dicts)
    - ``chosen`` / ``rejected``: just the assistant turn (list of role/content)

    For UltraFeedback's single-turn shape, the user turn lives in ``row["prompt"]``
    (a string) and the chosen/rejected assistant content is the last message of
    ``row["chosen"]`` / ``row["rejected"]``.
    """
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"][-1]["content"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"][-1]["content"]}],
    }


def load_ultrafeedback_prefs(
    n_samples: int | None = 5000,
    seed: int = 42,
    revision: str | None = None,
) -> Dataset:
    """Load a shuffled, capped slice of UltraFeedback-binarized for DPO.

    Args:
        n_samples: row cap; ``None`` uses the entire split (~62k pairs).
        seed: shuffle seed. Pin in the experiment config so the same YAML
            reproduces the same training slice.
        revision: dataset commit SHA on HF Hub; ``None`` = latest. Pin once a
            config is locked in to avoid silent upstream drift.

    Returns:
        ``Dataset`` with columns ``prompt`` / ``chosen`` / ``rejected``, each
        a list of role/content dicts — the conversational preference format
        TRL's ``DPOTrainer`` expects when a chat template is in play.
    """
    ds = load_dataset(ULTRAFEEDBACK_REPO, split=ULTRAFEEDBACK_SPLIT, revision=revision)
    ds = ds.shuffle(seed=seed)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
    return ds.map(_project_to_conversational, remove_columns=ds.column_names)
