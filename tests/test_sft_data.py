"""Tests for atlas.data.sft_data — UltraChat loader.

The unit tests stub out ``load_dataset`` so they don't hit the network. One
``@pytest.mark.slow`` test exercises the real Hub to catch upstream schema
drift; CI excludes it via ``-m 'not slow'``.
"""

from __future__ import annotations

import pytest

from atlas.data import sft_data


def _fake_ultrachat(n: int):
    """In-memory Dataset shaped like UltraChat's train_sft split."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "prompt": [f"p{i}" for i in range(n)],
            "prompt_id": [f"id{i}" for i in range(n)],
            "messages": [
                [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ]
                for i in range(n)
            ],
        }
    )


# --- unit tests (no network) ----------------------------------------------------

def test_loader_projects_to_messages_only(monkeypatch):
    """Strips prompt + prompt_id, keeps only the column SFTTrainer consumes."""
    monkeypatch.setattr(
        sft_data,
        "load_dataset",
        lambda repo, split, revision=None: _fake_ultrachat(100),
    )
    ds = sft_data.load_ultrachat_sft(n_samples=10)
    assert ds.column_names == ["messages"]
    assert len(ds) == 10


def test_loader_seed_is_deterministic(monkeypatch):
    """Same seed → identical row order, every run."""
    monkeypatch.setattr(
        sft_data,
        "load_dataset",
        lambda repo, split, revision=None: _fake_ultrachat(50),
    )
    a = sft_data.load_ultrachat_sft(n_samples=20, seed=42)
    b = sft_data.load_ultrachat_sft(n_samples=20, seed=42)
    assert [r["messages"] for r in a] == [r["messages"] for r in b]


def test_loader_different_seeds_yield_different_orders(monkeypatch):
    monkeypatch.setattr(
        sft_data,
        "load_dataset",
        lambda repo, split, revision=None: _fake_ultrachat(50),
    )
    a = sft_data.load_ultrachat_sft(n_samples=20, seed=42)
    b = sft_data.load_ultrachat_sft(n_samples=20, seed=7)
    assert [r["messages"] for r in a] != [r["messages"] for r in b]


def test_loader_n_samples_caps_at_dataset_size(monkeypatch):
    """Asking for more rows than exist returns everything, not an IndexError."""
    monkeypatch.setattr(
        sft_data,
        "load_dataset",
        lambda repo, split, revision=None: _fake_ultrachat(15),
    )
    ds = sft_data.load_ultrachat_sft(n_samples=100)
    assert len(ds) == 15


def test_loader_n_samples_none_uses_full_split(monkeypatch):
    monkeypatch.setattr(
        sft_data,
        "load_dataset",
        lambda repo, split, revision=None: _fake_ultrachat(15),
    )
    ds = sft_data.load_ultrachat_sft(n_samples=None)
    assert len(ds) == 15


def test_loader_passes_revision_through(monkeypatch):
    """Pinned dataset commit SHA flows through to load_dataset."""
    seen: dict = {}

    def fake_load(repo, split, revision=None):
        seen["repo"] = repo
        seen["split"] = split
        seen["revision"] = revision
        return _fake_ultrachat(5)

    monkeypatch.setattr(sft_data, "load_dataset", fake_load)
    sft_data.load_ultrachat_sft(n_samples=5, revision="abc123")
    assert seen == {
        "repo": sft_data.ULTRACHAT_REPO,
        "split": sft_data.ULTRACHAT_SPLIT,
        "revision": "abc123",
    }


# --- integration (hits the real Hub) -------------------------------------------

@pytest.mark.slow
def test_loader_against_real_hub():
    """Schema-drift canary. Skipped in CI; run manually before Phase 1 lands."""
    ds = sft_data.load_ultrachat_sft(n_samples=5)
    assert len(ds) == 5
    row = ds[0]
    assert isinstance(row["messages"], list) and row["messages"]
    first = row["messages"][0]
    assert "role" in first and "content" in first
