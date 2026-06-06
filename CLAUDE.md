# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A controlled comparison of post-training methods (SFT, DPO, KTO/ORPO, RLHF/PPO, GRPO/RLVR) on **one** small base model (`Qwen/Qwen2.5-0.5B`, pretrained — not the `-Instruct` variant) with **one** evaluation harness. The comparison — not any individual method — is the contribution. See [PROJECT.md](PROJECT.md) for the full plan-of-record (phases, datasets per phase, compute budget, anti-goals).

Code lives in `src/atlas/` and is importable as `atlas.*` (the project is named `post-training-lab`; the package is `atlas`).

Currently in **Phase 1** (mid-flight): config system, eval harness, and SFT pipeline all in place. After `sft_v1` (trained on `-Instruct`) regressed uniformly on every eval — diagnosed in `writeups/sft_regression_diagnosis.html` and captured in `LESSONS.md` — the base was switched to pretrained `Qwen/Qwen2.5-0.5B`, `assistant_only_loss` enabled, and the chat_template patched for TRL's mask path. New pretrained-base eval row landed (`config_hash fde0720e`). `sft_v2` to be trained next on the new base; `sft_v1` (`agaonker/atlas-sft-qwen05b-v1`) is kept on the Hub as historical reference but is no longer the active adapter.

## Common commands

Use `make <target>` — the Makefile auto-detects whether to prefix with `uv run`.

```
make install        # uv sync --extra dev --extra eval
make fmt            # ruff format
make lint           # ruff check  (CI runs this)
make typecheck      # mypy on src/atlas/  (CI runs this)
make test           # full pytest
make test-fast      # pytest -m 'not slow'  (CI runs this)
make eval-baseline  # full lm-eval on un-tuned Qwen2.5-0.5B → results/metrics.json
make eval-smoke     # limit=10 per task → results/metrics_smoke.json (no GPU needed)
```

Single test: `uv run pytest tests/test_eval.py::test_flatten_metrics_keeps_scalars_and_drops_stderr`

### RUNNER auto-detect (Makefile)

`RUNNER` is `uv run` only when **both** `uv` is on PATH **and** `.venv` exists in the cwd; otherwise it's empty so commands use the active environment. This exists because Colab now ships `uv` on PATH but installs the project via `pip install -e .[eval]` into the system Python — a PATH-only check would route through a starved uv venv. Override with `make test RUNNER=python3` if needed.

## Architecture

### Config system — `atlas.utils.config`

Every experiment is a YAML in `configs/`. `load_config(path)` deep-merges the experiment file over `configs/base.yaml`, validates against the pydantic `Config` model, and attaches a stable 8-char `config_hash` (sha256 of canonical JSON). Two design choices matter:

- **`Config` uses `extra="forbid"`** — typo'd top-level keys raise. But **`TrainCfg` uses `extra="allow"`** as a deliberate passthrough: its `model_dump()` is fed straight to a TRL `*Config` (e.g. `SFTConfig(**cfg.train.model_dump())`), so we never re-declare TRL's knob surface and the schema survives TRL minor bumps.
- **YAML float gotcha**: write `2.0e-4`, not `2e-4` — PyYAML parses the dot-less form as a string.

### Eval harness — `atlas.eval.harness`

One entry point used by every phase: baseline calls it with `adapter=None`; trained runs pass an HF Hub adapter id. Same task list, same metric extraction, same JSON schema — apples-to-apples by construction.

- `run_eval` appends one entry per run to `results/metrics.json` (`{schema_version, runs: [...]}`). Schema-version mismatch raises rather than silently overwriting.
- **lm-eval is imported lazily** inside `_build_lm` / `_evaluate_one_task`. These two functions are the test seams — monkeypatch them and the full `run_eval` pipeline runs with no lm-eval install and no model weights. **CI does this**: it installs only `--extra dev`, not `[eval]`, so do not move lm-eval imports to module top-level. The `[eval]` extra (lm-eval, anthropic) is only needed to actually run evals.
- `_flatten_metrics` keeps the `acc,none` / `strict-match,none` style keys verbatim (the comma is lm-eval's filter suffix) and drops anything with `stderr` or non-numeric values.

### Package layout

```
src/atlas/
  data/         dataset loaders (preference_data.py, sft_data.py, verifiable_data.py) — Phase 1+
  models/       base loaders + LoRA/QLoRA helpers — Phase 1+
  train/        sft.py / dpo.py / ppo.py / grpo.py / kto.py / reward_model.py — Phase 1+
  eval/         harness.py (built), judge.py / compare.py — Phase 2+/6
  utils/        config.py (built), logging_utils.py, checkpoint.py
  cloud/        Modal entrypoints (same code path runs local or on Modal)
```

Trained adapters are pushed to HF Hub (not committed). The repo only stores `results/metrics.json` and `results/plots/`.

## Conventions worth knowing

- **`bitsandbytes` is Linux-only** in `pyproject.toml` (`platform_system == 'Linux'`) so `uv sync` works on Mac. Quantization paths run on Colab/Modal, not locally.
- **vLLM is deliberately not in `pyproject.toml`** — no macOS build, only installed in the Modal image. Carrying it as an extra forced unsatisfiable resolves.
- **trl pinned `>=1.4,<2`** with `transformers>=4.56.2`. The PROJECT.md text mentions older floors (trl>=0.12, transformers>=4.45); the pyproject is the source of truth — it was corrected after pinning.
- **`lm-eval[ifeval]` extra is required**, not plain `lm-eval`. The `[ifeval]` extra pulls langdetect/immutabledict/nltk which IFEval's checkers import at task-load time; without them the eval crashes mid-run.
- **Custom test marker**: `@pytest.mark.slow` for network/training tests. CI runs `-m 'not slow'`.
- `ruff` line length is 100; mypy is `check_untyped_defs=true` but `disallow_untyped_defs=false` (loose on purpose — study repo).
- Secrets via `.env` (gitignored, copy from `.env.example`): `HUGGING_FACE_HUB_TOKEN`, `WANDB_API_KEY`, `ANTHROPIC_API_KEY` (judge only), optional `HF_HOME`.

## Anti-goals (from PROJECT.md §10)

- Notebooks are for exploration only, never the source of truth — code lives in `src/atlas/`.
- Don't train on paid compute (Modal) before smoke-testing on free tier — `make eval-smoke` and ~50-step local runs catch silent dataset/tokenization bugs.
- Don't skip the per-phase writeups in `writeups/` — those are the artifact, not the code.
