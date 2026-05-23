# Convenience targets. See PROJECT.md section 2 for which targets each phase relies on.
#
# RUNNER prefixes every tool invocation. Auto-detected from two signals:
#   - `uv` on PATH      (else: no uv available)
#   - `.venv` in cwd    (else: not a uv-managed project root, e.g. fresh Colab
#                        clone where `pip install -e .[eval]` populates system
#                        Python without ever creating `.venv`)
# Both true  -> `uv run`   (local dev)
# Either false -> empty    (Colab/Kaggle/CI — use the active environment)
#
# Why both? Colab ships `uv` on PATH (since ~2026-Q1 images), so a PATH-only
# check picked `uv run` on Colab and starved the venv of the [eval] extra
# installed via pip. Pairing with `.venv` distinguishes "uv-managed project"
# from "uv just happens to be installed."
#
# Override explicitly with e.g. `make test RUNNER=python3` if needed.
UV     ?= uv
RUNNER ?= $(if $(and $(shell command -v uv 2>/dev/null),$(wildcard .venv)),uv run,)

.PHONY: help install fmt lint typecheck test test-fast eval-baseline eval-smoke eval-modal-smoke eval-modal sft sft-smoke docs-serve docs-build clean

help:
	@echo "Targets:"
	@echo "  install        uv sync with [dev] + [eval] extras"
	@echo "  fmt            ruff format src/ tests/ scripts/"
	@echo "  lint           ruff check src/ tests/ scripts/"
	@echo "  typecheck      mypy on src/atlas/"
	@echo "  test           full pytest suite (incl. slow)"
	@echo "  test-fast      pytest, excluding @pytest.mark.slow (what CI runs)"
	@echo "  eval-baseline  full lm-eval on un-tuned Qwen2.5-0.5B — Phase 0 deliverable"
	@echo "                 designed for Colab/GPU; slow on Mac CPU"
	@echo "  eval-smoke     limit=10 per task; proves the harness wiring without compute"
	@echo "  eval-modal-smoke  Modal+vLLM wiring probe (limit=50, base only) — cheap"
	@echo "  eval-modal     Modal+vLLM base + sft_v1, fresh metrics.json (the clean re-baseline)"
	@echo "  sft            Phase 1: full SFT run from configs/sft_qwen05b.yaml; pushes adapter to HF Hub"
	@echo "  sft-smoke      50-step SFT smoke (no Hub push); proves the train wiring on free tier"
	@echo "  docs-serve     mkdocs serve on http://127.0.0.1:8000 (live reload)"
	@echo "  docs-build     mkdocs build --strict; mirrors what CI does before deploying to Pages"

install:
	$(UV) sync --extra dev --extra eval

fmt:
	$(RUNNER) ruff format src tests scripts

lint:
	$(RUNNER) ruff check src tests scripts

typecheck:
	$(RUNNER) mypy src/atlas

test:
	$(RUNNER) pytest

test-fast:
	$(RUNNER) pytest -m 'not slow'

# Phase 0 deliverable: populates results/metrics.json with the _base_ row.
# Run on Colab / a GPU per PROJECT.md. Locally on Mac CPU this completes but
# IFEval (which generates) takes hours.
eval-baseline:
	$(RUNNER) python -m atlas.eval.harness \
	    --config configs/baseline.yaml \
	    --name base --method none

# Local smoke: 10 samples per task, separate JSON path. Used to verify the
# harness end-to-end without paying for compute. Not the canonical metrics file.
eval-smoke:
	$(RUNNER) python -m atlas.eval.harness \
	    --config configs/baseline.yaml \
	    --name base_smoke --method none --limit 10 \
	    --metrics-path results/metrics_smoke.json

# Modal + vLLM eval (scripted; see src/atlas/cloud/eval_modal.py). Runs on a Modal
# GPU, not via RUNNER — `modal` is a separate CLI you install on your laptop:
#   pip install modal && modal token new
#   modal secret create hf-token HUGGING_FACE_HUB_TOKEN=hf_xxx   # reuses the SFT secret
# GPU defaults to L4 (bf16-native); override with ATLAS_EVAL_GPU=A10G. A T4 would force fp16.
eval-modal-smoke:
	modal run src/atlas/cloud/eval_modal.py --name base --method none --limit 50

# The clean re-baseline: base + sft_v1 on one backend/dtype, fresh two-row metrics.json.
eval-modal:
	modal run src/atlas/cloud/eval_modal.py::suite --fresh

# Phase 1 deliverable: full SFT run on UltraChat-200k. Pushes the adapter to
# the HF Hub repo defined in configs/sft_qwen05b.yaml's output.hub_repo.
sft:
	$(RUNNER) python -m atlas.train.sft \
	    --config configs/sft_qwen05b.yaml

# Local SFT smoke: 50 steps, no Hub push. Proves wiring on free tier per
# PROJECT.md §5.4 — "never launch paid GPU on code you haven't run 50 steps
# of on free tier." Runs on CPU; tiny in practice because each step is small.
sft-smoke:
	$(RUNNER) python -m atlas.train.sft \
	    --config configs/sft_qwen05b.yaml \
	    --max-steps 50 \
	    --no-push-to-hub

# Docs site (mkdocs-material → GitHub Pages). Install the [docs] extra first:
#   uv sync --extra docs   (or pip install -e .[docs])
docs-serve:
	$(RUNNER) mkdocs serve

docs-build:
	$(RUNNER) mkdocs build --strict

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache site
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
