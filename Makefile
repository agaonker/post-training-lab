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

.PHONY: help install fmt lint typecheck test test-fast eval-baseline eval-smoke clean

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

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
