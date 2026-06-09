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

.PHONY: help install fmt lint typecheck test test-fast eval-baseline eval-smoke eval-modal-check eval-modal-probe eval-modal-shell eval-modal-smoke eval-modal eval-modal-sft sft sft-smoke sft-modal-smoke sft-modal dpo dpo-smoke dpo-modal-smoke dpo-modal rm rm-smoke rm-modal-smoke rm-modal rloo rloo-smoke rloo-modal-smoke rloo-modal docs-serve docs-build clean

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
	@echo "  Modal eval — tiered loop, cheapest first (only pay for what a change can break):"
	@echo "  eval-modal-check  local preflight: imports/entrypoints/config — seconds, no GPU"
	@echo "  eval-modal-probe  ~1-2 min: one task (gsm8k), limit 5 — proves the vLLM path"
	@echo "  eval-modal-shell  interactive container in the eval image (vllm+atlas+GPU)"
	@echo "  eval-modal-smoke  Modal+vLLM wiring probe (limit=50, base only) — runs all 4 tasks"
	@echo "  eval-modal     Modal+vLLM base + sft_v1, fresh metrics.json (the clean re-baseline)"
	@echo "  eval-modal-sft Modal+vLLM sft_v1 only; APPENDS to metrics.json (preserves the base row)"
	@echo "  sft            Phase 1: full SFT run from configs/sft_qwen05b.yaml; pushes adapter to HF Hub"
	@echo "  sft-smoke      50-step local SFT smoke (no Hub push); slow on Mac/CPU"
	@echo "  sft-modal-smoke 50-step SFT on Modal L4 (no Hub push), ~5 min, ~$$0.30 — pre-flight for sft-modal"
	@echo "  sft-modal      Full SFT on Modal L4, pushes adapter to HF Hub, ~30-60 min, ~$$1-2"
	@echo "  dpo            Phase 2: full DPO run from configs/dpo_qwen05b.yaml; pushes adapter to HF Hub"
	@echo "  dpo-smoke      50-step local DPO smoke (no Hub push); slow on Mac/CPU, fast on Linux GPU"
	@echo "  dpo-modal-smoke 50-step DPO on Modal L4 (no Hub push), ~5 min, ~$$0.30 — pre-flight for dpo-modal"
	@echo "  dpo-modal      Full DPO on Modal L4, pushes adapter to HF Hub, ~30-60 min, ~$$1-2"
	@echo "  rm             Phase 3A: full RM run from configs/rm_qwen05b.yaml; pushes adapter to HF Hub"
	@echo "  rm-smoke       50-step local RM smoke (no Hub push); slow on Mac/CPU, fast on Linux GPU"
	@echo "  rm-modal-smoke 50-step RM on Modal L4 (no Hub push), ~5 min, ~$$0.30 — pre-flight for rm-modal"
	@echo "  rm-modal       Full RM on Modal L4, pushes adapter to HF Hub, ~30-45 min, ~$$1-2"
	@echo "  rloo           Phase 3B: full RLOO run from configs/rloo_qwen05b.yaml; pushes adapter to HF Hub"
	@echo "  rloo-smoke     50-step local RLOO smoke (no Hub push); slow on Mac/CPU"
	@echo "  rloo-modal-smoke 50-step RLOO on Modal L4 (no Hub push), ~5-10 min, ~$$0.30"
	@echo "  rloo-modal     Full RLOO on Modal L4, pushes adapter to HF Hub, ~45-90 min, ~$$2-4"
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
# Tiered fast loop. eval-modal-check runs OUTSIDE the GPU path: it must use the same Python
# as the `modal` CLI (which holds the modal package), NOT `uv run` (.venv has no modal, and
# using it would hide the local-import bugs this check exists to catch). The `modal` shim is
# a pyenv bash wrapper (no python shebang to read), so default to the active pyenv interpreter
# that backs it; fall back to python3 off-pyenv. Override: `make eval-modal-check MODAL_PY=...`.
MODAL_PY ?= $(shell pyenv which python 2>/dev/null || command -v python3)
eval-modal-check:
	$(MODAL_PY) scripts/eval_modal_check.py

# ~1-2 min: one small task, limit 5. The everyday "did my change break the wiring?" loop.
eval-modal-probe:
	modal run src/atlas/cloud/eval_modal.py::probe --task gsm8k --limit 5

# Interactive shell in the eval image (same vllm+atlas+GPU+secret as run_eval_remote). Boot
# once, iterate inside — `python -c "import vllm, atlas"`, rerun the harness, etc. Holds a GPU
# until you exit, so it's for debugging, not idle sessions. `--cmd python` for a Python REPL.
eval-modal-shell:
	modal shell src/atlas/cloud/eval_modal.py::run_eval_remote

eval-modal-smoke:
	modal run src/atlas/cloud/eval_modal.py::main --name base --method none --limit 50

# The clean re-baseline: base + sft_v1 on one backend/dtype, fresh two-row metrics.json.
eval-modal:
	modal run src/atlas/cloud/eval_modal.py::suite --fresh

# Append the SFT eval as a second row WITHOUT wiping the committed base row. Use this
# (not eval-modal, which is suite --fresh) once the base row is locked in: same image /
# backend / dtype / config_hash as base, only the adapter differs -> still apples-to-apples.
eval-modal-sft:
	modal run src/atlas/cloud/eval_modal.py::main \
	    --name sft_v1 --method sft \
	    --adapter agaonker/atlas-sft-qwen05b-v1

# Phase 1 deliverable: full SFT run on UltraChat-200k. Pushes the adapter to
# the HF Hub repo defined in configs/sft_qwen05b.yaml's output.hub_repo.
sft:
	$(RUNNER) python -m atlas.train.sft \
	    --config configs/sft_qwen05b.yaml

# Local SFT smoke: 50 steps, no Hub push. Proves wiring on free tier per
# PROJECT.md §5.4 — "never launch paid GPU on code you haven't run 50 steps
# of on free tier." Runs on CPU; tiny in practice because each step is small.
# WANDB_MODE=disabled avoids the no-API-key crash on dev machines (see LESSONS.md).
sft-smoke:
	WANDB_MODE=disabled $(RUNNER) python -m atlas.train.sft \
	    --config configs/sft_qwen05b.yaml \
	    --max-steps 50 \
	    --no-push-to-hub

# Modal SFT smoke: 50 steps on L4, no Hub push. ~5 min, ~$0.30. The right "smoke"
# tier when the local sft-smoke would take 4+ hours on MPS (Mac M-series).
sft-modal-smoke:
	modal run src/atlas/cloud/sft_modal.py::smoke

# Modal full SFT: per configs/sft_qwen05b.yaml's dataset.n_samples + train.* knobs.
# Pushes the trained adapter to cfg.output.hub_repo. ~30-60 min on L4, ~$1-2.
sft-modal:
	modal run src/atlas/cloud/sft_modal.py::main

# Phase 2 deliverable: full DPO run from configs/dpo_qwen05b.yaml. Pushes the
# adapter to the HF Hub repo defined in cfg.output.hub_repo.
dpo:
	$(RUNNER) python -m atlas.train.dpo \
	    --config configs/dpo_qwen05b.yaml

# Local DPO smoke: 50 steps, no Hub push. Same WANDB_MODE=disabled escape hatch
# as sft-smoke (see LESSONS.md). Designed for a linux GPU; on Mac MPS it's
# slow because of the policy + ref forward duplication DPO requires.
dpo-smoke:
	WANDB_MODE=disabled $(RUNNER) python -m atlas.train.dpo \
	    --config configs/dpo_qwen05b.yaml \
	    --max-steps 50 \
	    --no-push-to-hub

# Modal DPO smoke: 50 steps on L4, no Hub push. ~5 min, ~$0.30. Same tier as
# sft-modal-smoke — proves the merge-then-DPO + new-LoRA wiring without
# committing to the full ~$1-2 training run.
dpo-modal-smoke:
	modal run src/atlas/cloud/dpo_modal.py::smoke

# Modal full DPO: per configs/dpo_qwen05b.yaml. Pushes the trained adapter
# to cfg.output.hub_repo. ~30-60 min on L4, ~$1-2.
dpo-modal:
	modal run src/atlas/cloud/dpo_modal.py::main

# Phase 3A deliverable: full Reward Model run from configs/rm_qwen05b.yaml.
# Pushes the RM adapter to the HF Hub repo defined in cfg.output.hub_repo;
# Phase 3B PPO consumes that adapter to score rollouts.
rm:
	$(RUNNER) python -m atlas.train.reward_model \
	    --config configs/rm_qwen05b.yaml

# Local RM smoke: 50 steps, no Hub push. Same WANDB_MODE=disabled escape hatch
# as sft-smoke / dpo-smoke (see LESSONS.md).
rm-smoke:
	WANDB_MODE=disabled $(RUNNER) python -m atlas.train.reward_model \
	    --config configs/rm_qwen05b.yaml \
	    --max-steps 50 \
	    --no-push-to-hub

# Modal RM smoke: 50 steps on L4, no Hub push. ~5 min, ~$0.30. Proves the
# SequenceClassification head + LoRA SEQ_CLS task_type + regression head
# wire correctly before the full run.
rm-modal-smoke:
	modal run src/atlas/cloud/rm_modal.py::smoke

# Modal full RM: per configs/rm_qwen05b.yaml. Pushes the trained adapter
# to cfg.output.hub_repo. ~30-45 min on L4, ~$1-2.
rm-modal:
	modal run src/atlas/cloud/rm_modal.py::main

# Phase 3B deliverable: full RLOO run from configs/rloo_qwen05b.yaml. Loads
# the sft_v2 + rm_v1 adapters, generates rollouts, REINFORCE-updates the
# policy with the RM scoring + KL penalty against ref.
rloo:
	$(RUNNER) python -m atlas.train.rloo \
	    --config configs/rloo_qwen05b.yaml

# Local RLOO smoke: 50 steps, no Hub push. WANDB_MODE=disabled escape hatch.
rloo-smoke:
	WANDB_MODE=disabled $(RUNNER) python -m atlas.train.rloo \
	    --config configs/rloo_qwen05b.yaml \
	    --max-steps 50 \
	    --no-push-to-hub

# Modal RLOO smoke: 50 steps on L4, no Hub push. ~5-10 min, ~$0.30. Proves
# the three-model setup (policy + ref-via-LoRA-toggle + RM) loads, rollouts
# generate, rewards score, REINFORCE update lands.
rloo-modal-smoke:
	modal run src/atlas/cloud/rloo_modal.py::smoke

# Modal full RLOO: per configs/rloo_qwen05b.yaml. ~45-90 min on L4 because
# of generation-bound rollouts (no vLLM yet in the smoke config).
rloo-modal:
	modal run src/atlas/cloud/rloo_modal.py::main

# Docs site (mkdocs-material → GitHub Pages). Install the [docs] extra first:
#   uv sync --extra docs   (or pip install -e .[docs])
docs-serve:
	$(RUNNER) mkdocs serve

docs-build:
	$(RUNNER) mkdocs build --strict

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache site
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
