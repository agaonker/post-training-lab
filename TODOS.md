# TODOS

Punch list of deferred work and findings surfaced by real runs. Priority labels:
**P1** = address before the next phase; **P2** = address before the named phase
it would bite; **P3** = nice-to-have, schedule later.

---

## From the first baseline-eval run on Kaggle (2026-05-16)

### P1 — Incremental persistence in the eval harness
**Where:** [src/atlas/eval/harness.py:122-149](src/atlas/eval/harness.py#L122-L149)
**Why:** `run_eval` runs all tasks in a loop and only calls `append_run` ONCE at the
very end. The first baseline run took ~5h on Kaggle (P100/T4-class GPU) — if the
kernel had died in IFEval (the last task), MMLU + GSM8K + TruthfulQA would all be
lost. Four hours of compute, zero rows on disk.
**Fix:** write a partial-run JSON after each task completes (e.g.
`results/metrics.<run_name>.partial.json`), then promote to `metrics.json` once all
tasks finish. Resume logic (below) consumes the same partial file.

### P1 — Wall-clock estimates in PROJECT.md and notebooks are wildly off
**Where:** [PROJECT.md §6 Phase 0](PROJECT.md), [notebooks/colab/run_eval_baseline_colab.ipynb](notebooks/colab/run_eval_baseline_colab.ipynb) cell 15, [notebooks/kaggle/run_eval_baseline_kaggle.ipynb](notebooks/kaggle/run_eval_baseline_kaggle.ipynb) cell "Full baseline eval"
**Why:** Both notebooks claim "~30-60 min on free T4." Reality on free Kaggle (P100):
GSM8K alone was 1h 42m, IFEval ~2h 15m, total ~5h. The "30-60 min" number predates
ever having actually run it.
**Fix:** Replace the estimate with the observed timing table (per task) plus a note
that GSM8K and IFEval are generation-bound and dominate. Also: this invalidates the
Phase 4 GRPO budget envelope (§5.2) which assumed similar throughput — re-check.

### P2 — Add `--resume` to the eval harness
**Where:** [src/atlas/eval/harness.py](src/atlas/eval/harness.py), CLI in `main()`
**Why:** With incremental persistence in place, a crashed run shouldn't have to
re-do completed tasks. For Phase 4 GRPO (multi-hour) and Phase 6 (judge calls
across 200 prompts × N methods) this becomes load-bearing.
**Fix:** `--resume` flag reads the partial JSON, skips tasks already present in it,
runs only the missing ones, finalizes.

### P2 — Kaggle notebook should default to Save & Run All, not interactive
**Where:** [notebooks/kaggle/run_eval_baseline_kaggle.ipynb](notebooks/kaggle/run_eval_baseline_kaggle.ipynb) intro markdown
**Why:** Interactive runs require the browser tab to stay open; for a 5h eval this
is operationally fragile. The right pattern is "Save Version → Save & Run All"
(detached, 9h cap). Currently mentioned only in cell 10 as a download trick.
**Fix:** Promote Save & Run All to the up-front checklist; demote interactive to
"if you're babysitting."

### P3 — Per-task wallclock + ETA logging during the run
**Where:** [src/atlas/eval/harness.py](src/atlas/eval/harness.py), `run_eval` loop
**Why:** `lm-eval` prints its own per-task progress bars, but the harness gives no
signal of "task 3 of 4 done, ~2h remaining" — you only know by reading lm-eval's
nested bars. For long runs it'd be nice to know your elapsed and ETA at the
harness level.
**Fix:** Log `[task_name] done in HH:MM:SS, N/M tasks complete` after each task.

---

## From the Phase 0 plan-eng-review (2026-05-16, in progress)

### P1 — Pin one SFT checkpoint as the anchor for Phases 2-5
**Where:** [PROJECT.md §6](PROJECT.md), and every downstream config in `configs/`
**Why:** DPO, PPO, GRPO, KTO/ORPO are all supposed to start from the SAME SFT
checkpoint so the comparison is fair. Currently implicit ("the SFT checkpoint") —
re-training SFT mid-project would silently change what the comparison means.
**Fix:** Add to §6: "Phases 2-5 all start from the Phase 1 SFT adapter merged into
the base. Pin its HF Hub revision SHA in every downstream YAML's
`model.name` + `model.revision`." Optional belt-and-suspenders: assert it in
train scripts.
**Source:** Eng review 1A decision.

### P1 — vLLM-on-Modal is required for Phase 4 GRPO, not optional
**Where:** [PROJECT.md §3](PROJECT.md) (current "vllm (optional)") and §5.3 Modal recipe
**Why:** GRPO rollouts on plain HF `.generate()` are 5-10x slower than vLLM —
Phase 4's $5-20 budget assumes vLLM throughput. Without it the budget triples
or the run doesn't finish.
**Fix:** Reclassify vllm as "required for Phase 4" in §3; bake into the Modal
image in §5.3. Update Phase 4 budget if it doesn't account for this.
**Source:** Eng review 1C decision.

### P2 — Reward model architecture decision (deferred to Phase 3)
**Where:** [PROJECT.md §9 Open Decisions](PROJECT.md)
**Why:** Phase 3 says "train RM, ≥65% accuracy" but doesn't specify base model
(same Qwen2.5-0.5B + regression head? Bigger? LoRA-on-head?). Affects training
time AND PPO stability downstream. Deferred but must be decided before Phase 3
starts.
**Fix:** Add a bullet to §9: "RM architecture: same base + full FT via TRL
`RewardTrainer` / same base + LoRA-on-head / bigger base (1.5B). Decide
before Phase 3-A."
**Source:** Eng review 1B decision (deferred).

### P2 — `TrainCfg.output_dir` default is a shared-collision footgun
**Where:** [src/atlas/utils/config.py:96](src/atlas/utils/config.py#L96)
**Why:** Every experiment that doesn't override `output_dir` writes to
`outputs/run` — two parallel runs clobber each other. Easy to forget when
copying a config.
**Fix:** Either drop the default (make it required) or default to
`outputs/${run_name}` (requires Pydantic interpolation or a post-validator).

### P3 — Decide Modal vs Colab Pro before Phase 3 (with virtual card if Modal)
**Where:** [PROJECT.md §5](PROJECT.md) compute strategy
**Why:** Phase 3 PPO and Phase 4 GRPO need either Modal (vLLM-bound for GRPO
per the 1C decision) or Colab Pro ($10/mo, accepts PayPal — no card-on-file).
Modal has $30 free credit but requires a card; runaway-bill anxiety is a real
constraint. Decision is reversible (sign up later) but the plumbing in
[src/atlas/cloud/modal_train.py](src/atlas/cloud/modal_train.py) only makes
sense if Modal wins.
**Fix:** Before Phase 3 starts, pick one:
- **Modal** — sign up with a Privacy.com virtual card capped at $30; set a
  Modal dashboard spending limit ($10/mo to start); confirm every Modal
  function has `timeout=`; confirm every train config has `max_steps`. Then
  stand up `src/atlas/cloud/modal_train.py` per PROJECT.md §5.3.
- **Colab Pro** — accept the ~2-3x throughput hit vs Modal A10G; lose Phase 4
  GRPO with vLLM (drop to slower TRL-default rollouts); drop the
  `src/atlas/cloud/` scaffold from the plan.
- **Defer** — finish through Phase 2 on free tier; revisit when Phase 3
  blocks. PROJECT.md §10 anti-goal "don't do Phase 7 before Phase 6"
  generalizes: don't pay for compute before you need it.

---

## Add new TODOs here as they surface

Format each entry as: priority, where, why, fix. Link file paths with
`[path](path#Lline)`. Keep it actionable — vague bullets rot.
