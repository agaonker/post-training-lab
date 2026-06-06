# TODOS

Punch list of deferred work and findings surfaced by real runs. Priority labels:
**P1** = address before the next phase; **P2** = address before the named phase
it would bite; **P3** = nice-to-have, schedule later.

---

## From the first baseline-eval run on Kaggle (2026-05-16)

### ~~P1 — Incremental persistence in the eval harness~~ (DONE 2026-05-17)
**Landed in:** [src/atlas/eval/harness.py](src/atlas/eval/harness.py) — `_partial_path`, `_write_partial`, `_load_partial`, and the modified `run_eval` loop.
Per-task results are written to `results/metrics.<run_name>.partial.json` via
atomic temp+rename after each task completes. The partial is cleaned up only on
successful `append_run` — if `append_run` raises (e.g. schema mismatch), the
partial sticks around for manual recovery.

### P1 — Wall-clock estimates in PROJECT.md and notebooks are wildly off
**Where:** [PROJECT.md §6 Phase 0](PROJECT.md), [notebooks/colab/run_eval_baseline_colab.ipynb](notebooks/colab/run_eval_baseline_colab.ipynb) cell 15, [notebooks/kaggle/run_eval_baseline_kaggle.ipynb](notebooks/kaggle/run_eval_baseline_kaggle.ipynb) cell "Full baseline eval"
**Why:** Both notebooks claim "~30-60 min on free T4." Reality on free Kaggle (P100):
GSM8K alone was 1h 42m, IFEval ~2h 15m, total ~5h. The "30-60 min" number predates
ever having actually run it.
**Fix:** Replace the estimate with the observed timing table (per task) plus a note
that GSM8K and IFEval are generation-bound and dominate. Also: this invalidates the
Phase 4 GRPO budget envelope (§5.2) which assumed similar throughput — re-check.

### ~~P2 — Add `--resume` to the eval harness~~ (DONE 2026-05-17)
**Landed in:** [src/atlas/eval/harness.py](src/atlas/eval/harness.py) — resume is
the default behavior of `run_eval` (`resume=True`); `main()` accepts `--no-resume`
to force a clean re-run. The partial file is scoped by `--name` so a smoke run
and a baseline run on the same `metrics_path` don't collide. A `config_hash`
mismatch between the partial and the current config refuses to resume rather
than silently mixing runs.

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

### P1 — Pin the Phase-1 SFT adapter as the anchor for Phases 2-5
**Where:** [PROJECT.md §6](PROJECT.md), and every downstream config in `configs/`
**Why:** DPO, PPO, GRPO, KTO/ORPO are all supposed to start from the SAME SFT
checkpoint so the comparison is fair. With the base-model swap (2026-06-06,
pretrained `Qwen2.5-0.5B`), the anchor is `sft_v2` (to-be-trained). `sft_v1`
(`agaonker/atlas-sft-qwen05b-v1`, on `-Instruct`) is historical only — do NOT
reference it from downstream configs.
**Fix:** Once `sft_v2` is trained: add to §6: "Phases 2-5 all start from
`agaonker/atlas-sft-qwen05b-v2`. Pin its HF Hub revision SHA in every
downstream YAML's `model.name` + `model.revision`."
**Source:** Eng review 1A decision; reframed after the base swap.

### ~~P1 — vLLM-on-Modal is required for Phase 4 GRPO, not optional~~ (DONE 2026-05-23)
**Landed in:** vLLM is in the Modal eval image
([src/atlas/cloud/eval_modal.py](src/atlas/cloud/eval_modal.py)) and proved
end-to-end on every base + sft_v1 + new pretrained-base eval. Phase 4 GRPO can
re-use the same image. The pyproject.toml deliberately doesn't pin vllm
(no macOS build); the Modal image carries it.

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

## From the base-model swap + sft_v1 regression debugging (2026-06-06)

### P1 — Train `sft_v2` on the pretrained base
**Where:** Phase 1 deliverable; [configs/sft_qwen05b.yaml](configs/sft_qwen05b.yaml),
[src/atlas/train/sft.py](src/atlas/train/sft.py).
**Why:** The diff (`8c21e04`) landed the fix (pretrained base + `assistant_only_loss`
+ chat-template patch) but no adapter exists yet for the new pipeline.
**Fix:** `modal run` the SFT training; push to `agaonker/atlas-sft-qwen05b-v2`;
eval on Modal; append the new row.

### P1 — Rewrite [experiments/002_sft_qwen05b.md](experiments/002_sft_qwen05b.md) once `sft_v2` lands
**Why:** Current file leads with a refuted hypothesis ("chat-template drift
(most likely)") and reports the old `-Instruct` numbers. It's not just stale —
the lead diagnosis is wrong (see [writeups/sft_regression_diagnosis.html](writeups/sft_regression_diagnosis.html)).
**Fix:** Wait for `sft_v2` numbers, then rewrite end-to-end with: pretrained-base
framing, the real diagnosis (no `assistant_only_loss` + re-SFT-ing aligned
weights), `sft_v2` results vs both pretrained base and `-Instruct` reference.

### P2 — Update [README.md](README.md) results table after `sft_v2`
**Why:** Currently shows only the old `-Instruct` base + `sft_v1` rows and cites
the refuted "chat-template drift" cause. Misleads anyone landing on the repo.
**Fix:** Once `sft_v2` exists: replace with three rows (`base` pretrained,
`sft_v1` historical with a note, `sft_v2`); drop the wrong cause sentence; link
to `writeups/sft_regression_diagnosis.html` and `LESSONS.md`.

### P3 — [scripts/explore_model.py](scripts/explore_model.py) hardcodes `Qwen2.5-0.5B-Instruct`
**Why:** Inconsistent with the new pretrained base; if anyone runs it they'll
explore the wrong model. Not load-bearing — script is for one-shot inspection
only — but it'll confuse a future reader.
**Fix:** Switch the constant to `Qwen/Qwen2.5-0.5B`. Trivial.

### P3 — Author `writeups/01_sft_and_qlora.md` once `sft_v2` lands
**Why:** Phase 1 deliverable per [PROJECT.md §6](PROJECT.md); the polished
writeup hasn't been written yet. The diagnosis HTML records the *fix*; the Phase 1
writeup should record the *story* with final numbers.

---

## Add new TODOs here as they surface

Format each entry as: priority, where, why, fix. Link file paths with
`[path](path#Lline)`. Keep it actionable — vague bullets rot.
