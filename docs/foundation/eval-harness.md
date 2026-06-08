# Evaluation harness

A single eval harness that every phase runs through, producing a row in the
same `results/metrics.json` file with the same schema. Phase 0 calls it on
the un-tuned base; Phase 1+ calls it with `--adapter <hf-repo>` pointing at a
trained LoRA adapter. **Apples-to-apples by construction** — methods can't
choose their own tasks, their own few-shot setup, or their own metric
extraction.

Code: [`src/atlas/eval/harness.py`](https://github.com/agaonker/post-training-lab/blob/main/src/atlas/eval/harness.py)

## Tasks

Four tasks chosen to probe different capabilities. None are perfect; the
mix is the point — a method that improves IFEval but tanks MMLU is a
different story from one that improves both.

| Task | What it measures | Few-shot | Limit | Why it's in the mix |
|------|------------------|----------|-------|--------------------|
| **MMLU** | World knowledge across 57 subjects (multiple choice) | 5 | 1000 | Catastrophic forgetting check. Post-training shouldn't *lose* pretrained knowledge. |
| **GSM8K** | Grade-school math word problems (generation, exact match) | 8 | full | Reasoning that requires multi-step generation, not pattern matching. Hard for 0.5B; we expect small movement. |
| **TruthfulQA-MC2** | Resistance to plausible-sounding false statements | 0 | full | Honesty / calibration probe. Instruction-tuning can make this go *down*. |
| **IFEval** | Strict instruction following ("answer in 3 bullet points", "no commas") | 0 | full | The headline Phase 1 metric. SFT *should* move this clearly. |

MMLU is capped at 1000 questions per subject to keep eval wall-clock under
~30 min on a T4. The other three run in full because they're smaller.

## Why these four (and not others)

- **HumanEval / MBPP** (code generation) — excluded because 0.5B can barely
  generate compilable Python; signal would be near-zero.
- **HellaSwag / ARC** (commonsense) — excluded as redundant with MMLU for
  this size class.
- **MT-Bench / AlpacaEval** (LLM-judge win rates) — used separately for
  Phase 2+ DPO comparisons (judge-pairwise), not in the harness. The harness
  is for *deterministic, reproducible* metrics; LLM-judge results live in
  their own pipeline with explicit cost tracking.

## Why the same `eval:` block across phases

If SFT used IFEval@strict and DPO used IFEval@loose, we couldn't compare
them. The eval config lives in `configs/base.yaml` and is inherited by every
experiment YAML — it's not method-tunable. The same task list, the same
`num_fewshot`, the same `limit`.

## Implementation

The harness is a thin wrapper over [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
that handles three things lm-eval doesn't:

1. **Adapter loading.** `--adapter <hf-repo>` attaches a PEFT adapter to
   the base model before eval. The base model name comes from the same
   `cfg.model.name` SFT trained on, so the eval can't accidentally drift
   to a different base.
2. **Metric flattening + JSON schema.** lm-eval returns nested dicts; the
   harness flattens to `task/metric` keys (e.g. `ifeval/prompt_level_strict_acc,none`),
   drops `stderr` and non-numeric fields, and appends one entry per run to
   `results/metrics.json` under a stable schema (`{schema_version, runs: [...]}`).
3. **Incremental persistence.** Each task's results are written to a
   `metrics.<name>.partial.json` as soon as it completes, then promoted to
   `metrics.json` only on full success. A mid-run Colab disconnect doesn't
   lose completed tasks — the [Phase 0 lesson][p0] that prompted the
   `--resume` flag.

[p0]: https://github.com/agaonker/post-training-lab/blob/main/PROJECT.md

## Running it

```bash
# Phase 0: un-tuned pretrained baseline (Modal L4, vLLM/bf16)
make eval-modal

# Smoke test (10 samples per task — no GPU needed)
make eval-smoke

# Phase 1+ with a trained adapter (pass --tokenizer too so generation
# terminates on the saved Instruct-template's <|im_end|>, not the base's
# <|endoftext|>):
modal run src/atlas/cloud/eval_modal.py::main \
    --name sft_v2 --method sft \
    --adapter agaonker/atlas-sft-qwen05b-v2 \
    --tokenizer agaonker/atlas-sft-qwen05b-v2
```

The `--name` and `--method` flags label the row in `metrics.json`; the
[Results page](../results.md) reads from there.

## What's *not* tested by the harness

These need other instrumentation:

- **Win rate vs SFT** (Phase 2 DPO success criterion) — pairwise LLM judge on
  held-out UltraFeedback prompts. Lives in [`src/atlas/eval/judge.py`](https://github.com/agaonker/post-training-lab/blob/main/src/atlas/eval/judge.py)
  (Phase 2+).
- **KL divergence from reference policy** (PPO sanity check) — only meaningful
  during training and lives in the trainer's W&B logs, not the eval harness.
- **Reward model accuracy** (Phase 3 deliverable) — separate sanity-check
  script against held-out preference pairs.
