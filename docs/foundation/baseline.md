# Baseline (Phase 0)

The un-tuned Qwen2.5-0.5B-Instruct evaluated through the same harness every
subsequent method will use. This is the **floor**: every post-training method
has to clear this to be worth running.

Source: [`results/metrics.json`](https://github.com/agaonker/post-training-lab/blob/main/results/metrics.json),
row `name: base`, evaluated 2026-05-17.

## Numbers

| Task | Metric | Score |
|------|--------|-------|
| MMLU (5-shot, limit=1000) | acc | **0.474** |
| GSM8K (8-shot) | exact-match, strict | **0.265** |
| GSM8K (8-shot) | exact-match, flexible | **0.335** |
| TruthfulQA-MC2 | acc | **0.418** |
| IFEval | prompt-strict | **0.201** |
| IFEval | inst-strict | **0.347** |
| IFEval | prompt-loose | **0.248** |
| IFEval | inst-loose | **0.388** |

Run metadata: `Qwen/Qwen2.5-0.5B-Instruct`, `dtype=float16`,
`config_hash=2a8b4c64`, [`configs/baseline.yaml`](https://github.com/agaonker/post-training-lab/blob/main/configs/baseline.yaml).

## How to read these

- **MMLU 0.474** — roughly the published Qwen2.5-0.5B-Instruct number; sanity-checks
  that the harness is wired correctly. The base model already knows things; the
  bar for SFT/DPO is *don't lose this*.
- **GSM8K 0.265 strict / 0.335 flexible** — 0.5 B is at the floor of where
  multi-step arithmetic starts to work. The 7-point gap between strict and
  flexible is mostly formatting — the model often gets the answer right but
  doesn't follow the expected `#### N` output convention. SFT could close
  that gap by teaching the format.
- **TruthfulQA-MC2 0.418** — slightly above random (0.40 chance baseline
  given the multi-choice setup). The model is mildly *less* fooled by
  plausible falsehoods than chance, but not by much. Watch this in Phase 1+;
  instruction-tuning can drag it down.
- **IFEval 0.201 prompt-strict / 0.347 inst-strict** — the headline Phase 1
  target. The model follows ~35% of *individual* instructions strictly but
  only 20% of *full prompts* (which require following every instruction in
  the prompt). The gap between strict and loose (~5 pts) is small.

## Phase 1 target

From PROJECT.md §6: *"SFT model beats base on IFEval by a clear margin."*
Concretely:

| Metric | Base | Phase 1 target | What "clear margin" means |
|--------|------|----------------|---------------------------|
| IFEval prompt-strict | 0.201 | **> 0.25** | ~25% relative improvement |
| IFEval inst-strict | 0.347 | **> 0.40** | ~15% relative improvement |
| MMLU acc | 0.474 | **≥ 0.46** | "don't lose" — within 1 pt of base |
| TruthfulQA-MC2 acc | 0.418 | **≥ 0.39** | "don't tank" — within 3 pts of base |
| GSM8K (either) | 0.265 / 0.335 | _no strong prior_ | small movement either direction is fine |

The IFEval bars are set to be **clear**, not just statistically significant.
If Phase 1's adapter doesn't clear them, that's a recipe issue
(data, hparams, masking), not measurement noise.

## Why these specific numbers

These are not arbitrary thresholds. They come from two places:

1. **The gap to known good models in this size class.** Qwen2.5-1.5B-Instruct
   scores ~0.32 on IFEval prompt-strict; that's the next-rung-up benchmark.
   We want the SFT'd 0.5B to close ~50% of the gap to the 1.5B base —
   roughly 0.25.
2. **The recipe headroom in the literature.** Zephyr-style SFT recipes on
   UltraChat reliably move IFEval prompt-strict by +5–10 absolute points on
   sub-1B models. 0.20 → 0.25 is the conservative end of that range; if we
   miss it, something is wrong.

## How to reproduce

```bash
make install
make eval-baseline    # Colab/Kaggle T4 — takes ~30 min, mostly IFEval
```

For a no-GPU sanity check that the wiring works (10 samples per task):

```bash
make eval-smoke
```

The smoke run writes to `results/metrics_smoke.json` rather than the canonical file.
