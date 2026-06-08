# Baseline (Phase 0)

The un-tuned pretrained `Qwen/Qwen2.5-0.5B` evaluated through the same harness
every subsequent method will use. This is the **floor**: every post-training
method has to clear this to be worth running.

Source: [`results/metrics.json`](https://github.com/agaonker/post-training-lab/blob/main/results/metrics.json),
row `name: base` with `config_hash: fde0720e` (the pretrained-base row;
the older `-Instruct` rows with `config_hash 6af9a640` remain in
`metrics.json` for historical reference but are not the comparison anchor —
see [Mistakes & learnings on the README](https://github.com/agaonker/post-training-lab#learnings)
and `experiments/002` for why we switched).

## Numbers

| Task | Metric | Score |
|------|--------|-------|
| MMLU (5-shot, limit=1000) | acc | **0.4813** |
| GSM8K (8-shot) | exact-match, strict | **0.3389** |
| GSM8K (8-shot) | exact-match, flexible | **0.3419** |
| TruthfulQA-MC2 | acc | **0.3988** |
| IFEval | prompt-strict | **0.1238** |
| IFEval | inst-strict | **0.2278** |
| IFEval | prompt-loose | (see metrics.json) |
| IFEval | inst-loose | (see metrics.json) |

Run metadata: `Qwen/Qwen2.5-0.5B`, `dtype=bfloat16`, `backend=vllm`,
`config_hash=fde0720e`, [`configs/baseline.yaml`](https://github.com/agaonker/post-training-lab/blob/main/configs/baseline.yaml).

## How to read these

- **MMLU 0.4813** — pretrained `Qwen2.5-0.5B` actually slightly *outperforms*
  the `-Instruct` variant on MMLU (0.4732). Alignment can mildly compress raw
  recall. Sanity-checks that the harness is wired correctly. The bar for
  SFT/DPO is *don't lose this*.
- **GSM8K 0.3389 strict / 0.3419 flexible** — 0.5B is at the floor of where
  multi-step arithmetic starts to work. The small gap between strict and
  flexible says the model usually emits the answer in the expected format
  when it gets it right.
- **TruthfulQA-MC2 0.3988** — slightly below random (0.40 chance baseline
  given the multi-choice setup). The pretrained model is essentially at chance
  on this honesty/calibration probe. The `-Instruct` variant (0.4190) does
  better here — alignment buys some honesty calibration.
- **IFEval 0.1238 prompt-strict / 0.2278 inst-strict** — the headline Phase 1
  target. The pretrained model follows ~23% of *individual* instructions
  strictly but only 12% of *full prompts*. The gap to `-Instruct` (0.1885 /
  0.3070) is the headroom Qwen's own SFT recovered — and what the methods in
  this lab can be measured against.

## Phase 1 target (reframed for pretrained base)

From PROJECT.md §6: *"SFT model beats base on IFEval `prompt_level_strict_acc`
by a clear margin."*

| Metric | Pretrained base | Phase 1 target | `-Instruct` reference (ceiling) |
|--------|----------------:|---------------:|--------------------------------:|
| IFEval prompt-strict | 0.1238 | **> base** by a clear margin | 0.1885 (Qwen's own SFT) |
| IFEval inst-strict | 0.2278 | **> base** by a clear margin | 0.3070 |
| MMLU acc | 0.4813 | **≥ 0.47** | 0.4732 (alignment cost) |
| TruthfulQA-MC2 acc | 0.3988 | **≥ 0.38** | 0.4190 |
| GSM8K (either) | 0.3389 / 0.3419 | _no strong prior_ | 0.3404 / 0.3472 |

A successful SFT should ideally land near or above the `-Instruct` numbers —
that is, recover what Qwen's own supervised tuning achieved. Phase 1's
`sft_v2` came in **flat on IFEval prompt-strict** (0.1201). That's a Phase 1
finding, not a bug: see [`experiments/002`](https://github.com/agaonker/post-training-lab/blob/main/experiments/002_sft_qwen05b.md)
for the analysis and Phase 2's [`dpo_v1`](https://github.com/agaonker/post-training-lab/blob/main/experiments/003_dpo_qwen05b.md)
for the metric that did move.

## How to reproduce

```bash
make install
make eval-modal       # Modal L4 (vLLM/bf16) — produces the canonical row above
```

For a no-GPU sanity check that the wiring works (10 samples per task):

```bash
make eval-smoke
```

The smoke run writes to `results/metrics_smoke.json` rather than the canonical file.
