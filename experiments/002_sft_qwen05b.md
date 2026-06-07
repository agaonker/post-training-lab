# Experiment 002 — SFT on Qwen2.5-0.5B (Phase 1)

## TL;DR

Phase 1 ran twice. `sft_v1` (on `-Instruct`) regressed uniformly; we diagnosed
two structural causes, switched the base to pretrained `Qwen2.5-0.5B`, fixed
the pipeline, and trained `sft_v2`. The Phase 1 success criterion (IFEval
`prompt_level_strict_acc` clearly above base) is **not met** — `sft_v2` is
essentially flat on that headline metric (−0.37pp vs base). Several other
metrics moved positive (GSM8K, IFEval inst-strict), the pipeline is now
structurally correct, and `sft_v2` is a clean anchor for Phase 2 DPO.

## The journey

### Attempt 1 — `sft_v1` on `Qwen2.5-0.5B-Instruct`

- **Adapter:** `agaonker/atlas-sft-qwen05b-v1` (HF Hub, kept as historical record)
- **Result:** uniform regression on every eval (−1 to −2.6pp).
- **Diagnosis:** `writeups/sft_regression_diagnosis.html`. Two structural causes:
  1. `cfg.train.assistant_only_loss` was never set; TRL 1.4 defaults to `False`,
     so loss was computed over the whole rendered conversation including user
     turns. The 0.5B spent capacity learning to generate user questions.
  2. Re-SFT-ing already-Instruct weights had nothing for SFT to do. The base
     was already aligned by Qwen — additional alignment on top mildly forgets.
- **Reframe:** Phase 1's controlled comparison needs a *pretrained* base so each
  post-training method has room to actually move. Switched
  `configs/base.yaml`'s `model.name` to `Qwen/Qwen2.5-0.5B`.

### Pipeline fixes (committed `8c21e04`)

- `cfg.train.assistant_only_loss: true` — masks loss to assistant turns only.
- `cfg.model.tokenizer_name: Qwen/Qwen2.5-0.5B-Instruct` — same byte-identical
  vocab as the pretrained base but with correct `eos=<|im_end|>` and the
  Instruct chat_template (different default system message; otherwise the
  branches needed for the patcher are present).
- New `patch_chat_template_for_assistant_mask` in `src/atlas/models/base.py`:
  Qwen's chat_template doesn't ship `{% generation %}` markers (verified via
  exact substring match — the unrelated `add_generation_prompt` variable led
  the original audit astray). The patcher splits the combined `user / system /
  assistant-no-tools` branch in two and injects markers around assistant
  content. Without this, `assistant_only_loss=True` silently produces a mask
  of all zeros.
- `--dump-template-audit` in `src/atlas/train/sft.py`: byte-level confirmation
  the mask flips at `<|im_start|>assistant\n` boundaries before any Modal spend.
  Wrote `results/sft_template_audit.jsonl` (debug artifact, gitignored).
- New `scripts/audit_chat_template.py`: local hygiene + token + parity check.

### Attempt 2 — `sft_v2` on pretrained `Qwen2.5-0.5B`

- **Adapter:** `agaonker/atlas-sft-qwen05b-v2`
- **Dataset:** UltraChat-200k `train_sft`, 5,000-row slice (`seed=42`)
- **Trainer:** TRL `SFTTrainer` + LoRA r=16 + 4-bit QLoRA on Modal L4
- **Steps:** 313 micro-batches × (batch=4 × grad_accum=4) = 5,008 samples = 1 epoch
- **Training loss:** 1.479 → 1.382 (clean, no NaN, grad_norm bounded)
- **Mean assistant-token accuracy:** 0.6296 → 0.6538 (the mask is real and learning)
- **Config hash:** `b133712d` (different from base hash `fde0720e` because
  `cfg.model.tokenizer_name` is set for sft_v2 eval — same fingerprint
  semantics, just different tokenizer source)

## Results

All eval rows in `results/metrics.json` (vLLM/bf16, Modal L4):

| Row | Model | Adapter | hash | MMLU | GSM8K strict | TruthfulQA | IFEval prompt-strict | IFEval inst-strict |
|---|---|---|---|---:|---:|---:|---:|---:|
| `base` (historical) | -Instruct | — | 6af9a640 | 0.4732 | 0.3404 | 0.4190 | 0.1885 | 0.3070 |
| `sft_v1` (historical) | -Instruct | qwen05b-v1 | 6af9a640 | 0.4595 | 0.3207 | 0.4073 | 0.1719 | 0.2986 |
| `base` (canonical) | **pretrained** | — | fde0720e | 0.4813 | 0.3389 | 0.3988 | 0.1238 | 0.2278 |
| **`sft_v2`** | **pretrained** | **qwen05b-v2** | **b133712d** | **0.4713** | **0.3450** | **0.3893** | **0.1201** | **0.2398** |

### Deltas vs the canonical pretrained base

| Metric | pretrained base | sft_v2 | Δ | vs Qwen's own -Instruct |
|---|---:|---:|---:|---:|
| MMLU | 0.4813 | 0.4713 | **−1.00pp** | 0.4732 (Qwen ≈ ours +0.19pp) |
| GSM8K strict | 0.3389 | 0.3450 | **+0.61pp** | 0.3404 (we edge it) |
| GSM8K flexible | 0.3419 | 0.3495 | **+0.76pp** | 0.3472 (we edge it) |
| TruthfulQA MC2 | 0.3988 | 0.3893 | **−0.95pp** | 0.4190 (Qwen ahead) |
| **IFEval prompt-strict** | **0.1238** | **0.1201** | **−0.37pp** | **0.1885** (Qwen +6.84pp) |
| IFEval inst-strict | 0.2278 | 0.2398 | **+1.20pp** | 0.3070 (Qwen +6.72pp) |

## Outcome

**Phase 1 success criterion: NOT MET.** PROJECT.md §6 sets the bar at "SFT model
beats the pretrained base on IFEval `prompt_level_strict_acc` by a clear
margin." sft_v2 is −0.37pp — flat within noise.

## Analysis (honest)

The pipeline is **structurally correct**, evidenced by:
- Training loss decreased monotonically with healthy grad_norm
- Mean assistant-token accuracy ticked up — the mask trained the right tokens
- Multiple downstream metrics moved positive (GSM8K both filters, IFEval
  inst-strict)
- Generation terminates cleanly on `<|im_end|>` (the tokenizer override worked)

**What didn't work:** the headline IFEval prompt-strict metric. Plausible causes,
none of which are bugs:

1. **Data scale.** 5,000 rows is the smallest Phase 1 starting point (PROJECT.md
   §4.1). Qwen's own -Instruct tuning used vastly more. At 64% of Qwen's
   prompt-strict score (0.1201 vs 0.1885), there's clear data-budget headroom.
2. **prompt-strict vs inst-strict.** prompt-strict requires *every* instruction
   in a prompt to be followed. inst-strict counts individual instructions. The
   fact that inst-strict moved +1.20pp while prompt-strict stayed flat means
   sft_v2 is *better* at following instructions on average — just not better
   enough to clear the harder all-or-nothing bar.
3. **Recipe.** LoRA r=16, lr=2e-4, 1 epoch. These are the defaults from
   `configs/sft_qwen05b.yaml`. We haven't searched them.

## What this means for the project

The lab's contribution is the **controlled comparison**, not beating Qwen at
their own game. We now have a clean SFT anchor (`sft_v2`) and an honest result.
Phase 2 DPO will train on top of `sft_v2`; the comparison question becomes:
**can DPO move IFEval prompt-strict more than SFT did?** That's exactly the
Phase 2 success criterion in PROJECT.md §6.

If we want a stronger Phase 1 result later, `sft_v3` on 20k+ rows or a tuned
recipe would be the move — but doing that *before* Phase 2 conflates two
variables (data scale and method). Keep sft_v2 as the fixed anchor; revisit
sft scale once the methods comparison is complete (Phase 7 stretch goal).

## Anchor for Phases 2-5

Per TODOS.md (P1): downstream phases pin this adapter as the start point.

- **Anchor:** `agaonker/atlas-sft-qwen05b-v2`
- **Hub revision SHA:** TBD — pin once we're sure we won't retrain
- **Pin it in:** every `configs/{dpo,rm,ppo,grpo,kto}_qwen05b.yaml`'s
  `model.name` + `model.revision`, plus
  `model.tokenizer_name = Qwen/Qwen2.5-0.5B-Instruct` (the SFT was trained
  with this; downstream methods inherit).

## Engineering learnings

- **OOM fix:** `gpu_memory_utilization` 0.9 → 0.6. lm-eval's loglikelihood
  path spikes to fp32 for `log_softmax(vocab)`, exhausting 22 GiB L4 at 0.9.
- **Token-scope footgun:** Modal `hf-token` secret with read-only scope
  preflight-fails in ~200ms — exactly what fail-fast is for. See
  `_preflight_hub_access` and the `LESSONS.md` entries.
- **wandb auth crash:** `WANDB_MODE=disabled` in `make sft-smoke` and the
  Modal SFT image avoids a `trainer.train()` crash on `on_train_begin` when
  `WANDB_API_KEY` isn't set.
- **Append-only metrics:** `results/metrics.json` carries all four rows
  (historical -Instruct + canonical pretrained) so the comparison story is
  reproducible. `config_hash` discriminates apparent name collisions.
- **Modal CLI heartbeat:** long-running `modal run` from a sandboxed shell can
  drop the local heartbeat after ~30 min, killing the run. Local terminal on a
  stable network is the reliable path; the eval was finally landed by running
  via `! modal run ...` from the user's Mac shell.

## Next

- Phase 2 plan: `configs/dpo_qwen05b.yaml`, `src/atlas/train/dpo.py`,
  `src/atlas/cloud/dpo_modal.py` (mirroring `sft_modal.py`). DPO on
  UltraFeedback-binarized, anchored to `sft_v2`.
- Polished writeup: `writeups/01_sft_and_qlora.md` — once Phase 2 is in flight
  and we have something to contrast SFT against.
