# post-training-lab

A controlled, reproducible comparison of modern LLM post-training methods — SFT, DPO, KTO,
full RLHF (reward model + PPO), and GRPO/RLVR — applied to the same small base model
(`Qwen/Qwen2.5-0.5B`, pretrained — *not* the `-Instruct` variant) with the same evaluation
harness.

📖 **Docs:** [agaonker.github.io/post-training-lab](https://agaonker.github.io/post-training-lab/) — method explainers, paper links, and hyperparameter reasoning.

See [`PROJECT.md`](PROJECT.md) for the full charter and phase plan.

> **Status:** Phase 2 (DPO) closed. Phase 3 (Reward Model + PPO) next. Three methods rows in
> the comparison table below; each phase adds one. The README is updated as each phase lands.

## Quickstart

```bash
uv sync --extra dev --extra eval
```

## Results

All runs: vLLM/bf16, Modal L4. IFEval = `prompt_level_strict_acc,none`. GSM8K =
`exact_match,strict-match`. Each row carries its own `config_hash` — same fingerprint
means apples-to-apples; different fingerprints mean the model / tokenizer / adapter
config differs (deliberately).

| Method | config_hash | MMLU | GSM8K | TruthfulQA | IFEval (prompt strict) | Judge win-rate vs SFT |
|---|---|---:|---:|---:|---:|---:|
| **base** (pretrained Qwen2.5-0.5B) | `fde0720e` | 0.4813 | 0.3389 | 0.3988 | 0.1238 | — |
| **sft_v2** (UltraChat-200k, 5k, QLoRA r=16, 1 epoch) | `b133712d` | 0.4713 | 0.3450 | 0.3893 | 0.1201 | — |
| **dpo_v1** (UltraFeedback, 5k pairs, β=0.1, 1 epoch) | `d53fd258` | **0.4802** | **0.3495** | **0.3958** | **0.1275** | *Phase 6* |

### Deltas

| Metric | sft_v2 − base | dpo_v1 − sft_v2 | dpo_v1 − base |
|---|---:|---:|---:|
| MMLU | −1.00pp | **+0.89pp** | −0.11pp |
| GSM8K strict | +0.61pp | **+0.45pp** | +1.06pp |
| TruthfulQA | −0.95pp | **+0.65pp** | −0.30pp |
| **IFEval prompt-strict** | −0.37pp (flat) | **+0.74pp** | **+0.37pp** |
| IFEval inst-strict | +1.20pp | +0.24pp | +1.44pp |

**The headline finding so far:** SFT on a 0.5B with 5k UltraChat rows was *flat* on the
hardest IFEval metric. DPO on top of that SFT moved every lm-eval metric in the right
direction and put `dpo_v1` slightly above the pretrained base on IFEval prompt-strict —
the first policy in this lab to clear that bar. The Phase 2 *formal* success criterion
(LLM-judge pairwise win-rate) is gated on Phase 6 and not yet measured. See
[`experiments/003_dpo_qwen05b.md`](experiments/003_dpo_qwen05b.md).

## Mistakes, dead-ends, and learnings

This lab is partly *about* getting these right, so they're documented as they're caught:

- **First SFT attempt was on `Qwen2.5-0.5B-Instruct`** — already chat-tuned by Qwen.
  Re-SFT-ing already-aligned weights regressed uniformly on every eval (−1 to −2.6pp).
  Diagnosed in
  [`writeups/sft_regression_diagnosis.html`](writeups/sft_regression_diagnosis.html);
  pipeline rebuilt on pretrained `Qwen2.5-0.5B`. The historical `-Instruct` rows from
  that attempt (`agaonker/atlas-sft-qwen05b-v1`) still live in
  [`results/metrics.json`](results/metrics.json) for reference but are not the
  active comparison.
- **`assistant_only_loss` defaults to `False`** in TRL 1.4 — without setting it, loss
  was computed over user turns too. The 0.5B was learning to generate user questions.
- **Qwen2.5's chat template doesn't ship `{% generation %}` markers** — required for
  TRL's assistant-only mask. Patched at load time in
  [`src/atlas/models/base.py`](src/atlas/models/base.py) via
  `patch_chat_template_for_assistant_mask`.
- **Pretrained `Qwen2.5-0.5B` has `pad == eos == <|endoftext|>`** — classic SFT
  supervision footgun. Solved by loading the `-Instruct` tokenizer (same vocab; pad
  and eos correctly distinct).
- **An HF token that worked for `sft_v1` didn't work for `sft_v2`** — Modal secret had
  a read-only token. The fail-fast preflight in
  [`src/atlas/train/sft.py:_preflight_hub_access`](src/atlas/train/sft.py) caught it
  in ~200ms instead of after ~45 min of training.
- **`max_prompt_length` was removed from `DPOConfig` in TRL 1.4** — only `max_length`
  governs the combined sequence now. The Modal smoke caught this in ~30s.
- **`merge_and_unload` the SFT adapter into the base before attaching the DPO LoRA**
  is the simplest reliable QLoRA-DPO recipe for TRL 1.4 — trades ~750 MB memory for
  never having to think about multi-adapter `model_adapter_name`/`ref_adapter_name`.
- **Long-running `modal run` from the sandboxed shell dropped the local heartbeat
  multiple times** mid-eval (~30 min in). User-terminal launches via `! modal run …`
  were reliable. Future long runs should prefer that.

Full one-liner-per-finding list with dates and why-it-matters lines:
[`LESSONS.md`](LESSONS.md). Live punch list of things to fix: [`TODOS.md`](TODOS.md).

## Writeups

One polished writeup per phase is the final deliverable per [`PROJECT.md`](PROJECT.md) §6;
those land at the end of each phase and live in [`writeups/`](writeups/). In flight:

- [`writeups/sft_regression_diagnosis.html`](writeups/sft_regression_diagnosis.html) —
  full diagnosis of the `sft_v1` regression that drove the base swap.
- Phase 1 polished writeup `writeups/01_sft_and_qlora.md` — pending.
- Phase 2 polished writeup `writeups/02_dpo.md` — pending Phase 6 LLM-judge numbers.

Per-experiment logs (hypothesis → config_hash → results → learnings) live in
[`experiments/`](experiments/):

- [`experiments/002_sft_qwen05b.md`](experiments/002_sft_qwen05b.md) — Phase 1 SFT.
- [`experiments/003_dpo_qwen05b.md`](experiments/003_dpo_qwen05b.md) — Phase 2 DPO.
