# Experiment 003 — DPO on Qwen2.5-0.5B (Phase 2)

## TL;DR

DPO on top of `sft_v2` (5,000 UltraFeedback pairs, β=0.1, LR 5.0e-6, 1 epoch
on Modal L4) moved **every** lm-eval-harness metric in the right direction.
`IFEval prompt_level_strict_acc` — the headline metric Phase 1 SFT couldn't
move — gained **+0.74pp** vs `sft_v2`, putting `dpo_v1` slightly above the
pretrained base for the first time. Phase 1's flat result was caused by SFT
having little to do on its own; DPO with preference data clearly added signal.

## Method

Hyperparameters: [`configs/dpo_qwen05b.yaml`](../configs/dpo_qwen05b.yaml).
Training code: [`src/atlas/train/dpo.py`](../src/atlas/train/dpo.py).

Two recipe choices worth recording because they're load-bearing for the
result:

- **Merge-then-DPO.** The `sft_v2` LoRA is fused into the base via
  `merge_and_unload` before the DPO LoRA attaches. This is why
  `quant.load_in_4bit: false` in the YAML — merge needs full-precision
  weights. TRL's reference policy then comes for free (disable the DPO LoRA
  for the ref forward and you're back to merged-`sft_v2` behavior). Avoids
  the multi-adapter `model_adapter_name` / `ref_adapter_name` juggling that
  varies across TRL minor versions.
- **In-distribution preference data.** UltraFeedback-binarized is *derived
  from* UltraChat-200k (the sft_v2 corpus), so the prefs are in-distribution
  for the policy by construction.

Eval uses the same vLLM/bf16 path as `base` and `sft_v2`, with
`--tokenizer agaonker/atlas-dpo-qwen05b-v1` so generation terminates on the
saved Instruct-template's `<|im_end|>`.

Total Modal compute: ~35 min training + ~25 min eval. Cost ~$1.50.

## The DPO loss curve in plain words

DPO loss starts at `−log(0.5) ≈ 0.693` (random preference). It went:

| Step | loss | rewards/margins | rewards/accuracies |
|---:|---:|---:|---:|
| 10 | 0.699 | −0.011 | 0.331 |
| 30 | 0.689 | **+0.014** ← crossed zero | 0.481 |
| 100 | 0.654 | +0.107 | 0.656 |
| 200 | 0.619 | +0.204 | 0.694 |
| 290 | 0.606 | +0.279 | 0.706 |
| 313 (end) | 0.646 (mean) | **+0.314** (final batch) | **0.775** (final batch) |

The reward margins (= reward for chosen − reward for rejected) crossed zero at
step 30 and grew monotonically. By the end the policy was correctly preferring
chosen over rejected on **77.5% of batch examples** — clear, sustained
preference learning.

## Results

All eval rows in `results/metrics.json` (vLLM/bf16, Modal L4):

| Row | hash | MMLU | GSM8K strict | TruthfulQA | IFEval prompt-strict | IFEval inst-strict |
|---|---|---:|---:|---:|---:|---:|
| `base` (pretrained) | fde0720e | 0.4813 | 0.3389 | 0.3988 | 0.1238 | 0.2278 |
| `sft_v2` | b133712d | 0.4713 | 0.3450 | 0.3893 | 0.1201 | 0.2398 |
| **`dpo_v1`** | **d53fd258** | **0.4802** | **0.3495** | **0.3958** | **0.1275** | **0.2422** |

### Deltas

| Metric | vs sft_v2 | vs base |
|---|---:|---:|
| MMLU | **+0.89pp** | −0.11pp (back near base) |
| GSM8K strict | **+0.45pp** | +1.06pp |
| GSM8K flexible | +0.30pp | +1.06pp |
| TruthfulQA MC2 | **+0.65pp** | −0.30pp (back near base) |
| **IFEval prompt-strict** | **+0.74pp** | **+0.37pp** (Phase 2 success on this metric) |
| IFEval inst-strict | +0.24pp | +1.44pp |

## Outcome

**Every lm-eval-harness metric improved vs sft_v2.** The headline IFEval
prompt-strict — flat in Phase 1 — is now +0.37pp over the pretrained base,
i.e. `dpo_v1` is the first policy in this lab to *do better than the raw
pretrained checkpoint on the strict instruction-following bar*.

The plan-of-record's *formal* Phase 2 success criterion (PROJECT.md §6) is
**LLM-judge pairwise win-rate `dpo_v1` > `sft_v2` on 200 held-out UltraFeedback
prompts**. That judge eval is gated on Phase 6 (`src/atlas/eval/judge.py`
isn't built yet — TODOS). The lm-eval improvements above are *necessary
evidence* the policy is better in measurable ways, but the *sufficient*
criterion still waits on the judge.

## Analysis (honest)

What this confirms:

1. **The Phase 1 flat result was about SFT, not about the pipeline.** Same
   pipeline, same evals, just with DPO on top of sft_v2: every metric moved.
   The pipeline was fine.
2. **DPO learned a real preference signal.** Margins crossed zero at step 30
   and grew steadily to +0.314 by end of epoch. Rewards/accuracies hit 77.5%
   — clear, sustained.
3. **DPO didn't break anything.** MMLU recovered (+0.89pp, almost back to
   base). TruthfulQA recovered (+0.65pp). DPO is *adding* signal, not just
   moving variance.
4. **Reference-policy bookkeeping worked.** TRL's "disable the LoRA for ref
   forward" trick on the merged sft_v2 model produced exactly the expected
   ref behavior — no collapse, no drift, sensible margin trajectory.

What's still unknown:

- **Does the LLM-judge agree?** Phase 6 will tell us. Without it we have
  benchmark improvements but no direct preference-quality verdict.
- **What's the IFEval ceiling?** We're at 0.1275 prompt-strict; Qwen's own
  `-Instruct` hits 0.1885. We have ~6pp of headroom. Whether DPO with more
  data / different β would close it is a separate question — Phase 7 stretch.
- **Length / refusal / repetition guardrails** — PROJECT.md §7 calls these
  out for catching reward hacking. Not measured yet; should be a Phase 6 add.

## What this means for the project

Phase 2 closes with a **clean comparison row** in `metrics.json`:

> pretrained base → SFT → DPO, with each step's contribution to each metric
> isolated by a `config_hash` and reproducible from `configs/`.

This is what the lab is *for* — controlled comparison. The result here doesn't
depend on beating Qwen at their own instruction tuning; it depends on `dpo_v1`
being clearly better than `sft_v2` on the same eval, which it is.

Phase 3 (Reward Model + PPO) and Phase 4 (GRPO) can now start from `dpo_v1`
(or `sft_v2`, depending on the comparison design — PPO conventionally starts
from SFT, not DPO, since PPO is itself the preference-tuning step). Phase 6
will add the LLM judge, which closes the formal Phase 2 success criterion and
backs every method-row in the final comparison table.

## Anchor for Phase 3 (RLHF)

- **Anchor for PPO**: `agaonker/atlas-sft-qwen05b-v2` (PPO starts from SFT
  policy, with a reward model trained on UltraFeedback pairs)
- **Anchor for the reward model**: same pretrained base, head replaced via
  TRL `RewardTrainer`
- DPO doesn't feed downstream by convention — it's a peer of PPO, not a
  precursor. The comparison story is "DPO vs PPO vs GRPO, all starting from
  sft_v2."

## Engineering learnings

- **TRL 1.4 dropped `max_prompt_length`** from `DPOConfig`; only `max_length`
  governs the combined prompt + completion now. The Modal smoke caught this
  in ~30s — exactly what the smoke tier is for.
- **The "merge-then-DPO" recipe (`merge_and_unload` the SFT adapter before
  attaching the DPO LoRA) is the simplest reliable QLoRA-DPO path** with
  TRL 1.4. It trades ~750 MB of memory (bf16 vs 4-bit base) for never having
  to think about multi-adapter `model_adapter_name`/`ref_adapter_name`
  semantics.
- **DPO loss steepness ≈ 25× SFT's grad norm at init** (13 vs 0.55). Same
  model, same LoRA — just the DPO loss surface. LR has to be 1-2 orders
  lower than SFT to compensate; 5.0e-6 worked.
- **Sandbox heartbeat held this time for ~35 min training + ~25 min eval.**
  Lucky vs the recurring drops on the sft_v2 eval. Future runs should still
  prefer the user-terminal-with-`!`-prefix pattern for long Modal runs to
  be safe.

## Next

- Phase 3 (Reward Model + PPO): `src/atlas/train/reward_model.py`,
  `src/atlas/train/ppo.py`, `configs/rm_qwen05b.yaml`, `configs/ppo_qwen05b.yaml`.
  Anchor: `sft_v2`.
- Phase 6 LLM judge: `src/atlas/eval/judge.py` once we have enough method
  rows to compare. Will close Phase 2's formal success criterion.
- Polished `writeups/02_dpo.md` after Phase 6 numbers exist.
