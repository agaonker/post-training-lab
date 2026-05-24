# Experiment 002 — SFT on Qwen2.5-0.5B-Instruct (Phase 1)

## Hypothesis

Supervised fine-tuning on UltraChat-200k (5 000 steps, LoRA) will improve
instruction-following (IFEval) while preserving MMLU/GSM8K within acceptable
bounds.

## Setup

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter:** `agaonker/atlas-sft-qwen05b-v1`
  - Hub revision SHA: `1586ae5ef98fb525452da54fbc2b3c6f04e19fbd` ← pinned anchor for Phases 2-5
- **Config hash:** `6af9a640` (both base and sft_v1 — identical config, only `adapter` differs)
- **Eval backend:** vLLM 0.21.0, bfloat16, Modal L4
- **Tasks:** MMLU (5-shot, limit 1000), GSM8K (8-shot, default), TruthfulQA MC2, IFEval

## Results

| Metric | base | sft_v1 | delta |
|--------|------|--------|-------|
| `mmlu/acc,none` | 0.4732 | 0.4595 | −0.0137 |
| `gsm8k/exact_match,strict-match` | 0.3404 | 0.3207 | −0.0197 |
| `truthfulqa_mc2/acc,none` | 0.4190 | 0.4073 | −0.0117 |
| `ifeval/prompt_level_strict_acc,none` | 0.1885 | 0.1719 | **−0.0166** |
| `ifeval/inst_level_strict_acc,none` | 0.3070 | 0.2986 | −0.0084 |
| `ifeval/prompt_level_loose_acc,none` | 0.2163 | 0.1941 | −0.0222 |
| `ifeval/inst_level_loose_acc,none` | 0.3369 | 0.3177 | −0.0192 |

## Outcome

**Phase 1 criterion: FAIL.** The success criterion was `sft_v1 > base` on
IFEval prompt-level strict accuracy. SFT regressed on every metric.

## Analysis

Three candidate causes (not mutually exclusive):

1. **Chat-template drift (most likely).** UltraChat-200k training likely used a
   different system/user/assistant template than the one IFEval's prompts assume.
   If the fine-tuning hard-baked a slightly different template, the eval prompts
   land in an off-distribution region at inference time.

2. **Task-distribution mismatch.** 5k steps of open-ended UltraChat conversation
   may have shifted the model's prior toward verbose multi-turn responses, away
   from the short, tightly-formatted outputs IFEval rewards.

3. **Under-training.** 5k steps on a 0.5B model may be insufficient to shift
   IFEval meaningfully — the adapter may not have converged on the instruction
   pattern.

## Learnings / engineering notes

- **OOM fix:** `gpu_memory_utilization` dropped 0.9 → 0.6. The lm-eval
  loglikelihood path spikes to fp32 for log_softmax(vocab), which exhausted the
  22 GiB L4 at 0.9. At 0.6 this headroom avoids the spike.
- **fp16 → vLLM/bf16 re-baseline:** an earlier eval used HF/fp16; the committed
  base row (`bc931c1`) is the clean vLLM/bf16 reference.
- **Append workflow:** `make eval-modal-sft` uses `main` (appends), not `suite
  --fresh` (wipes), so the committed base row is preserved.

## Next steps

- Inspect chat template used during SFT training vs eval prompt format.
- Try `make eval-modal-sft` after re-training with correct Qwen chat template
  enforcement (Phase 2 DPO may inherit this bug — fix it first).
