# Experiment 004 — Reward Model on Qwen2.5-0.5B (Phase 3A)

## TL;DR

Reward Model trained on UltraFeedback-binarized preference pairs from the
pretrained `Qwen/Qwen2.5-0.5B` base with a fresh regression head + QLoRA r=16.
Train-batch accuracy peaked at **0.725** and sustained 0.6–0.7 in the second
half of training, with margin (chosen − rejected) climbing from −0.027 (init)
to **+0.59** (final batch). Phase 3 success criterion (PROJECT.md §6:
≥0.65 on held-out pairs) is likely met on the train distribution; a held-out
eval is still pending. RM adapter pushed to `agaonker/atlas-rm-qwen05b-v1`.

## Method

Hyperparameters: [`configs/rm_qwen05b.yaml`](../configs/rm_qwen05b.yaml).
Training code: [`src/atlas/train/reward_model.py`](../src/atlas/train/reward_model.py).

Two recipe choices worth flagging:

- **No SFT warm-start.** Unlike DPO, the RM is a classifier — it doesn't need
  to be a good generator. Starting from the raw pretrained base + a fresh
  regression head is the canonical RM recipe (Stiennon et al. 2020 §3.3,
  InstructGPT). This is the *one* phase that doesn't anchor to `sft_v2`.
- **LoRA + 4-bit + `task_type="SEQ_CLS"`.** Same QLoRA setup as SFT, but the
  PEFT task type targets the classification head instead of the LM head. The
  base stays in 4-bit; Phase 3B (PPO) loads this adapter directly with no
  merging.

The pretrained-base / Instruct-tokenizer pattern from sft_v2/dpo_v1 carries
over: same tokenizer override (`cfg.model.tokenizer_name`), same patched
chat_template at load time (harmless for RM since there's no
`assistant_only_loss` mask, but consistency with the eval pipeline).

Total Modal compute: ~17 min training + Hub push. Cost ~$0.40.

## Training trajectory

| Step | loss | accuracy | margin |
|---:|---:|---:|---:|
| 10 (init) | 1.337 | 0.463 | −0.027 |
| 20 | 0.681 | 0.500 | +0.114 |
| 60 | 0.621 | 0.594 | +0.378 |
| 100 | 0.629 | **0.663** ← first crossing of 0.65 | +0.507 |
| 200 | 0.549 | 0.669 | +0.612 |
| 220 | 0.556 | **0.725** ← peak | +0.696 |
| 292 (end) | 0.649 (mean) | 0.625 (final batch) | +0.593 |

The reward scale settled into the 7–11 range (min 6.7, mean 9.4, max 11.0 on
the last batch). That's expected for a randomly-initialized regression head
finding a useful operating point — what matters for PPO is the *margin*
(separation between chosen and rejected), not the absolute scale.

## Outcome

**Train-batch accuracy: 0.6–0.72 throughout the second half of training.**
PROJECT.md §6's bar is ≥0.65 on *held-out* pairs — that eval isn't built yet.
The trajectory looks like it meets the bar, but until we score the RM on
`test_prefs` (UltraFeedback's held-out split, ~2k pairs), the success
criterion is formally unverified.

## What's still TBD

- **Held-out RM eval.** Add `src/atlas/eval/rm_accuracy.py` (or similar) that
  loads the RM adapter, scores `test_prefs` chosen/rejected, computes the %
  where chosen > rejected. Quick (~5 min on L4), small script. Closes the
  Phase 3 success criterion formally.
- **Reward calibration check.** Length / refusal / repetition guardrails
  (PROJECT.md §7) — does the RM disproportionately favor longer responses?
  Should be measured before PPO uses these rewards to update a policy. The
  classic RM failure mode is length bias.

## What this means for the project

Phase 3A delivers the **input** to Phase 3B (PPO). PPO loads:
- **Policy**: pretrained base + `sft_v2` adapter (Phase 1 anchor)
- **Reference**: same base + same `sft_v2` (frozen)
- **Reward model**: pretrained base + `rm_v1` adapter (this experiment)

The PPO loop samples rollouts from the policy, scores them with the RM, and
updates the policy with a KL penalty against the reference. Classic failure
modes (KL blowup, reward hacking, length bias) are well-documented; the
controlled comparison story turns on whether PPO can match what DPO already
achieved at far lower operational complexity.

## Anchor

- **RM adapter**: `agaonker/atlas-rm-qwen05b-v1`
- **Hub revision SHA**: TBD — pin once Phase 3B's PPO run locks in the policy
- **Pin in**: `configs/ppo_qwen05b.yaml` (`model.rm_adapter`)

## Engineering learnings

- **`AutoModelForSequenceClassification` + LoRA needs `task_type="SEQ_CLS"`**
  on the PEFT config. The `make_lora_config` default (`CAUSAL_LM`) targets
  the wrong head; without the override PEFT wires nothing useful onto the
  classifier.
- **`model.config.pad_token_id` must be set explicitly for SequenceClassification
  heads.** The head reads garbage from pad positions otherwise. The
  `_build_rm_model_and_tokenizer` helper does this after the tokenizer routing.
- **Grad-norm spikes (~25× the baseline 5) on hard preference pairs are normal
  early in RM training.** They don't propagate — the head finds a stable
  region within 2–3 steps. If they sustain across many steps, that's a
  signal for `max_grad_norm` clipping (TRL's default is 1.0, fine here).
- **RM trained in ~17 min on Modal L4** — faster than SFT (~25 min) and DPO
  (~35 min) because there's no ref forward and the loss is per-pair rather
  than per-token. The compute-cost story for full RLHF (Phase 3A + 3B vs
  Phase 2 DPO) starts to add up here.

## Next

- Phase 3B (PPO): `configs/ppo_qwen05b.yaml`, `src/atlas/train/ppo.py`,
  `src/atlas/cloud/ppo_modal.py`, mirroring the rm/dpo scaffolding pattern.
  The big code piece; KL/reward-hacking/length-bias are the lessons to expect.
- Held-out RM eval (`src/atlas/eval/rm_accuracy.py` or similar) — small,
  ~5 min on L4. Closes PROJECT.md §6 Phase 3 RM criterion formally.
