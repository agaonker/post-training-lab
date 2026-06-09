# LESSONS

Mistakes and gotchas found by actually running things. One-liner each;
expand only when a future reader can't recover the rule from the codebase.
Group by category. Date each entry so stale ones are easy to retire.

---

## Base model

- **Don't SFT an already-Instruct model and call it the comparison's "base".**
  Re-SFT-ing aligned weights has nothing to do, regresses uniformly across evals
  (sft_v1 vs Qwen2.5-0.5B-Instruct: −1 to −2.6pp on every metric). For a
  methods comparison, start from the pretrained variant so each post-training
  method has room to actually move. (2026-06-06)

- **Pretrained Qwen2.5-0.5B has `pad == eos == <|endoftext|>` (id 151643).**
  TRL masks `pad_token_id` in labels, so the model never sees `eos` in
  supervision → it never learns to stop. Solved by loading the `-Instruct`
  tokenizer (same vocab; `pad=<|endoftext|>`, `eos=<|im_end|>` 151645). See
  `src/atlas/utils/config.py:ModelCfg.tokenizer_name`. (2026-06-06)

- **Pretrained and `-Instruct` Qwen2.5 share a byte-identical vocab.**
  Verified via `sha256(get_vocab())` — safe to mix model from base repo with
  tokenizer from `-Instruct` repo. Don't assume this for non-Qwen models. (2026-06-06)

## Tokenizer / chat template

- **Qwen2.5's `chat_template` lacks `{% generation %}` markers — both base and
  `-Instruct`.** Without them TRL's `assistant_only_loss=True` (and
  `return_assistant_tokens_mask=True`) silently return a mask of all zeros. We
  inject markers via `src/atlas/models/base.py:patch_chat_template_for_assistant_mask`.
  Be wary when adopting a new model: a missing marker is silent. (2026-06-06)

- **Don't grep `"generation"` to test whether `{% generation %}` is present.**
  Qwen's template contains the word inside `{%- if add_generation_prompt %}`,
  which is unrelated. The real check is exact substring match for
  `{% generation %}` or `{%- generation`. (2026-06-06)

- **The audit dump pattern (`--dump-template-audit` flag in `src/atlas/train/sft.py`)
  catches mask bugs locally for $0 before Modal spend.** It renders the first N
  rows of the dataset, dumps `assistant_mask` byte-for-byte alongside decoded
  tokens. If the mask is empty or off-by-one, you see it in the jsonl. (2026-06-06)

## Training framework (TRL 1.4)

- **`SFTConfig.assistant_only_loss` defaults to `False`.** Without setting it
  in YAML, loss is computed over user/system turns too — the 0.5B spends
  capacity learning to generate user questions. Always set
  `assistant_only_loss: true` for multi-turn SFT. (2026-06-06)

- **`trainer.save_model()` may or may not persist the tokenizer depending on
  TRL version.** Always call `tokenizer.save_pretrained(output_dir)` explicitly
  after `save_model()` — relevant when the tokenizer was modified at load time
  (patched template, eos override). See `src/atlas/train/sft.py`. (2026-06-06)

- **TRL's `processing_class` auto-save behavior has drifted across versions.**
  Don't rely on side effects you can't see in `git diff` — make persistence
  explicit. (2026-06-06)

## Eval / Modal

- **`make eval-modal-check` is the cheap escape hatch.** Local preflight,
  no GPU, validates config loads + harness imports + Modal app registration.
  Catches a config_hash drift or a broken refactor before $$ flow to L4. (2026-06-06)

- **`metrics.json` has no name-uniqueness constraint.** Two rows with
  `name="base"` are allowed; `config_hash` is what disambiguates. Rename rows
  explicitly when the experiment family changes, or readers downstream will
  guess. (2026-06-06)

- **vLLM `gpu_memory_utilization=0.9` OOMs the lm-eval loglikelihood spike on
  L4 22GB.** The fp32 `log_softmax(vocab)` spike during MMLU/TruthfulQA needs
  headroom. Cap at `0.6`. See [configs/base.yaml](configs/base.yaml). (Already
  in CLAUDE.md; restated here so the lesson lives in one place.) (2026-05-23)

## HF Hub / secrets

- **An HF token that worked for `sft_v1` may not work for `sft_v2`.** A re-minted
  fine-grained token loses Write scope unless explicitly re-granted. The Modal
  `hf-token` secret stores whatever you set at creation time — it doesn't
  auto-rotate. If the secret stales out, the fail-fast preflight in
  `src/atlas/train/sft.py:_preflight_hub_access` rejects in ~200ms (`whoami`
  passes, `create_repo` returns 403). Refresh:
  `modal secret create hf-token HUGGING_FACE_HUB_TOKEN=hf_NEW --force`. (2026-06-06)

- **The fail-fast preflight saved a full SFT run.** Without
  `_preflight_hub_access`, the 403 would have surfaced only after `trainer.train()`
  completed — ~45 min and ~$1 of L4 wasted. The preflight is ~200ms. The cost of
  the extra two API calls (`whoami` + `create_repo(exist_ok=True)`) is rounding
  error against the cost of one wasted training run. Generalize this pattern
  before every paid run that *ends* with a Hub push. (2026-06-06)

## TRL surface drift

- **TRL 1.4 removed `PPOTrainer` / `PPOConfig`.** The modern RLHF surface in
  TRL is `SFTTrainer` + `DPOTrainer` + `RewardTrainer` + `RLOOTrainer` +
  `GRPOTrainer` + `KTOTrainer`. PPO is gone. This matches the field shift
  PROJECT.md §8 describes ("PPO-RLHF → DPO → GRPO/RLVR"). For "RLHF with a
  reward model" use `RLOOTrainer` (REINFORCE Leave-One-Out) — it's the
  natural replacement: same RM-scored rollouts + KL penalty, simpler than
  PPO (no value head, no separate critic). (2026-06-08)

- **`RLOOTrainer` takes `reward_funcs=[rm_model]`, not `reward_model`.** Plural,
  list of callables or models. Each can be an `AutoModelForSequenceClassification`
  (a real reward model), a string path, or a Python callable. Mismatching this
  signature against the docs (which sometimes still reference PPO) wastes a
  smoke run. (2026-06-08)

## wandb / training callbacks

- **A missing `WANDB_API_KEY` on a dev machine crashes `trainer.train()` mid-init.**
  HF Trainer calls wandb's `on_train_begin` callback when `report_to: wandb` is
  set; wandb then refuses to start without auth. The fix is to set
  `WANDB_MODE=disabled` (or unset `report_to`). The `sft-smoke` Makefile target
  now does this; `src/atlas/cloud/sft_modal.py` bakes it into the image env so
  the Modal container doesn't need a wandb secret to run. Re-enable by creating
  a `wandb` Modal secret and dropping the override. (2026-06-06)

## Process / tooling

- **`make fmt` reformats every file in scope, not just the ones you edited.**
  Ruff-format collateral on unrelated files will sneak into your commit. Revert
  the unrelated files (`git checkout --`) before `git add`. (2026-06-06)

- **Tests that hardcode model names couple your test suite to one experiment.**
  When the base swapped, 6 tests broke on the model string. Either centralize
  the model name (e.g. a `TEST_MODEL` constant) or assert on `cfg.model.name`
  not literals. (2026-06-06)

- **A wrong lead hypothesis in an `experiments/*.md` writeup is expensive.**
  `experiments/002` lead with "chat-template drift" — chasing that would have
  meant rewriting the data loader and tokenizer setup. The real bug was loss
  masking. Audit hypotheses against the code path before publishing. (2026-06-06)

- **Don't widen scope mid-fix.** A 1-line YAML change (`assistant_only_loss: true`)
  grew into a 7-file diff because the chat_template needed patching first.
  That's fine *if* each step is gated and verified locally — but it requires
  the audit-dump pattern to keep the scope honest. (2026-06-06)

---

Add new lessons as one-liner + 1-2 lines of context + date. Group above.
Strike entries when they no longer apply (the underlying knob changed,
upstream fixed it, etc.) but keep the strike in place — the history is the
point.
