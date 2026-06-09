# post-training-lab

A controlled, reproducible comparison of modern LLM post-training methods — SFT, DPO, KTO/ORPO, full RLHF (reward model + PPO), and GRPO/RLVR — applied to the same small base model with the same evaluation harness.

**Goals**

1. *Learn*: hands-on with every major post-training technique.
2. *Update knowledge*: stay current with the 2024–2026 shift from PPO-RLHF → DPO → GRPO/RLVR.
3. *Showcase*: a Staff-level portfolio artifact — one repo, one writeup, one comparison table.

---

## 1. Scope and non-goals

**In scope**

- Small open base models (0.5B–3B parameters)
- Parameter-efficient fine-tuning (QLoRA primarily, full FT where it fits)
- Six training methods: SFT, DPO, KTO *or* ORPO (pick one), Reward Model + PPO, GRPO
- Standardized evaluation across all methods
- Per-phase writeups + a final comparison study

**Out of scope (deliberately)**

- Frontier-scale models (7B+); pipeline scales but the comparison stays at ≤3B
- Pretraining or continued pretraining
- Multimodal, vision, or speech
- Production serving (this is a research/study repo, not a product)
- Novel research contributions — this is a study of existing methods

---

## 2. Repository structure

```
post-training-lab/
  README.md                  # narrative + headline results table + writeup links
  PROJECT.md                 # this file (plan-of-record)
  pyproject.toml             # uv-managed, hatchling backend
  uv.lock                    # checked in for reproducible installs
  Makefile                   # make sft / make dpo / make eval / make compare
  .github/workflows/ci.yml   # lint + smoke test
  .gitignore
  .python-version            # pinned via uv

  src/atlas/
    __init__.py
    data/                    # dataset loading, prep, shared across experiments
      __init__.py
      sft_data.py
      preference_data.py
      verifiable_data.py     # GSM8K-style tasks with checkers
    models/
      base.py                # base model + tokenizer loaders
      adapters.py            # LoRA/QLoRA helpers, merge utilities
    train/
      sft.py
      dpo.py
      kto.py                 # or orpo.py — pick one
      reward_model.py
      ppo.py
      grpo.py
    eval/
      harness.py             # lm-eval-harness wrapper
      custom_evals.py        # task-specific evals
      judge.py               # LLM-as-judge pairwise win-rates
      compare.py             # produces results/metrics.json + plots
    utils/
      config.py              # YAML loading + validation (pydantic)
      logging_utils.py       # wandb + local logging
      checkpoint.py

  configs/                   # one YAML per experiment, version controlled
    base.yaml                # shared defaults
    sft_qwen05b.yaml
    dpo_qwen05b.yaml
    rm_qwen05b.yaml
    ppo_qwen05b.yaml
    grpo_qwen05b.yaml
    kto_qwen05b.yaml

  experiments/               # markdown log, one per training run
    001_baseline_eval.md
    002_sft_qwen05b.md
    ...

  notebooks/
    colab/                   # thin launchers that clone + install + run
      run_sft_colab.ipynb
    demo.ipynb               # final side-by-side completions demo
    scratch/                 # exploration, gitignored content

  writeups/                  # polished blog posts (markdown, publishable)
    01_sft_and_qlora.md
    02_dpo.md
    03_rlhf_ppo.md
    04_grpo_rlvr.md
    05_kto_or_orpo.md
    06_comparison.md

  results/
    metrics.json             # canonical comparison table
    plots/
    adapters/                # gitignored; HF Hub is source of truth
  
  tests/
    test_data.py
    test_eval.py
    test_smoke.py            # 50-step training smoke test for CI
```

**Source of truth rules**

- Code: `src/atlas/`. Importable as `atlas.train.sft`, etc.
- Experiment definition: `configs/*.yaml`. Every run reproducible from a YAML.
- Experiment log: `experiments/*.md`. Hypothesis, config hash, results, learnings.
- Trained artifacts: Hugging Face Hub. The repo only holds metrics + plots.
- Writeups: `writeups/*.md`. Linkable from the README and resume.

---

## 3. Tooling and environment

- **Python**: 3.11 (3.10 ok; 3.13 still has ML lib gaps as of early 2026)
- **Package/env manager**: `uv` + `pyproject.toml` + `uv.lock`
- **Build backend**: `hatchling`
- **Lint/format**: `ruff` (replaces black/isort/flake8)
- **Type-check**: `mypy` on `src/atlas/` (loose mode is fine)
- **Test**: `pytest` with a smoke test in CI
- **Experiment tracking**: Weights & Biases (free tier)
- **Secrets**: `.env` via `python-dotenv` (HF token, WANDB_API_KEY, OpenAI/Anthropic key for judge)

Core ML stack (versions are floors; let uv resolve):

- `torch>=2.4`, `transformers>=4.45`, `trl>=0.12`, `peft>=0.13`
- `datasets>=3.0`, `accelerate>=1.0`, `bitsandbytes>=0.44`
- `unsloth` (optional extra — speeds up SFT/DPO ~2x, finicky install)
- `lm-eval>=0.4` (eval harness)
- `vllm` (optional, for fast GRPO rollouts later)

**Colab/Kaggle bootstrap** (uv not preinstalled there):

```python
!git clone https://github.com/<you>/post-training-lab && cd post-training-lab && pip install -e .[eval]
```

---

## 4. Hugging Face datasets — usage plan

Hugging Face `datasets` is the spine of every phase. The pattern below stays consistent across all training scripts.

### 4.1 Datasets per phase

| Phase | Method | Dataset | Size used | Notes |
|---|---|---|---|---|
| 1 | SFT | `HuggingFaceH4/ultrachat_200k` or `teknium/OpenHermes-2.5` | 5k–10k samples | Start small for iteration speed |
| 2 | DPO | `HuggingFaceH4/ultrafeedback_binarized` | 5k preference pairs | Chosen/rejected pairs ready-made |
| 3 | Reward Model | Same as DPO | 5k pairs | Train RM on pairs, val on held-out |
| 3 | PPO | Prompts only from `HuggingFaceH4/ultrafeedback_binarized` | 2k prompts | RM scores rollouts; no need for labels |
| 4 | GRPO | `openai/gsm8k` | full train (~7.5k) | Verifiable: numeric answer checker |
| 5 | KTO | `HuggingFaceH4/ultrafeedback_binarized` (re-binarized) | 5k samples | Unpaired binary signal |
| 6 | Eval / judge | Held-out slice of UltraFeedback prompts (200) | 200 | LLM-as-judge win rates |

### 4.2 Standard loading pattern

```python
# src/atlas/data/preference_data.py
from datasets import load_dataset

def load_ultrafeedback(split="train_prefs", n_samples=5000, seed=42):
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return ds.map(
        lambda x: {
            "prompt": x["prompt"],
            "chosen": x["chosen"][-1]["content"],
            "rejected": x["rejected"][-1]["content"],
        },
        remove_columns=ds.column_names,
    )
```

### 4.3 Caching and offline use

- Set `HF_HOME=/path/to/cache` once (a 50GB volume on cloud, persistent dir locally). Datasets and model weights both land here.
- `datasets` caches to disk by default. Second load is instant. This matters when iterating on training scripts.
- For Colab: mount Google Drive and point `HF_HOME` there to survive runtime resets.
- For Modal/RunPod: use a persistent volume mounted at `/cache`; saves 5–15 min of redownload per cold start.

### 4.4 Authentication

- Create a HF account, generate a read+write token, store in `.env` as `HUGGING_FACE_HUB_TOKEN`.
- `huggingface-cli login` once locally; on cloud, set the env var.
- Push adapters (LoRA weights, ~10–100MB each) to your own HF Hub namespace, e.g. `<you>/atlas-sft-qwen05b-v1`. Repo stays small; results stay shareable.

### 4.5 Versioning and reproducibility

- Every YAML config pins: dataset name, split, `n_samples`, `seed`, and a `revision` (commit SHA on the dataset repo). HF datasets are versioned — pinning the revision avoids silent drift.
- Log the exact dataset SHA and row count to wandb at training start.

### 4.6 Data-leakage discipline

- Hold out a fixed 200-prompt slice of UltraFeedback (`seed=42`, last 200 after sorting by prompt hash) for the LLM-judge eval. Never train on these. Document in `experiments/000_data_splits.md`.
- GSM8K has its own test split — use it as-is for eval.

---

## 5. Compute and cloud strategy ($50–80 total budget)

The goal: do as much as possible on free tiers; spend money only where free tiers can't deliver. The $5 floor is enough to *start*, but realistically PPO and any 3B-scale work need a bit more. Plan for $50, cap at $80.

### 5.1 Compute tiers

| Tier | What you get | Cost | Use for |
|---|---|---|---|
| **Colab free** | T4 16GB, ~12hr sessions, can disconnect | $0 | Phase 0 setup, smoke tests, SFT/DPO on 0.5B |
| **Kaggle free** | P100 16GB or T4 x2, 30hr/week, more stable than Colab | $0 | Same as Colab; better for longer runs |
| **Colab Pro** | A100 40GB sometimes, longer sessions | $10/mo | Worth it for one month during PPO/GRPO phases |
| **Modal** | Pay-per-second, A10G $1.10/hr, A100 $3.40/hr | ~$1–3/hr | PPO, GRPO with vLLM rollouts, batch evals |
| **RunPod** | Spot A40 ~$0.40/hr, A100 ~$1.50/hr | Cheap | Long training runs if you tolerate preemption |
| **vast.ai** | Cheapest spot, RTX 4090 ~$0.30/hr | Cheapest | Same as RunPod, less polished UX |

### 5.2 Recommended mapping to phases

| Phase | Compute | Est. cost |
|---|---|---|
| 0 — Scaffolding | Local laptop + Colab free | $0 |
| 1 — SFT (Qwen2.5-0.5B, QLoRA) | Colab free / Kaggle free | $0 |
| 2 — DPO | Colab free / Kaggle free | $0 |
| 3 — Reward Model | Colab free | $0 |
| 3 — PPO | Modal A10G, ~3–5 hrs | $5–15 |
| 4 — GRPO (GSM8K) | Modal A10G, ~4–8 hrs | $5–20 |
| 5 — KTO/ORPO | Kaggle free | $0 |
| 6 — Comparison + LLM judge | Modal for batch inference + judge API calls | $5–15 |
| 7 — Stretch (3B scale) | Modal A100, ~2–4 hrs | $10–15 |

**Realistic total: $25–60.** $5 alone is not enough to comfortably do PPO + GRPO; if budget is truly capped at $5, drop Phase 3-PPO (the most expensive, also the one DPO replaced) and reallocate that $5 to GRPO, which is more interview-relevant.

### 5.3 Modal as the primary paid option (recommended)

Modal is the best pick for this project:

- Pay-per-second, no minimum, no idle cost
- Python-native: decorate a function, `modal run script.py`, done
- Easy GPU selection (`gpu="A10G"`)
- Persistent volumes for HF cache (huge time saver)
- Free $30 credit on signup (verify current offer) — likely covers Phase 3 entirely

Pattern for a Modal training entrypoint:

```python
# src/atlas/cloud/modal_train.py
import modal

app = modal.App("post-training-lab")
image = modal.Image.debian_slim().pip_install_from_pyproject("pyproject.toml")
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": vol},
    timeout=60 * 60 * 6,  # 6h
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb")],
)
def train(config_path: str):
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    from atlas.train.ppo import main
    main(config_path)
```

Then `modal run -m atlas.cloud.modal_train --config-path configs/ppo_qwen05b.yaml`.

The same code path runs locally (`python -m atlas.train.ppo --config configs/...`) and on Modal. No code duplication.

### 5.4 Cost-control discipline

- **Smoke-test locally / on Colab first.** Never launch a paid GPU on code you haven't run 50 steps of on free tier. Most expensive bugs are silent dataset/tokenization issues caught in the first 100 steps.
- **Cap wallclock in code.** Every paid run has `max_steps` set to a known-good value. No "let it run overnight."
- **Spot/preemptible by default** on RunPod and vast.ai. Checkpoint every 100–200 steps so preemption costs minutes, not hours.
- **One paid run per session.** Launch, watch the first 200 steps, walk away. Don't queue up speculative runs.
- **wandb alerts** on training divergence (KL blowup in PPO especially). Saves you from paying for runs that already failed.

### 5.5 What stays local

- All editing, config tweaking, eval analysis, plot generation, writeup drafting
- Inference on trained adapters (a 0.5B + LoRA runs on CPU at 5–10 tok/s; fine for spot checks)
- The LLM-judge step uses API calls to Claude/GPT — no GPU needed locally

---

## 6. Phase plan (10 weekends)

Each phase has: goal, deliverable, success criterion. Skip none; resist scope creep.

### Phase 0 — Scaffolding (1 weekend)
- **Goal**: repo, env, baseline eval working end-to-end.
- **Deliverable**: green CI; `results/metrics.json` populated with un-tuned `Qwen2.5-0.5B` (pretrained, not `-Instruct`) scores; README with the empty results table scaffolded.
- **Success**: `make eval-baseline` runs to completion on free Colab.

### Phase 1 — SFT + QLoRA (1 weekend)
- **Goal**: SFT pretrained `Qwen2.5-0.5B` on a 5–10k slice of UltraChat/OpenHermes.
- **Deliverable**: trained adapter pushed to HF Hub (planned: `agaonker/atlas-sft-qwen05b-v2`); `writeups/01_sft_and_qlora.md`; metrics row added.
- **Success**: SFT model beats the pretrained base on IFEval `prompt_level_strict_acc` by a clear margin. Pretrained-base reference (committed `d8365bd`): IFEval prompt-strict `0.1238`, inst-strict `0.2278`. A successful SFT should land above the corresponding `-Instruct` numbers (prompt-strict `0.1885`, inst-strict `0.3070`) — i.e. it should at least recover what Qwen's own Instruct tuning got out of supervised data.
- **Anchor for downstream phases**: this SFT adapter (and its Hub revision SHA) becomes the fixed starting point for Phases 2–5. Pin it in every downstream YAML. See TODOS.md.

### Phase 2 — DPO (1 weekend)
- **Goal**: DPO on the SFT checkpoint with UltraFeedback-binarized.
- **Deliverable**: adapter on Hub; `writeups/02_dpo.md`; metrics row.
- **Success**: DPO model has higher LLM-judge win rate vs SFT model on held-out prompts.

### Phase 3 — RLHF: Reward Model + RLOO (2 weekends)
- **Weekend A**: train reward model, achieve ≥65% accuracy on held-out preference pairs.
- **Weekend B**: **RLOO** on the SFT policy with the RM. Stable training, KL stays bounded.
- **Deliverable**: both adapters on Hub; `writeups/03_rlhf_rloo.md` including failure modes you hit.
- **Success**: RLOO model is competitive with DPO on win rate; writeup honestly compares operational complexity.
- **Note**: this phase was originally planned as PPO. TRL 1.4 deprecated `PPOTrainer`
  upstream — only `RLOOTrainer` and `GRPOTrainer` remain in the modern RLHF surface.
  This actually aligns with the field shift documented in §8 ("PPO-RLHF → DPO →
  GRPO/RLVR"). See `LESSONS.md` for the discovery.

### Phase 4 — GRPO + RLVR on GSM8K (1–2 weekends)
- **Goal**: GRPO on Qwen2.5-0.5B SFT using GSM8K with a numeric-answer verifier reward.
- **Deliverable**: adapter on Hub; `writeups/04_grpo_rlvr.md`; metrics row.
- **Success**: GRPO model substantially beats SFT and DPO on GSM8K. (Expected: this is GRPO's home turf.)

### Phase 5 — KTO or ORPO (1 weekend, pick one)
- **Goal**: train one of KTO or ORPO on the same preference data for direct comparison.
- **Deliverable**: adapter, writeup `05_*.md`, metrics row.
- **Success**: method runs cleanly and the comparison table has a fifth row.

### Phase 6 — Comparison study (1 weekend)
- **Goal**: the capstone. Synthesize everything.
- **Deliverable**:
  - `eval/compare.py` produces the final results table and plots
  - `writeups/06_comparison.md`: the headline writeup — when to use each method, what you'd reach for at a real job, what surprised you
  - LLM-judge head-to-head win rates on 200 held-out prompts across all methods
  - README updated with the polished results table and writeup links
- **Success**: a hiring manager can skim the README in 60 seconds and understand the project. The writeup answers "what would you do at scale?"

### Phase 7 — Stretch (optional)
Pick at most two of: RLAIF preference generation; Constitutional-AI-lite; process reward model; multi-task GRPO; rerun comparison at Qwen2.5-1.5B or 3B.

---

## 7. Evaluation: what counts as a result

Every method gets the same evaluation. Apples-to-apples is the whole point of the project.

**Standard benchmarks** (via `lm-eval-harness`):
- MMLU (5-shot, 1k-sample subset for speed)
- GSM8K (8-shot CoT)
- TruthfulQA (MC2)
- IFEval (instruction-following)

**Custom evals**:
- 200-prompt held-out slice from UltraFeedback, scored by Claude or GPT as judge in pairwise win-rates vs the SFT baseline
- Length, refusal rate, repetition rate as guardrails (catch reward hacking)

**Per-method extras**:
- PPO: KL divergence trajectory, reward over time, win rate vs SFT
- GRPO: GSM8K pass@1 and pass@8, format-reward vs answer-reward decomposition

**Output**: one JSON in `results/metrics.json`, one plot per benchmark in `results/plots/`, one summary table in the README.

---

## 8. Showcase and interview narrative

**Headline pitch** (1 sentence):
> A controlled comparison of post-training methods — SFT, DPO, PPO-RLHF, GRPO — on the same base model with the same evals; punch line: DPO matches PPO at a fraction of the complexity, and GRPO+verifiable rewards beats both on reasoning.

**Resume bullet**:
> Built `post-training-lab`: end-to-end implementations and a controlled benchmark of SFT, DPO, full RLHF (reward model + PPO), GRPO, and KTO on Qwen2.5, with a six-part technical writeup and apples-to-apples evaluation harness.

**Interview talking points to rehearse**:
1. Why the field moved from PPO-RLHF to DPO — and what DPO trades away
2. Why GRPO works for verifiable tasks and what its limits are
3. A specific failure mode you debugged in PPO (KL blowup, reward hacking, length bias — whichever you actually hit)
4. When you'd choose each method at a real company with real constraints
5. What you'd do differently with 100x compute

**Distribution**:
- Pin repo on GitHub profile
- Publish writeups on a personal site or Substack — each writeup is a linkable artifact
- One LinkedIn post per major phase ("just finished the GRPO writeup, here's what surprised me")

---

## 9. Foundational decisions (resolved)

Decisions that locked the comparison's structure. Items still open are tracked
as P-tagged TODOs in [TODOS.md](TODOS.md).

- **Repo name**: `post-training-lab`.
- **Base model**: pretrained `Qwen/Qwen2.5-0.5B` (decided 2026-06-06 after
  `sft_v1` on `-Instruct` regressed uniformly — there's nothing for SFT to do
  on already-aligned weights; see `LESSONS.md`). Tokenizer is borrowed from
  `-Instruct` at SFT time (`cfg.model.tokenizer_name`). Stretch: Qwen2.5-1.5B.

---

## 10. Anti-goals (things to actively avoid)

- Jupyter notebooks as the source of truth — they're for exploration only
- Chasing a moving SOTA — small models, small budgets, the *comparison* is the contribution
- Skipping the writeups — the writeups are the artifact, not the code
- Training on Modal before smoke-testing on Colab — burns money on silent bugs
- Doing Phase 7 stretch goals before Phase 6 is wrapped — finished beats fancy
