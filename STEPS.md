# Runbook — complete the base + SFT comparison

**Goal:** a clean, apples-to-apples two-row `results/metrics.json` (base vs SFT
adapter), evaluated on one backend — vLLM/bf16 on a Modal L4.

**Key facts:** the SFT adapter `agaonker/atlas-sft-qwen05b-v1` already exists on
the Hub → **no retraining**. The `base` row has been re-evaluated on vLLM/bf16 and
**committed** (`config_hash 6af9a640`, commit `bc931c1`) → it is the fixed
reference. We do **not** re-eval base; we only run SFT now and **append** it as a
second row (`make eval-modal-sft`), so the committed base numbers are preserved.

**How to run:** one command at a time, in order. In Claude Code, prefix with `!`
to run in-session so output is captured. Each step lists what it does, the
expected outcome, and the gate before moving on.

---

## Stage A — validate the eval infra (cheapest first; STOP on any failure)

### Step A1 — local preflight  ·  `$0`  ·  DONE ✓
```
make eval-modal-check
```
- Does: imports/entrypoints/config check, no GPU, no container.
- Expect: `preflight passed`, `config_hash=6af9a640`, entrypoints `{main, probe, suite}`.
- Status: ✓ passed.

### Step A2 — CI parity (tests + lint + types)  ·  `$0`  ·  DONE ✓
```
make test-fast
make lint
make typecheck
```
- Does: validates the `gpu_memory_utilization == 0.6` change + no regressions.
- Expect: `63 passed`, ruff clean, mypy clean.
- Status: ✓ all passed.

### Step A3 — vLLM path probe  ·  Modal L4, ~2 min, a few cents
```
make eval-modal-probe
```
- Does: boots L4, loads model, runs gsm8k @ limit 5 — proves the whole vLLM path.
- Expect: a `probe` row appended; gsm8k exact_match ≈ 0.4; no errors.
- Gate: must succeed before any bigger run.

### Step A4 — base OOM canary (OPTIONAL)  ·  Modal L4, ~10-15 min
```
make eval-modal-smoke
```
- Does: base only, all 4 tasks @ limit 50. Slow because MMLU fans out to 57 subtasks.
- Expect: completes all 4 tasks with **no CUDA OOM** (the 0.9→0.6 fix).
- Skippable: Step B2's suite re-tests OOM at a larger limit anyway.

---

## Stage B — produce the clean two-row comparison

### Step B1 — adapter de-risk  ·  Modal L4, ~5 min
```
modal run src/atlas/cloud/eval_modal.py::main --name sft_v1 --method sft --adapter agaonker/atlas-sft-qwen05b-v1 --limit 5
```
- Does: proves the LoRA adapter **downloads + applies under vLLM** before the full SFT run.
- Expect: a throwaway `sft_v1` row (dropped in B2's first line); no adapter/download errors.
- Note: MUST start with `modal run` (NOT `modal app stop run`).
- Gate: must succeed before the full SFT run (B2).

### Step B2 — append the full SFT row  ·  Modal L4, ~15-30 min, `< $1`
```
git checkout -- results/metrics.json   # drop B1's throwaway sft_v1 row → committed base-only file
make eval-modal-sft
```
- Does: `git checkout` restores `metrics.json` to the **committed base row**
  (discarding B1's throwaway `sft_v1 --limit 5` row; a no-op if you skipped B1).
  Then `make eval-modal-sft` evals `sft_v1` (the adapter) on vLLM/bf16 across the
  full task list (mmlu@5-shot/limit-1000, gsm8k@8-shot, truthfulqa_mc2, ifeval)
  and **appends** it. The HF-cache volume is already warm from the base run, so
  only the LoRA adapter downloads.
- Expect: exactly two rows in `results/metrics.json` — `base` (preserved,
  `config_hash 6af9a640`) + `sft_v1` (new), identical `config_hash` + `model_args`
  except `adapter`.
- Note: this does **not** wipe. The committed base row is the fixed reference.
  `make eval-modal` (`suite --fresh`) stays available if you ever want a full clean
  re-baseline of *both* rows from scratch — but that re-evals base.

---

## Stage C — verify the result (read-only)

### Step C1 — inspect the table
```
cat results/metrics.json
```
- Expect: two rows; identical `config_hash` + `model_args` except `adapter`
  (null for base, the Hub id for sft_v1) → apples-to-apples confirmed.

### Step C2 — check the Phase 1 success criterion
- PROJECT.md §6: SFT beats base on **IFEval** by a clear margin — specifically
  `ifeval/prompt_level_strict_acc,none` and `ifeval/inst_level_strict_acc,none`
  higher for sft_v1 than base.
- Some MMLU/GSM8K dip from a 5k UltraChat SFT is acceptable; IFEval is the target.
- If SFT does NOT beat base on IFEval → stop, treat as a real finding (adapter
  quality / chat-template / eval wiring), don't paper over it.

---

## Stage D — land the artifacts (done in-session by Claude)

### Step D1 — git hygiene
```
git status
```
- Confirm only intended files changed: the eval-vllm-modal edits + regenerated `metrics.json`.

### Step D2 — README results table
- Fill `README.md` lines 24-27: replace the `_base_` placeholder with real base
  numbers and add an `SFT` row from `metrics.json`. Leave "Judge win-rate" as `—`
  (Phase 6).

### Step D3 — experiments log
- Add `experiments/002_sft_qwen05b.md`: hypothesis (SFT > base on IFEval), the
  eval `config_hash`, the adapter id, the two-row results, learnings (the OOM
  fix, the fp16→vLLM/bf16 re-baseline).

### Step D4 — commit
```
git add -A && git commit
```
- Commit on branch `eval-vllm-modal`. (Push/PR only if explicitly asked.)
- Optional: fix the stale "training not yet implemented" line in `CLAUDE.md`.

---

## Fast path (skip the optional smoke)
A1 → A2 (done) → **A3 → B1 → B2 → C1**, then Claude does C2 + Stage D.

## Gotchas
- `modal run <file>::<entrypoint>` launches a run. `modal app stop <APP_ID>`
  kills a running app — different subcommand, takes an app ID, not `--name`.
- A cancelled run shows "exit code 1" locally — that's the cancel, not a bug.
- Stop a runaway GPU run: `modal app list`, then `modal app stop <APP_ID>`.
