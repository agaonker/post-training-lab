# Runbook — finish the SFT eval (base is done)

**Goal:** a clean, apples-to-apples two-row `results/metrics.json` (`base` vs
`sft_v1`), evaluated on one backend — vLLM/bf16 on a Modal L4. The base half is
done; this runbook finishes the SFT half and lands the artifacts.

---

## Progress so far (committed + pushed on `eval-vllm-modal`)

- **Base eval — DONE.** `base` row committed (`bc931c1`), `config_hash 6af9a640`,
  vLLM/bf16 on L4, full 4-task list. This is the **fixed reference**; we do not
  re-run it.
- **Append-SFT tooling — DONE** (`58d7286`): the `make eval-modal-sft` target
  (appends, no wipe), this runbook, `scripts/eval_modal_check.py` preflight, the
  scripted `probe`/`main`/`suite` entrypoints, and the
  `gpu_memory_utilization 0.9 → 0.6` OOM fix.
- **Branch is even with `origin/eval-vllm-modal`** (2 commits pushed).
- **Adapter** `agaonker/atlas-sft-qwen05b-v1` already exists on the Hub → **no
  retraining**.

**Remaining:** run the SFT eval (B) → verify (C) → land docs + PR (D).

```
results/metrics.json — state machine for the work that's left
──────────────────────────────────────────────────────────────────────────
[base]                       committed @ bc931c1, pushed         ← WE ARE HERE
   │
   │  B1  modal run ...::main --name sft_v1 --method sft --adapter <id> --limit 5
   ▼
[base, sft_v1(limit5)]       throwaway de-risk row (proves adapter loads)
   │
   │  B2.1  git checkout -- results/metrics.json   (base is committed → safe restore)
   ▼
[base]                       back to the committed reference
   │
   │  B2.2  make eval-modal-sft                     (full task list, appends)
   ▼
[base, sft_v1]               the clean two-row comparison        ← GOAL
```

**How to run:** one command at a time, in order. In Claude Code, prefix with `!`
to run in-session so output is captured.

---

## Stage A — infra validation  ·  DONE / now optional

The committed base run already exercised the full vLLM path end-to-end on this
exact Modal image, including the OOM headroom fix. So the infra is proven; these
steps are kept for reference and as a fallback if Modal misbehaves.

- **A1 — local preflight** (`make eval-modal-check`)  ·  DONE ✓ — `preflight
  passed`, `config_hash=6af9a640`, entrypoints `{main, probe, suite}`.
- **A2 — CI parity** (`make test-fast` / `lint` / `typecheck`)  ·  DONE ✓ —
  `63 passed`, ruff + mypy clean.
- **A3 — vLLM path probe** (`make eval-modal-probe`)  ·  OPTIONAL — the base run
  already proved this path. Run only if you suspect the Modal image drifted.
- **A4 — base OOM canary** (`make eval-modal-smoke`)  ·  OPTIONAL — the base run
  already completed all 4 tasks with no OOM at full scale.

---

## Stage B — run the SFT eval (the real next action)

### Step B1 — adapter de-risk  ·  Modal L4, ~5 min
```
modal run src/atlas/cloud/eval_modal.py::main --name sft_v1 --method sft --adapter agaonker/atlas-sft-qwen05b-v1 --limit 5
```
- Does: the one path the base run did NOT exercise — `snapshot_download` of the
  LoRA adapter + vLLM `lora_local_path` load. Fail-fast in 5 min before the
  15-30 min full run (catches a private/gated repo or a bad HF token cheaply).
- Expect: a throwaway `sft_v1` row appended; no adapter/download errors.
- Note: MUST start with `modal run` (NOT `modal app stop run`).
- Gate: must succeed before B2.

### Step B2 — append the full SFT row  ·  Modal L4, ~15-30 min, `< $1`
```
git checkout -- results/metrics.json   # drop B1's throwaway row → back to committed base-only
make eval-modal-sft
```
- Why the `git checkout` is safe: the `base` row is committed at `bc931c1`, so
  this restores `metrics.json` to exactly `[base]` and discards only B1's
  throwaway `sft_v1 --limit 5` row. It's a no-op if you skipped B1.
- Then `make eval-modal-sft` evals `sft_v1` on vLLM/bf16 across the full task
  list (mmlu@5-shot/limit-1000, gsm8k@8-shot, truthfulqa_mc2, ifeval) and
  **appends** it. The HF-cache Volume is already warm from the base run, so only
  the LoRA adapter downloads.
- Expect: exactly two rows — `base` (preserved, `6af9a640`) + `sft_v1` (new),
  identical `config_hash` + `model_args` except `adapter`.
- Note: `make eval-modal` (`suite --fresh`) stays available for a full clean
  re-baseline of *both* rows — but that re-evals base, which we don't want here.

---

## Stage C — verify the result (read-only)

### Step C1 — inspect the table
```
cat results/metrics.json
```
- Expect: two rows; identical `config_hash` + `model_args` except `adapter`
  (null for base, the Hub id for sft_v1) → apples-to-apples confirmed.

### Step C2 — check the Phase 1 success criterion (PROJECT.md §6)
SFT must beat base on **IFEval** by a clear margin. The base bar to beat:

| IFEval metric                      | base (`6af9a640`) | sft_v1 target |
|------------------------------------|-------------------|---------------|
| `prompt_level_strict_acc,none`     | 0.1885            | **> base**    |
| `inst_level_strict_acc,none`       | 0.3070            | **> base**    |
| `prompt_level_loose_acc,none`      | 0.2163            | (watch)       |
| `inst_level_loose_acc,none`        | 0.3369            | (watch)       |

- Some MMLU/GSM8K dip from a 5k UltraChat SFT is acceptable; IFEval is the target.
  (base MMLU 0.4732, GSM8K strict 0.3404, TruthfulQA 0.4190 — for context.)
- If SFT does NOT beat base on IFEval → stop, treat it as a real finding (adapter
  quality / chat-template / eval wiring), don't paper over it.

---

## Stage D — land the artifacts (done in-session by Claude)

### Step D1 — git hygiene
```
git status
```
- Confirm only intended files changed: the regenerated `metrics.json` + the doc
  edits below.

### Step D2 — README results table
- In `README.md`'s `## Results` table, replace the `_base_` placeholder row with
  the real base numbers and add an `sft_v1` row from `metrics.json`. Match on the
  table content, not line numbers (they drift). Leave "Judge win-rate vs SFT" as
  `—` (Phase 6).

### Step D3 — experiments log
- Create `experiments/002_sft_qwen05b.md` (the `experiments/` dir doesn't exist
  yet): hypothesis (SFT > base on IFEval), the eval `config_hash 6af9a640`, the
  adapter id **and its Hub revision SHA** (pin it — TODOS.md P1: this adapter is
  the anchor for Phases 2-5), the two-row results, and learnings (the OOM fix,
  the fp16→vLLM/bf16 re-baseline).

### Step D4 — commit
```
git status        # inspect first; stage only the metrics + docs
git add results/metrics.json README.md experiments/002_sft_qwen05b.md
git commit
```
- Commit on `eval-vllm-modal`. Optional: fix the stale "training not yet
  implemented" line in `CLAUDE.md` in the same commit.

### Step D5 — push + PR  (only when you say so)
```
! git push origin eval-vllm-modal
gh pr create --base main
```
- The push needs `!` (in-session) — the sandbox blocks it from a tool call.
- Open the PR against `main` once the two-row result + docs are in.

---

## Fast path
A1 + A2 (done) → **B1 → B2 → C1 → C2** → Claude does Stage D.

## Gotchas
- `modal run <file>::<entrypoint>` launches a run. `modal app stop <APP_ID>`
  kills a running app — different subcommand, takes an app ID, not `--name`.
- A cancelled run shows "exit code 1" locally — that's the cancel, not a bug.
- Stop a runaway GPU run: `modal app list`, then `modal app stop <APP_ID>`.
- `gpu_memory_utilization` (0.6) only affects OOM headroom, not the metric values
  — so base (run earlier) and sft_v1 stay comparable regardless.
