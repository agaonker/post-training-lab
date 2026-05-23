"""Local preflight for the Modal eval — seconds, no GPU, no `modal run`, $0.

Validates the wiring that does NOT need a container, so the cheap bug classes never cost a
GPU run: the file imports, the config loads, the local result-append path imports, and all
three local entrypoints register. (Most of this session's failures were exactly these:
entrypoint ambiguity and the `_append_local` import — both catchable here in seconds.)

Run with the SAME interpreter as the `modal` CLI (NOT `uv run`): the local entrypoints
execute in modal's Python, which isn't this project's .venv. The src/ bootstrap below mirrors
what eval_modal.py does, so `atlas` imports regardless of where it's pip-installed.

    make eval-modal-check        # or:  python3 scripts/eval_modal_check.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # make `atlas` importable like eval_modal.py does


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")
    sys.exit(1)


def main() -> None:
    print("eval-modal preflight (local, no GPU):")

    # 1. Config loads + canonical fingerprint (catches YAML/schema typos).
    from atlas.utils.config import load_config

    cfg = load_config("configs/baseline.yaml")
    print(f"  ok    config loads             config_hash={cfg.config_hash}")

    # 2. The local result-append path imports — this is the import that broke _append_local
    #    after a full 10-min remote eval had already succeeded.
    from atlas.eval.harness import SCHEMA_VERSION, append_run  # noqa: F401

    print(f"  ok    atlas.eval.harness imports (local append path)  schema=v{SCHEMA_VERSION}")

    # 3. The modal module imports and every local entrypoint registers (catches the
    #    "Specify a Modal Function" ambiguity and any decorator/typo regressions).
    try:
        import modal  # noqa: F401
    except ModuleNotFoundError:
        _fail(
            "`modal` not importable by this interpreter. Install it, or run this with the "
            "same Python as the `modal` CLI (this check must NOT use `uv run`)."
        )

    path = REPO / "src/atlas/cloud/eval_modal.py"
    spec = importlib.util.spec_from_file_location("eval_modal_preflight", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    entrypoints = set(mod.app.registered_entrypoints)
    expected = {"main", "suite", "probe"}
    if not expected <= entrypoints:
        _fail(f"missing entrypoints {sorted(expected - entrypoints)} (found {sorted(entrypoints)})")
    if "run_eval_remote" not in mod.app.registered_functions:
        _fail("GPU function `run_eval_remote` not registered")
    print(f"  ok    modal app registers      entrypoints={sorted(entrypoints)}")

    print("\npreflight passed — wiring is sound; safe to spend a GPU run.")


if __name__ == "__main__":
    main()
