"""Local chat-template + tokenizer sanity audit.

Runs three checks against a base model (and optionally an attached PEFT
adapter), entirely on CPU/MPS — no Modal, no quantization. Designed to catch
the failure modes diagnosed in writeups/sft_regression_diagnosis.html before
spending Modal compute on a retrain.

Scope is honest: this script only exercises generation tasks (gsm8k / ifeval
shape). MMLU and TruthfulQA MC2 are loglikelihood (no generation), so eos /
chat-template issues affecting them won't show here. The full eval still has
to run on Modal — this is the cheap pre-check.

Checks
------
1. Hygiene: greedy-generate a short reply to a one-shot user message and
   inspect the tail for unterminated turns, run-on garbage, or visible
   special-token leakage (``<|im_start|>`` appearing inside the reply, no
   ``<|im_end|>`` at the end, etc.).
2. Parity: if --adapter is given, render the same prompt with and without the
   adapter and report whether outputs diverge meaningfully. SKIP otherwise.
3. Tokens: print the tokenizer's eos / pad / bos / chat_template length and
   verify pad != eos (Qwen2.5-0.5B pretrained ships pad == eos, which is the
   SFT-supervision footgun this audit catches).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Write a haiku about debugging."


def _device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _load(model_name: str, adapter: str | None) -> tuple[Any, Any]:
    device, dtype = _device_and_dtype()
    # device_map={"": 0} matches the repo's deliberate single-card pin for CUDA;
    # on Mac/CPU we let transformers place the model implicitly.
    model_kwargs: dict[str, Any] = {"dtype": dtype}
    if device == "cuda":
        model_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device != "cuda":
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(adapter or model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model, tokenizer


def _generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 96) -> str:
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, return_tensors="pt").to(model.device)
    # do_sample=False is greedy; temperature is intentionally not passed (it
    # would warn under do_sample=False and be ignored either way).
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0][ids.input_ids.shape[1] :], skip_special_tokens=False)


def _check_hygiene(reply: str) -> tuple[str, list[str]]:
    """Return (status, notes). status ∈ {CLEAN, DIRTY}."""
    notes: list[str] = []
    tail = reply.rstrip()
    # Real failure: trailing garbage that isn't terminated by im_end / endoftext.
    if not (tail.endswith("<|im_end|>") or tail.endswith("<|endoftext|>")):
        notes.append(
            f"tail not terminated by <|im_end|> or <|endoftext|> (last 60 chars: {tail[-60:]!r})"
        )
    # Real failure: an extra <|im_start|> snuck into the assistant reply,
    # indicating the model is hallucinating new turns.
    if "<|im_start|>" in reply:
        notes.append("found <|im_start|> inside assistant reply (run-on turn)")
    return ("CLEAN" if not notes else "DIRTY"), notes


def _check_parity(model_name: str, adapter: str) -> tuple[str, list[str]]:
    """Compare base vs base+adapter on the same prompt."""
    base_model, tokenizer = _load(model_name, adapter=None)
    base_reply = _generate(base_model, tokenizer, PROMPT)
    del base_model

    adapter_model, tok2 = _load(model_name, adapter=adapter)
    adapter_reply = _generate(adapter_model, tok2, PROMPT)
    diverged = base_reply.strip() != adapter_reply.strip()
    notes = [
        f"base reply head: {base_reply.strip()[:120]!r}",
        f"adapter reply head: {adapter_reply.strip()[:120]!r}",
    ]
    return ("DIVERGED" if diverged else "IDENTICAL"), notes


def _check_tokens(tokenizer: Any) -> tuple[str, list[str]]:
    notes = [
        f"eos: {tokenizer.eos_token!r} ({tokenizer.eos_token_id})",
        f"pad: {tokenizer.pad_token!r} ({tokenizer.pad_token_id})",
        f"bos: {tokenizer.bos_token!r}",
        f"chat_template length: {len(tokenizer.chat_template or '')} chars",
    ]
    has_gen = bool(tokenizer.chat_template) and (
        "{% generation %}" in tokenizer.chat_template or "{%- generation" in tokenizer.chat_template
    )
    notes.append(f"chat_template has {{% generation %}} markers: {has_gen}")
    pad_eos_collision = tokenizer.pad_token_id == tokenizer.eos_token_id
    notes.append(f"pad == eos: {pad_eos_collision} (True is the SFT footgun)")
    status = "FAIL" if pad_eos_collision else "PASS"
    return status, notes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local chat-template + tokenizer audit (no Modal needed)."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B",
        help="Base model repo id (default: Qwen/Qwen2.5-0.5B).",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional PEFT adapter repo id. Enables the parity check.",
    )
    args = parser.parse_args()

    print(f"=== audit: model={args.model} adapter={args.adapter} ===\n")

    model, tokenizer = _load(args.model, args.adapter)

    print("[Check 3] Tokens / template")
    token_status, token_notes = _check_tokens(tokenizer)
    print(f"  status: {token_status}")
    for n in token_notes:
        print(f"    - {n}")
    print()

    print("[Check 1] Hygiene (greedy generation tail)")
    reply = _generate(model, tokenizer, PROMPT)
    print(f"  reply head: {reply.strip()[:200]!r}")
    hygiene_status, hygiene_notes = _check_hygiene(reply)
    print(f"  status: {hygiene_status}")
    for n in hygiene_notes:
        print(f"    - {n}")
    print()

    print("[Check 2] Adapter parity")
    if args.adapter is None:
        print("  status: SKIP (no --adapter passed)")
    else:
        del model
        parity_status, parity_notes = _check_parity(args.model, args.adapter)
        print(f"  status: {parity_status}")
        for n in parity_notes:
            print(f"    - {n}")

    return 0 if hygiene_status == "CLEAN" and token_status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
