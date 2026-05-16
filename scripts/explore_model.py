"""Module 0 — exploration: meet the base model.

A cumulative script across Module 0. Each step adds one inspection:
  Step 1 — load Qwen2.5-0.5B-Instruct and generate one response (greedy).
  Step 2 — print the prompt as token ids — see special tokens and sub-word splits.
  Step 3 — try harder prompts; note where 0.5B struggles.
  Step 4 — peek at the next-token probability distribution; vary temperature.

Run:  uv run python scripts/explore_model.py
First run downloads the model (~1 GB) into the Hugging Face cache (~/.cache/huggingface).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def generate_response(model, tokenizer, user_message: str, max_new_tokens: int = 256) -> str:
    """One user-message → assistant-response cycle. Greedy, deterministic."""
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    # --- load ----------------------------------------------------------------
    # tokenizer: converts text <-> integer token ids.
    # model:     the network that, given tokens, predicts the next one.
    # float32 on CPU is the simplest, most reliable path on a Mac. (4-bit QLoRA
    # needs bitsandbytes, which is CUDA-only — that arrives later, on Colab.)
    print(f"Loading {MODEL_NAME} ... (first run downloads ~1 GB)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()  # inference mode — disables dropout, no gradient tracking

    # --- build the prompt ----------------------------------------------------
    # This is the *Instruct* variant: Qwen already fine-tuned it to follow a chat
    # format. `apply_chat_template` wraps our message in the special tokens the
    # model was trained to expect. Feeding raw text instead would work far worse.
    messages = [
        {"role": "user", "content": "In two sentences, what is supervised fine-tuning?"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,             # return a string, so we can see what we built
        add_generation_prompt=True,  # append the marker that says "assistant, your turn"
    )
    print("\n--- formatted prompt sent to the model ---")
    print(prompt)

    # --- tokenize ------------------------------------------------------------
    # The model only sees integers. Encode the prompt and look at the result:
    # a flat list of token ids. The chat-format markers like <|im_start|> are
    # *single* special tokens, not the characters that spell them. Regular
    # words may be one token or split into several sub-word pieces.
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()
    special_ids = set(tokenizer.all_special_ids)

    print(f"\n--- tokenization ({len(token_ids)} tokens) ---")
    print(f"special tokens this tokenizer knows: {tokenizer.special_tokens_map}")
    print("\n  idx  token_id  decoded                       special?")
    for i, tid in enumerate(token_ids):
        piece = repr(tokenizer.decode([tid]))
        mark = " ✓" if tid in special_ids else ""
        print(f"  {i:3d}  {tid:>8d}  {piece:<28s}{mark}")

    # --- generate ------------------------------------------------------------
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,  # greedy decoding: deterministic, take the top token each step
        )

    # generate() returns prompt tokens + new tokens; slice off the prompt so we
    # print only what the model actually wrote.
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("\n--- model response ---")
    print(response)

    # --- Step 3: probe where the base model is weak --------------------------
    # The point is qualitative: get a feel for what 0.5B *can't* do reliably, so
    # you know what each post-training method later is aiming to fix.
    probes = [
        ("easy general knowledge — baseline (should work)",
         "What is the capital of France? Answer in one sentence."),
        ("multi-step math word problem — GRPO's target (Module 5)",
         "A baker makes 36 cookies. She gives 1/4 to her neighbor and eats 3 "
         "herself. Her friend then doubles what's left. How many cookies does "
         "she have now? Show your work."),
        ("multi-constraint instruction following — SFT/DPO's target (Modules 1, 3)",
         "List exactly three planets in our solar system. Use bullet points. "
         "After the list, add ONE sentence in formal English."),
    ]
    print("\n\n========== Step 3: probing the base model ==========")
    for label, prompt_text in probes:
        print(f"\n--- prompt: {label} ---")
        print(prompt_text)
        print("--- response ---")
        print(generate_response(model, tokenizer, prompt_text))

    # --- Step 4: peek at the next-token distribution -------------------------
    # Greedy decoding hides what the model *considered*. One forward pass gives
    # logits at every position; we take the LAST position — the score vector
    # the model would softmax to pick the very next token after our original
    # prompt. This is the same object DPO compares via log-prob ratios and the
    # one PPO's KL term measures distance from. Worth seeing concretely once.
    print("\n\n========== Step 4: next-token distribution ==========")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]  # shape: [vocab_size]

    # Temperature reshapes the SAME logits before softmax:
    #   T < 1   sharpens — peakier, more deterministic
    #   T = 1   the model's own distribution
    #   T > 1   flattens — more diversity, more nonsense
    for T in (0.1, 1.0, 2.0):
        probs = torch.softmax(logits / T, dim=-1)
        top = torch.topk(probs, k=8)
        print(f"\n--- top-8 at temperature {T} ---")
        print("  rank  token_id  prob     decoded")
        for rank, (p, tid) in enumerate(
            zip(top.values.tolist(), top.indices.tolist(), strict=True), 1
        ):
            print(f"  {rank:>4d}  {tid:>8d}  {p:.4f}   {tokenizer.decode([tid])!r}")


if __name__ == "__main__":
    main()
