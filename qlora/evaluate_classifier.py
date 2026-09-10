"""
Load the trained QLoRA adapter and test its actual classification accuracy
on real examples - not just loss, but does it predict the right category.

Usage:
    modal run qlora/evaluate_classifier.py
"""

import json
from pathlib import Path

import modal

app = modal.App("runbook-copilot-qlora-eval")

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
CATEGORIES = ["active-directory", "virtualization", "networking", "storage", "backup"]

volume = modal.Volume.from_name("runbook-copilot-adapter", create_if_missing=False)
ADAPTER_DIR = "/adapter_output"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.44.0",
        "peft>=0.12.0",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.33.0",
    )
)


def build_prompt(text: str) -> str:
    categories_str = ", ".join(CATEGORIES)
    return (
        f"Classify the following IT infrastructure incident into exactly one "
        f"of these categories: {categories_str}.\n\n"
        f"Incident: {text}\n\n"
        f"Category:"
    )


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={ADAPTER_DIR: volume},
)
def evaluate():
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    hf_token = os.environ["HF_TOKEN"]

    # Load the EXACT held-out eval set saved during training - not a
    # re-derived split, so we're guaranteed to be testing on examples the
    # model never saw during training.
    with open(f"{ADAPTER_DIR}/eval_split.json", "r", encoding="utf-8") as f:
        test_examples = json.load(f)
    print(f"Loaded {len(test_examples)} held-out examples from training's saved split")

    print(f"Loading base model + adapter from {ADAPTER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", token=hf_token
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    results = []
    correct = 0
    for ex in test_examples:
        prompt = build_prompt(ex["text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()

        # Match the predicted text against known category strings
        predicted = None
        for cat in CATEGORIES:
            if cat in generated:
                predicted = cat
                break

        is_correct = predicted == ex["label"]
        correct += int(is_correct)
        results.append({
            "text": ex["text"][:80],
            "true_label": ex["label"],
            "predicted_raw": generated,
            "predicted_label": predicted,
            "correct": is_correct,
        })
        print(f"{'✓' if is_correct else '✗'} true={ex['label']:20s} pred={str(predicted):20s} raw='{generated}'")

    accuracy = correct / len(test_examples) if test_examples else 0.0
    print(f"\nAccuracy: {correct}/{len(test_examples)} = {accuracy:.1%}")
    return {"accuracy": accuracy, "results": results}


@app.local_entrypoint()
def main():
    print("Evaluating against the exact held-out split saved during training...")
    result = evaluate.remote()
    print(f"\nFinal accuracy: {result['accuracy']:.1%}")