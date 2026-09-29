"""
Modal-based QLoRA fine-tuning job for the Runbook Copilot incident
classifier. Trains unsloth/Llama-3.2-1B-Instruct (ungated mirror) to
classify an incident description into one of 5 categories:
active-directory, virtualization, networking, storage, backup.

Training data: data/qlora_training/classifier_dataset.jsonl (72 examples,
built in a prior step from real ingested docs + synthetic tickets).

Usage:
    modal run qlora/train_modal.py

Output: LoRA adapter weights saved to a Modal Volume, downloadable
afterward via `modal volume get`.
"""

import json
import os
from pathlib import Path

import modal

app = modal.App("runbook-copilot-qlora")

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
CATEGORIES = ["active-directory", "virtualization", "networking", "storage", "backup", "other"]

# Persistent volume to store the trained adapter
volume = modal.Volume.from_name("runbook-copilot-adapter", create_if_missing=True)
ADAPTER_DIR = "/adapter_output"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.44.0",
        "peft>=0.12.0",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.33.0",
        "datasets",
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
    cpu=16,
    memory=32768,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={ADAPTER_DIR: volume},
)
def train(dataset_jsonl: str):
    import random
    from collections import defaultdict
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        default_data_collator,
    )
    from peft import LoraConfig, get_peft_model

    hf_token = os.environ["HF_TOKEN"]

    # Parse the dataset passed in as a string (one JSON object per line)
    examples = [json.loads(line) for line in dataset_jsonl.strip().split("\n")]
    print(f"Loaded {len(examples)} training examples")

    # Build full training texts: prompt + label
    prompts = [build_prompt(ex["text"]) for ex in examples]
    full_texts = [p + f" {ex['label']}" for p, ex in zip(prompts, examples)]

    print(f"Loading tokenizer and base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        token=hf_token,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_fn(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        labels = []
        for i, full_text in enumerate(batch["text"]):
            prompt_text = batch["prompt"][i]
            prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=512)["input_ids"])
            ids = tokenized["input_ids"][i]
            # Mask prompt tokens (-100 = ignored by loss) and padding tokens
            label_row = [-100] * prompt_len + ids[prompt_len:]
            label_row = label_row[:len(ids)]
            label_row = [
                (tok if tok != tokenizer.pad_token_id else -100)
                for tok in label_row
            ]
            labels.append(label_row)
        tokenized["labels"] = labels
        return tokenized

    # Leak-proof source-level train/eval split:
    # Group examples by their originating document or ticket ID so no two rows
    # from the same source document land on opposite sides.
    source_to_indices = defaultdict(list)
    for idx, ex in enumerate(examples):
        src_id = ex.get("source_id", f"src_{idx}")
        source_to_indices[src_id].append(idx)

    sources = sorted(source_to_indices.keys())
    rng = random.Random(42)
    rng.shuffle(sources)

    target_eval_count = int(len(examples) * 0.20)
    eval_indices = []
    train_indices = []

    for src in sources:
        indices = source_to_indices[src]
        if len(eval_indices) + len(indices) <= target_eval_count:
            eval_indices.extend(indices)
        else:
            train_indices.extend(indices)

    if not eval_indices and len(sources) > 1:
        eval_indices.extend(source_to_indices[sources[0]])
        train_indices = [i for i in range(len(examples)) if i not in eval_indices]

    train_sources = set(examples[i].get("source_id") for i in train_indices)
    eval_sources = set(examples[i].get("source_id") for i in eval_indices)
    overlap = train_sources.intersection(eval_sources)
    assert len(overlap) == 0, f"Source leakage detected! Overlapping sources: {overlap}"

    print(f"Leak-proof split confirmed: 0 source overlap.")
    print(f"Train: {len(train_indices)} rows across {len(train_sources)} sources.")
    print(f"Eval:  {len(eval_indices)} rows across {len(eval_sources)} sources.")

    train_ds = Dataset.from_dict({
        "text": [full_texts[i] for i in train_indices],
        "prompt": [prompts[i] for i in train_indices],
    }).map(tokenize_fn, batched=True, remove_columns=["text", "prompt"])

    eval_ds = Dataset.from_dict({
        "text": [full_texts[i] for i in eval_indices],
        "prompt": [prompts[i] for i in eval_indices],
    }).map(tokenize_fn, batched=True, remove_columns=["text", "prompt"])

    training_args = TrainingArguments(
        output_dir="/tmp/qlora_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        use_cpu=True,
        bf16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {ADAPTER_DIR}")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # Save the exact eval split used (with source_id tracking)
    eval_examples_raw = [
        {
            "text": examples[i]["text"],
            "label": examples[i]["label"],
            "source_id": examples[i].get("source_id"),
        }
        for i in eval_indices
    ]
    with open(f"{ADAPTER_DIR}/eval_split.json", "w", encoding="utf-8") as f:
        json.dump(eval_examples_raw, f, indent=2)

    volume.commit()

    eval_result = trainer.evaluate()
    print(f"Final eval loss: {eval_result}")

    return eval_result


@app.local_entrypoint()
def main():
    dataset_path = Path("data/qlora_training/classifier_dataset.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    dataset_content = dataset_path.read_text(encoding="utf-8")
    print(f"Sending dataset ({len(dataset_content.splitlines())} lines) to Modal for training...")

    result = train.remote(dataset_jsonl=dataset_content)
    print(f"Training complete. Result: {result}")