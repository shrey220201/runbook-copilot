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
CATEGORIES = ["active-directory", "virtualization", "networking", "storage", "backup"]

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
    gpu="A10G",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={ADAPTER_DIR: volume},
)
def train(dataset_jsonl: str):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        default_data_collator,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    hf_token = os.environ["HF_TOKEN"]

    # Parse the dataset passed in as a string (one JSON object per line)
    examples = [json.loads(line) for line in dataset_jsonl.strip().split("\n")]
    print(f"Loaded {len(examples)} training examples")

    # Build full training texts: prompt + label (what the model should learn
    # to generate after "Category:"). Track prompt lengths separately so we
    # can mask them out of the loss - we only want the model learning to
    # predict the category word, not reconstruct the incident text itself.
    prompts = [build_prompt(ex["text"]) for ex in examples]
    full_texts = [p + f" {ex['label']}" for p, ex in zip(prompts, examples)]

    print(f"Loading tokenizer and base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    model = prepare_model_for_kbit_training(model)

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
            # Mask prompt tokens (-100 = ignored by loss) and padding tokens,
            # so the model only learns to predict the category label itself.
            label_row = [-100] * prompt_len + ids[prompt_len:]
            label_row = label_row[:len(ids)]
            label_row = [
                (tok if tok != tokenizer.pad_token_id else -100)
                for tok in label_row
            ]
            labels.append(label_row)
        tokenized["labels"] = labels
        return tokenized

    dataset = Dataset.from_dict({"text": full_texts, "prompt": prompts})
    # Small held-out eval split (80/20), given the small dataset size
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"].map(tokenize_fn, batched=True, remove_columns=["text", "prompt"])
    eval_ds = split["test"].map(tokenize_fn, batched=True, remove_columns=["text", "prompt"])

    print(f"Train examples: {len(train_ds)}, Eval examples: {len(eval_ds)}")

    training_args = TrainingArguments(
        output_dir="/tmp/qlora_checkpoints",
        num_train_epochs=8,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=True,
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

    # Save the exact eval split used, so evaluation later tests against the
    # real held-out set rather than trying to reproduce this split separately
    eval_examples_raw = [
        {"text": examples[i]["text"], "label": examples[i]["label"]}
        for i in range(len(examples))
        if full_texts[i] in split["test"]["text"]
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