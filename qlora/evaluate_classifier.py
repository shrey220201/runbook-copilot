"""
Load the trained LoRA adapter and test its actual classification accuracy
on real examples - not just loss, but does it predict the right category.

Usage:
    modal run qlora/evaluate_classifier.py
"""

import json
from pathlib import Path

import modal

app = modal.App("runbook-copilot-qlora-eval")

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
CATEGORIES = ["active-directory", "virtualization", "networking", "storage", "backup", "other"]

volume = modal.Volume.from_name("runbook-copilot-adapter", create_if_missing=False)
ADAPTER_DIR = "/adapter_output"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.44.0",
        "peft>=0.12.0",
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
    cpu=8,
    memory=16384,
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={ADAPTER_DIR: volume},
)
def evaluate():
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    hf_token = os.environ["HF_TOKEN"]

    # Load the EXACT held-out eval set saved during training
    with open(f"{ADAPTER_DIR}/eval_split.json", "r", encoding="utf-8") as f:
        test_examples = json.load(f)
    print(f"Loaded {len(test_examples)} held-out examples from training's saved split")

    print(f"Loading base model + adapter from {ADAPTER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map="cpu", token=hf_token
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    results = []
    correct = 0
    for ex in test_examples:
        prompt = build_prompt(ex["text"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
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
            "source_id": ex.get("source_id"),
        })
        print(f"{'✓' if is_correct else '✗'} true={ex['label']:20s} pred={str(predicted):20s} raw='{generated}'")

    accuracy = correct / len(test_examples) if test_examples else 0.0

    # Build Confusion Matrix
    eval_cols = CATEGORIES + ["unknown"]
    confusion_matrix = {true_c: {pred_c: 0 for pred_c in eval_cols} for true_c in CATEGORIES}

    for r in results:
        t_lbl = r["true_label"]
        p_lbl = r["predicted_label"] if r["predicted_label"] in CATEGORIES else "unknown"
        if t_lbl in confusion_matrix:
            confusion_matrix[t_lbl][p_lbl] += 1

    # Per-Class Metrics: Precision, Recall, F1
    per_class_metrics = {}
    for cat in CATEGORIES:
        tp = confusion_matrix[cat].get(cat, 0)
        fp = sum(confusion_matrix[other_cat].get(cat, 0) for other_cat in CATEGORIES if other_cat != cat)
        fn = sum(confusion_matrix[cat].get(other_col, 0) for other_col in eval_cols if other_col != cat)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_metrics[cat] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    # Macro & Weighted Averages
    total_support = sum(m["support"] for m in per_class_metrics.values())
    macro_precision = sum(m["precision"] for m in per_class_metrics.values()) / len(CATEGORIES) if CATEGORIES else 0.0
    macro_recall = sum(m["recall"] for m in per_class_metrics.values()) / len(CATEGORIES) if CATEGORIES else 0.0
    macro_f1 = sum(m["f1"] for m in per_class_metrics.values()) / len(CATEGORIES) if CATEGORIES else 0.0

    weighted_precision = sum(m["precision"] * m["support"] for m in per_class_metrics.values()) / total_support if total_support > 0 else 0.0
    weighted_recall = sum(m["recall"] * m["support"] for m in per_class_metrics.values()) / total_support if total_support > 0 else 0.0
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_class_metrics.values()) / total_support if total_support > 0 else 0.0

    summary = {
        "accuracy": round(accuracy, 4),
        "total_examples": len(test_examples),
        "correct_predictions": correct,
        "macro_avg": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
        },
        "weighted_avg": {
            "precision": round(weighted_precision, 4),
            "recall": round(weighted_recall, 4),
            "f1": round(weighted_f1, 4),
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
        "results": results,
    }

    return summary


@app.local_entrypoint()
def main():
    print("Evaluating against the exact held-out split saved during training...")
    metrics = evaluate.remote()

    print("\n" + "=" * 65)
    print("CLASSIFIER EVALUATION REPORT (Held-Out Test Split)")
    print("=" * 65)
    print(f"Overall Accuracy: {metrics['accuracy']:.1%} ({metrics['correct_predictions']}/{metrics['total_examples']})\n")

    print(f"{'Category':20s} {'Precision':10s} {'Recall':10s} {'F1':10s} {'Support':8s}")
    print("-" * 65)
    for cat, m in metrics["per_class_metrics"].items():
        print(f"{cat:20s} {m['precision']:<10.2%} {m['recall']:<10.2%} {m['f1']:<10.2%} {m['support']:<8d}")
    print("-" * 65)
    print(f"{'Macro Avg':20s} {metrics['macro_avg']['precision']:<10.2%} {metrics['macro_avg']['recall']:<10.2%} {metrics['macro_avg']['f1']:<10.2%}")
    print(f"{'Weighted Avg':20s} {metrics['weighted_avg']['precision']:<10.2%} {metrics['weighted_avg']['recall']:<10.2%} {metrics['weighted_avg']['f1']:<10.2%}\n")

    print("Confusion Matrix (rows: True Label, cols: Predicted Label):")
    cols = CATEGORIES + ["unknown"]
    true_pred_label = "True \\ Pred"
    header = f"{true_pred_label:20s} " + " ".join(f"{c[:8]:>8s}" for c in cols)
    print(header)
    for true_c in CATEGORIES:
        row = f"{true_c:20s} " + " ".join(f"{metrics['confusion_matrix'][true_c].get(pred_c, 0):>8d}" for pred_c in cols)
        print(row)

    out_dir = Path("results/v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "classifier_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out_path}")