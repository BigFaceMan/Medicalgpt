# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Evaluate Reward Model - randomly sample 10 examples and check if chosen > rejected
"""

import json
import os
import random
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "/lfs1/users/spsong/Code/MedicalGPT/output/qwen0.5B-instruct-rm-merge"
DATA_PATH = "/lfs1/users/spsong/Code/MedicalGPT/data/reward/dpo_zh_500.jsonl"
OUTPUT_DIR = os.path.join(MODEL_PATH, "eval")
NUM_SAMPLES = 10
MAX_SOURCE_LENGTH = 2048
MAX_TARGET_LENGTH = 512
FULL_MAX_LENGTH = MAX_SOURCE_LENGTH + MAX_TARGET_LENGTH


def build_qwen_prompt(system, history, question, answer):
    system_prompt = system if system else "You are a helpful assistant."
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    if history:
        for h in history:
            prompt += f"<|im_start|>user\n{h[0]}<|im_end|>\n<|im_start|>assistant\n{h[1]}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
    return prompt


def get_reward(model, tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=FULL_MAX_LENGTH,
        padding="max_length",
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.item()


def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Model loaded successfully.")

    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    print(f"Total data samples: {len(data)}")

    random.seed(42)
    samples = random.sample(data, NUM_SAMPLES)

    results = []
    correct_count = 0

    for i, item in enumerate(samples):
        chosen_text = build_qwen_prompt(
            item["system"], item["history"], item["question"], item["response_chosen"]
        )
        rejected_text = build_qwen_prompt(
            item["system"], item["history"], item["question"], item["response_rejected"]
        )

        chosen_score = get_reward(model, tokenizer, chosen_text)
        rejected_score = get_reward(model, tokenizer, rejected_text)
        is_correct = chosen_score > rejected_score

        if is_correct:
            correct_count += 1

        result = {
            "question": item["question"][:200] + "..."
            if len(item["question"]) > 200
            else item["question"],
            "system": item["system"],
            "history": item["history"],
            "response_chosen": item["response_chosen"][:200] + "..."
            if len(item["response_chosen"]) > 200
            else item["response_chosen"],
            "response_rejected": item["response_rejected"][:200] + "..."
            if len(item["response_rejected"]) > 200
            else item["response_rejected"],
            "chosen_score": round(chosen_score, 6),
            "rejected_score": round(rejected_score, 6),
            "is_correct": is_correct,
        }
        results.append(result)

        print(
            f"[{i + 1}/{NUM_SAMPLES}] chosen={chosen_score:.4f}, rejected={rejected_score:.4f}, correct={is_correct}"
        )
        print(f"    Q: {result['question'][:80]}...")

    accuracy = correct_count / NUM_SAMPLES

    output_data = {
        "model_path": MODEL_PATH,
        "data_path": DATA_PATH,
        "num_samples": NUM_SAMPLES,
        "correct_count": correct_count,
        "accuracy": round(accuracy, 4),
        "results": results,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "reward_eval_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Evaluation Results:")
    print(f"  Total samples: {NUM_SAMPLES}")
    print(f"  Correct: {correct_count}")
    print(f"  Accuracy: {accuracy * 100:.1f}%")
    print(f"  Results saved to: {output_file}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
