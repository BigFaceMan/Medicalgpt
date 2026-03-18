# -*- coding: utf-8 -*-
"""
@description: 标准PPL评估（无量化版本）

usage:
python eval_fp.py \
  --model_path /path/to/model \
  --data_path data.jsonl
"""

import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from loguru import logger
import os


parser = argparse.ArgumentParser(description="========FP模型PPL评估========")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_jsonl_data(file_path):
    logger.info(f"Loading data from {file_path}")
    conversations = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            human, gpt = None, None

            for conv in data["conversations"]:
                if conv["from"] == "human":
                    human = conv["value"]
                elif conv["from"] == "gpt":
                    gpt = conv["value"]

            if human and gpt:
                conversations.append((human, gpt))

    logger.info(f"Loaded {len(conversations)} samples")
    return conversations


def build_inputs(tokenizer, prompt, answer, debug=False):
    """
    构造 input_ids + labels，并做decode验证
    """

    full_text = prompt + answer

    full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]

    labels = full_ids.clone()
    labels[:len(prompt_ids)] = -100

    if debug:
        # ===== decode =====
        decoded_full = tokenizer.decode(full_ids, skip_special_tokens=False)
        decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)

        # 只保留answer部分
        answer_ids = full_ids[len(prompt_ids):]
        decoded_answer = tokenizer.decode(answer_ids, skip_special_tokens=False)

        print("\n================ DEBUG ================")
        print("PROMPT:")
        print(prompt)
        print("\nDECODED PROMPT:")
        print(decoded_prompt)

        print("\nANSWER:")
        print(answer)
        print("\nDECODED ANSWER:")
        print(decoded_answer)

        print("\nFULL TEXT:")
        print(decoded_full)

        # ===== mask可视化 =====
        visible_labels = [
            tokenizer.decode([tid]) if lid != -100 else "[MASK]"
            for tid, lid in zip(full_ids, labels)
        ]

        print("\nLABEL VISUALIZATION:")
        print("".join(visible_labels))

        print("=====================================\n")

        # ===== assert =====
        # 1️⃣ prompt对齐
        assert decoded_prompt in decoded_full, "❌ prompt不在full_text中"

        # 2️⃣ answer对齐
        assert decoded_answer.strip() in decoded_full, "❌ answer不在full_text中"

        # 3️⃣ mask长度正确
        assert (labels[:len(prompt_ids)] == -100).all(), "❌ prompt没有正确mask"

        # 4️⃣ answer没有被mask
        assert (labels[len(prompt_ids):] != -100).all(), "❌ answer被错误mask"

        print("✅ ALL ASSERT PASSED")

    return full_ids, labels

def evaluate_perplexity(model, tokenizer, dataset):
    model.eval()
    device = get_device()

    total_loss = 0.0
    total_tokens = 0

    debug_flg = True
    for prompt, answer in tqdm(dataset, desc="Evaluating PPL"):
        input_ids, labels = build_inputs(tokenizer, prompt, answer, debug_flg)
        debug_flg = False  # 只调试第一个样本

        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)

        loss = outputs.loss

        valid_tokens = (labels != -100).sum().item()

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    ppl = torch.exp(torch.tensor(total_loss / total_tokens))

    logger.info(f"PPL: {ppl.item():.4f}")
    return ppl.item()


if __name__ == "__main__":
    args = parser.parse_args()

    # if not os.path.exists(args.model_path):
    #     raise ValueError(f"Model path not found: {args.model_path}")

    device = get_device()

    logger.info("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    # ✅ FP16 / FP32 自动选择
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    dataset = load_jsonl_data(args.data_path)

    evaluate_perplexity(model, tokenizer, dataset)

    logger.info("Done.")