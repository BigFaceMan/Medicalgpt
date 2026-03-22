# -*- coding: utf-8 -*-
"""
@description: 计算模型的困惑度 (Perplexity)

Usage:
python evaluate_perplexity.py \
    --model_path ./my_model \
    --data_path ./data/tianlongbabu/tianlongbabu.txt \
    --max_length 512 \
    --batch_size 4
"""

import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
import math


def load_text_data(file_path):
    """加载纯文本文件，每行作为一个样本"""
    logger.info(f"Loading data from {file_path}")
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                texts.append(line)
    logger.info(f"Loaded {len(texts)} samples")
    return texts


def calculate_perplexity(
    model, tokenizer, texts, max_length=512, batch_size=4, device="cuda"
):
    """计算困惑度"""
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating Perplexity"):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss

            # 计算有效 token 数量（排除 padding）
            # 注意：这里简化处理，实际可以根据 attention_mask 计算
            token_count = input_ids.numel() - (attention_mask == 0).sum().item()

            total_loss += loss.item() * token_count
            total_tokens += token_count

    # 计算平均 loss 和困惑度
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss


def main():
    parser = argparse.ArgumentParser(description="计算模型困惑度")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument(
        "--data_path", type=str, required=True, help="测试数据路径 (txt文件)"
    )
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Batch size: {args.batch_size}")

    # 加载 tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 加载模型
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载数据
    texts = load_text_data(args.data_path)

    # 计算困惑度
    logger.info("Starting perplexity calculation...")
    perplexity, avg_loss = calculate_perplexity(
        model,
        tokenizer,
        texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    logger.info("=" * 50)
    logger.info(f"Average Loss: {avg_loss:.4f}")
    logger.info(f"Perplexity: {perplexity:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
