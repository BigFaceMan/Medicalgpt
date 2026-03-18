#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Datasets 库使用示例
演示如何加载本地文本数据并进行预处理
"""

from datasets import load_dataset
from transformers import AutoTokenizer


def load_local_text_dataset():
    """加载本地 txt 数据集"""
    print("=" * 50)
    print("1. 加载本地 txt 数据集")
    print("=" * 50)

    # 加载本地 txt 文件（每行是一个样本）
    raw_datasets = load_dataset("text", data_files="data/pretrain/tianlongbabu.txt")

    print(f"数据集结构: {raw_datasets}")
    print(f"样本数量: {len(raw_datasets['train'])}")
    print(f"\n前3个样本:")
    for i in range(3):
        print(f"  [{i}]: {raw_datasets['train'][i]['text'][:50]}...")
    
    print(f"第一个样本")
    print(raw_datasets['train'][0]['text'])
    print(raw_datasets['train'][1]['text'])
    print(raw_datasets['train'][2]['text'])
    print(raw_datasets['train'][3]['text'])
    print(raw_datasets['train'][4]['text'])


def split_train_val():
    """划分训练集和验证集"""
    print("\n" + "=" * 50)
    print("2. 划分训练集和验证集")
    print("=" * 50)

    raw_datasets = load_dataset("text", data_files="data/pretrain/tianlongbabu.txt")

    # 划分 90% 训练，10% 验证
    split = raw_datasets["train"].train_test_split(test_size=0.1)

    print(f"训练集: {len(split['train'])} 条")
    print(f"验证集: {len(split['test'])} 条")


def tokenize_example():
    """Tokenize 处理"""
    print("\n" + "=" * 50)
    print("3. Tokenize 处理")
    print("=" * 50)

    raw_datasets = load_dataset("text", data_files="data/pretrain/tianlongbabu.txt")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    # 批量 tokenize
    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],  # 移除原文列
    )

    print(f"Tokenize 后:")
    print(f"  列名: {tokenized['train'].column_names}")
    print(f"  第一个样本 input_ids 长度: {len(tokenized['train'][0]['input_ids'])}")
    print(f"  第一个样本: {tokenized['train'][0]['input_ids'][:20]}...")


def filter_empty_lines():
    """过滤空行"""
    print("\n" + "=" * 50)
    print("4. 过滤空行")
    print("=" * 50)

    raw_datasets = load_dataset("text", data_files="data/pretrain/tianlongbabu.txt")

    print(f"过滤前: {len(raw_datasets['train'])} 条")

    # 过滤空行
    filtered = raw_datasets["train"].filter(lambda x: x["text"].strip() != "")

    print(f"过滤后: {len(filtered)} 条")


def packing_example():
    """Packing 示例（预训练常用）"""
    print("\n" + "=" * 50)
    print("5. Packing 示例")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )

    block_size = 128

    def tokenize_wo_pad(examples):
        return tokenizer(examples["text"])

    def group_texts(examples):
        """将多个文档拼接后按 block_size 切分"""
        eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id

        # 拼接所有文本
        all_ids = []
        for ids in examples["input_ids"]:
            all_ids.extend(ids)
            if len(ids) == 0 or ids[-1] != eos_id:
                all_ids.append(eos_id)

        # 按 block_size 切分
        total_length = len(all_ids)
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            "input_ids": [
                all_ids[i : i + block_size] for i in range(0, total_length, block_size)
            ],
            "labels": [
                all_ids[i : i + block_size] for i in range(0, total_length, block_size)
            ],
        }
        return result

    # 加载数据
    raw_datasets = load_dataset("text", data_files="data/pretrain/tianlongbabu.txt")

    # Tokenize（不 padding）
    tokenized = raw_datasets.map(tokenize_wo_pad, batched=True, remove_columns=["text"])

    # Packing
    packed = tokenized.map(group_texts, batched=True, desc="Packing texts")

    print(f"Packing 后样本数: {len(packed['train'])}")
    print(f"每个样本长度: {len(packed['train'][0]['input_ids'])}")


def main():
    # 1. 加载数据集
    load_local_text_dataset()

    # 2. 划分训练/验证集
    split_train_val()

    # 3. Tokenize
    tokenize_example()

    # 4. 过滤空行
    filter_empty_lines()

    # 5. Packing
    packing_example()

    print("\n" + "=" * 50)
    print("Demo 完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
