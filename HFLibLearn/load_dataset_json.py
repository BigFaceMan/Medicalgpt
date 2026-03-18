
# -*- coding: utf-8 -*-
"""
@description: eval quantize for jsonl format data

usage:
python eval_quantize.py --bnb_path /path/to/your/bnb_model --data_path data/finetune/medical_sft_1K_format.jsonl
"""
from xml.parsers.expat import model

from pyexpat import model

import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from loguru import logger
import os

parser = argparse.ArgumentParser(description="========量化困惑度测试========")
parser.add_argument(
    "--bnb_path",
    type=str,
    required=True,  # 设置为必须的参数
    help="bnb量化后的模型路径。"
)
parser.add_argument(
    "--data_path",
    type=str,
    required=True,  # 设置为必须的参数
    help="jsonl数据集路径。"
)


# 设备选择函数
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


# 清理GPU缓存函数
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 从jsonl文件中加载数据
def load_jsonl_data(file_path):
    logger.info(f"Loading data from {file_path}")
    conversations = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 提取 human 和 gpt 部分的文本
                for conv in data['conversations']:
                    if conv['from'] == 'human':
                        input_text = conv['value']
                    elif conv['from'] == 'gpt':
                        target_text = conv['value']
                        conversations.append((input_text, target_text))
        return conversations
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []


# 困惑度评估函数
def evaluate_perplexity(model, tokenizer, conversation_pairs):
    def _perplexity(nlls, n_samples, seqlen):
        try:
            return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float('inf')

    model = model.eval()
    nlls = []
    # 获取设备
    device = get_device()

    # 确保 tokenizer 和 model 使用相同的设备
    model = model.to(device)
    # 遍历每个对话，基于 human 部分生成并与 gpt 部分计算困惑度
    for input_text, target_text in tqdm(conversation_pairs, desc="Perplexity Evaluation"):
        # Tokenize input and target
        inputs = tokenizer(input_text, return_tensors="pt", padding='max_length', truncation=True,
                           max_length=512).input_ids.to(get_device())
        target_ids = tokenizer(target_text, return_tensors="pt", padding='max_length', truncation=True,
                               max_length=512).input_ids.to(get_device())

        # Ensure both inputs and target have the same length
        if inputs.size(1) != target_ids.size(1):
            logger.warning(f"Input length {inputs.size(1)} and Target length {target_ids.size(1)} are not equal.")

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=inputs, labels=target_ids)
            loss = outputs.loss
            nlls.append(loss * target_ids.size(1))  # loss * sequence length

    # 计算最终困惑度
    total_samples = len(conversation_pairs)
    total_length = sum([len(pair[1]) for pair in conversation_pairs])
    ppl = _perplexity(nlls, total_samples, total_length)
    logger.info(f"Final Perplexity: {ppl:.3f}")

    return ppl.item()


# 主函数
if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.bnb_path):
        logger.error(f"Model path {args.bnb_path} does not exist.")
        exit(1)

    try:
      
        tokenizer = AutoTokenizer.from_pretrained(args.bnb_path, use_fast=True)
        # 加载jsonl数据 ds: [(input_text, target_text), ...]
        conversation_pairs = load_jsonl_data(args.data_path)

        for input_text, target_text in tqdm(conversation_pairs, desc="Perplexity Evaluation"):
            # Tokenize input and target
            print(f"padding token{tokenizer.pad_token_id}")
            print(f"eos token{tokenizer.eos_token_id}")
            print(f"inputs: {input_text}")
            print(f"targets: {target_text}")
            print(f"tokenize encode !!! {tokenizer.tokenize(input_text)}")
            print(f"tokenize decode !!! {tokenizer.decode(tokenizer.encode(input_text))}")

            inputs = tokenizer(input_text, return_tensors="pt", padding='max_length', truncation=True,
                            max_length=512).input_ids.to(get_device())
            print(type(inputs))
            print(inputs[0][:10])  # 打印前10个token id
            target_ids = tokenizer(target_text, return_tensors="pt", padding='max_length', truncation=True,
                                max_length=512).input_ids.to(get_device())

            print(target_ids.shape)

            # Ensure both inputs and target have the same length
            if inputs.size(1) != target_ids.size(1):
                logger.warning(f"Input length {inputs.size(1)} and Target length {target_ids.size(1)} are not equal.")

            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids=inputs, labels=target_ids)
                loss = outputs.loss

            if not conversation_pairs:
                logger.error("No valid conversation pairs found.")
                exit(1)

            # 开始评估
            # evaluate_perplexity(model, tokenizer, conversation_pairs)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
