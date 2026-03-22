#!/bin/bash

# 设置离线模式（避免网络问题）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 可选：设置 CUDA 设备
# export CUDA_VISIBLE_DEVICES=0

    # --model_path /lfs1/users/spsong/Code/MedicalGPT/outputs-pt-qwen-tl \
DATAPAHT=/lfs3/users/spsong/dataset/LLMData/finetune/valid/valid_zh_0_convert.json
python HFLibLearn/evaluate_perplexity.py \
    --model_path Qwen/Qwen2.5-0.5B \
    --data_path ./data/tianlongbabu/tianlongbabu.txt \
    --max_length 512 \
    --batch_size 4
