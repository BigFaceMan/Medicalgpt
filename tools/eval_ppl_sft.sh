#!/bin/bash

# 设置离线模式（避免网络问题）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0


DATAPAHT=/lfs3/users/spsong/dataset/LLMData/finetune/valid/valid_zh_0_convert.json
# MODELPATH=Qwen/Qwen2.5-3B-instruct
MODELPATH=/lfs1/users/spsong/Code/MedicalGPT/output/qwen3B-instruct-sft-merge
python ./tools/eval/eval_ppl_sft.py \
    --model_path $MODELPATH \
    --data_path $DATAPAHT
