#!/bin/bash

CUDA_VISIBLE_DEVICES=${1:-$CUDA_VISIBLE_DEVICES}
MODEL_PATH=${2:-"/lfs1/users/spsong/Code/MedicalGPT/output/qwen0.5Binstruct-sft"}
DATA_PATH=${3:-"data/finetune/medical_sft_1K_format.jsonl"}

export CUDA_VISIBLE_DEVICES
python ../tools/eval/eval_ppl.py --model_path "$MODEL_PATH" --data_path "$DATA_PATH"
