#!/bin/bash

# Inference script for single GPU
# Usage: bash run_single_gpu.sh [device_id] [base_model] [lora_model] [data_file]

CUDA_VISIBLE_DEVICES=${1:-$CUDA_VISIBLE_DEVICES}
BASE_MODEL=${2:-"/lfs1/users/spsong/Code/MedicalGPT/output/qwen3B-instruct-sft-merge"}
LORA_MODEL=${3:-"/lfs1/users/spsong/Code/MedicalGPT/output/qwen0.5B-instruct-sft"}
DATA_FILE=${4:-"/lfs1/users/spsong/Code/MedicalGPT/scripts/test_questions.txt"}
OUTPUT_FILE=${5:-"./predictions_result.jsonl"}

export CUDA_VISIBLE_DEVICES

python ./src/inference/single_gpu.py \
    --base_model "$BASE_MODEL" \
    --data_file "$DATA_FILE" \
    --output_file "$OUTPUT_FILE" \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --eval_batch_size 4
