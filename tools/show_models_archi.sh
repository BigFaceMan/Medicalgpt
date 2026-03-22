#!/bin/bash

MODEL_PATHS=(
    "/lfs1/users/spsong/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    "/lfs1/users/spsong/Code/MedicalGPT/output/qwen0.5B-instruct-merge"
)

if [ $# -gt 0 ]; then
    MODEL_PATHS=("$@")
fi

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "=========================================="
    echo "Model: $MODEL_PATH"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=1 python ./tools/show_model_archi.py \
        --model_path "$MODEL_PATH"
    echo ""
done
