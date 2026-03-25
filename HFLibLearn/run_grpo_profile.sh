#!/bin/bash

# GRPO Training with Profiling
# Usage: bash run_grpo_profile.sh [optional_args]

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 HFLibLearn/grpo_demo.py \
    --per_device_train_batch_size=8 \
    --num_generations=2 \
    --dataloader_num_workers=8 \
    --gradient_accumulation_steps=2 \
    "$@"
