#!/bin/bash
# GRPO Training with YAML config
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 ./src/trainer/grpo_training.py --config configs/grpo_qwen3b.yaml
