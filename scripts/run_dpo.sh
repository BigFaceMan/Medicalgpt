#!/bin/bash
# DPO Training with YAML config
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 ./src/trainer/dpo_training.py --config configs/dpo_qwen3b.yaml
