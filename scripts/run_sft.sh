#!/bin/bash
# SFT Training with YAML config
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 ./src/trainer/supervised_finetuning.py --config configs/sft_qwen3b.yaml
