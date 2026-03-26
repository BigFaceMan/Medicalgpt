#!/bin/bash
# ORPO Training with YAML config
CUDA_VISIBLE_DEVICES=0,1 python ../src/trainer/orpo_training.py --config configs/orpo_qwen3b.yaml
