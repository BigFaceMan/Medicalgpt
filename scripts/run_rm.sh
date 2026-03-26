#!/bin/bash
# Reward Model Training with YAML config
CUDA_VISIBLE_DEVICES=0,1 python ./src/trainer/reward_modeling.py --config configs/rm_qwen3b.yaml
