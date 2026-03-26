#!/bin/bash
# PPO Training with YAML config
CUDA_VISIBLE_DEVICES=0,1 python ./src/trainer/ppo_training.py --config configs/ppo_qwen3b.yaml
