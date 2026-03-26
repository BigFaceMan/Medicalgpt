#!/bin/bash
# Pretraining with YAML config
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node 2 ../src/trainer/pretrain.py --config configs/pretrain_qwen3b.yaml
