#!/bin/bash
# SFT Training with Accelerate + YAML config
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --num_processes=2 ../src/trainer/supervised_finetuning_accelerate.py --config configs/sft_accelerate_qwen3b.yaml
