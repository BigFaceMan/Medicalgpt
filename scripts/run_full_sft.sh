#!/bin/bash
# Full SFT Training (without PEFT, with DeepSpeed) + YAML config
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 ../src/trainer/supervised_finetuning.py --config configs/full_sft_qwen3b.yaml
