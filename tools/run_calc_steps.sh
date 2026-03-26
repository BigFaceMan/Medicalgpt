#!/bin/bash
# Calculate training steps from YAML config

# 使用默认配置
# python ../tools/calc_training_steps.py --config ../configs/sft_qwen3b.yaml

# 指定 GPU 数量
# python ../tools/calc_training_steps.py --config ../configs/sft_qwen3b.yaml --num-gpus 4

# 使用全部数据计算步数
# python ../tools/calc_training_steps.py --config ../configs/sft_qwen3b.yaml --max-samples -1

# 计算并打印详细信息
# python ../tools/calc_training_steps.py --config ../configs/sft_qwen3b.yaml --verbose

# 计算步数并更新配置文件
# python ../tools/calc_training_steps.py --config ../configs/sft_qwen3b.yaml --output update

# 计算并设置目标步数（自动计算需要的 epoch 数）
# python ../tools/calc_training_steps.py --config ../configs/sft_qwen3b.yaml --output update --target-steps 5000

# 示例：计算 SFT 配置的步数
python ./tools/calc_training_steps.py --config ./configs/sft_qwen3b.yaml --verbose --num-gpus 1
