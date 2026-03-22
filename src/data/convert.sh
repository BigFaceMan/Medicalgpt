#!/bin/bash
# 转换 alpaca 格式
python /lfs1/users/spsong/Code/MedicalGPT/src/data/converter.py \
    --in_dir /lfs3/users/spsong/dataset/LLMData/finetune \
    --data_type alpaca \
    --file_type json