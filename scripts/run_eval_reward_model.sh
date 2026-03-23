#!/bin/bash

# Run Reward Model Evaluation
# Author: XuMing
# Usage: bash scripts/run_eval_reward_model.sh

cd /lfs1/users/spsong/Code/MedicalGPT

source ~/anaconda3/etc/profile.d/conda.sh
conda activate MedicalLLM

python tools/eval_reward_model.py