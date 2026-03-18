#!/bin/bash

MODEL="hf"
MODEL_ARGS="pretrained=/lfs1/users/spsong/Code/MedicalGPT/output/qwen0.5B-instruct-merge"
TASKS="ceval-valid_basic_medicine"
DEVICE="cuda:4"
BATCH_SIZE="8"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model_args)
            MODEL_ARGS="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

lm_eval \
    --model "$MODEL" \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"
