BASE_MODEL=/lfs1/users/spsong/Code/MedicalGPT/output/qwen0.5B-instruct-merge
LORA_MODEL=/lfs1/users/spsong/Code/MedicalGPT/output/qwen0.5B-instruct-rm
OUTPUT_DIR=./output/qwen0.5B-instruct-rm-merge

CUDA_VISIBLE_DEVICES=1 python ./src/trainer/merge_peft_adapter.py \
    --base_model $BASE_MODEL \
    --lora_model $LORA_MODEL \
    --output_dir $OUTPUT_DIR \

