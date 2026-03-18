BASE_MODEL=Qwen/Qwen2.5-0.5B-instruct
OUTPUT_DIR=./output/qwen0.5B-instruct-merge
LORA_MODEL=/lfs1/users/spsong/Code/MedicalGPT/output/qwen0.5Binstruct-sft/checkpoint-1140

python ../src/trainer/merge_peft_adapter.py \
    --base_model $BASE_MODEL \
    --lora_model $LORA_MODEL \
    --output_dir $OUTPUT_DIR \

