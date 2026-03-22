# MODEL=Qwen/Qwen2.5-3B-instruct
# MODEL=/lfs1/users/spsong/Code/MedicalGPT/output/qwen0.5B-instruct-merge
# MODEL=/lfs1/users/spsong/Code/MedicalGPT/model/Llama-3.2-3B-Instruct
MODEL=/lfs1/users/spsong/Code/MedicalGPT/output/qwen3B-instruct-sft-merge

  # --tasks ceval-valid_basic_medicine,ceval-valid_clinical_medicine,ceval-valid_physician  \
  # --tasks ceval-test_basic_medicine,ceval-test_clinical_medicine,ceval-test_physician  \
lm_eval \
  --model hf \
  --model_args "pretrained=$MODEL,trust_remote_code=True" \
  --tasks ceval-valid_basic_medicine,ceval-valid_clinical_medicine,ceval-valid_physician,ceval-test_basic_medicine,ceval-test_clinical_medicine,ceval-test_physician  \
  --num_fewshot 0 \
  --device cuda:1 \
  --batch_size 64 \
  --output_path ./tools/eval/outputs/ceval_medicine_qwen2.5_3B_instruct_sft


# lm_eval \
#   --model hf \
#   --model_args "pretrained=$MODEL,trust_remote_code=True" \
#   --tasks ceval-valid_basic_medicine,ceval-valid_clinical_medicine,ceval-valid_physician  \
#   --num_fewshot 0 \
#   --device cuda:0 \
#   --batch_size 64 \
#   --output_path ./tools/eval/outputs/ceval_medicine_qwen2.5_3B_instruct_val
