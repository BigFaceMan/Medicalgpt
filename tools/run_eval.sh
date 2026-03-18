python eval/local_ceval_zero_shot.py \
  --model_path /lfs1/users/spsong/Code/MedicalGPT/output/qwen0.5B-instruct-merge \
  --subject basic_medicine \
  --split test \
  --device cuda:1 \
  --dtype bfloat16 \
  --target_delimiter " " \
  --save_path ./sft_medicine_test_0.5B.jsonl