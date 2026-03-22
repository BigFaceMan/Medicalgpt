python tools/eval/local_ceval_zero_shot.py \
  --model_path meta-llama/Llama-3.2-3B-Instruct \
  --subject basic_medicine \
  --split test \
  --device cuda:0 \
  --dtype bfloat16 \
  --target_delimiter " " \
  --trust_remote_code \
  --save_path ./tools/eval/outputs/basic_medicine_test_llama3.2_3B.jsonl