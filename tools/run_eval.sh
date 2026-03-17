python local_ceval_zero_shot.py \
  --model_path Qwen/Qwen2.5-0.5B-instruct \
  --subject basic_medicine \
  --split val \
  --device cuda:4 \
  --dtype bfloat16 \
  --target_delimiter " " \
  --save_path ./basic_medicine_val.jsonl