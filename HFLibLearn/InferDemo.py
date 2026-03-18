import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("outputs-pt-qwen-tl")
model = AutoModelForCausalLM.from_pretrained("outputs-pt-qwen-tl")
text = "你好，你是谁"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
