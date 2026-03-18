from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# model_path = "/lfs1/users/spsong/Code/MedicalGPT/outputs-pt-qwen-tl"
model_path = "/lfs1/users/spsong/Code/MedicalGPT/output/qwen0.5B-instruct-merge"


tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto"
)

model.eval()

while True:

    prompt = input("请输入: ")

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print(response)