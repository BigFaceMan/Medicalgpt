from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/lfs1/users/spsong/Code/MedicalGPT/output/qwen0.5B-instruct-merge"
# model_path = "Qwen/Qwen2.5-0.5B-instruct"  # 🤗 hub 上的模型路径

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.eval()

# ✅ system prompt（很重要）
system_prompt = "You are a helpful assistant."

# ✅ 多轮对话缓存
messages = [
    {"role": "system", "content": system_prompt}
]

while True:
    user_input = input("请输入: ")

    messages.append({"role": "user", "content": user_input})

    # ✅ 核心：构造 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # 让模型生成 assistant
    )
    print("tokenizer applied text:", text)

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )

    print("助手:", response)

    # ✅ 记录 assistant 回复（用于多轮）
    messages.append({"role": "assistant", "content": response})