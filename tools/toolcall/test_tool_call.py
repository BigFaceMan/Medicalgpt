# -*- coding: utf-8 -*-
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，如：北京、上海、纽约",
                        }
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_location",
                "description": "获取用户当前所在的城市位置",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "description": "返回格式，可选值：city、coordinates、full",
                            "enum": ["city", "coordinates", "full"],
                        }
                    },
                    "required": ["format"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_date",
                "description": "获取当前日期",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "description": "日期格式，可选值：date、datetime、timestamp",
                            "enum": ["date", "datetime", "timestamp"],
                        }
                    },
                },
            },
        },
    ]


def main():
    model_path = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"chat template loaded: {tokenizer.chat_template}\n")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    model.eval()
    print("Model loaded successfully!\n")

    tools = get_tools()
    print("Tools defined:")
    print(json.dumps(tools, indent=2, ensure_ascii=False))
    print("\n" + "=" * 60 + "\n")

    test_prompts = [
        "我在哪个城市？",
        "请告诉我北京的天气",
        "今天的日期是什么？"
    ]

    for prompt_text in test_prompts:
        print(f"Prompt: {prompt_text}")
        print("-" * 40)

        messages = [{"role": "user", "content": prompt_text}]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )
        print(f"Template applied:\n{text}\n")

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
        )

        print(f"Model Response:\n{response}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
