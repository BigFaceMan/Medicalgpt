#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tokenizer 使用示例
演示如何加载和使用 Hugging Face 的 AutoTokenizer
"""

from transformers import AutoTokenizer


def main():
    # ==================== 1. 加载 Tokenizer ====================
    print("=" * 50)
    print("1. 加载 Tokenizer")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

    
    print(f"toenizer chat template: {tokenizer.chat_template}")

    # ==================== 2. 基础编码 ====================
    print("\n" + "=" * 50)
    print("2. 基础编码 (encode)")
    print("=" * 50)

    text = "Hello, how are you?"
    inputs = tokenizer(text)
    print(f"原文: {text}")
    print(f"inputs type : {type(inputs)}")
    print(f"input_ids: {inputs['input_ids']}")
    print(f"attention_mask: {inputs['attention_mask']}")

    # ==================== 3. 解码 ====================
    print("\n" + "=" * 50)
    print("3. 解码 (decode)")
    print("=" * 50)

    decoded = tokenizer.decode(inputs["input_ids"])
    print(f"解码后: {decoded}")

    # 跳过特殊 token 解码
    decoded_skip = tokenizer.decode(inputs["input_ids"], skip_special_tokens=True)
    print(f"跳过特殊token解码: {decoded_skip}")

    # ==================== 4. 批量编码 ====================
    print("\n" + "=" * 50)
    print("4. 批量编码 (batch_encode)")
    print("=" * 50)

    texts = ["Hello", "Hi there", "Good morning!"]
    batch = tokenizer(
        texts,
        padding=True,  # 填充到最长
        truncation=True,  # 截断超长文本
        max_length=10,  # 最大长度
        return_tensors="pt",  # 返回 PyTorch tensor
    )
    print(f"输入文本: {texts}")
    print(f"input_ids:\n{batch['input_ids']}")
    print(f"attention_mask:\n{batch['attention_mask']}")

    # ==================== 5. Chat Template ====================
    print("\n" + "=" * 50)
    print("5. Chat Template (对话模板)")
    print("=" * 50)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "Thanks!"},
    ]

    # 应用 chat template (生成文本)
    output_text = tokenizer.apply_chat_template(messages, tokenize=False)
    print("Chat template 生成的文本:")
    print(output_text)

    # 应用 chat template (生成 token IDs)
    output_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    print(f"\nChat template 生成的 input_ids: {output_ids}")

    # 解码验证
    decoded_chat = tokenizer.decode(output_ids)
    print(f"\n解码验证:\n{decoded_chat}")

    # ==================== 6. 特殊 Token 操作 ====================
    print("\n" + "=" * 50)
    print("6. 特殊 Token 操作")
    print("=" * 50)

    # 添加特殊 token
    custom_text = "Hello [USER] how are you? [ASSISTANT]"
    custom_tokens = ["[USER]", "[ASSISTANT]"]
    tokenizer.add_special_tokens({"additional_special_tokens": custom_tokens})

    inputs = tokenizer(custom_text)
    print(f"原文: {custom_text}")
    print(f"input_ids: {inputs['input_ids']}")
    print(f"解码: {tokenizer.decode(inputs['input_ids'])}")

    # ==================== 7. 获取 Token 信息 ====================
    print("\n" + "=" * 50)
    print("7. 获取 Token 信息")
    print("=" * 50)

    # 获取 token 对应的 ID
    print(f"'hello' -> {tokenizer.encode('hello', add_special_tokens=False)}")
    print(f"'world' -> {tokenizer.encode('world', add_special_tokens=False)}")

    # 获取 ID 对应的 token
    print(f"72825 -> {tokenizer.decode([72825])}")
    print(f"271 -> {tokenizer.decode([271])}")

    # 批量获取
    ids = [9906, 374, 4981, 30, 3152, 25]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"IDs {ids} -> Tokens: {tokens}")

    # ==================== 8. 截断与溢出处理 ====================
    print("\n" + "=" * 50)
    print("8. 长文本处理")
    print("=" * 50)

    long_text = "This is a very long text. " * 100

    # 默认截断
    truncated = tokenizer(long_text, truncation=True, max_length=20)
    print(f"原文长度: {len(long_text)}")
    print(f"截断后长度: {len(truncated['input_ids'])}")

    # 返回 overflow token
    overflow = tokenizer(
        long_text,
        truncation=True,
        max_length=20,
        return_overflowing_tokens=True,
        stride=10,
    )
    print(f"溢出 tokens 数量: {len(overflow['input_ids'])}")
    for i, chunk in enumerate(overflow["input_ids"]):
        print(f"  Chunk {i + 1}: 长度={len(chunk)}")

    # ==================== 9. 保存与加载 ====================
    print("\n" + "=" * 50)
    print("9. 保存与加载 Tokenizer")
    print("=" * 50)

    save_path = "./tokenizer_test"
    tokenizer.save_pretrained(save_path)
    print(f"Tokenizer 已保存到: {save_path}")

    # 重新加载
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)
    print(f"重新加载: {loaded_tokenizer.__class__.__name__}")

    # 验证
    test_text = "Hello world"
    original_ids = tokenizer.encode(test_text)
    loaded_ids = loaded_tokenizer.encode(test_text)
    print(f"原始: {original_ids}")
    print(f"加载: {loaded_ids}")
    print(f"一致: {original_ids == loaded_ids}")

    # ==================== 10. 完整训练数据准备示例 ====================
    print("\n" + "=" * 50)
    print("10. 训练数据准备示例")
    print("=" * 50)

    dataset = [
        {"text": "The sky is blue."},
        {"text": "The grass is green."},
        {"text": "The sun is bright."},
    ]

    def preprocess_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128
        )

    # 模拟处理
    processed = preprocess_function({"text": "The sky is blue."})
    print(f"处理后 keys: {processed.keys()}")
    print(f"input_ids: {processed['input_ids'][:20]}...")
    print(f"attention_mask: {processed['attention_mask'][:20]}...")
    print(f"labels: {processed.get('labels', 'N/A')}")

    print("\n" + "=" * 50)
    print("Demo 完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
