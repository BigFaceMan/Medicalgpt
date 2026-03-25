# -*- coding: utf-8 -*-
"""
对比 Conversation 模板和 apply_chat_template 两种渲染方式的差异
"""

import json
import random
from datasets import load_dataset
from transformers import AutoTokenizer


def load_data():
    """从文件夹加载原始数据"""
    data_files = [
        "/lfs1/users/spsong/Code/MedicalGPT/data/finetune/sharegpt_zh_1K_format.jsonl"
    ]
    raw_datasets = load_dataset("json", data_files=data_files)
    return raw_datasets["train"]


def convert_conversations_to_messages(conversations):
    """
    将原始数据格式转换为 Qwen2.5 的 messages 格式
    原始格式: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
    目标格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    messages = []
    for conv in conversations:
        if conv["from"] == "human":
            messages.append({"role": "user", "content": conv["value"]})
        elif conv["from"] == "gpt":
            messages.append({"role": "assistant", "content": conv["value"]})
    return messages


def render_with_conversation_template(messages, tokenizer):
    """
    使用 Conversation 类的方式渲染（参考 template.py 中的 qwen 模板）
    """
    from template import get_conv_template

    prompt_template = get_conv_template("qwen")

    history_messages = []
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            history_messages.append(
                [messages[i]["content"], messages[i + 1]["content"]]
            )

    dialog = prompt_template.get_dialog(history_messages, system_prompt="")
    text = "".join(dialog)
    return text


def render_with_apply_chat_template(messages, tokenizer):
    """
    使用 Qwen2.5 tokenizer 的 apply_chat_template 方式渲染
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return text


def main():
    print("=" * 80)
    print("Conversation 模板 vs apply_chat_template 对比")
    print("=" * 80)

    # 加载 tokenizer
    print("\n[1] 加载 Qwen2.5-3B-Instruct tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True
    )

    # 加载数据
    print("[2] 加载数据...")
    dataset = load_data()
    print(f"    数据集大小: {len(dataset)} 条")

    # 找到第一条多轮对话
    for idx in range(len(dataset)):
        sample = dataset[idx]
        messages = convert_conversations_to_messages(sample["conversations"])
        if len(messages) > 2:
            print(f"[3] 选择第 {idx} 条数据 (多轮对话: {len(messages)} 条消息)")
            break

    # 转换为 messages 格式
    messages = convert_conversations_to_messages(sample["conversations"])
    print(f"    转换后 messages 数量: {len(messages)} 条")
    print(f"    第一条 user message: {messages[0]['content'][:50]}...")

    # 两种渲染方式
    print("\n[4] 开始渲染...")
    print("-" * 80)

    text1 = render_with_conversation_template(messages, tokenizer)
    print("\n=== 方式1: Conversation 模板渲染结果 ===")
    print(text1)
    print()

    text2 = render_with_apply_chat_template(messages, tokenizer)
    print("\n=== 方式2: apply_chat_template 渲染结果 ===")
    print(text2)
    print()

    # 对比差异
    print("\n" + "=" * 80)
    print("差异分析")
    print("=" * 80)

    if text1 == text2:
        print("\n✅ 两种方式渲染结果完全相同")
    else:
        print("\n❌ 两种方式渲染结果存在差异")
        print(f"\n方式1 长度: {len(text1)} 字符")
        print(f"方式2 长度: {len(text2)} 字符")

        print("\n--- 方式1独有内容 ---")
        diff1 = set(text1) - set(text2)
        if diff1:
            print(f"字符差异: {diff1}")

        print("\n--- 方式2独有内容 ---")
        diff2 = set(text2) - set(text1)
        if diff2:
            print(f"字符差异: {diff2}")

        # 逐行对比
        lines1 = text1.split("\n")
        lines2 = text2.split("\n")
        print(f"\n--- 逐行对比 (方式1 {len(lines1)} 行 vs 方式2 {len(lines2)} 行) ---")
        max_lines = max(len(lines1), len(lines2))
        for i in range(max_lines):
            l1 = lines1[i] if i < len(lines1) else "<空>"
            l2 = lines2[i] if i < len(lines2) else "<空>"
            if l1 != l2:
                print(f"行{i}: 方式1: {repr(l1[:80])}")
                print(f"行{i}: 方式2: {repr(l2[:80])}")
                print()

    # 写入文件
    output_file = (
        "/lfs1/users/spsong/Code/MedicalGPT/src/trainer/compare_output_multiturn.txt"
    )
    print(f"\n[5] 写入文件: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Conversation 模板 vs apply_chat_template 对比\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"数据索引: {idx}\n")
        f.write(f"数据集大小: {len(dataset)} 条\n\n")

        f.write("--- 原始数据 ---\n")
        f.write(json.dumps(sample, ensure_ascii=False, indent=2))
        f.write("\n\n")

        f.write("--- 转换后的 messages ---\n")
        f.write(json.dumps(messages, ensure_ascii=False, indent=2))
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("方式1: Conversation 模板渲染结果\n")
        f.write("=" * 80 + "\n\n")
        f.write(text1)
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("方式2: apply_chat_template 渲染结果\n")
        f.write("=" * 80 + "\n\n")
        f.write(text2)
        f.write("\n\n")

        if text1 != text2:
            f.write("=" * 80 + "\n")
            f.write("差异分析\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"方式1 长度: {len(text1)} 字符\n")
            f.write(f"方式2 长度: {len(text2)} 字符\n")
            f.write(f"长度差异: {abs(len(text1) - len(text2))} 字符\n")

    print(f"\n✅ 完成！结果已写入 {output_file}")


if __name__ == "__main__":
    main()
