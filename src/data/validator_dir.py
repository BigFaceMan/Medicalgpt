# -*- coding: utf-8 -*-
"""
Validate all files in a directory to check if they conform to sharegpt format.
"""

import json
import os
import argparse


def validate_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        line_number = 0
        valid_lines = 0
        total_lines = 0
        invalid_lines = []

        for line in file:
            total_lines += 1
            line_number += 1
            try:
                data = json.loads(line)

                if "conversations" not in data:
                    invalid_lines.append((line_number, "缺少 'conversations' 键"))
                    continue

                conversations = data["conversations"]
                if not isinstance(conversations, list):
                    invalid_lines.append((line_number, "'conversations' 应为列表格式"))
                    continue

                conversation_valid = True
                for conv in conversations:
                    if "from" not in conv or "value" not in conv:
                        invalid_lines.append((line_number, "缺少 'from' 或 'value' 键"))
                        conversation_valid = False
                        continue

                    if conv["from"] not in ["system", "human", "gpt"]:
                        invalid_lines.append(
                            (line_number, f"'from' 字段值无效: {conv['from']}")
                        )
                        conversation_valid = False

                if conversation_valid:
                    valid_lines += 1

            except json.JSONDecodeError:
                invalid_lines.append((line_number, "JSON 格式无效"))

    return {
        "total": total_lines,
        "valid": valid_lines,
        "invalid": total_lines - valid_lines,
        "invalid_details": invalid_lines,
    }


def validate_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            return {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "invalid_details": [(0, "JSON 格式无效")],
            }

    if not isinstance(data, list):
        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "invalid_details": [(0, "JSON 应为列表格式")],
        }

    valid_lines = 0
    invalid_lines = []

    for line_number, item in enumerate(data, 1):
        try:
            if "conversations" not in item:
                invalid_lines.append((line_number, "缺少 'conversations' 键"))
                continue

            conversations = item["conversations"]
            if not isinstance(conversations, list):
                invalid_lines.append((line_number, "'conversations' 应为列表格式"))
                continue

            conversation_valid = True
            for conv in conversations:
                if "from" not in conv or "value" not in conv:
                    invalid_lines.append((line_number, "缺少 'from' 或 'value' 键"))
                    conversation_valid = False
                    continue

                if conv["from"] not in ["system", "human", "gpt"]:
                    invalid_lines.append(
                        (line_number, f"'from' 字段值无效: {conv['from']}")
                    )
                    conversation_valid = False

            if conversation_valid:
                valid_lines += 1

        except Exception as e:
            invalid_lines.append((line_number, str(e)))

    total = len(data)
    return {
        "total": total,
        "valid": valid_lines,
        "invalid": total - valid_lines,
        "invalid_details": invalid_lines,
    }


def validate_dir(in_dir):
    supported_extensions = {".json", ".jsonl"}

    files_to_validate = []
    for root, dirs, files in os.walk(in_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_extensions:
                files_to_validate.append(os.path.join(root, filename))

    if not files_to_validate:
        print(f"目录 '{in_dir}' 中没有找到 .json 或 .jsonl 文件。")
        return

    print(f"验证目录: {in_dir}\n")
    print("=" * 60)

    total_files = len(files_to_validate)
    valid_files = 0
    invalid_files = 0
    total_lines = 0
    total_valid = 0
    total_invalid = 0

    for file_path in sorted(files_to_validate):
        rel_path = os.path.relpath(file_path, in_dir)
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".jsonl":
            result = validate_jsonl(file_path)
        else:
            result = validate_json(file_path)

        total_lines += result["total"]
        total_valid += result["valid"]
        total_invalid += result["invalid"]

        status = "✓" if result["invalid"] == 0 else "✗"
        print(f"File: {rel_path}")
        print(f"  有效: {result['valid']} 行, 无效: {result['invalid']} 行 {status}")

        if result["invalid_details"]:
            for line_num, error_msg in result["invalid_details"][:5]:
                print(f"    - 第 {line_num} 行: {error_msg}")
            if len(result["invalid_details"]) > 5:
                print(f"    - ... 还有 {len(result['invalid_details']) - 5} 个错误")

        print()

        if result["invalid"] == 0:
            valid_files += 1
        else:
            invalid_files += 1

    print("=" * 60)
    print(
        f"总计: {total_files} 个文件, 有效: {valid_files} 个, 无效: {invalid_files} 个"
    )
    print(f"总行数: {total_lines} 行, 有效: {total_valid} 行, 无效: {total_invalid} 行")

    if invalid_files == 0:
        print("\n恭喜！所有文件格式都正确。")
    else:
        print(f"\n有 {invalid_files} 个文件存在问题，请检查并修复。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate all files in a directory.")
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Directory containing files to validate",
    )
    args = parser.parse_args()

    validate_dir(args.in_dir)
