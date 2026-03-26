#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Steps Calculator
根据 YAML 配置文件和数据目录样本数量，自动计算训练步数
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import yaml


def count_samples_in_dir(data_dir: str) -> int:
    """统计目录中的样本数量（支持 jsonl 和 json 文件）"""
    if not os.path.exists(data_dir):
        print(f"  [Warning] Data directory not found: {data_dir}")
        return 0

    sample_count = 0

    # 统计 jsonl 文件
    jsonl_files = glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True)
    for f in jsonl_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                sample_count += sum(1 for _ in fp)
        except Exception as e:
            print(f"  [Warning] Failed to read {f}: {e}")

    # 统计 json 文件
    json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    sample_count += len(data)
                elif isinstance(data, dict):
                    sample_count += 1
        except Exception as e:
            print(f"  [Warning] Failed to read {f}: {e}")

    return sample_count


def format_number(n: int) -> str:
    """格式化数字，使用千位分隔符"""
    return f"{n:,}"


def parse_bool(value) -> bool:
    """解析布尔值"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return bool(value)


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config_path: str, config: dict):
    """保存 YAML 配置文件"""
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)


def detect_num_gpus() -> int:
    """自动检测 GPU 数量"""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 1


def calculate_steps(
    config: dict, num_gpus: int, max_samples_override: int = None
) -> dict:
    """计算训练步数"""

    # 数据信息
    data_dir = config.get("train_file_dir", "")
    config_max_samples = config.get("max_train_samples", -1)
    validation_split = config.get("validation_split_percentage", 0)

    # 使用样本数
    if max_samples_override is not None and max_samples_override != -1:
        use_samples = max_samples_override
    else:
        use_samples = config_max_samples if config_max_samples != -1 else None

    # 实际样本数
    if data_dir and os.path.exists(data_dir):
        actual_samples = count_samples_in_dir(data_dir)
    else:
        actual_samples = 0

    # 确定最终使用的样本数
    if use_samples is None:
        final_samples = actual_samples
    else:
        final_samples = (
            min(use_samples, actual_samples) if actual_samples > 0 else use_samples
        )

    # Batch Size 信息
    per_device_bs = config.get("per_device_train_batch_size", 1)
    gradient_accum = config.get("gradient_accumulation_steps", 1)
    effective_batch_size = per_device_bs * gradient_accum * num_gpus

    # 计算 steps
    if effective_batch_size > 0 and final_samples > 0:
        steps_per_epoch = final_samples // effective_batch_size
    else:
        steps_per_epoch = 0

    # Epoch 和 Steps
    num_epochs = config.get("num_train_epochs", 1)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = config.get("warmup_steps", 0)

    return {
        "data_dir": data_dir,
        "actual_samples": actual_samples,
        "config_max_samples": config_max_samples,
        "use_samples": use_samples,
        "final_samples": final_samples,
        "validation_split": validation_split,
        "per_device_batch_size": per_device_bs,
        "gradient_accumulation_steps": gradient_accum,
        "num_gpus": num_gpus,
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "num_train_epochs": num_epochs,
        "total_training_steps": total_steps,
        "warmup_steps": warmup_steps,
        "max_train_samples": config.get("max_train_samples", -1),
        "max_eval_samples": config.get("max_eval_samples", -1),
        "learning_rate": str(config.get("learning_rate", "N/A")),
        "output_dir": config.get("output_dir", "N/A"),
    }


def print_report(info: dict, verbose: bool = False):
    """打印训练步数报告"""
    separator = "=" * 50

    print(f"\n{separator}")
    print(f"{'Training Steps Calculator':^50}")
    print(f"{separator}")

    # Data Information
    print(f"\n[Data Information]")
    print(f"  Data Directory:      {info['data_dir'] or 'N/A'}")
    print(f"  Actual Samples:       {format_number(info['actual_samples'])}")

    if info["config_max_samples"] == -1:
        print(
            f"  Use Samples:          {format_number(info['final_samples'])} (all data)"
        )
    else:
        print(
            f"  Use Samples:          {format_number(info['final_samples'])} "
            f"(limited by max_train_samples={info['config_max_samples']})"
        )
    print(f"  Validation Split:    {info['validation_split']}%")

    # Batch Size Information
    print(f"\n[Batch Size Information]")
    print(f"  Per Device BS:        {info['per_device_batch_size']}")
    print(f"  Gradient Accum:       {info['gradient_accumulation_steps']}")
    print(f"  Num GPUs:             {info['num_gpus']}")
    print(f"  Effective Batch Size: {info['effective_batch_size']}")

    # Step Calculation
    print(f"\n[Step Calculation]")
    print(f"  Steps per Epoch:      {format_number(info['steps_per_epoch'])}")
    print(f"  Num Epochs:          {info['num_train_epochs']}")
    print(f"  Warmup Steps:         {info['warmup_steps']}")
    print(f"  Total Training Steps: {format_number(info['total_training_steps'])}")

    # Key Config
    print(f"\n[Key Config]")
    print(f"  Max Train Samples:   {info['max_train_samples']}")
    print(f"  Max Eval Samples:    {info['max_eval_samples']}")
    print(f"  Learning Rate:       {info['learning_rate']}")
    print(f"  Output Dir:          {info['output_dir']}")

    # Additional info
    if verbose:
        print(f"\n[Additional Info]")
        for key, value in info.items():
            if key not in [
                "data_dir",
                "actual_samples",
                "config_max_samples",
                "use_samples",
                "final_samples",
                "validation_split",
                "per_device_batch_size",
                "gradient_accumulation_steps",
                "num_gpus",
                "effective_batch_size",
                "steps_per_epoch",
                "num_train_epochs",
                "warmup_steps",
                "total_training_steps",
                "max_train_samples",
                "max_eval_samples",
                "learning_rate",
                "output_dir",
            ]:
                print(f"  {key}: {value}")

    print(f"\n{separator}\n")


def reverse_calculate_epochs(total_target_steps: int, info: dict) -> float:
    """根据目标步数反推需要的 epoch 数"""
    if info["steps_per_epoch"] <= 0:
        return 0
    return total_target_steps / info["steps_per_epoch"]


def main():
    parser = argparse.ArgumentParser(
        description="Training Steps Calculator - Calculate training steps from YAML config"
    )
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--num-gpus",
        "-g",
        type=int,
        default=None,
        help="Number of GPUs (default: auto-detect)",
    )
    parser.add_argument(
        "--max-samples",
        "-m",
        type=int,
        default=None,
        help="Override max_train_samples (-1 means use all data)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        choices=["print", "update"],
        default="print",
        help="Output mode: print (default) or update (modify config file)",
    )
    parser.add_argument(
        "--target-steps",
        "-t",
        type=int,
        default=None,
        help="Target training steps (used with --output update to calculate epochs)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # 加载配置
    config = load_config(args.config)

    # 检测 GPU 数量
    num_gpus = args.num_gpus if args.num_gpus else detect_num_gpus()

    # 计算步数
    info = calculate_steps(config, num_gpus, args.max_samples)

    # 打印报告
    print_report(info, args.verbose)

    # 更新配置文件
    if args.output == "update":
        if args.target_steps:
            # 根据目标步数反推 epoch
            new_epochs = reverse_calculate_epochs(args.target_steps, info)
            if new_epochs > 0:
                config["num_train_epochs"] = round(new_epochs, 2)
                print(
                    f"[Update] Set num_train_epochs to {config['num_train_epochs']} "
                    f"(to achieve ~{args.target_steps} steps)"
                )
            else:
                print(
                    f"[Warning] Cannot calculate epochs from steps_per_epoch={info['steps_per_epoch']}"
                )
        else:
            # 直接使用计算出的总步数更新 num_train_epochs
            if info["steps_per_epoch"] > 0:
                config["num_train_epochs"] = info["num_train_epochs"]
                print(
                    f"[Update] Set num_train_epochs to {config['num_train_epochs']} "
                    f"(calculated from {info['final_samples']} samples)"
                )
            else:
                print(f"[Warning] Cannot update: steps_per_epoch is 0")

        # 保存配置
        config_abs_path = os.path.abspath(args.config)
        save_config(config_abs_path, config)
        print(f"[Update] Config saved to: {config_abs_path}")


if __name__ == "__main__":
    main()
