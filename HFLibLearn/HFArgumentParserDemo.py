#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HfArgumentParser 使用示例
"""

from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-0.5B",
        metadata={"help": "Model name or path"}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Model dtype",
            "choices": ["auto", "bfloat16", "float16", "float32"]
        }
    )
    device_map: str = field(
        default="auto",
        metadata={"help": "Device map"}
    )


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default="output",
        metadata={"help": "Output directory"}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "Learning rate"}
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=3,
        metadata={"help": "Train batch size per device"}
    )
    save_steps: int = field(
        default=50,
        metadata={"help": "Save checkpoint every X steps"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X steps"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )[:2]
    
    print("=" * 50)
    print("Model Arguments:")
    print(f"  model_name_or_path: {model_args.model_name_or_path}")
    print(f"  torch_dtype: {model_args.torch_dtype}")
    print(f"  device_map: {model_args.device_map}")
    print("=" * 50)
    print("Training Arguments:")
    print(f"  output_dir: {training_args.output_dir}")
    print(f"  learning_rate: {training_args.learning_rate}")
    print(f"  num_train_epochs: {training_args.num_train_epochs}")
    print(f"  per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    print(f"  save_steps: {training_args.save_steps}")
    print(f"  logging_steps: {training_args.logging_steps}")
    print("=" * 50)


if __name__ == "__main__":
    main()