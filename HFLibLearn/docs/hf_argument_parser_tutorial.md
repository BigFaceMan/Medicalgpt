# HfArgumentParser 使用教程

本教程介绍如何在项目中使用 Hugging Face 的 `HfArgumentParser` 进行命令行参数解析。

## 目录

1. [什么是 HfArgumentParser](#1-什么是-hfargumentparser)
2. [基本用法](#2-基本用法)
3. [定义参数类](#3-定义参数类)
4. [解析参数](#4-解析参数)
5. [项目中的实际使用](#5-项目中的实际使用)
6. [常见参数类型](#6-常见参数类型)
7. [高级用法](#7-高级用法)

---

## 1. 什么是 HfArgumentParser

`HfArgumentParser` 是 Hugging Face **transformers** 库中的一个类，用于**命令行参数解析**。

### 作用

- 类似 Python 的 `argparse`，但专为 Hugging Face 训练脚本设计
- 自动从命令行解析参数，生成对应的 dataclass 类型的配置对象
- 支持多个 dataclass 组合解析
- 自动生成 `--help` 参数说明

### 安装

```python
from transformers import HfArgumentParser
```

---

## 2. 基本用法

### 简单示例

```python
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class TrainingArguments:
    learning_rate: float = field(default=2e-4, metadata={"help": "Learning rate"})
    num_train_epochs: int = field(default=1, metadata={"help": "Number of training epochs"})
    output_dir: str = field(default="output", metadata={"help": "Output directory"})

parser = HfArgumentParser(TrainingArguments)
args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

print(args.learning_rate)  # 2e-4
print(args.num_train_epochs)  # 1
print(args.output_dir)  # "output"
```

### 命令行调用

```bash
python train.py --learning_rate 1e-4 --num_train_epochs 3 --output_dir my_output
```

---

## 3. 定义参数类

### 使用 `@dataclass` 装饰器

```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The tokenizer for weights initialization."},
    )
    load_in_8bit: bool = field(
        default=False, 
        metadata={"help": "Whether to load the model in 8bit mode."}
    )
    load_in_4bit: bool = field(
        default=False, 
        metadata={"help": "Whether to load the model in 4bit mode."}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
```

### `field()` 参数说明

| 参数 | 说明 |
|------|------|
| `default` | 默认值 |
| `default_factory` | 默认值工厂（用于可变对象如 list） |
| `metadata` | 帮助信息，`help` 用于 `--help` 输出 |
| `init` | 是否作为 `__init__` 参数 |

### `metadata` 常用字段

```python
@dataclass
class ExampleArguments:
    # 简单类型
    num: int = field(default=10, metadata={"help": "A number"})
    
    # 字符串选项
    mode: str = field(
        default="train",
        metadata={
            "help": "Mode to run",
            "choices": ["train", "eval", "test"]
        }
    )
    
    # 布尔类型
    use_peft: bool = field(
        default=True,
        metadata={"help": "Whether to use PEFT"}
    )
    
    # 可选字符串
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory"}
    )
    
    # 列表类型
    target_modules: Optional[str] = field(
        default="all",
        metadata={"help": "Target modules for LoRA, comma separated"}
    )
```

---

## 4. 解析参数

### 方法一：`parse_args_into_dataclasses`

```python
from transformers import HfArgumentParser

# 单个 dataclass
parser = HfArgumentParser(TrainingArguments)
args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

# 多个 dataclass
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses(
    return_remaining_strings=True
)[:3]
```

### 方法二：`parse_args`

```python
# 返回 namespace 对象
parser = HfArgumentParser(TrainingArguments)
args = parser.parse_args()

# 访问属性
print(args.learning_rate)
```

### 保留剩余字符串

```python
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args, remaining = parser.parse_args_into_dataclasses(
    return_remaining_strings=True
)

# remaining 包含未解析的参数
print(remaining)  # ['--extra_arg', 'value']
```

---

## 5. 项目中的实际使用

### 5.1 预训练脚本 (pretraining.py)

文件位置: `pretraining.py:353-356`

```python
from transformers import HfArgumentParser

# 定义多个参数类
parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))
model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses(
    return_remaining_strings=True
)[:4]

# 打印参数
logger.info(f"Model args: {model_args}")
logger.info(f"Data args: {data_args}")
logger.info(f"Training args: {training_args}")
logger.info(f"Script args: {script_args}")
```

### 5.2 SFT 脚本 (supervised_finetuning.py)

```python
parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))
model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses(
    return_remaining_strings=True
)[:4]
```

### 5.3 DPO 训练 (dpo_training.py)

```python
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
```

### 5.4 PPO 训练 (ppo_training.py)

```python
parser = HfArgumentParser((PPOArguments, PPOConfig, ModelConfig))
ppo_args, ppo_config, model_config = parser.parse_args_into_dataclasses(
    return_remaining_strings=True
)[:3]
```

---

## 6. 常见参数类型

### 6.1 ModelArguments - 模型参数

```python
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The tokenizer for weights initialization."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models."}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to."}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code."}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use fast tokenizer."}
    )
```

### 6.2 DataArguments - 数据参数

```python
@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use."}
    )
    train_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The train data file folder."}
    )
    validation_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The validation data file folder."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Max training samples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Max evaluation samples."}
    )
    block_size: Optional[int] = field(
        default=1024,
        metadata={"help": "Input sequence length after tokenization."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes for preprocessing."}
    )
```

### 6.3 ScriptArguments - 脚本自定义参数

```python
@dataclass
class ScriptArguments:
    use_peft: bool = field(
        default=True,
        metadata={"help": "Whether to use PEFT"}
    )
    target_modules: Optional[str] = field(
        default="all",
        metadata={"help": "Target modules for LoRA."}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "LoRA rank."}
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "LoRA dropout."}
    )
    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={"help": "LoRA alpha."}
    )
    modules_to_save: Optional[str] = field(
        default=None,
        metadata={"help": "Modules to save besides LoRA."}
    )
    peft_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to PEFT adapter."}
    )
```

---

## 7. 高级用法

### 7.1 条件验证

```python
@dataclass
class TrainingArguments:
    learning_rate: float = field(default=2e-4)
    
    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
```

### 7.2 互斥参数

```python
@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None)
    train_file_dir: Optional[str] = field(default=None)
    
    def __post_init__(self):
        if self.dataset_name is None and self.train_file_dir is None:
            raise ValueError("Must specify either dataset_name or train_file_dir")
```

### 7.3 从文件加载参数

```python
# 可以将参数保存到 JSON 文件
import json

# 保存
with open("args.json", "w") as f:
    json.dump(vars(training_args), f, indent=2)

# 加载
parser = HfArgumentParser(TrainingArguments)
args = parser.parse_json_file("args.json")
```

### 7.4 生成帮助信息

```bash
python pretraining.py --help
```

输出示例：
```
--model_name_or_path MODEL_NAME_OR_PATH
                        The model checkpoint for weights initialization.
--tokenizer_name_or_path TOKENIZER_NAME_OR_PATH
                        The tokenizer for weights initialization.
--learning_rate LEARNING_RATE
                        Learning rate to use for training.
...
```

---

## 完整示例

```python
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
```

运行：
```bash
python example.py --model_name_or_path Qwen/Qwen2.5-1.8B --learning_rate 1e-4
```

输出：
```
==================================================
Model Arguments:
  model_name_or_path: Qwen/Qwen2.5-1.8B
  torch_dtype: bfloat16
  device_map: auto
==================================================
Training Arguments:
  output_dir: output
  learning_rate: 0.0001
  num_train_epochs: 1
  per_device_train_batch_size: 3
  save_steps: 50
  logging_steps: 10
==================================================
```

---

## 相关链接

- [Hugging Face Transformers - ArgumentParser](https://huggingface.co/docs/transformers/en/internal/trainer_utils#transformers.HfArgumentParser)
- [transformers/examples](https://github.com/huggingface/transformers/tree/main/examples)
- [MedicalGPT 训练脚本](./pretraining.py)
