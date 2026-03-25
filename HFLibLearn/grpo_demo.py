import time
from contextlib import contextmanager

import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward


class Timer:
    def __init__(self, name):
        self.name = name
        self.times = {}

    @contextmanager
    def measure(self, step_name):
        start = time.time()
        yield
        elapsed = time.time() - start
        if step_name not in self.times:
            self.times[step_name] = []
        self.times[step_name].append(elapsed)

    def summary(self):
        for name, times in self.times.items():
            avg = sum(times) / len(times)
            max_t = max(times)
            min_t = min(times)
            print(
                f"[Timer] {name}: avg={avg:.3f}s, min={min_t:.3f}s, max={max_t:.3f}s, count={len(times)}"
            )


class ProfilingGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = Timer("GRPO_Training")
        self.gpu_util_history = []

    def training_step(self, model, inputs, num_items_in_batch=None):
        with self.timer.measure("total_step"):
            result = super().training_step(model, inputs, num_items_in_batch)

        if self.state.global_step % 10 == 0:
            print(f"\n{'=' * 50}")
            print(f"Step {self.state.global_step} Summary:")
            self.timer.summary()
            self._print_gpu_stats()
            print(f"{'=' * 50}\n")

        return result

    def _print_gpu_stats(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(
                    f"[GPU {i}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
                )


dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

print(f"数据集大小: {len(dataset)}")
print(f"数据集列名: {dataset.column_names}")
print(f"数据集特征: {dataset.features}")

print("\n=== 第一个样本 ===")
print(dataset[0])

print("\n=== 查看 prompt/question 字段 ===")
for i in range(min(3, len(dataset))):
    sample = dataset[i]
    print(f"\n--- 样本 {i} ---")
    print(
        f"prompt/question: {sample.get('prompt', sample.get('question', 'N/A'))[:200]}..."
    )
    print(
        f"answer/solution: {sample.get('answer', sample.get('solution', 'N/A'))[:200]}..."
    )

for sample in tqdm(dataset):
    if "prompt" in sample:
        for question in sample["prompt"]:
            if question["role"] != "user":
                print(f"role : {question['role']}question: {question['content']}")

parser = argparse.ArgumentParser()
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--num_generations", type=int, default=4)
parser.add_argument("--dataloader_num_workers", type=int, default=8)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
args = parser.parse_args()

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)

trainer.train()

