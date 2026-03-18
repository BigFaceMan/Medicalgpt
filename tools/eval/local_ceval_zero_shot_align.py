import argparse
import json
from typing import Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

CHOICES = ["A", "B", "C", "D"]


def build_prompt(doc: Dict) -> str:
    return (
        f"{doc['question'].strip()}\n"
        f"A. {doc['A']}\n"
        f"B. {doc['B']}\n"
        f"C. {doc['C']}\n"
        f"D. {doc['D']}\n"
        f"答案："
    )


def get_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {dtype_str}")


@torch.no_grad()
def score_completion(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
    add_special_tokens: bool = False,
) -> Dict[str, float]:
    """
    更贴近 lm-eval 的做法：
    1. 对 prompt + completion 整体 tokenize
    2. completion 单独 tokenize 只用于确定 continuation 长度
    3. 从 full_ids 的末尾 comp_len 个 token 作为 continuation
    """

    full_text = prompt + completion

    full_ids = tokenizer(
        full_text,
        add_special_tokens=add_special_tokens,
        return_tensors="pt"
    ).input_ids.to(device)

    completion_ids = tokenizer(
        completion,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids.to(device)

    comp_len = completion_ids.size(1)
    seq_len = full_ids.size(1)

    if comp_len <= 0:
        raise ValueError("completion 被 tokenize 后长度为 0，请检查 target_delimiter 或 tokenizer。")

    # 模型输出 logits[:, t, :] 预测的是 input_ids[:, t+1]
    outputs = model(input_ids=full_ids)
    logits = outputs.logits  # [1, seq_len, vocab]

    # continuation 是 full_ids 的最后 comp_len 个 token
    continuation_ids = full_ids[:, -comp_len:]  # [1, comp_len]

    # 对应这些 continuation token 的预测位置
    # 第一个 continuation token 由它前一个位置的 logits 来预测
    target_logits = logits[:, seq_len - comp_len - 1: seq_len - 1, :]  # [1, comp_len, vocab]

    if target_logits.size(1) != comp_len:
        raise RuntimeError(
            f"target_logits 长度 {target_logits.size(1)} != continuation 长度 {comp_len}"
        )

    log_probs = torch.log_softmax(target_logits, dim=-1)

    token_log_probs = log_probs.gather(
        dim=-1,
        index=continuation_ids.unsqueeze(-1)
    ).squeeze(-1)  # [1, comp_len]

    sum_logprob = token_log_probs.sum().item()
    avg_logprob = token_log_probs.mean().item()

    return {
        "sum_logprob": sum_logprob,
        "avg_logprob": avg_logprob,
        "num_tokens": comp_len,
        "continuation_token_ids": continuation_ids[0].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="本地模型目录")
    parser.add_argument("--subject", type=str, default="basic_medicine", help="C-Eval 科目名")
    parser.add_argument("--split", type=str, default="val", choices=["dev", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"]
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--target_delimiter",
        type=str,
        default=" ",
        help='默认用空格，更贴近 lm-eval；若想试严格模板可设为 ""'
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--save_path", type=str, default=None, help="可选，保存逐题结果到 jsonl")
    parser.add_argument(
        "--add_special_tokens",
        action="store_true",
        help="是否在 full_text 编码时加入 special tokens；默认不加"
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(args.dtype)

    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {args.model_path}")
    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.to(device)
    model.eval()

    print(f"Loading dataset: ceval/ceval-exam, subject={args.subject}, split={args.split}")
    ds = load_dataset("ceval/ceval-exam", name=args.subject)[args.split]

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"Num samples: {len(ds)}")

    results: List[Dict] = []
    correct_sum = 0
    correct_avg = 0

    for i, doc in enumerate(tqdm(ds, desc="Evaluating")):
        prompt = build_prompt(doc)

        choice_scores = {}
        for choice in CHOICES:
            completion = args.target_delimiter + choice
            s = score_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                completion=completion,
                device=device,
                add_special_tokens=args.add_special_tokens,
            )
            choice_scores[choice] = s

        pred_sum = max(CHOICES, key=lambda c: choice_scores[c]["sum_logprob"])
        pred_avg = max(CHOICES, key=lambda c: choice_scores[c]["avg_logprob"])
        gold = doc["answer"]

        is_correct_sum = int(pred_sum == gold)
        is_correct_avg = int(pred_avg == gold)

        correct_sum += is_correct_sum
        correct_avg += is_correct_avg

        row = {
            "idx": i,
            "id": doc.get("id", i),
            "question": doc["question"],
            "gold": gold,
            "pred_sum": pred_sum,
            "pred_avg": pred_avg,
            "correct_sum": is_correct_sum,
            "correct_avg": is_correct_avg,
            "scores": {
                c: {
                    "sum_logprob": choice_scores[c]["sum_logprob"],
                    "avg_logprob": choice_scores[c]["avg_logprob"],
                    "num_tokens": choice_scores[c]["num_tokens"],
                    "continuation_token_ids": choice_scores[c]["continuation_token_ids"],
                }
                for c in CHOICES
            }
        }
        results.append(row)

        if i < 5:
            print("=" * 80)
            print(f"[{i}] gold={gold} pred_sum={pred_sum} pred_avg={pred_avg}")
            print(prompt)
            for c in CHOICES:
                print(
                    f"  {c}: sum={choice_scores[c]['sum_logprob']:.6f} "
                    f"avg={choice_scores[c]['avg_logprob']:.6f} "
                    f"tokens={choice_scores[c]['num_tokens']} "
                    f"ids={choice_scores[c]['continuation_token_ids']}"
                )

    acc_sum = correct_sum / len(ds)
    acc_avg = correct_avg / len(ds)

    results.append({
        "idx": "final",
        "correct_sum": correct_sum,
        "correct_avg": correct_avg,
        "accuracy_sum": acc_sum,
        "accuracy_avg": acc_avg,
    })

    print("\nFinal Results")
    print(f"split={args.split}, subject={args.subject}, samples={len(ds)}")
    print(f"acc (sum logprob)      = {acc_sum:.4f}")
    print(f"acc_norm (avg logprob) = {acc_avg:.4f}")

    if args.save_path:
        with open(args.save_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved per-sample results to: {args.save_path}")


if __name__ == "__main__":
    main()