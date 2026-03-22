import argparse
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "clm", "seq_cls"],
        help="模型类型: clm(因果语言模型), seq_cls(序列分类), auto(自动检测)",
    )
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)


    print("=== Config Info ===")
    print(f"model_type: {config.model_type}")
    print(f"hidden_size: {config.hidden_size}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"vocab_size: {config.vocab_size}")
    print()

    if args.model_type == "auto":
        model_type = (
            "clm"
            if config.model_type
            in ["qwen2", "llama", "gpt2", "bloom", "gptj", "gpt_neox", "qwen2_5"]
            else "seq_cls"
        )
    else:
        model_type = args.model_type

    print(f"=== Loading model as: {model_type} ===")
    print()

    if model_type == "clm":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",
        )

    print("=== Model Architecture ===")
    print(model)
    print()

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")


if __name__ == "__main__":
    main()
