"""
Convert alpaca dataset into sharegpt format.

Usage: python convert_dataset.py --in_dir /path/to/input_folder --out_dir /path/to/output_folder --data_type alpaca --file_type json
"""

import argparse
import os

from datasets import load_dataset


def process_file(in_file, out_file, data_type, file_type):
    data_files = {"train": in_file}
    if file_type == "csv":
        if data_type in ["qa"]:
            column_names = ["input", "output"]
        else:
            column_names = ["instruction", "input", "output"]
        raw_datasets = load_dataset(
            "csv", data_files=data_files, column_names=column_names, delimiter="\t"
        )
    elif file_type in ["json", "jsonl"]:
        raw_datasets = load_dataset("json", data_files=data_files)
    else:
        raise ValueError("File type not supported")
    ds = raw_datasets["train"]

    def process_qa(examples):
        convs = []
        for q, a in zip(examples["input"], examples["output"]):
            convs.append([{"from": "human", "value": q}, {"from": "gpt", "value": a}])
        return {"conversations": convs}

    def process_alpaca(examples):
        convs = []
        for instruction, inp, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            if inp and len(inp.strip()) > 0:
                instruction = instruction + "\n\n" + inp
            q = instruction
            a = output
            convs.append([{"from": "human", "value": q}, {"from": "gpt", "value": a}])
        return {"conversations": convs}

    if data_type in ["alpaca"]:
        ds = ds.map(
            process_alpaca,
            batched=True,
            remove_columns=ds.column_names,
            desc="Running process",
        )
    elif data_type in ["qa"]:
        ds = ds.map(
            process_qa,
            batched=True,
            remove_columns=ds.column_names,
            desc="Running process",
        )
    else:
        if "items" in ds.column_names:
            ds = ds.rename(columns={"items": "conversations"})
        columns_to_remove = ds.column_names.copy()
        columns_to_remove.remove("conversations")
        ds = ds.remove_columns(columns_to_remove)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    ds.to_json(out_file, lines=True, force_ascii=False)
    print(f"Converted: {in_file} -> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="input directory containing files to convert",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="output directory, default same as input",
    )
    parser.add_argument(
        "--data_type", type=str, default="alpaca", help="alpaca, qa, or sharegpt"
    )
    parser.add_argument(
        "--file_type",
        type=str,
        default="json",
        help="input file type, json, jsonl or csv",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.in_dir

    supported_extensions = {".json", ".jsonl", ".csv"}
    file_type_map = {".json": "json", ".jsonl": "jsonl", ".csv": "csv"}

    converted_count = 0
    for root, dirs, files in os.walk(args.in_dir):
        rel_dir = os.path.relpath(root, args.in_dir)

        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported_extensions:
                continue

            in_file = os.path.join(root, filename)
            name_without_ext = os.path.splitext(filename)[0]
            out_filename = f"{name_without_ext}_convert{ext}"

            if rel_dir == ".":
                out_file = os.path.join(args.out_dir, out_filename)
            else:
                out_file = os.path.join(args.out_dir, rel_dir, out_filename)

            try:
                file_type = file_type_map.get(ext, args.file_type)
                process_file(in_file, out_file, args.data_type, file_type)
                converted_count += 1
            except Exception as e:
                print(f"Error converting {in_file}: {e}")

    print(f"\nConversion complete! {converted_count} files converted.")
