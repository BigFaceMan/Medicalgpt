from datasets import load_dataset
from loguru import logger
from glob import glob

extension = "text"
train_file_dir = "./data/pretrain"
valid_file_dir = "./data/pretrain"
data_files = {}
dataset_args = {}

train_data_files = glob(f'{train_file_dir}/**/*.txt', recursive=True) + glob(
    f'{train_file_dir}/**/*.json', recursive=True) + glob(
    f'{train_file_dir}/**/*.jsonl', recursive=True)
logger.info(f"train files: {train_data_files}")
# Train data files must be same type, e.g. all txt or all jsonl
types = [f.split('.')[-1] for f in train_data_files]
if len(set(types)) > 1:
    raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")
data_files["train"] = train_data_files


valid_data_files = glob(f'{valid_file_dir}/**/*.txt', recursive=True) + glob(
    f'{valid_file_dir}/**/*.json', recursive=True) + glob(
    f'{valid_file_dir}/**/*.jsonl', recursive=True)
logger.info(f"valid files: {valid_data_files}")
# Valid data files must be same type, e.g. all txt or all jsonl
types = [f.split('.')[-1] for f in valid_data_files]
if len(set(types)) > 1:
    raise ValueError(f"valid files must be same type, e.g. all txt or all jsonl, but got {types}")
data_files["validation"] = valid_data_files



dataset_args["keep_linebreaks"] = True



raw_datasets = load_dataset(
    extension,
    data_files=data_files,
    cache_dir=False,
    **dataset_args
)


print(f"数据集结构: {raw_datasets}")



print("train type")
print(type(raw_datasets["train"]))

# dict
print(type(raw_datasets["train"][0]))


print(raw_datasets["train"][0]["text"])

print(type(raw_datasets["train"][0]["text"]))

def tokenize_wo_pad_function(examples):
    logger.info(f"encoder input type: {type(examples['text'])}")
    logger.info(f"encoder input len: {len(examples['text'])}")
    logger.info(f"encoder input 0 type: {type(examples['text'][0])}")
    return examples


tokenized_datasets = raw_datasets.map(
    tokenize_wo_pad_function,
    batched=True,
)


