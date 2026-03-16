# MedicalGPT 模块分析

本文档详细分析 MedicalGPT 项目的各个模块实现原理和使用方法。

## 目录

- [一、训练模块](#一训练模块)
  - [1.1 预训练 (pretraining.py)](#11-预训练-pretrainingpy)
  - [1.2 有监督微调 SFT (supervised_finetuning.py)](#12-有监督微调-sft-supervised_finetuningpy)
  - [1.3 奖励建模 (reward_modeling.py)](#13-奖励建模-reward_modelingpy)
  - [1.4 PPO强化学习 (ppo_training.py)](#14-ppo强化学习-ppo_trainingpy)
  - [1.5 DPO训练 (dpo_training.py)](#15-dpo训练-dpo_trainingpy)
  - [1.6 ORPO训练 (orpo_training.py)](#16-orpo训练-orpo_trainingpy)
  - [1.7 GRPO训练 (grpo_training.py)](#17-grpo训练-grpo_trainingpy)
- [二、推理部署模块](#二推理部署模块)
  - [2.1 推理 (inference.py)](#21-推理-inferencepy)
  - [2.2 OpenAI兼容API (openai_api.py)](#22-openai兼容api-openai_apipy)
  - [2.3 Gradio Web UI (gradio_demo.py)](#23-gradio-web-ui-gradio_demopy)
  - [2.4 FastAPI服务 (fastapi_server_demo.py)](#24-fastapi服务-fastapi_server_demopy)
- [三、工具模块](#三工具模块)
  - [3.1 聊天模板 (template.py)](#31-聊天模板-templatepy)
  - [3.2 RAG问答 (chatpdf.py)](#32-rag问答-chatpdfpy)
  - [3.3 模型量化 (model_quant.py)](#33-模型量化-model_quantpy)
  - [3.4 LoRA合并 (merge_peft_adapter.py)](#34-lora合并-merge_peft_adapterpy)
- [四、数据流程](#四数据流程)
- [五、技术栈总结](#五技术栈总结)

---

## 一、训练模块

### 1.1 预训练 (pretraining.py)

**作用**：增量预训练/Continual Pretraining，在领域数据上继续训练基础模型，让模型学习领域知识分布。

**实现原理**：

```python
# 核心训练流程
1. 加载基础模型: AutoModelForCausalLM.from_pretrained()
2. 加载 tokenizer，设置 block_size 分块
3. 加载文本数据，按 block_size 分词
4. 使用 CausalLanguageModelingTrainer 进行训练
5. 预测下一个 token，计算交叉熵损失
```

**关键参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_name_or_path` | 基础模型路径 | 必填 |
| `train_file_dir` | 训练数据目录 | 必填 |
| `block_size` | 训练块大小 | 1024 |
| `learning_rate` | 学习率 | 1e-4 |
| `load_in_4bit` | 4bit量化 | False |
| `load_in_8bit` | 8bit量化 | False |

**数据格式**：原始文本文件 (.txt)，每行一条数据。

**使用示例**：

```bash
python pretraining.py \
    --model_name_or_path /path/to/base/model \
    --train_file_dir data/pretrain \
    --block_size 1024 \
    --output_dir outputs/pt
```

---

### 1.2 有监督微调 SFT (supervised_finetuning.py)

**作用**：指令微调 (Supervised Fine-tuning)，让模型学习遵循指令，理解任务意图。

**实现原理**：

```python
# 核心训练流程
1. 加载基础模型和 tokenizer
2. 使用模板 (template.py) 格式化对话数据
3. 使用 DataCollatorForSeq2Seq 批量处理
4. 计算 seq2seq 损失 (只计算 assistant 回复部分)
5. 使用 HuggingFace Trainer 进行训练
```

**支持的高级训练技术**：

| 技术 | 参数 | 说明 |
|------|------|------|
| FlashAttention-2 | `flash_attn` | 加速 attention 计算 |
| S²-Attn | `shift_attn` | LongLoRA 移位稀疏注意力 |
| NEFTune | `neft_alpha` | 给 embedding 加噪声防止过拟合 |
| RoPE Scaling | `rope_scaling` | 扩展上下文长度 |
| Gradient Checkpointing | `gradient_checkpointing` | 节省显存 |

**数据格式**：JSONL，每条包含多轮对话：

```json
{
  "conversation": [
    {"role": "user", "content": "问题1"},
    {"role": "assistant", "content": "回答1"},
    {"role": "user", "content": "问题2"},
    {"role": "assistant", "content": "回答2"}
  ]
}
```

**使用示例**：

```bash
python supervised_finetuning.py \
    --model_name_or_path /path/to/pt/model \
    --train_file_dir data/finetune \
    --template_name vicuna \
    --output_dir outputs/sft \
    --flash_attn true \
    --neft_alpha 5
```

---

### 1.3 奖励建模 (reward_modeling.py)

**作用**：训练奖励模型 (Reward Model)，用于评估生成文本的质量和人类偏好。

**实现原理**：

```python
# 核心训练流程
1. 加载预训练模型，替换为 SequenceClassification 头
2. 构造 pairwise 数据 (chosen, rejected)
3. 使用对比损失训练：让 chosen 分数高于 rejected
4. 输出奖励分数用于后续 PPO 训练
```

**模型结构**：

```python
# 奖励模型输出单分数
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    num_labels=1  # 输出单分数
)
```

**数据格式**：

```json
{
  "prompt": "用户问题",
  "chosen": "好的回答",
  "rejected": "差的回答"
}
```

**损失函数**：

```python
# Pairwise 损失
loss = -log(sigmoid(chosen_reward - rejected_reward))
# 或使用 margin ranking loss
```

---

### 1.4 PPO强化学习 (ppo_training.py)

**作用**：基于人类反馈的强化学习 (RLHF)，使用 PPO 算法让模型生成更高质量的回答。

**实现原理**：

```python
# 核心训练流程 (使用 TRL PPOTrainer)
1. 加载三个模型:
   - Policy: 要训练的 SFT 模型
   - Reference: 参考模型 (防止偏离 SFT 太远)
   - Reward: 奖励模型
2. PPO 训练循环:
   for batch in dataloader:
     - Policy 生成 response
     - Reward 模型评分
     - PPO 更新策略 (最大化 reward + 最小化 KL 散度)
```

**训练流程图**：

```
Prompt → Policy 生成 → Reward 评分 → PPO 更新策略
         ↓
    Reference Model (KL 约束)
```

**关键配置**：

```python
ppo_config = {
    "learning_rate": 1e-5,
    "ppo_epochs": 4,
    "mini_batch_size": 1,
    "batch_size": 16,
}
```

**使用示例**：

```bash
python ppo_training.py \
    --sft_model_path outputs/sft \
    --reward_model_path outputs/rm \
    --output_dir outputs/ppo
```

---

### 1.5 DPO训练 (dpo_training.py)

**作用**：直接偏好优化 (Direct Preference Optimization)，无需显式训练奖励模型，直接从偏好数据学习。

**实现原理**：

```python
# 核心训练流程 (使用 TRL DPOTrainer)
1. 加载 SFT 模型
2. 构造偏好数据 (chosen, rejected)
3. 使用 DPO 损失直接优化:
   loss = -log(sigmoid(logits_chosen - logits_rejected))
4. 不需要单独的奖励模型
```

**DPO 损失函数**：

```python
# DPO 目标函数
π_ref = reference_model(prompt)
loss = -E[(prompt,chosen,rejected)] [
    log(σ(r_chosen - r_rejected)) +
    β * (log(π_ref(chosen|prompt)) - log(π(chosen|prompt)))
]
```

**与 PPO 对比**：

| 特性 | DPO | PPO |
|------|-----|-----|
| 需要奖励模型 | 否 | 是 |
| 需要参考模型 | 可选 | 必须 |
| 训练复杂度 | 简单 | 复杂 |
| 显存需求 | 低 | 高 |
| 训练稳定性 | 好 | 一般 |

**数据格式**：

```json
{
  "prompt": "用户问题",
  "chosen": "好的回答",
  "rejected": "差的回答"
}
```

**使用示例**：

```bash
python dpo_training.py \
    --model_name_or_path outputs/sft \
    --train_file_dir data/dpo \
    --output_dir outputs/dpo
```

---

### 1.6 ORPO训练 (orpo_training.py)

**作用**：单阶段偏好优化 (Odds Ratio Preference Optimization)，无需参考模型，将 SFT 和对齐合并为单一步骤。

**实现原理**：

```python
# 核心训练流程 (使用 TRL ORPOTrainer)
1. 加载基础模型
2. 同时进行:
   - SFT 损失: 预测 assistant 回复
   - ORPO 损失: odds ratio 偏好优化
3. 缓解灾难性遗忘问题
```

**ORPO 损失**：

```python
# Odds Ratio 损失
odds_ratio = (p_chosen / (1 - p_chosen)) / (p_rejected / (1 - p_rejected))
loss = -log(sigmoid(log(odds_ratio)))
```

**特点**：

- 不需要参考模型 (ref_model)
- 单阶段完成 SFT + 对齐
- 缓解灾难性遗忘

**使用示例**：

```bash
python orpo_training.py \
    --model_name_or_path /path/to/base/model \
    --train_file_dir data/orpo \
    --output_dir outputs/orpo
```

---

### 1.7 GRPO训练 (grpo_training.py)

**作用**：组相对偏好优化 (Group Relative Preference Optimization)，用于 R1 模型训练，支持纯 RL 训练体验 "aha moment"。

**实现原理**：

```python
# 核心训练流程 (使用 TRL GRPOTrainer)
1. 加载 SFT 模型
2. 对每个 prompt 生成多个 response
3. 使用自定义奖励函数评分
4. GRPO 更新: 相对偏好优化
```

**自定义奖励函数**：

```python
# 准确性奖励
def accuracy_reward(completions, answer):
    # 提取 <answer> 标签内容
    # 与标准答案比较
    return reward  # 1.0 或 0.0

# 格式奖励
def format_reward(completions):
    # 检查是否包含 <answer> 标签
    return reward
```

**特点**：

- 无需参考模型
- 支持 GSM8K、MATH 等数学推理数据集
- 可实现纯 RL 训练

**使用示例**：

```bash
python grpo_training.py \
    --model_name_or_path outputs/sft \
    --dataset_name openai/gsm8k \
    --output_dir outputs/grpo
```

---

## 二、推理部署模块

### 2.1 推理 (inference.py)

**作用**：模型推理，支持单句和批量推理。

**实现方式**：

```python
# 流式输出
@torch.inference_mode()
def stream_generate_answer(model, tokenizer, prompt, ...):
    streamer = TextIteratorStreamer(tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt")
    thread = Thread(target=model.generate, kwargs={..., streamer})
    thread.start()
    for new_text in streamer:
        print(new_text, end="")

# 批量推理
def batch_generate_answer(sentences, model, tokenizer, ...):
    # 使用 apply_chat_template 构造 prompt
    # 批量生成
    return generated_texts
```

**支持的功能**：

- LoRA 权重加载 (`PeftModel`)
- 4bit/8bit 量化加载
- 流式输出
- 批量推理
- 自定义停止符

**使用示例**：

```python
from inference import stream_generate_answer

model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)

prompt = "用户: 你好\n助手:"
stream_generate_answer(model, tokenizer, prompt)
```

---

### 2.2 OpenAI兼容API (openai_api.py)

**作用**：提供 OpenAI 兼容的 REST API 服务，支持 ChatGPT 调用方式。

**实现方式**：

```python
# 基于 FastAPI
app = FastAPI()

# 支持的接口
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 处理 ChatML 格式
    # 流式/非流式生成
    return response

@app.get("/v1/models")
async def list_models():
    return ModelList(data=[...])
```

**支持的 API**：

| 接口 | 说明 |
|------|------|
| `/v1/chat/completions` | 聊天完成 (支持流式) |
| `/v1/completions` | 文本补全 |
| `/v1/models` | 模型列表 |
| `/v1/embeddings` | 向量嵌入 |

**使用示例**：

```bash
# 启动服务
python openai_api.py --model_path /path/to/model --port 8000

# 调用 API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "medical-gpt",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false
  }'
```

---

### 2.3 Gradio Web UI (gradio_demo.py)

**作用**：基于 Gradio 的交互式 Web 界面。

**实现方式**：

```python
import gradio as gr

def predict(message, history):
    # 流式生成
    for token in model.generate(message):
        yield token

demo = gr.ChatInterface(
    fn=predict,
    title="MedicalGPT",
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="请输入问题..."),
)
demo.launch()
```

---

### 2.4 FastAPI服务 (fastapi_server_demo.py)

**作用**：轻量级 FastAPI 推理服务。

**使用示例**：

```bash
python fastapi_server_demo.py --model_path /path/to/model
```

---

## 三、工具模块

### 3.1 聊天模板 (template.py)

**作用**：定义不同模型的对话格式模板，确保模型正确理解对话结构。

**实现方式**：

```python
# 注册模板
conv_templates = {}

def register_conv_template(template):
    conv_templates[template.name] = template

# Vicuna 模板
register_conv_template(Conversation(
    name="vicuna",
    system_prompt="A chat between a curious user...",
    prompt="USER: {query} ASSISTANT:",
    sep="</s>",
))
```

**支持的模板**：

- `vicuna` - Vicuna 系列
- `chatglm` - ChatGLM 系列
- `baichuan` - Baichuan 系列
- `qwen` - Qwen 系列
- `llama` - LLaMA 系列
- 等等...

**使用示例**：

```python
from template import get_conv_template

template = get_conv_template("vicuna")
prompt = template.get_prompt(messages)
```

---

### 3.2 RAG问答 (chatpdf.py)

**作用**：基于 PDF 文档的检索增强生成 (RAG) 问答系统。

**实现方式**：

```python
# 1. 文档加载与分块
class SentenceSplitter:
    def split_text(self, text):
        # 中文: jieba 分词
        # 英文: 正则按句子分割
        return chunks

# 2. 向量检索
similarity = EnsembleSimilarity(
    BertSimilarity(),
    BM25Similarity()
)

# 3. RAG 生成
RAG_PROMPT = """基于以下已知信息回答用户问题。
已知内容: {context_str}
问题: {query_str}
"""
```

**RAG 流程**：

```
PDF 文档
    ↓
SentenceSplitter 分块
    ↓
BertSimilarity + BM25Similarity 向量化
    ↓
向量数据库存储
    ↓
用户查询 → 向量检索 → Top-K 相关块
    ↓
拼接 prompt → LLM 生成答案
```

---

### 3.3 模型量化 (model_quant.py)

**作用**：模型量化，减小模型体积和显存占用，提升推理速度。

**实现方式**：

```python
# 4bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)
```

**量化类型**：

| 类型 | 显存减少 | 精度损失 |
|------|----------|----------|
| FP16 | - | 基准 |
| INT8 | ~50% | 较小 |
| INT4 | ~75% | 中等 |

---

### 3.4 LoRA合并 (merge_peft_adapter.py)

**作用**：将 LoRA 适配器权重合并回基础模型，生成可直接使用的完整模型。

**实现方式**：

```python
# 加载 LoRA
peft_config = PeftConfig.from_pretrained(lora_path)
base_model = AutoModelForCausalLM.from_pretrained(base_path)
model = PeftModel.from_pretrained(base_model, lora_path)

# 合并权重
merged_model = model.merge_and_unload()
merged_model.save_pretrained(output_dir)
```

---

## 四、数据流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      MedicalGPT 完整训练流程                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                                               │
│  │  1. 预训练   │  pretraining.py                               │
│  │  PT/Continual│  领域文档 → 增量预训练 → 领域基础模型          │
│  └──────┬───────┘                                               │
│         ↓                                                        │
│  ┌──────────────┐                                               │
│  │  2. SFT      │  supervised_finetuning.py                     │
│  │  指令微调    │  指令数据 → 有监督微调 → SFT模型               │
│  └──────┬───────┘                                               │
│         ↓                                                        │
│  ┌──────────────────────────────────────────┐                   │
│  │           3. 对齐训练 (三选一)             │                   │
│  ├─────────────┬─────────────┬──────────────┤                   │
│  │  RLHF       │  DPO/ORPO   │  GRPO        │                   │
│  │  奖励模型+PPO│  直接偏好   │  组相对偏好  │                   │
│  │  (复杂稳定)  │  (简单高效)  │  (R1模型)    │                   │
│  └─────────────┴─────────────┴──────────────┘                   │
│         ↓                                                        │
│  ┌──────────────┐                                               │
│  │  4. 部署推理  │  inference.py / openai_api.py                │
│  │  模型服务    │  API / Gradio / FastAPI                       │
│  └──────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、技术栈总结

### 核心框架

| 框架 | 用途 |
|------|------|
| HuggingFace Transformers | 模型加载、训练、推理 |
| TRL (Transformer Reinforcement Learning) | RLHF/DPO/ORPO/GRPO 训练 |
| PEFT | LoRA/QLoRA 高效微调 |
| DeepSpeed | 分布式训练优化 |
| Accelerate | 分布式训练 |

### 推理加速

| 技术 | 说明 |
|------|------|
| FlashAttention-2 | 高效 attention 计算 |
| vLLM | PagedAttention 高吞吐推理 |
| BitsAndBytes | 4bit/8bit 量化 |

### Web 框架

| 框架 | 用途 |
|------|------|
| FastAPI | REST API 服务 |
| Gradio | Web 交互界面 |

### 向量检索

| 库 | 说明 |
|-----|------|
| similarities | Bert + BM25 组合检索 |
| LangChain | RAG 框架 |

### 依赖库

```
transformers
torch
trl
peft
deepspeed
accelerate
fastapi
gradio
scikit-learn
datasets
loguru
```

---

*文档生成时间: 2025*
