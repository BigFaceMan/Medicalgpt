#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查看 Qwen2.5 Tokenizer 的具体实现
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

print("=" * 60)
print("Tokenizer 基础信息")
print("=" * 60)
print(f"类名: {tokenizer.__class__.__name__}")
print(f"模块: {tokenizer.__class__.__module__}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Model max length: {tokenizer.model_max_length}")

print("\n" + "=" * 60)
print("特殊 Token")
print("=" * 60)
print(f"BOS: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
print(f"EOS: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
print(f"PAD: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"UNK: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")

print("\n" + "=" * 60)
print("Tokenizer 类型判断")
print("=" * 60)

# 判断底层算法
if hasattr(tokenizer, "word_tokenizer"):
    print("Word-level tokenizer")
elif hasattr(tokenizer, "subword_tokenizer"):
    print("Subword tokenizer")

# BPE 相关属性
if hasattr(tokenizer, "byte_encoder"):
    print("有 byte_encoder -> Byte-level BPE")
if hasattr(tokenizer, "bpe_ranks"):
    print(f"BPE ranks 数量: {len(tokenizer.bpe_ranks)}")
if hasattr(tokenizer, "bpe"):
    print("使用 BPE 合并规则")

# SentencePiece 相关
if hasattr(tokenizer, "sp_model"):
    print("使用 SentencePiece (SPM)")

print("\n" + "=" * 60)
print("Tokenize 示例")
print("=" * 60)

text = "Hello, 你好 world!"
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"原文: {text}")
print(f"Tokens: {tokens}")
print(f"IDs: {ids}")

# 详细查看每个 token
print("\n" + "=" * 60)
print("每个 Token 详情")
print("=" * 60)
for i, (t, tid) in enumerate(zip(tokens, ids)):
    print(f"  {i}: '{t}' -> {tid}")

# 特殊token合并
print("\n" + "=" * 60)
print("BPE 合并规则查看 (前10个)")
print("=" * 60)
if hasattr(tokenizer, "bpe_ranks"):
    sorted_ranks = sorted(tokenizer.bpe_ranks.items(), key=lambda x: x[1])[:10]
    for (t1, t2), rank in sorted_ranks:
        print(f"  {rank}: {t1} + {t2} -> {t1 + t2}")

# vocabulary 样例
print("\n" + "=" * 60)
print("Vocabulary 样例 (前20个)")
print("=" * 60)
vocab = tokenizer.get_vocab()
for i, (k, v) in enumerate(list(vocab.items())[:20]):
    print(f"  '{k}': {v}")
