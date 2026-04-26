# ChineseBabyLM 模型训练深度分析与优化方案

> **文档版本**: v1.0 | **日期**: 2026-04-19 | **作者**: 项目组

---

## 目录

1. [训练结果概览](#1-训练结果概览)
2. [关键问题诊断](#2-关键问题诊断)
3. [Tokenizer 深度分析](#3-tokenizer-深度分析)
4. [架构升级方案](#4-架构升级方案)
5. [训练策略优化](#5-训练策略优化)
6. [数据优化方案](#6-数据优化方案)
7. [SOTA 实施路线图](#7-sota-实施路线图)
8. [总结与优先级排序](#8-总结与优先级排序)

---

## 1. 训练结果概览

### 1.1 训练配置

| 配置项 | 当前值 | 说明 |
|--------|--------|------|
| 模型架构 | GPT-2 Small | 768d, 12层, 12头 |
| 参数量 | ~109.9M | 含 embedding 层 |
| 词表大小 | 32,000 BPE | WhitespaceSplit + Punctuation |
| 序列长度 | 512 tokens | 学习到的绝对位置编码 |
| 训练数据 | 82.3M tokens | babylm-zho-100M |
| 训练样本 | 160,790 | block_size=512 |
| GPU | 3× RTX A6000 | 49GB VRAM each |
| 有效 Batch | 96 | 16 × 3 GPU × 2 grad_accum |
| 学习率 | 6e-4 → 周期性变化 | Cosine Scheduler（有Bug） |
| 训练时长 | ~3.8 小时 | 30,140 steps, 10 epochs |

### 1.2 Loss / PPL 演变

```
Epoch | Train Loss | Val Loss  | Val PPL  | 趋势
------|-----------|----------|---------|-------
  1   |   7.60    |   6.78   |   880   | ↓ 快速下降
  2   |   6.62    |   6.39   |   595   | ↓ 持续下降
  3   |   6.31    |   6.25   |   518   | ↓ 减速
  4   |   ~6.18   |   ~6.20  |   ~490  | → 平台期（LR→0）
  5   |   ~6.18   |   ~6.20  |   ~490  | → 平台期
  6   |   ~6.18   |   6.15   |   ~468  | → 平台期
  7   |   6.06    |   6.03   |   414   | ↓ 新LR周期下降
  8   |   5.88    |   5.90   |   365   | ↓ 持续下降
  9   |   5.68    |  5.84    |   343   | ↓ 最佳
  10  |   5.57    |   5.85   |   347   | → 轻微过拟合
```

**Best Model**: Epoch 9, Val Loss = 5.8419, Val PPL ≈ 343

### 1.3 实际评测结果（evaluate_model.py）

**评测配置**: 2000 行训练数据, block_size=512, 8235 chunks, 4,216,320 tokens

| 指标 | 值 | 说明 |
|------|-----|------|
| **实际 PPL** | **1352.16** | 极其差（比训练时 Val PPL 343 还差 4 倍） |
| **实际 Loss** | **7.2095** | 比训练时 Val Loss 5.84 高出很多 |

> ⚠️ **注意**: PPL 在训练数据上居然高达 1352，而训练结束时 Val PPL 报告为 343。这个巨大差异可能是由于：
> 1. 训练时的 loss 计算方式与评测时不一致（padding/mask 处理差异）
> 2. 学习率 bug 导致模型欠拟合
> 3. Tokenizer 编码/解码问题

**Tokenizer 分析**:

| 测试文本 | Token 数 | Tokens |
|----------|----------|--------|
| "这是一个测试句子。" | 4 | ['这是一个', '测试', '句子', '。'] |
| "今天天气真好，我想出去玩。" | 6 | ['今天天气', '真好', **'\<unk\>'**, '我想', '出去玩', '。'] ⚠️ |
| "小猫在阳光下睡觉。" | 5 | ['小猫', '在', '阳光下', '睡觉', '。'] |
| "The quick brown fox..." | 25+ | 字符级分割 |
| "我喜欢吃苹果和香蕉。" | 6 | ['我喜欢', '吃', '苹果', '和', '香蕉', '。'] |

**关键发现**: 中文逗号 "，" 被编码为 `<unk>`！这说明 tokenizer 词表不完整。

**Token/字符比**: 0.541（看似合理，但隐藏了 `<unk>` 问题）

**文本生成质量**: 极差 ❌

```
Prompt: "今天"
生成: 今天,,来的人多呀有的,排队还有人有的。。,说,,排,是啊好像是,排,,,排是,,,,排嗯是。,的是,,,是嗯是不是哦还是,,,,有呀都有不是就是。,是一个。一个?,,,,,,,,一个,,,不是

Prompt: "中国的首都是"
生成: 中国的首都是,。,那个好像首是队对的那对对对对就是比赛对冠军?,首队美国的对...
```

**问题**:
1. 大量重复标点符号（，，，。。。）
2. 生成内容毫无语义连贯性
3. 无法回答简单事实问题（"中国的首都是" → 应该是"北京"）
4. 严重的重复循环问题

###

**修复方案**:

```python
# 正确的计算方式
num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
max_train_steps = args.num_epochs * num_update_steps_per_epoch
```

### 2.2 🔴 问题二：Tokenizer 对中文不友好（严重）

**当前 Tokenizer**: 使用 `WhitespaceSplit + Punctuation` 预分词。

**问题**: 
- 中文没有空格分隔词语，`WhitespaceSplit` 会将整个句子（到下一个空格或换行符）当作一个 "word"
- BPE 无法在单个超长 token 上有效学习
- 导致 tokenization 效率极低
- 32K 词表大量浪费在罕见长 token 上

**具体表现**:
- 整个中文句子可能被 tokenize 为少量超长 token
- Token/字符比异常
- 模型输入的信息密度降低

**修复方案**: 使用 `ByteLevel` 或字符级预分词。

### 2.3 🟡 问题三：GPT-2 架构过时

| 特性 | GPT-2 (当前) | LLaMA/Qwen2 (推荐) |
|------|-------------|-------------------|
| 位置编码 | 学习式绝对位置 | RoPE（旋转位置编码） |
| 注意力机制 | 标准多头注意力 | GQA (分组查询注意力) |
| 归一化 | LayerNorm (pre-norm) | RMSNorm |
| 激活函数 | GELU | SwiGLU |
| FFN 维度 | 4× hidden | 8/3 × hidden (更高效) |
| 上下文长度 | 512 (学习式) | 2048+ (RoPE 扩展) |
| 注意力缩放 | 无 | RoPE 自然衰减 |
| KV Cache | 标准实现 | GQA 减少内存 |

**影响**: 在相同参数量下，现代架构可降低 20-40% 的 PPL。

### 2.4 🟡 问题四：训练不充分

- 82.3M tokens, 10 epochs → 总共 ~823M token exposures
- 对于 110M 参数模型，数据量偏少
- 但比赛限制为 100M words，不能增加数据
- 需要通过更好的利用现有数据来提升效果

### 2.5 🟢 问题五：Dropout 偏高

当前 dropout = 0.1 对于小数据集训练是合理的，但在后期 epochs 可能限制了模型容量。
建议在后期训练阶段（如最后 3 epochs）降低 dropout 到 0.05 或 0。

---

## 3. Tokenizer 深度分析

### 3.1 当前 Tokenizer 问题详解

**当前预分词策略**: `WhitespaceSplit → Punctuation`

**中文 tokenize 过程**:

```
输入: "今天天气真好，我想出去玩。"
WhitespaceSplit: ["今天天气真好，我想出去玩。"]  ← 整句是一个 "word"!
Punctuation:     ["今天天气真好", "，", "我想出去玩", "。"]
BPE merge:       对上述 4 段分别进行 BPE
```

这意味着：
1. "今天天气真好" 作为一个整体被 BPE 处理
2. 如果这个词组在训练语料中出现不足 2 次，它会被逐字符拆分
3. 但这完全依赖于 BPE 的合并操作在训练语料中的频率
4. 中文字符有数千个常见字，32K 词表在字符级别是够用的
5. 但词级别的信息完全丢失

### 3.2 推荐 Tokenizer 方案

#### 方案 A: ByteLevel BPE（推荐）

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"],
    min_frequency=2,
    show_progress=True,
)
```

**优势**:
- ByteLevel 自动处理 UTF-8 编码，天然支持中文
- 不需要预先分词，直接在字节级别学习 BPE
- 与 GPT-2/Qwen 等模型完全兼容
- 每个中文字符约 3 个 UTF-8 字节，32K 词表足够覆盖

#### 方案 B: SentencePiece BPE（备选）

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='data/processed/train.txt',
    model_prefix='data/tokenizer/spiece',
    vocab_size=32000,
    character_coverage=0.9995,
    model_type='bpe',
    unk_id=0, bos_id=1, eos_id=2, pad_id=3,
)
```

**优势**:
- SentencePiece 是中文 NLP 的工业标准
- 自动学习子词分割，无需预分词
- `character_coverage=0.9995` 适合中文

#### 方案 C: 复用 Qwen2.5 Tokenizer（最佳但需确认合规）

直接使用 Qwen2.5-0.5B 的 tokenizer（151,643 词表），在 100M 词限制下可能有合规性问题。

### 3.3 词表大小建议

| 词表大小 | 参数影响 | 推荐度 |
|----------|---------|--------|
| 16,000 | embedding 少，但覆盖率可能不足 | ⭐⭐ |
| 32,000 | 平衡选择 | ⭐⭐⭐⭐ |
| 48,000 | 更好的中文覆盖 | ⭐⭐⭐ |
| 64,000 | 覆盖好但 embedding 开销大 | ⭐⭐⭐ |

**建议**: 维持 32K 词表但改用 ByteLevel BPE。对于 110M 参数模型，32K 词表的 embedding 层占 32K × 768 = 24.6M 参数，约占总参数的 22%，是合理的比例。

---

## 4. 架构升级方案

### 4.1 方案概览

从当前 GPT-2 架构升级到 LLaMA 风格架构，预计可将 Val PPL 从 ~343 降低到 ~30-80。

### 4.2 推荐: LLaMA-Small 架构

```python
from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    vocab_size=32000,
    hidden_size=768,        # 与当前一致
    intermediate_size=2048, # SwiGLU: 768 × 8/3 ≈ 2048 (而非 GPT-2 的 3072)
    num_hidden_layers=12,   # 与当前一致
    num_attention_heads=12, # 与当前一致
    num_key_value_heads=4,  # GQA: 12/4=3, 每个 KV 头服务 3 个 Q 头
    max_position_embeddings=2048,  # RoPE 支持更长上下文
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    tie_word_embeddings=False,     # 不共享 embedding（LLaMA 标准）
)
```

### 4.3 参数量对比

| 组件 | GPT-2 (当前) | LLaMA-Small |
|------|-------------|-------------|
| Token Embedding | 32K × 768 = 24.6M | 32K × 768 = 24.6M |
| Position Embedding | 512 × 768 = 0.4M | 0 (RoPE 无参数) |
| Per Layer | ~7.1M | ~6.3M (GQA 节省) |
| 12 Layers | ~85.2M | ~75.6M |
| LM Head | 0 (tied) | 24.6M (untied) |
| **Total** | **~110M** | **~125M** |

### 4.4 关键改进详解

#### (1) RoPE (旋转位置编码)

```
优势:
- 无需学习位置参数，节省 0.4M 参数
- 自然支持长度外推（训练 512，可推理 2048）
- 相对位置感知能力更强
- 对中文这种词序重要的语言特别有利
```

#### (2) GQA (分组查询注意力)

```
优势:
- KV Cache 减少 3× (12头→4 KV头)
- 推理速度更快
- 减少注意力计算量
- 在 110M 参数规模下不影响性能
```

#### (3) SwiGLU 激活

```
优势:
- 比 GELU 收敛更快
- 在相同计算量下 PPL 更低
- 已被 LLaMA/Mistral/Qwen 等验证
```

#### (4) RMSNorm

```
优势:
- 比 LayerNorm 计算更快
- 不需要计算均值，训练更稳定
- 已成为现代 LLM 标配
```

### 4.5 替代架构方案

| 方案 | 架构 | 参数量 | 预期PPL | 实现难度 |
|------|------|--------|---------|---------|
| **A (推荐)** | LLaMA-Small | ~125M | 30-60 | ⭐⭐ |
| B | Qwen2-style | ~130M | 25-55 | ⭐⭐⭐ |
| C | GPT-2 + 修复 | ~110M | 50-100 | ⭐ |
| D | Encoder-Decoder (T5) | ~120M | 30-50 | ⭐⭐⭐⭐ |

**推荐方案 A**: LLaMA-Small，实现最简单，效果提升最大。

### 4.6 实现代码骨架

```python
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

def create_model(vocab_size=32000):
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    return model

def create_tokenizer(data_path, save_path, vocab_size=32000):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        min_frequency=2,
    )
    tokenizer.train([data_path], trainer=trainer)
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    
    # 创建 HF 格式配置
    from transformers import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(save_path, "tokenizer.json"),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    hf_tokenizer.save_pretrained(save_path)
    return hf_tokenizer
```

---

## 5. 训练策略优化

### 5.1 LR Scheduler 修复

**当前 Bug**:
```python
# 错误: DDP 下 len(train_loader) 已经是 per-process 的
num_update_steps_per_epoch = len(train_loader) // accelerator.num_processes // args.gradient_accumulation_steps
```

**修复**:
```python
# 正确: 不需要除以 num_processes
num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
max_train_steps = args.num_epochs * num_update_steps_per_epoch
```

### 5.2 最优超参数建议

| 超参数 | 当前值 | 建议值 | 理由 |
|--------|--------|--------|------|
| learning_rate | 6e-4 | **3e-4** | LLaMA 架构适合稍低 LR |
| warmup_ratio | 0.1 | **0.05** | 5% warmup 足够 |
| lr_scheduler | cosine | **cosine** | 保持（但修复步数计算） |
| weight_decay | 0.1 | **0.1** | 合理 |
| max_grad_norm | 1.0 | **1.0** | 合理 |
| batch_size | 16/GPU | **24/GPU** | A6000 49GB 可容纳更大 batch |
| gradient_accumulation | 2 | **1** | 增大 per-GPU batch 后减少累积 |
| max_length | 512 | **1024** | RoPE 支持更长序列 |
| num_epochs | 10 | **15-20** | 更多 epoch 充分学习 |
| dropout | 0.1 | **0.1 → 0** | 后期退火 |

### 5.3 训练策略改进

#### (1) 学习率 Warmup + Cosine Decay（正确版）

```python
# 正确计算总步数
total_steps = num_epochs * steps_per_epoch
warmup_steps = int(0.05 * total_steps)

# 使用 linear warmup + cosine decay
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)
```

#### (2) 梯度累积优化

```python
# 更大的有效 batch size 有助于训练稳定
# 目标: effective_batch = 256 tokens × 序列长度
effective_batch = 24 * 1 * 4_gpus * 1024_tokens = ~98,304 tokens per step
```

#### (3) 混合精度训练

```python
# A6000 支持 bf16
accelerator = Accelerator(mixed_precision="bf16")
```

#### (4) Dropout 退火

```python
# 训练后半段降低 dropout
if epoch > num_epochs * 0.7:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
```

### 5.4 数据采样策略

```python
# 使用动态 batch 采样: 长度相近的样本放在同一 batch
# 减少 padding，提高训练效率
from torch.utils.data import Sampler
import numpy as np

class LengthBasedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        # 按样本长度排序，相近长度分到同一 batch
        ...
```

---

## 6. 数据优化方案

### 6.1 数据质量提升

```python
# 1. 去重
def deduplicate_texts(texts, similarity_threshold=0.8):
    """使用 MinHash 去除相似文本"""
    from datasketch import MinHash, MinHashLSH
    ...

# 2. 清洗
def clean_text(text):
    """清洗文本数据"""
    # 去除 HTML 标签
    # 去除异常字符
    # 标准化标点
    # 去除过短/过长文本
    ...
```

### 6.2 数据增强（在 100M 词限制内）

```python
# 1. 句子打乱（文档内句子重排）
def sentence_shuffle(text, prob=0.1):
    """以一定概率打乱句内词序，增加鲁棒性"""
    ...

# 2. 随机插入/删除/替换
def random_augmentation(text, aug_prob=0.05):
    """随机文本增强"""
    ...

# 3. BPE dropout（训练时随机跳过 BPE merge）
# 在 tokenizer encode 时启用
tokenizer.enable_dropout(p=0.1)
```

### 6.3 序列构造优化

```python
# 当前: 简单分块（可能跨文档边界）
# 改进: 文档感知的序列构造

def pack_sequences(texts, tokenizer, max_length):
    """将多个短文档 pack 进同一序列，用 EOS 分隔"""
    sequences = []
    current_seq = []
    
    for text in texts:
        tokens = tokenizer.encode(text)
        tokens.append(tokenizer.eos_token_id)
        
        if len(current_seq) + len(tokens) <= max_length:
            current_seq.extend(tokens)
        else:
            if current_seq:
                sequences.append(current_seq)
            current_seq = tokens
    
    return sequences
```

---

## 7. SOTA 实施路线图

### Phase 1: 快速修复（1-2天）

**优先级最高，预计 PPL 改善 50%+**

1. ✅ 修复 LR Scheduler Bug
2. ✅ 替换为 ByteLevel BPE Tokenizer
3. ✅ 增大 batch size 和序列长度
4. ✅ 启用 bf16 混合精度
5. ✅ 重新训练 10 epochs

**预期结果**: Val PPL ~100-150

### Phase 2: 架构升级（2-3天）

**最大提升，预计 PPL 再降低 50%+**

1. ✅ 将 GPT-2 替换为 LLaMA 架构
2. ✅ 实现 RoPE + GQA + SwiGLU + RMSNorm
3. ✅ 序列长度扩展到 1024
4. ✅ 训练 15-20 epochs
5. ✅ 使用 HuggingFace LlamaForCausalLM

**预期结果**: Val PPL ~30-60

### Phase 3: 精细优化（1-2天）

**锦上添花**

1. ✅ 超参数搜索（LR, batch size, warmup）
2. ✅ Dropout 退火
3. ✅ 数据增强（BPE dropout）
4. ✅ 文档感知序列构造
5. ✅ 数据去重和清洗
6. ✅ 训练 20+ epochs

**预期结果**: Val PPL ~25-40

### Phase 4: 评测和提交（1天）

1. ✅ 下载并准备中文评测数据
2. ✅ 运行 ZhoBLiMP 评测（句法理解）
3. ✅ 运行 CLUE 子任务评测（NLU）
4. ✅ 生成文本质量评估
5. ✅ 准备提交格式

---

## 8. 总结与优先级排序

### 8.1 问题严重程度排序

| 排名 | 问题 | 严重度 | 预计PPL改善 |
|------|------|--------|-----------|
| 1 | LR Scheduler Bug | 🔴 致命 | 50%+ |
| 2 | Tokenizer 不适合中文 | 🔴 严重 | 30-50% |
| 3 | GPT-2 架构过时 | 🟡 中等 | 20-40% |
| 4 | 训练策略次优 | 🟡 中等 | 10-20% |
| 5 | 数据利用不充分 | 🟢 轻微 | 5-10% |

### 8.2 投入产出比排序

1. **修复 LR Scheduler** — 代码改动最小，效果最大
2. **升级 Tokenizer** — 几十行代码，效果显著
3. **升级到 LLaMA 架构** — 需要较大改动，但收益最大
4. **优化训练策略** — 中等改动，中等收益
5. **数据增强** — 最后优化，锦上添花

### 8.3 建议的最终模型规格

```
架构: LlamaForCausalLM
参数: ~125M
词表: 32K ByteLevel BPE
序列: 1024 tokens
训练: 20 epochs, bf16
GPU: 4× A6000 (全部使用)
预计训练时间: ~8 小时
目标 Val PPL: < 40
```

### 8.4 与 SOTA 的差距分析

在 BabyLM 挑战赛（英文）中：
- 2024 冠军 GPT-BERT 在 100M 词数据上 PPL 约 25-35
- 纯 GPT-2 baseline PPL 约 40-60
- 中文由于 tokenization 和语言复杂性，PPL 会偏高

我们的目标：
- **最低目标**: Val PPL < 80（修复 Bug 后）
- **理想目标**: Val PPL < 40（架构升级后）
- **SOTA 目标**: 综合评测分数进入 Top 3

---

## 附录: 评测任务说明

ChineseBabyLM 评测包含三个 Track:

### NLU Track
- CLUE benchmark 子任务（文本分类、推理等）
- ZhoBLiMP（最小对语法判断，类似 BLiMP）
- 需要微调或 zero-shot 评测

### Cognitive Modeling Track
- MulCogBench（行为和神经信号对齐）
- 衡量模型表征与人类认知信号的对齐程度

### HANZI Track
- PinyinBench（拼音知识）
- HanziBench（汉字结构知识）
- 测试汉字特有知识

**注意**: 比赛不限制模型架构，也不限制训练 epoch 数，只限制总数据量 ≤ 100M 词。

---

*文档结束。建议立即执行 Phase 1 快速修复，然后进入 Phase 2 架构升级。*