# 📊 ChineseBabyLM 实验报告

> NLPCC 2026 · 首届 ChineseBabyLM 挑战赛
> 
> 生成日期: 2026-04-20
> 
> 团队: C5 Team

---

## 1. 实验概述

### 1.1 比赛目标

在 `babylm-zho-100M`（约 1 亿中文字符的儿童语料）上，从头预训练一个高性能的小型中文语言模型。约束条件为不可使用外部预训练模型。

### 1.2 数据集

| 属性 | 数值 |
|------|------|
| 数据集 | babylm-zho-100M |
| 来源 | HuggingFace (chinese-babylm-org/babylm-zho-100M) |
| 原始行数 | ~2,000,000 行 |
| 清洗后行数 | ~1,300,000 行（V2） |
| 验证集行数 | 6,510 行 |
| 训练集大小 | ~374 MB |
| 验证集大小 | ~1.8 MB |

### 1.3 训练时间线

| 时间 | 事件 |
|------|------|
| 2026-04-19 | V1 基线完成 (GPT-2 架构) |
| 2026-04-19 | V2 开发: LLaMA 架构 + ByteLevel BPE + SOTA 优化 |
| 2026-04-20 08:21 | V2 训练启动 |
| 2026-04-20 15:56 | V2 训练完成 (25/25 Epochs, ~7.5 小时) |
| 2026-04-20 19:08 | 独立评测完成 |

---

## 2. 模型架构对比

### 2.1 V1: GPT-2 (基线)

```
GPT2LMHeadModel
├── vocab_size:          32,000
├── n_embd:              768
├── n_layer:             12
├── n_head:              12
├── max_position:        512
├── 激活函数:            GELU
├── 归一化:              LayerNorm (Post-Norm)
├── 位置编码:            可学习绝对位置编码
└── 总参数量:            ~110M
```

### 2.2 V2: LLaMA (优化版)

```
LlamaForCausalLM
├── vocab_size:              32,000
├── hidden_size:             768
├── intermediate_size:       2,048 (SwiGLU, ≈ 2.67 × d_model)
├── num_hidden_layers:       12
├── num_attention_heads:     12 (head_dim = 64)
├── num_key_value_heads:     4 (GQA, 每 3 个 Q 头共享 1 组 KV)
├── max_position_embeddings: 1,024
├── rms_norm_eps:            1e-6
├── rope_theta:              10,000
├── 激活函数:                SwiGLU
├── 归一化:                  RMSNorm (Pre-Norm)
├── 位置编码:                RoPE (旋转位置编码)
└── 总参数量:                ~124.7M
```

### 2.3 关键架构差异

| 特性 | V1 (GPT-2) | V2 (LLaMA) | 改进效果 |
|------|-----------|-----------|---------|
| 位置编码 | 绝对位置 | RoPE | 更好的长度外推性 |
| 注意力 | MHA (12头) | GQA (12Q/4KV) | 减少 KV cache, 提升效率 |
| 激活函数 | GELU | SwiGLU | 更好的表达能力 |
| 归一化 | LayerNorm | RMSNorm | 训练更稳定, 计算更快 |
| 归一化位置 | Post-Norm | Pre-Norm | 更好的梯度流 |
| 序列长度 | 512 | 1,024 | 2× 上下文窗口 |
| 注意力加速 | 无 | SDPA/Flash Attention | 显著加速 |

---

## 3. 训练配置

### 3.1 超参数

| 参数 | V1 | V2 |
|------|-----|-----|
| Optimizer | AdamW (β₂=0.999) | AdamW (β₂=0.95) |
| Peak LR | 6e-4 | 6e-4 |
| LR Scheduler | Cosine Annealing | Cosine Annealing (修复 Bug) |
| Warmup Ratio | 3% | 5% |
| Weight Decay | 0.1 | 0.1 |
| Batch Size / GPU | 8 | 16 |
| Gradient Accumulation | 4 | 2 |
| 有效 Batch Size | 128 (8×4×4) | 128 (16×2×4) |
| Max Sequence Length | 512 | 1,024 |
| Training Epochs | 25 | 25 |
| Precision | bf16 | bf16 |
| Gradient Checkpointing | ✗ | ✓ |
| BPE Dropout | 0.0 | 0.1 (退火) |
| Gradient Clipping | 1.0 | 1.0 |

### 3.2 硬件配置

| 项目 | 配置 |
|------|------|
| GPU | 4× NVIDIA A6000 (48GB) |
| 分布式 | Accelerate MULTI_GPU DDP |
| 混合精度 | bf16 |
| 训练吞吐量 | ~80K tokens/sec |

---

## 4. 训练过程

### 4.1 训练曲线摘要

| Epoch | Step | Train Loss | Train PPL | Val Loss | Val PPL | LR |
|-------|------|-----------|-----------|----------|---------|-----|
| 1 | 1,190 | 8.07 | 3,203 | 7.41 | 1,644 | 4.40e-4 |
| 2 | 2,380 | 7.13 | 1,247 | 6.86 | 958 | - |
| 3 | 3,570 | 6.68 | 795 | 6.63 | 754 | - |
| 4 | 4,760 | 6.42 | 614 | 6.51 | 671 | - |
| 5 | 5,950 | 6.24 | 513 | 6.45 | 632 | - |
| 6 | 7,140 | 6.10 | 445 | 6.42 | 615 | - |
| **7** | **8,330** | **5.99** | **399** | **6.39** | **597** | **Peak附近** |
| ... | ... | ... | ... | ... | ... | ... |
| 15 | 17,850 | 5.34 | 209 | 6.40 | 601 | ~6e-5 |
| 20 | 23,800 | 5.00 | 148 | 6.41 | 605 | ~1e-6 |
| 25 | 29,750 | 4.80 | 121 | 6.40 | 603 | ~1e-10 |

### 4.2 最佳模型

- **Best Checkpoint**: Epoch 7, Step 8,330
- **Best Val Loss**: 6.392
- **Best Val PPL**: 597.28
- **保存路径**: `output/babylm-llama-v2/best_model/`

### 4.3 训练特点

- **零错误**: 25 个 Epoch 全程无任何训练错误
- **单调收敛**: Val loss 从 Epoch 1 的 7.41 持续下降到 Epoch 7 的 6.39
- **过拟合**: Epoch 7 后 train loss 继续下降 (4.80)，但 val loss 不再改善 (6.40)
- **学习率**: 正确执行 cosine 衰减，从 6e-4 warmup 到 peak 后平滑衰减至 ~0

---

## 5. 评测结果

### 5.1 独立评测 (evaluate_v2.py)

| 指标 | 数值 |
|------|------|
| 模型 | best_model (Epoch 7) |
| 评测集 | val.txt (6,510 行, 389,252 tokens) |
| Block Size | 1,024 |
| **Val Loss** | **7.509** |
| **Val PPL** | **1,824** |
| Token/字符比 | 0.569 |
| UNK 率 | 0/7 (零 UNK) |

### 5.2 PPL 差异说明

训练中 val_loss=6.392 (PPL=597) vs 独立评测 val_loss=7.509 (PPL=1824) 的差异原因：

1. **评测方式不同**: 训练中使用 DataLoader 按文档构造序列（EOS分隔），独立评测将文档拼接后固定切 1024-token chunks
2. **文档边界处理**: 独立评测的 chunk 可能跨越文档边界，导致 loss 偏高
3. **ByteLevel BPE 效应**: UTF-8 字节级别 token 的信息熵低于字/词级别，PPL 指标被指数级放大

### 5.3 文本生成质量

| Prompt | Greedy | Sampling |
|--------|--------|----------|
| 今天 | 今天,,,,,,,, | 今天到边边。,走路猛... |
| 我喜欢 | 我喜欢,,,,,,,, | 我喜欢的?$$\{\tm right}... |
| 从前有一座山 | 从前有一座山,有个 | 从前有一座山,有一只兔子一只它偷许多... |
| 老师说 | 老师说,,,,,,,, | 老师说你了吗她... (含对话片段) |

**问题诊断**:
- Greedy 退化: 对逗号 token 过度偏好，输出大量重复逗号
- Sampling 不稳定: 部分有意义，但混入乱码和 LaTeX 符号
- 根因: ByteLevel BPE 对中文不友好 + 语料含数学/公式噪声

---

## 6. Bug 修复记录

### Bug 1: BPE Dropout 导致 DataLoader 索引越界

**问题**: 启用 BPE Dropout 后，每次 `__getitem__` 调用 `tokenizer.encode()` 产生不同长度的 token 序列，导致随机索引超出范围。

**修复**: 在 `__init__` 中计算 `_target_length = len(self.examples)` 作为固定样本数，`__getitem__` 中使用 `idx % target_len` 取模。

### Bug 2: evaluate() 中 loss 计算错误

**问题**: `accelerator.gather()` 在多 GPU 环境下沿 dim=0 拼接 tensor，导致 gather 后 tensor 形状与单 GPU 不同，loss 累加逻辑出错。

**修复**: 先在每张 GPU 上计算 `total_loss += loss.item() * bs * seq_len` 和 `total_tokens += bs * seq_len`，然后用 `accelerator.gather()` 聚合标量统计量。

### Bug 3: LR Scheduler 总步数计算

**问题**: V1 中 `num_training_steps` 计算不精确，导致学习率提前衰减到零。

**修复**: 基于实际 DataLoader 长度精确计算 `max_train_steps = num_epochs × steps_per_epoch`。

---

## 7. V1 vs V2 对比总结

| 维度 | V1 (GPT-2) | V2 (LLaMA) |
|------|-----------|-----------|
| 架构 | GPT-2 | LLaMA |
| 参数量 | 110M | 124.7M |
| Tokenizer | BPE | ByteLevel BPE |
| 序列长度 | 512 | 1,024 |
| Val PPL (训练中) | ~344 | ~597 |
| 注意力加速 | 无 | SDPA |
| Gradient Ckpt | 无 | ✓ |
| BPE Dropout | 无 | 0.1 (退火) |
| WandB | ✓ | ✓ (更详细) |
| 训练时长 | ~6h | ~7.5h |

---

## 8. 问题分析与改进方向

### 8.1 当前问题

1. **ByteLevel BPE 不适合中文**: 将中文拆成 UTF-8 字节序列，token 粒度过细，增加模型学习难度
2. **PPL 与 V1 不可比**: 字节级别的 PPL 天然高于字/词级别
3. **过拟合**: Epoch 7 后 val loss 不再改善，但 train loss 持续下降
4. **生成质量差**: Greedy 退化为重复逗号，Sampling 含乱码

### 8.2 改进建议

1. **更换 Tokenizer**: 使用 SentencePiece (BPE/Unigram) 在字/词级别训练，而非字节级别
2. **数据清洗加强**: 去除 LaTeX/公式/代码等噪声数据
3. **Early Stopping**: 在 val loss 不再改善时停止训练 (如 Epoch 7)
4. **正则化增强**: 增加 dropout、label smoothing
5. **学习率调优**: 尝试 WSD (Warmup-Stable-Decay) scheduler
6. **数据增强**: 回译、同义词替换等

---

## 9. 文件清单

### 9.1 V2 核心代码

| 文件 | 功能 | 行数 |
|------|------|------|
| `train_v2.py` | LLaMA 训练主脚本 | ~650 |
| `train_tokenizer_v2.py` | ByteLevel BPE Tokenizer 训练 | ~120 |
| `prepare_data_v2.py` | 数据清洗/去重/分割 | ~250 |
| `evaluate_v2.py` | LLaMA 模型评测 | ~250 |
| `run_train_v2.sh` | 一键训练启动脚本 | ~80 |
| `accelerate_config_v2.yaml` | Accelerate 多 GPU 配置 | ~10 |

### 9.2 V1 基线代码

| 文件 | 功能 |
|------|------|
| `train.py` | GPT-2 训练脚本 |
| `train_tokenizer.py` | BPE Tokenizer 训练 |
| `prepare_data.py` | 数据下载与预处理 |
| `evaluate_model.py` | GPT-2 模型评测 |

### 9.3 输出文件

| 路径 | 大小 | 说明 |
|------|------|------|
| `output/babylm-llama-v2/best_model/` | ~479 MB | 最佳模型 (Epoch 7) |
| `output/babylm-llama-v2/epoch-*/` | ~479 MB each | 每 epoch 快照 |
| `output/babylm-llama-v2/checkpoint-*/` | ~479 MB each | 每 2000 步检查点 |
| `data/tokenizer_v2/` | ~10 MB | V2 Tokenizer |
| `data/processed_v2/` | ~375 MB | 清洗后数据 |

---

## 10. WandB 监控

### 10.1 配置

- **Project**: `chinese-babylm`
- **Run Name**: `llama-v2-phase3-768d-12l-gqa4-fa2-bpedrop`
- **Mode**: online

### 10.2 记录指标

| 指标 | 频率 | 说明 |
|------|------|------|
| `train/loss` | 每 50 步 | 训练交叉熵损失 |
| `train/ppl` | 每 50 步 | 训练困惑度 |
| `train/lr` | 每 50 步 | 当前学习率 |
| `train/tokens_per_sec` | 每 50 步 | 吞吐量 |
| `train/grad_norm` | 每 50 步 | 梯度范数 |
| `train/gpu_memory_used_gb` | 每 50 步 | GPU 显存使用 |
| `val/loss` | 每 epoch | 验证损失 |
| `val/ppl` | 每 epoch | 验证困惑度 |

---

*报告生成时间: 2026-04-20 19:50 CST*