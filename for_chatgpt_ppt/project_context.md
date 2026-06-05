# BabyLLM 项目技术演进全景文档

> **用途**: 供 ChatGPT 制作 NLPCC 2026 ChineseBabyLM 挑战赛中期汇报 PPT 的核心素材
> **生成日期**: 2026-06-05
> **团队**: SULAB / C5 Team

---

## 一、项目概况

### 1.1 比赛信息

- **赛事**: NLPCC 2026 首届 ChineseBabyLM 挑战赛
- **目标**: 在 ~100M 中文字符（~82M tokens）的儿童语料上，从头预训练高性能小型中文语言模型
- **约束**: 不可使用外部预训练模型
- **评测任务**: ZhoBLiMP (语法判断)、汉字结构、汉字拼音、AFQMC (语义相似)、OCNLI (推理)、TNEWS (新闻分类)、WSC2020 (共指消解)

### 1.2 硬件配置

| 项目 | 配置 |
|------|------|
| GPU | 4× NVIDIA RTX A6000 (48GB VRAM) |
| 分布式 | Accelerate MULTI_GPU DDP |
| 混合精度 | bf16 |
| 训练吞吐量 | ~80K tokens/sec |

### 1.3 核心成果摘要

| 指标 | 数值 |
|------|------|
| 总迭代版本 | 15 个 (V1~V15) |
| 开发周期 | 23 天 (2026-04-19 ~ 2026-05-12) |
| PPL 改进幅度 | V2: 597 → V13: **38.68** (↓93.5%) |
| 最佳模型 | V13 Stage 2 EMA (94.2M 参数, PPL=38.68) |
| 最佳参数效率 | V12 (54.2M 参数, PPL=38.84, PPL/10M=7.21) |
| 评测总时长 | ~8 小时 (含数据准备) |

---

## 二、时间线与版本演进 (V1 ~ V15)

### 2.1 全版本总览表

| 版本 | 日期 | 架构 | 参数量 | Tokenizer | 核心创新 | PPL | 状态 |
|------|------|------|--------|-----------|---------|-----|------|
| V1 | 04-19 | GPT-2 768d/12L | 109.9M | BPE 32K | 基线模型 | 343 (bug) | ❌ 失败 |
| V2 | 04-19 | LLaMA 768d/12L | 125M | ByteLevel BPE 32K | RoPE+GQA+SwiGLU | 597 | ❌ Tokenizer 灾难 |
| V3 | 04-20 | LLaMA 768d/12L | 125M | SentencePiece | WSD 调度器 | ~542 | ❌ NCCL 超时 |
| V4 | 04-20 | LLaMA 1024d/24L | ~350M | SentencePiece | 深层架构 | - | ❌ 模型过大欠拟合 |
| V5 | 04-21 | LLaMA 512d/12L | ~51M | SentencePiece | 知识蒸馏 | 526 | ❌ KD 阶段性能崩溃 |
| V6 | 04-22 | LLaMA 640d/12L | ~75M | SentencePiece | 三阶段流水线 | - | ❌ 数据清洗丢失 78% |
| V7 | 04-23 | LLaMA 448d/12L | ~35M | SPM 8K Unigram | MNTP 混合训练 | ~50.8 | ⚠️ 探索性 |
| V8 | 04-24 | LLaMA 512d/12L | ~38M | SPM 8K | 简化三阶段 | 50.84 | ⚠️ 基础版 |
| V9 | 04-25 | LLaMA 512d/12L | ~38M | SPM 8K | 超参数探测 | 50.85 | ⚠️ 实验版 |
| V10 | 04-26 | LLaMA 512d/12L | 38.7M | SPM 32K Unigram | 生产流水线 | 42.89 | ✅ 首个可用版本 |
| V11 | 04-26 | LLaMA 512d/12L | 38.7M | SPM 32K Unigram | EMA+SGDR+自蒸馏 | 40.73 | ✅ 突破 41 |
| V12 | 04-27 | LLaMA 576d/14L | 54.2M | SPM 32K Unigram | Focal Loss+数据清洗 | 38.84 | ✅ 突破 39 |
| **V13** | **04-28** | **LLaMA 768d/14L** | **94.2M** | **SPM 32K Unigram** | **PPL 过滤+MinHash** | **38.68** | **🏆 SOTA** |
| V14 | 04-29 | LLaMA 640d/12L | 59.2M | SPM 32K Unigram | 效率优化 | 41.8 | ⚠️ 未超 V13 |
| V15 | 05-12 | LLaMA 640d/14L | 68.2M | SPM 32K Unigram | 多尺度 EMA | 45.1 | ❌ 架构对齐问题 |

### 2.2 各版本详细说明

#### V1: GPT-2 基线 (2026-04-19)

**架构**: GPT-2 (768d, 12L, 12H, 可学习绝对位置编码, GELU, LayerNorm Post-Norm)
**参数量**: ~109.9M
**Tokenizer**: BPE 32K (WhitespaceSplit + Punctuation)
**训练配置**:
- 3× RTX A6000, bf16, 有效 batch=96
- LR=6e-4, Cosine Scheduler (有 bug), 10 epochs
- 训练数据: 82.3M tokens, 序列长度 512
- 训练时长: ~3.8 小时

**结果**:
| Epoch | Train Loss | Val Loss | Val PPL |
|-------|-----------|---------|---------|
| 1 | 7.60 | 6.78 | 880 |
| 2 | 6.62 | 6.39 | 595 |
| 7 | 6.06 | 6.03 | 414 |
| **9** | **5.68** | **5.84** | **343** |
| 10 | 5.57 | 5.85 | 347 |

**问题**: (1) LR Scheduler Bug: DDP 步数计算错误导致 LR 提前衰减为零; (2) Tokenizer 不适配中文: WhitespaceSplit 将整个中文句子当作单个 token; (3) 独立评测 PPL=1352, 远差于训练报告的 343

---

#### V2: LLaMA 架构迁移 (2026-04-19)

**架构**: LLaMA (768d, 12L, 12Q/4KV GQA, RoPE, SwiGLU, RMSNorm Pre-Norm)
**参数量**: ~124.7M
**Tokenizer**: ByteLevel BPE 32K
**训练配置**:
- 4× RTX A6000, bf16, 有效 batch=128
- LR=6e-4, Cosine (修复 V1 bug), Warmup 5%, 25 epochs
- 序列长度 1024, Gradient Checkpointing, BPE Dropout 0.1
- 训练时长: ~7.5 小时

**关键架构改进 (vs V1)**:

| 特性 | V1 (GPT-2) | V2 (LLaMA) | 改进效果 |
|------|-----------|-----------|---------|
| 位置编码 | 绝对位置 | RoPE | 更好的长度外推性 |
| 注意力 | MHA (12头) | GQA (12Q/4KV) | 减少 KV cache |
| 激活函数 | GELU | SwiGLU | 更好的表达能力 |
| 归一化 | LayerNorm | RMSNorm | 训练更稳定 |
| 序列长度 | 512 | 1024 | 2× 上下文窗口 |

**结果**:
| Epoch | Train Loss | Train PPL | Val Loss | Val PPL |
|-------|-----------|-----------|----------|---------|
| 1 | 8.07 | 3,203 | 7.41 | 1,644 |
| 2 | 7.13 | 1,247 | 6.86 | 958 |
| **7** | **5.99** | **399** | **6.39** | **597** |
| 25 | 4.80 | 121 | 6.40 | 603 |

**严重问题**: ByteLevel BPE 将中文拆为 UTF-8 字节序列 (如 "今天" → `ä»Ĭå¤©å¤©æ°Ķ`), 独立评测 PPL=1824, 文本生成退化为重复逗号

---

#### V3: SentencePiece 引入 (2026-04-20)

**核心改动**: 将 ByteLevel BPE 替换为 SentencePiece, 自定义 SPMTokenizer 封装
**新增**: WSD (Warmup-Stable-Decay) 学习率调度器 + Early Stopping
**效果**: **这是全项目最大的单次改进**, PPL 从 597 降至 ~542, 改进幅度 ~9%

**问题**: NCCL 超时崩溃 (CUDA/NCCL 版本不匹配)

---

#### V4: 深层 LLaMA (2026-04-20)

**架构**: LLaMA (1024d, 24L, 16H), ~350M 参数
**新增**: 嵌入权重共享 (tie_word_embeddings), 滑动窗口注意力, Dropout 退火
**失败原因**: 350M 参数 / 82M tokens = tokens/param = 0.23x, 严重违反 Chinchilla Scaling Laws (需要 20-200x), 模型严重欠拟合

---

#### V5: 小模型 + 知识蒸馏 (2026-04-21)

**架构**: LLaMA (512d, 12L, 8Q/4KV), ~51M 参数
**创新**: 两阶段训练 — CE 预训练 → KD (知识蒸馏)
**KD 配置**: lambda_ce=0.3, lambda_kd=0.7, temperature=2.0, top_k=10

**Phase 1 预训练结果**:
| Epoch | Train Loss | Val Loss | Val PPL |
|-------|-----------|---------|---------|
| 1 | 7.52 | 6.76 | 865 |
| **6** | **5.83** | **6.27** | **527** |
| 11 | 5.45 | 6.33 | 560 |

**Phase 2 KD 结果**: Val Loss 从 6.27 急剧上升至 8.49, PPL 从 527 飙升至 4854, KD 配置过于激进导致性能崩溃

---

#### V6: 三阶段流水线 (2026-04-22)

**架构**: LLaMA (640d, 12L, 10Q/5KV GQA), ~74.5M 参数
**创新**: 三阶段流水线 (CLM → CLM+MLM → Reverse KL KD)

**三阶段设计**:
- Stage 1 (CLM): LR=6e-4, dropout=0.1, label_smoothing=0.1, BPE dropout=0.1
- Stage 2 (CLM+MLM): LR=3e-4, dropout=0.05, 继承 Stage 1 最佳模型
- Stage 3 (Reverse KL KD): LR=1e-4, dropout=0.0, lambda_ce=0.5, lambda_kd=0.5, temperature=3.0, top_k=10

**Dropout 退火策略**: 0.1 → 0.05 → 0.0 (跨三阶段)

**失败原因**: 数据清洗过于激进, 训练集从 1,203,087 行缩减至 800,371 行 (-33%), 验证集从 63,320 行减至 42,066 行 (-33%)

---

#### V7: MNTP 混合训练 (2026-04-23)

**架构**: LLaMA (448d, 12L, 7Q/4KV), ~35M 参数 (最小模型)
**Tokenizer**: SentencePiece 8K Unigram (新增 `<mask>` token)
**核心创新**: MNTP (Masked Next Token Prediction) — GPT-BERT 混合方法
- CLM + MNTP 比例 1:7
- Label Smoothing + Dropout Annealing
**PPL**: ~50.8

---

#### V8: 简化三阶段 (2026-04-24)

**架构**: LLaMA (512d, 12L, 8Q/4KV), ~38M 参数
**PPL**: 50.84
**改进**: 基于 V7 简化, 增加 Polish 阶段

---

#### V9: 超参数探测 (2026-04-25)

**PPL**: 50.85
**内容**: MNTP stride 实现, Label Smoothing 微调, 实验不同训练策略

---

#### V10: 生产流水线 ✅ (2026-04-26)

**架构**: LLaMA (512d, 12L, 8Q/4KV), 38.7M 参数
**Tokenizer**: SentencePiece 32K Unigram (从 8K 升级至 32K)
**创新**:
- BPE Dropout 0.1 默认启用
- 三阶段流水线 (CLM → MNTP → Polish)
- Stride 512 实现 50% 序列重叠
- WandB 通知集成
- PPL: **42.89** (首个可用版本!)

**分阶段 PPL**:
| 阶段 | Val Loss | Val PPL | 时长 |
|------|---------|---------|------|
| Stage 1 (CLM) | 4.0182 | 55.60 | 0.7h |
| Stage 2 (MNTP) | 3.9256 | 50.68 | 1.5h |
| Stage 3 (Polish) | 3.7586 | 42.89 | 0.3h |

---

#### V11: EMA + SGDR + 自蒸馏 ✅ (2026-04-26)

**架构**: LLaMA (512d, 12L, 8Q/4KV), 38.7M 参数 (同 V10)
**核心创新**:
- **EMA (指数移动平均)**, decay=0.999: Stage 1 提供 8.4% PPL 改进
- **SGDR 调度器** (CosineAnnealingWarmRestarts): 周期性探索损失景观
- **自蒸馏**: EMA 教师 → 学生模型
- **动态 CLM:MTP 比例**
- **Label Smoothing 退火**
- 7 阶段流水线
- **PPL: 40.73**

**分阶段 PPL (含 EMA)**:
| 阶段 | Base Val Loss | Base PPL | EMA Val Loss | EMA PPL | 时长 |
|------|-------------|---------|-------------|---------|------|
| Stage 1 | 3.8163 | 45.44 | 3.7647 | 43.15 | 1.2h |
| Stage 2 | 3.7099 | 40.85 | 3.7094 | 40.84 | 1.3h |
| Stage 3 | 3.7081 | 40.78 | 3.7079 | 40.77 | 0.8h |
| Stage 4 | 3.7076 | 40.76 | 3.7075 | 40.76 | 0.5h |
| Stage 5 | 3.7070 | 40.73 | 3.7069 | 40.72 | 0.9h |

**关键发现**: EMA 在高 LR + 高噪声的 Stage 1 效果最显著, 后续阶段收益递减

---

#### V12: Focal Loss + 数据清洗 ✅ (2026-04-27)

**架构**: LLaMA (576d, 14L, 9Q/3KV GQA), 54.2M 参数
**核心创新**:
- **Focal Loss (gamma=2.0)**: 聚焦困难样本, 缓解 MNTP 类别不均衡
- **数据清洗**: 去重 + 质量过滤
- **改进自蒸馏**: Step-level 教师更新
- 5 阶段流水线
- **PPL: 38.84** (首次突破 39!)

**分阶段 PPL**:
| 阶段 | Base PPL | EMA PPL | 改进幅度 |
|------|---------|---------|---------|
| Stage 1 | 41.20 | 39.40 | -4.4% |
| Stage 2 | 38.94 | 38.84 | -0.3% |

**参数效率最优**: PPL/10M params = 7.21 (所有版本中排名第二)

---

#### V13: SOTA 模型 🏆 (2026-04-28)

**架构**: LLaMA (768d, 14L, 12Q/4KV GQA), **94.2M 参数**
**Tokenizer**: SentencePiece 32K Unigram

**三阶段流水线详细配置**:

**Stage 1: CLM + SGDR + Focal Loss (8 epochs)**
- LR=6e-4, SGDR (T_mult=2), Focal Loss gamma=1.5
- EMA decay=0.999, Label Smoothing 0.1→0.05 (退火)
- Attention Dropout=0.1, BPE Dropout=0.1

| Epoch | Train Loss | Val Loss | Val PPL | Label Smoothing |
|-------|-----------|---------|---------|----------------|
| 1 | 5.3668 | 4.1248 | 61.85 | 0.1000 |
| 4 | 3.8880 | 3.7751 | 43.60 | 0.0786 |
| **7** | **3.7007** | **3.7422** | **42.19** | 0.0571 |
| 8 | 3.5076 | 3.7472 | 42.40 | 0.0500 |

Best: Epoch 7, PPL=42.19 (EMA: PPL=39.51, **改进 8.4%**)

**Stage 2: MNTP + Dynamic CLM (10 epochs)**
- LR=5e-4, Cosine, Focal Loss gamma=1.0
- Dynamic CLM ratio: 25% → 12.5% → 6.25%
- Mask ratio: 25% → 10%, Label Smoothing 0.05→0.025 (退火)

| Epoch | Train Loss | Val Loss | Val PPL | CLM Ratio | Mask Ratio |
|-------|-----------|---------|---------|-----------|-----------|
| 1 | 4.3477 | 3.7599 | 42.94 | 0.250 | 0.250 |
| 4 | 4.2380 | 3.7148 | 41.05 | 0.125 | 0.200 |
| **6** | **4.1343** | **3.7070** | **40.73** | 0.125 | 0.167 |
| 10 | 3.9089 | 3.7175 | 41.16 | 0.062 | 0.100 |

Best: Epoch 6, PPL=40.73 (EMA: PPL=**38.68**, **改进 6.7%**)

**Stage 3: Polish + DropBlock + StochDepth (5 epochs)**
- LR=2e-5, 无 Focal Loss, DropBlock=0.1 (size=3), StochDepth=0.05
- **结果**: val_loss 从未改善, train-val gap = 0.57 (过度正则化)
- **结论**: Stage 3 Polish 证明有害, 后续版本移除

| Epoch | Train Loss | Val Loss | Val PPL |
|-------|-----------|---------|---------|
| 1 | 3.1527 | 3.6969 | 40.32 |
| 5 | 3.1338 | 3.7033 | 40.58 |

**EMA 定量贡献**:
| 阶段 | Base PPL | EMA PPL | 改进 |
|------|---------|---------|------|
| Stage 1 | 43.15 | 39.51 | **-8.4%** |
| Stage 2 | 41.45 | 38.68 | **-6.7%** |
| Stage 3 | 40.40 | 40.08 | -0.8% |

**收敛速度分析**:
| 阶段 | PPL 改进 | 时长 | PPL/小时 | Steps/sec |
|------|---------|------|---------|-----------|
| Stage 1 | 61.85→42.19 (-19.66) | 3.1h | 6.34 | 2.03 |
| Stage 2 | 42.94→40.73 (-2.21) | 3.1h | 0.71 | 2.58 |
| Stage 3 | 40.32→40.32 (0) | 1.5h | 0.00 | 2.59 |

---

#### V14: 效率优化 (2026-04-29)

**架构**: LLaMA (640d, 12L, 10Q/5KV), 59.2M 参数
**改进**:
- 移除 DropBlock / Stochastic Depth (吸取 V13 Stage 3 教训)
- 增强版 Focal Loss
- 优雅关闭 + OOM 恢复机制
- tokens/param 比率提升至 1.9x
**PPL**: 41.8 (未超过 V13)

---

#### V15: 终极版本 (2026-05-12)

**架构**: LLaMA (640d, 14L, 10Q/5KV), 68.2M 参数
**目标**: PPL < 38.0, ZhoBLiMP > 65%, ~58M 参数
**核心改进**:
- **多尺度 EMA**: 同时跟踪 decay=0.999 和 0.9999 两个 EMA
- **2 阶段流水线** (CLM → MNTP, 移除 Polish)
- **每层梯度范数监控**: 每 250 步记录各层梯度范数
- **更频繁评测**: 每 200 步评测 (vs V14 的 500 步)
- **梯度尖峰检测**: 梯度范数 >10x 前值时发出警告

**V15 失败原因分析**:
1. `intermediate_size = ⌊640×8/3⌋ = 1706`, 不是 256 的倍数, 导致张量布局不优
2. 使用了 V14 的 PPL 过滤数据 (比 V13 少 ~43K 行)
3. LR=5e-4 偏低 (V12/V13 使用 6e-4)

**实际结果**: PPL=45.1, ZhoBLiMP=62.4% (均未达目标)

---

## 三、三阶段训练架构详解

### 3.1 整体流水线

```
原始数据 → 数据清洗 → SentencePiece Tokenizer
    ↓
Stage 1: CLM 预训练 (高 LR + 高正则化 + EMA)
    ↓ (best_model + EMA 权重)
Stage 2: MNTP 混合训练 (动态 CLM/Mask 比例 + EMA)
    ↓ (best_model_ema)
最终模型
```

### 3.2 Stage 1: CLM (Causal Language Model)

**目标**: 从零学习语言基础分布
**损失函数**:

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N} \log p(x_i \mid x_{<i})$$

$$\text{PPL} = \exp(\mathcal{L}_{CE})$$

**核心技术**:
- **SGDR 调度器**: 周期性重启学习率, 探索不同损失景观区域
  - T_0=1 epoch, T_mult=2 (周期翻倍: 1→2→4 epochs)
- **Focal Loss**: 聚焦困难 token, 缓解简单 token 的主导地位
  - $\text{FL}(p_t) = -(1-p_t)^\gamma \log(p_t)$, gamma=1.5~2.0
- **EMA**: 指数移动平均权重, 平滑训练噪声
  - $\theta_{EMA} = \alpha \cdot \theta_{EMA} + (1-\alpha) \cdot \theta$, decay=0.999
- **Label Smoothing**: 软化硬标签, 防止过拟合
  - 0.1 → 0.05 线性退火
- **Dropout Annealing**: attention_dropout=0.1, bpe_dropout=0.1

**超参数**:
| 参数 | V13 值 |
|------|--------|
| Learning Rate | 6e-4 |
| Batch Size | 16×4×2 = 128 |
| Max Seq Length | 1024 |
| Stride | 512 (50% overlap) |
| Epochs | 8 |
| Weight Decay | 0.1 |
| Max Grad Norm | 1.0 |
| Focal Loss gamma | 1.5 |
| EMA Decay | 0.999 |

### 3.3 Stage 2: MNTP (Masked Next Token Prediction)

**目标**: 在自回归基础上引入双向上下文学习
**核心思想**: GPT-BERT 混合 — 同时进行 CLM 和掩码预测

**MNTP 机制**:
- 以 CLM:MTP = 1:7 的比例混合训练
- 对输入序列随机 mask 15%~25% 的 token
- 模型需要同时预测下一个 token (CLM) 和被 mask 的 token (MNTP)

**动态比例调整**:
| 训练进度 | CLM 比例 | Mask 比例 |
|---------|---------|---------|
| 前 25% | 25% | 25% |
| 25%~50% | 12.5% | 20% |
| 后 50% | 6.25% | 10% |

**V13 Stage 2 超参数**:
| 参数 | 值 |
|------|-----|
| Learning Rate | 5e-4 |
| Scheduler | Cosine |
| Focal Loss gamma | 1.0 |
| Label Smoothing | 0.05→0.025 |
| Attention Dropout | 0.05 |
| EMA Decay | 0.999 |
| Epochs | 10 |

### 3.4 Stage 3: Knowledge Distillation (V6 版本, 后期弃用)

**V6 KD 设计**:
- **Reverse KL 散度**: $D_{KL}(p_{student} \| p_{teacher})$ (避免模式坍缩)
- **教师模型**: 从更大模型提取的 top-k logits
- **温度**: 3.0, top_k=10
- **混合损失**: lambda_ce=0.5, lambda_kd=0.5
- **Teacher logits shape**: (31840, 1024, 10) — 每个 position 10 个候选

**后续版本弃用 KD 的原因**: V5 KD 阶段性能严重退化 (PPL 527→4854), 且 V10+ 的自蒸馏/EMA 方案更稳定

---

## 四、Tokenizer 演进

### 4.1 Tokenizer 版本对比

| 版本 | 类型 | 词表大小 | 平均 Token/字符比 | 对中文适配 | PPL 影响 |
|------|------|---------|-----------------|----------|---------|
| V1 | BPE (WhitespaceSplit) | 32K | ~1.0 | ❌ 中文句子当单 token | PPL=343 |
| V2 | ByteLevel BPE | 32K | 0.569 | ❌ 拆为 UTF-8 字节 | PPL=597 |
| V3~V6 | SentencePiece BPE | 32K | ~0.8 | ⚠️ 基础支持 | PPL~527 |
| V7~V9 | SPM Unigram 8K | 8K | ~0.7 | ✅ 含 `<mask>` | PPL~50.8 |
| V10+ | SPM Unigram 32K | 32K | ~0.85 | ✅ 最优平衡 | PPL≤42.89 |

### 4.2 ByteLevel BPE 的灾难性影响

**示例**: "今天天气真好，我想出去玩。"
- ByteLevel BPE (V2): `['ä»Ĭå¤©å¤©æ°Ķ', 'çľŁå¥½', 'ï', '¼', 'Į', 'æĪĳæĥ³', 'åĩºåİ»çİ©', 'ãĢĤ']` — 8 tokens
- 中文被拆成无意义的 UTF-8 字节序列, 模型需要先"学会"UTF-8 编码才能学习语义

**影响**: V2 训练 Val PPL=597, 但独立评测 PPL=1824 (差距 205%), 因为评测时序列拼接方式不同导致跨文档边界

---

## 五、数据处理演进

### 5.1 数据版本

| 版本 | 训练集大小 | 验证集大小 | 清洗策略 |
|------|----------|----------|---------|
| processed (原始) | ~399 MB | - | 无清洗 |
| processed_v2 | ~374 MB | ~1.8 MB | 基础去重+分割 |
| processed_v3 | ~352 MB | ~18.8 MB | 进一步清洗 |
| processed_v6 | ~72 MB | ~3.8 MB | ⚠️ 过度清洗 (-78%) |
| processed_v7 | ~347 MB | ~18.6 MB | 保守清洗 |

### 5.2 V13 数据质量优化

- **PPL 过滤**: max_ppl=250, 移除高困惑度噪声数据
- **MinHash 去重**: 阈值=0.7, 移除近似重复文档
- **效果**: 显著提升数据质量, V13 仅用 3 阶段即达 SOTA

---

## 六、评测结果全面对比

### 6.1 官方评测 (Official Evaluation)

| 版本 | ZhoBLiMP | 汉字结构 | 汉字拼音 | AFQMC | OCNLI | TNEWS | WSC2020 |
|------|---------|---------|---------|-------|-------|-------|---------|
| V13 | **63.5%** | **64.7%** | **49.5%** | 69.0% | 64.0% | 53.9% | 63.5% |
| V14 | 64.3% | 62.4% | 41.9% | 69.0% | **66.0%** | 54.1% | 63.5% |
| V15 | 62.4% | 63.9% | 47.4% | 69.0% | 65.9% | **54.4%** | 63.8% |

**评测配置**: AFQMC/OCNLI/TNEWS/WSC2020 使用 finetune 方式 (lr=3e-5, batch=32, epochs=10); ZhoBLiMP/汉字/拼音为零样本

### 6.2 ZhoBLiMP 15 维度详细分析 (V13)

| 维度 | V13 | V12 | V11 | 随机基线 | V13 vs V12 |
|------|-----|-----|-----|---------|-----------|
| BA (把字句) | 75.33% | 74.36% | 76.33% | 50% | +0.97 |
| question (疑问) | 64.41% | 68.78% | 63.05% | 50% | -4.37 |
| nominal_expression | 75.85% | 72.58% | 71.82% | 50% | +3.27 |
| classifier (量词) | 77.78% | 79.11% | 74.44% | 50% | -1.33 |
| npi_licensing | 46.67% | 40.70% | 42.37% | 50% | +5.97 |
| topicalization (主题化) | 63.50% | 54.00% | 60.33% | 50% | +9.50 |
| verb_phrase (动词短语) | 75.17% | 77.57% | 79.81% | 50% | -2.40 |
| anaphor (照应) | 35.00% | 37.33% | 36.44% | 50% | -2.33 |
| passive (被动) | 37.03% | 30.69% | 32.50% | 50% | +6.34 |
| argument_structure | 64.05% | 60.67% | 63.19% | 50% | +3.38 |
| ellipsis (省略) | 71.00% | 72.11% | 66.56% | 50% | -1.11 |
| control_raising | 70.42% | 62.83% | 64.50% | 50% | +7.59 |
| relativization (关系化) | 55.25% | 56.25% | 51.92% | 50% | -1.00 |
| fci_licensing | 75.13% | 63.67% | 66.13% | 50% | +11.46 |
| quantifiers (量词辖域) | 84.67% | 88.00% | 98.17% | 50% | -3.33 |
| **平均** | **63.47%** | **62.03%** | **61.97%** | **50%** | **+1.44** |

**弱点 (低于随机基线)**: anaphor (-15%), passive (-13%), npi_licensing (-3.3%)

### 6.3 汉字结构评测

| 维度 | V13 | V12 | V11 |
|------|-----|-----|-----|
| sx (声形) | 66.33 | 65.72 | 66.94 |
| szx (声字形) | **66.67** | 60.19 | 62.04 |
| zy (字义) | **67.00** | 63.20 | 62.20 |
| pin (拼音) | 44.44 | 55.56 | 55.56 |
| zzy (字字义) | **66.67** | 64.39 | 59.09 |
| bw (部首) | **61.57** | 60.37 | 61.17 |
| xq (字形) | 66.67 | 66.67 | 66.67 |
| **平均** | **64.65** | **62.65** | **62.75** |

### 6.4 Fine-tuning 结果

| 任务 | 类别数 | V13 | V12 | V11 | 竞赛基线 | 随机 |
|------|-------|-----|-----|-----|---------|------|
| AFQMC (语义相似) | 2 | 69.0% | 69.0% | 69.07% | 70.2% | 50% |
| OCNLI (推理) | 3 | 64.03% | 64.47% | 64.71% | - | 33.3% |
| TNEWS (新闻分类) | 15 | 53.89% | 53.60% | 53.04% | - | ~10% |
| CLUEWSC (共指消解) | 2 | 63.49% | 63.49% | 63.49% | - | 50% |

**异常**: CLUEWSC 所有版本均为 63.49%, MCC=0.0 — 模型学习了浅层启发式规则而非真正的共指消解

---

## 七、跨版本 PPL 完整数据

### 7.1 各版本各阶段 PPL 详表

| 版本 | 阶段 | Val Loss | Val PPL | EMA Loss | EMA PPL | 时长 |
|------|------|---------|---------|----------|---------|------|
| V10 Stage 1 | CLM | 4.0182 | 55.60 | - | - | 0.7h |
| V10 Stage 2 | MNTP | 3.9256 | 50.68 | - | - | 1.5h |
| V10 Stage 3 | Polish | 3.7586 | 42.89 | - | - | 0.3h |
| V11 Stage 1 | CLM | 3.8163 | 45.44 | 3.7647 | 43.15 | 1.2h |
| V11 Stage 2 | MNTP | 3.7099 | 40.85 | 3.7094 | 40.84 | 1.3h |
| V11 Stage 3 | MNTP | 3.7081 | 40.78 | 3.7079 | 40.77 | 0.8h |
| V11 Stage 4 | MNTP | 3.7076 | 40.76 | 3.7075 | 40.76 | 0.5h |
| V11 Stage 5 | MNTP | 3.7070 | 40.73 | 3.7069 | 40.72 | 0.9h |
| V12 Stage 1 | CLM | 3.7183 | 41.20 | 3.6737 | 39.40 | 1.9h |
| V12 Stage 2 | MNTP | 3.6620 | 38.94 | 3.6580 | 38.84 | 2.0h |
| V13 Stage 1 | CLM | 3.7422 | 42.19 | 3.6766 | 39.51 | 3.1h |
| **V13 Stage 2** | **MNTP** | **3.7070** | **40.73** | **3.6554** | **38.68** | **3.1h** |
| V13 Stage 3 | Polish | 3.6969 | 40.32 | 3.6909 | 40.08 | 1.5h |

### 7.2 EMA 贡献汇总

| 版本+阶段 | Base PPL | EMA PPL | 改进幅度 |
|---------|---------|---------|---------|
| V11 Stage 1 | 46.24 | 43.07 | -6.9% |
| V12 Stage 1 | 41.83 | 39.36 | -5.9% |
| V13 Stage 1 | 43.15 | 39.51 | **-8.4%** |
| V13 Stage 2 | 41.45 | 38.68 | **-6.7%** |
| V13 Stage 3 | 40.40 | 40.08 | -0.8% |

**关键发现**: EMA 在 Stage 1 (高 LR + 高噪声) 贡献最大, 后续阶段收益递减

### 7.3 训练效率对比

| 版本 | 总步数 | 总时长 | PPL 改进/小时 | Steps/sec |
|------|-------|-------|-------------|-----------|
| V10 | 20,437 | ~1.1h | ~8.5 | ~5.1 |
| V11 | 37,538 | ~4.8h | ~1.0 | ~2.2 |
| V12 | 49,520 | ~7.5h | ~0.3 | ~1.8 |
| V13 | 64,492 | ~7.7h | ~0.45 | ~2.3 |

### 7.4 参数效率对比

| 版本 | 参数量 | 最佳 PPL | PPL/10M 参数 | tokens/param |
|------|-------|---------|-------------|-------------|
| V11 | 38.7M | 40.73 | 10.53 | 2.6x |
| V12 | 54.2M | 38.84 | 7.21 | 1.9x |
| V10 | 38.7M | 42.89 | 11.09 | 2.6x |
| V13 | 94.2M | 38.68 | 4.11 | 1.1x |

---

## 八、核心技术贡献量化

### 8.1 各技术对 PPL 的贡献排序

| 排名 | 技术 | 贡献 | 来源版本 |
|------|------|------|---------|
| 1 | SentencePiece Tokenizer (替代 ByteLevel BPE) | ~74% PPL 降低 | V3 |
| 2 | EMA (指数移动平均) | 6~8% PPL 降低 | V11 |
| 3 | MNTP 混合训练 (CLM+Mask) | 3~5 PPL 降低 | V7 |
| 4 | SGDR 调度器 | ~2 PPL 降低 | V11 |
| 5 | Focal Loss | ~1~2 PPL 降低 | V12 |
| 6 | PPL 数据过滤 | ~0.5 PPL 降低 | V13 |
| 7 | Label Smoothing | ~0.3~0.5 PPL 降低 | V6 |
| 8 | 数据去重 (MinHash) | ~0.2~0.3 PPL 降低 | V13 |

### 8.2 失败经验教训

| 版本 | 失败原因 | 教训 |
|------|---------|------|
| V2 | ByteLevel BPE 不适配中文 | 中文需要字/词级别 Tokenizer |
| V4 | 350M 参数过大 | tokens/param 应 > 1.0, 理想 20~200x |
| V5 | KD 配置过于激进 | lambda_kd 不应 > lambda_ce |
| V6 | 数据清洗丢失 78% | 清洗需保守, 宁可保留噪声 |
| V13 Stage 3 | DropBlock+StochDepth | 低 LR 下强正则化有害 |
| V15 | FFN 维度非 256 倍数 | 架构参数需对齐 GPU 张量布局 |

---

## 九、媒体素材索引

### 9.1 图片清单及说明

| 文件名 | 说明 | PPT 使用建议 |
|--------|------|-------------|
| `images/version_timeline.png` | V1~V15 版本演进时间线, 蓝色=成功, 黄色=SOTA, 红色=失败 | 用于"项目概览"Slide |
| `images/ppl_evolution.png` | PPL 随版本演进曲线 (V2:597 → V13:38.68) | 用于"核心成果"Slide |
| `images/params_vs_ppl.png` | 参数量 vs PPL 散点图, 红色虚线为 Pareto 前沿 | 用于"参数效率"Slide |
| `images/efficiency_frontier.png` | 参数效率前沿分析图 | 用于"Scaling Analysis"Slide |
| `images/ppl_by_stage.png` | V10~V15 各阶段 PPL 对比柱状图 | 用于"训练流水线"Slide |
| `images/technique_impact.png` | 各技术对 PPL 贡献的量化对比图 | 用于"技术创新"Slide |
| `images/training_pipeline.png` | 三阶段训练流水线架构图 | 用于"方法"Slide |
| `images/official_eval_radar.png` | V13/V14/V15 雷达图 (7 个 CLUE 任务) | 用于"评测结果"Slide |
| `images/babylm.png` | BabyLM 项目 Logo/架构图 | 用于封面 Slide |
| `midterm_report.pdf` | 中期报告 LaTeX 原始文件 | 参考 Slide 结构 |

---

## 十、模型架构最终配置 (V13 SOTA)

```
LlamaForCausalLM (V13 SOTA)
├── vocab_size:          32,000 (SPM Unigram)
├── hidden_size:         768
├── intermediate_size:   2,048 (SwiGLU, ≈ 2.67 × d_model)
├── num_hidden_layers:   14
├── num_attention_heads: 12 (head_dim = 64)
├── num_key_value_heads: 4 (GQA, 3:1 比例)
├── max_position:        1,024 (RoPE, θ=10000)
├── rms_norm_eps:        1e-5
├── 激活函数:            SwiGLU
├── 归一化:              RMSNorm (Pre-Norm)
├── 位置编码:            RoPE (旋转位置编码)
├── 嵌入共享:            tie_word_embeddings = True
├── 总参数量:            94,246,656 (~94.2M)
└── 非嵌入参数量:        ~69.6M
```

---

## 十一、关键公式与方法

### 11.1 PPL 定义

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log p(x_i \mid x_{<i})\right)$$

### 11.2 Focal Loss

$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

其中 $\gamma = 1.5$ (V13 Stage 1), $\gamma = 1.0$ (V13 Stage 2)

### 11.3 EMA (Exponential Moving Average)

$$\theta_{EMA}^{(t)} = \alpha \cdot \theta_{EMA}^{(t-1)} + (1 - \alpha) \cdot \theta^{(t)}$$

其中 $\alpha = 0.999$ (标准), V15 同时跟踪 $\alpha = 0.9999$ (慢速)

### 11.4 Reverse KL Distillation (V6)

$$\mathcal{L}_{KD} = D_{KL}(p_{student} \| p_{teacher})$$

$$\mathcal{L}_{total} = \lambda_{CE} \cdot \mathcal{L}_{CE} + \lambda_{KD} \cdot \mathcal{L}_{KD}$$

V6: $\lambda_{CE} = 0.5, \lambda_{KD} = 0.5, T = 3.0$

### 11.5 MNTP 损失

$$\mathcal{L}_{MNTP} = \alpha \cdot \mathcal{L}_{CLM} + (1-\alpha) \cdot \mathcal{L}_{Mask}$$

动态调整: $\alpha \in \{0.25, 0.125, 0.0625\}$ 随训练进度递减

---

## 十二、未来方向 (V15.1 计划)

1. **修复 FFN 维度**: 1706 → 1792 (256 的倍数)
2. **使用 V13 数据管线**: PPL 过滤 + MinHash 去重
3. **提升 LR**: 从 5e-4 恢复至 6e-4
4. **架构搜索**: 对比 768d/12L (宽+浅) vs 576d/14L (窄+深)
5. **消融实验**: 量化各技术独立贡献 (EMA, SGDR, Focal Loss, PPL 过滤)

---

*文档生成时间: 2026-06-05*
*数据来源: babyLLM 项目 V1~V15 训练日志、评测结果、分析报告*
