# ChineseBabyLM V5 训练计划 — 知识蒸馏 + 小模型优化

> **版本**: V5  
> **创建日期**: 2026-04-23  
> **状态**: 🟡 待实施  
> **核心技术**: 白盒知识蒸馏 (White-Box KD) + 参数量匹配数据规模  
> **参考**: DistilQwen2.5 (Wang et al., ACL 2025)

---

## 1. 执行摘要

### 1.1 V1-V4 核心教训

| 问题 | 根因 | V5 解决方案 |
|------|------|-------------|
| 所有版本 Epoch 7+ 过拟合 | 100M 数据不足以支撑 110M+ 参数 | 缩小模型至 ~60M 参数 |
| V2 PPL 反而高于 V1 | ByteLevel BPE 对中文不友好 | 使用 SentencePiece Unigram |
| V4 350M 参数训练失败 | tokens/参数 = 0.23，严重欠训练 | 参数量匹配 Chinchilla 定律 |
| 生成质量差（重复逗号/乱码） | 模型未学到有效语言结构 | 知识蒸馏引入教师模型指导 |

### 1.2 V5 核心创新

1. **知识蒸馏 (Knowledge Distillation)**: 使用 Qwen2.5-0.5B 作为教师模型，通过白盒 KD（logit-level distillation）将知识迁移到 ~60M 学生模型
2. **参数量匹配数据规模**: ~60M 参数 / ~82M tokens ≈ 1.37 tokens/参数，接近 Chinchilla 最优
3. **SentencePiece Unigram tokenizer**: 词表缩至 16K，提高覆盖率
4. **两阶段训练**: Phase 1 标准 CE 预训练 → Phase 2 KD 微调

---

## 2. 技术方案设计

### 2.1 知识蒸馏架构

借鉴 DistilQwen2.5 的白盒蒸馏方法：

```
教师模型: Qwen2.5-0.5B-Instruct (或 Qwen2.5-1.5B-Instruct)
    │
    │  前向传播 → Top-K Logits (K=10)
    │
    ▼
学生模型: BabyLLM-V5 (~60M, LLaMA 架构)
    │
    │  Loss = λ_ce * L_ce + λ_kd * L_kd
    │  L_ce = CrossEntropy(labels)
    │  L_kd = KL_Divergence(teacher_logits/T, student_logits/T)
    │
    ▼
优化后的学生模型
```

#### 2.1.1 白盒知识蒸馏原理

根据 DistilQwen2.5 论文，教师模型 top-10 token 的概率之和几乎等于 1，即教师模型的几乎所有知识都包含在 top-10 token 中。因此：

1. **离线生成教师 logits**: 预先用教师模型对训练数据做前向推理，保存每个位置的 top-K logits 及其 token index
2. **在线 KD 训练**: 学生模型训练时，加载教师 logits，计算 KL 散度损失
3. **温度缩放**: 使用温度参数 T 平滑概率分布，使学生模型学到更丰富的知识

#### 2.1.2 损失函数设计

```python
# V5 混合损失函数
def compute_kd_loss(student_logits, teacher_logits_topk, teacher_indices_topk, 
                     labels, temperature=2.0, lambda_ce=0.5, lambda_kd=0.5):
    """
    student_logits: [batch, seq_len, vocab_size] - 学生模型完整 logits
    teacher_logits_topk: [batch, seq_len, K] - 教师模型 top-K logits
    teacher_indices_topk: [batch, seq_len, K] - 教师模型 top-K token indices
    labels: [batch, seq_len] - 真实标签
    """
    # 1. 标准 CE 损失
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # 2. KD 损失: 从学生 logits 中提取对应教师 top-K 位置的 logits
    # student_topk_logits: [batch*seq_len, K]
    student_topk = torch.gather(
        student_logits.view(-1, student_logits.size(-1)),
        dim=1,
        index=teacher_indices_topk.view(-1, teacher_indices_topk.size(-1))
    )
    
    # 温度缩放
    teacher_probs = F.softmax(teacher_logits_topk.view(-1, K) / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_topk / temperature, dim=-1)
    
    # KL 散度
    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    
    # 3. 混合损失
    total_loss = lambda_ce * ce_loss + lambda_kd * kd_loss
    return total_loss, {'ce_loss': ce_loss.item(), 'kd_loss': kd_loss.item()}
```

### 2.2 教师模型选择

| 候选教师 | 参数量 | 词表大小 | 优势 | 劣势 |
|----------|--------|----------|------|------|
| **Qwen2.5-0.5B** | 0.5B | 151,936 | 轻量，推理快 | 词表与学生不匹配 |
| **Qwen2.5-1.5B** | 1.5B | 151,936 | 更强能力 | 词表不匹配，推理慢 |
| **V4 best_model** | ~350M | 32,000 | 词表匹配 | 训练未完成，质量未知 |
| **V2 best_model** | ~125M | 32,000 | 词表匹配，已完成 | 能力有限 |

**推荐方案**: 使用 **Qwen2.5-0.5B-Instruct** 作为教师模型，通过 token alignment 处理词表不匹配问题。

#### 2.2.1 Token Alignment 策略

由于教师模型（Qwen2.5, 151K 词表）和学生模型（BabyLLM, 16K 词表）词表不同，需要：

1. 教师模型使用自己的 tokenizer 编码文本，生成 logits
2. 学生模型使用自己的 tokenizer 编码相同文本
3. 通过字符级别的对齐，建立两个 token 序列的映射关系
4. 在对齐的位置上计算 KD 损失

**简化方案（推荐）**: 由于词表不匹配处理复杂，V5 先使用 **V2 best_model (Epoch 7, val_loss=6.39)** 作为教师模型进行同词表 KD 验证，后续再扩展到 Qwen 教师。

### 2.3 学生模型架构

基于 Chinchilla 定律和 100M 数据约束，设计 ~60M 参数模型：

```
BabyLLM-V5 Student Model
├── vocab_size:              16,000 (SentencePiece Unigram)
├── hidden_size:             512
├── intermediate_size:       1,365 (8/3 × d_model, SwiGLU)
├── num_hidden_layers:       12
├── num_attention_heads:     8 (head_dim = 64)
├── num_key_value_heads:     4 (GQA, 2:1 ratio)
├── max_position_embeddings: 1,024
├── rms_norm_eps:            1e-5 (提升数值稳定性)
├── rope_theta:              10,000 (回归标准值)
├── 激活函数:                SwiGLU
├── 归一化:                  RMSNorm (Pre-Norm)
├── 位置编码:                RoPE
├── tie_word_embeddings:     True (节省参数)
└── 总参数量:                ~60M
```

#### 2.3.1 参数量估算

| 组件 | 计算 | 参数量 |
|------|------|--------|
| Token Embedding | 16,000 × 512 (tied) | 8.19M |
| 每层 Attention | 4 × (512×512) QKV + 512×512 O | ~1.31M |
| 每层 FFN (SwiGLU) | 3 × 512 × 1,365 | 2.10M |
| 每层 RMSNorm | 2 × 512 | 0.001M |
| 12 层合计 | 12 × (1.31 + 2.10 + 0.001) | ~40.9M |
| Final RMSNorm | 512 | 0.0005M |
| LM Head (tied) | 0 (共享 embedding) | 0 |
| **总计** | | **~49.1M** |

> 注: 实际参数量可能因实现略有差异，目标控制在 50-65M 范围内。

### 2.4 Tokenizer 方案

#### 2.4.1 SentencePiece Unigram (推荐)

| 参数 | V3/V4 (BPE) | V5 (Unigram) |
|------|-------------|--------------|
| 模型类型 | BPE | Unigram |
| 词表大小 | 32,000 | 16,000 |
| character_coverage | 0.9995 | 0.9995 |
| input_sentence_size | 10,000,000 | 10,000,000 |
| 预期 Token/字符比 | ~1.3 | ~1.1-1.2 |

**为什么选择 Unigram**:
1. Unigram 模型在中文上通常比 BPE 产生更合理的子词分割
2. 16K 词表更紧凑，减少稀疏 token 的嵌入训练负担
3. 更小的词表意味着 LM Head 参数更少，模型更高效

#### 2.4.2 备选: 复用 V3 Tokenizer

如果时间紧迫，可以直接复用 V3 的 SentencePiece BPE tokenizer (32K)，仅调整模型架构。这样可以避免重新训练 tokenizer 和重新预处理数据。

---

## 3. 训练策略

### 3.1 两阶段训练

```
Phase 1: 标准 CE 预训练 (Epochs 1-15)
├── 损失函数: CrossEntropy
├── 学习率: 6e-4 → Cosine Decay → 0
├── 目标: 让学生模型建立基础语言能力
└── Early Stopping: patience=5

Phase 2: 知识蒸馏微调 (Epochs 1-10)
├── 损失函数: λ_ce * CE + λ_kd * KL_div
├── λ_ce=0.3, λ_kd=0.7 (偏重 KD)
├── 学习率: 1e-4 → Cosine Decay → 0 (低 LR 微调)
├── 温度: T=2.0
├── 教师模型: V2 best_model 或 Qwen2.5-0.5B
└── Early Stopping: patience=3
```

### 3.2 超参数配置

#### Phase 1: 标准预训练

| 参数 | 值 | 说明 |
|------|-----|------|
| Optimizer | AdamW (β₁=0.9, β₂=0.95) | 标准 LLM 优化器 |
| Peak LR | 6e-4 | 小模型可用较高 LR |
| LR Scheduler | Cosine with Warmup | 稳定收敛 |
| Warmup Ratio | 5% | 充分预热 |
| Weight Decay | 0.1 | 标准正则化 |
| Batch Size / GPU | 32 | 小模型可用大 batch |
| Gradient Accumulation | 1 | 无需累积 |
| 有效 Batch Size | 128 (32×4) | 4 GPU |
| Max Sequence Length | 1,024 | 保持不变 |
| Training Epochs | 15 (max) | 配合 Early Stopping |
| Gradient Clipping | 1.0 | 防止梯度爆炸 |
| Dropout | 0.05 → 0 (退火) | 轻度正则化 |
| BPE Dropout | 0.1 | 数据增强 |
| Gradient Checkpointing | 否 | 小模型无需 |
| Mixed Precision | bf16 | A6000 原生支持 |

#### Phase 2: 知识蒸馏

| 参数 | 值 | 说明 |
|------|-----|------|
| Optimizer | AdamW (β₁=0.9, β₂=0.95) | 同 Phase 1 |
| Peak LR | 1e-4 | 低 LR 微调 |
| LR Scheduler | Cosine with Warmup | 同 Phase 1 |
| Warmup Ratio | 3% | 少量预热 |
| λ_ce | 0.3 | CE 损失权重 |
| λ_kd | 0.7 | KD 损失权重 |
| Temperature | 2.0 | 蒸馏温度 |
| K (top-K) | 10 | 教师 top-K logits |
| Training Epochs | 10 (max) | KD 不需要太多 epoch |
| Batch Size / GPU | 32 | 同 Phase 1 |

### 3.3 硬件利用

| GPU | 用途 | 显存估算 |
|-----|------|----------|
| GPU 0 | 学生模型训练 | ~8 GB |
| GPU 1 | 学生模型训练 | ~8 GB |
| GPU 2 | 学生模型训练 | ~8 GB |
| GPU 3 | 学生模型训练 | ~8 GB |
| - | 教师模型 (Phase 2, 离线) | ~2 GB (V2) / ~4 GB (Qwen-0.5B) |

> 60M 模型在 4×A6000 上训练速度预计 ~3-5 it/s，每 epoch 约 5-8 分钟。

---

## 4. 实施计划

### 4.1 文件结构

```
src/v5/
├── train_v5.py              # 主训练脚本 (含 KD)
├── train_tokenizer_v5.py    # Unigram tokenizer 训练 (可选)
├── generate_teacher_logits.py # 离线生成教师 logits
├── kd_dataset.py            # KD 数据集 (含教师 logits)
├── evaluate_v5.py           # 评测脚本
├── run_pipeline_v5.sh       # 一键训练 pipeline
└── README_V5.md             # V5 说明文档

plans/
└── V5_TRAINING_PLAN.md      # 本文档
```

### 4.2 实施步骤

#### Step 1: 准备 Tokenizer (可选，可复用 V3)

- 选项 A: 训练新的 16K Unigram tokenizer
- 选项 B: 直接复用 V3 的 32K BPE tokenizer（推荐，节省时间）

#### Step 2: Phase 1 标准预训练

```bash
# 使用 4 GPU 训练 ~60M 学生模型
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 --mixed_precision=bf16 \
    train_v5.py \
    --phase pretrain \
    --d_model 512 --n_layer 12 --n_head 8 --n_kv_heads 4 \
    --max_length 1024 --batch_size 32 \
    --learning_rate 6e-4 --num_epochs 15 \
    --output_dir ../../output/babylm-llama-v5
```

#### Step 3: 离线生成教师 Logits

```bash
# 使用 V2 best_model 作为教师
python generate_teacher_logits.py \
    --teacher_model_path ../../output/babylm-llama-v2/best_model \
    --data_dir ../../data/processed_v3 \
    --output_dir ../../output/teacher_logits_v2 \
    --top_k 10 \
    --batch_size 16
```

#### Step 4: Phase 2 知识蒸馏

```bash
# 加载 Phase 1 最佳模型 + 教师 logits 进行 KD
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 --mixed_precision=bf16 \
    train_v5.py \
    --phase kd \
    --student_model_path ../../output/babylm-llama-v5/best_model \
    --teacher_logits_dir ../../output/teacher_logits_v2 \
    --lambda_ce 0.3 --lambda_kd 0.7 \
    --temperature 2.0 \
    --learning_rate 1e-4 --num_epochs 10 \
    --output_dir ../../output/babylm-llama-v5-kd
```

---

## 5. 预期效果

### 5.1 性能目标

| 指标 | V3 (当前最佳) | V5 Phase 1 | V5 Phase 2 (KD) |
|------|---------------|------------|-----------------|
| Val PPL | ~542 | < 400 | < 300 |
| Val Loss | ~6.30 | < 5.99 | < 5.70 |
| 过拟合 Epoch | 7-10 | 10-15 | N/A |
| 训练时长 | ~2h | ~1.5h | ~0.5h |
| 参数量 | 125M | ~60M | ~60M |

### 5.2 为什么 V5 能超越 V3

1. **参数量匹配数据**: 60M / 82M tokens = 1.37 tokens/参数，远优于 V3 的 125M / 82M = 0.66
2. **知识蒸馏**: 教师模型的 soft logits 提供比 hard labels 更丰富的信息
3. **更紧凑的词表**: 16K 词表减少参数浪费，提高训练效率
4. **两阶段训练**: 先建立基础能力，再用 KD 精炼

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 教师模型词表不匹配 | 中 | 高 | 优先使用 V2 同词表教师；后续再做 Qwen 对齐 |
| 60M 模型容量不足 | 低 | 中 | 可扩展到 80M (d_model=576) |
| KD 效果不明显 | 中 | 低 | 调整 λ_ce/λ_kd 比例和温度 |
| 训练不稳定 | 低 | 中 | 降低 LR，增加 warmup |
| 磁盘空间不足 | 低 | 高 | 教师 logits 使用 float16 存储，定期清理 |

---

## 7. 与 DistilQwen2.5 的技术对照

| DistilQwen2.5 技术 | V5 适配 |
|---------------------|---------|
| Multi-Agent 数据增强 | ❌ 不适用（预训练阶段，非 instruction tuning） |
| 白盒 KD (Top-K Logits) | ✅ 核心采用，K=10 |
| Token Alignment | ⚠️ Phase 2 优先使用同词表教师避免此问题 |
| 温度缩放 | ✅ T=2.0 |
| 模型融合 | ❌ 暂不采用，复杂度过高 |
| 离线 logits 生成 | ✅ 采用，节省 GPU 显存 |

---

## 8. 快速启动指南

### 8.1 最简方案（复用 V3 tokenizer，仅 Phase 1）

如果时间紧迫，可以先只做 Phase 1 标准预训练（不含 KD），验证小模型效果：

```bash
cd /home/kehe/babyllm/babyLLM/src/v5
conda activate data

# 直接启动 Phase 1 训练
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 --mixed_precision=bf16 \
    train_v5.py \
    --phase pretrain \
    --data_dir ../../data \
    --tokenizer_dir ../../data/tokenizer_v3 \
    --output_dir ../../output/babylm-llama-v5 \
    --d_model 512 --n_layer 12 --n_head 8 --n_kv_heads 4 \
    --max_length 1024 --batch_size 32 \
    --learning_rate 6e-4 --weight_decay 0.1 \
    --num_epochs 15 --warmup_ratio 0.05 \
    --patience 5 --rope_theta 10000.0 \
    --wandb_project chinese-babylm \
    --wandb_run_name llama-v5-512d-12l-kd-pretrain
```

### 8.2 完整方案（含 KD）

先运行 Phase 1，完成后依次执行 Step 3-4。

---

*本计划基于 V1-V4 实验的深度分析和 DistilQwen2.5 (Wang et al., ACL 2025) 技术报告设计。*
