# 🏋️ Training Log V3 - Chinese BabyLM SOTA 冲刺

> **创建时间**: 2026-04-21 11:49 (北京时间)  
> **训练开始**: 2026-04-21 11:27  
> **状态**: 🟢 训练进行中 (Epoch 2)

---

## 📋 项目概览

| 项目 | 详情 |
|------|------|
| **比赛** | Chinese BabyLM Challenge |
| **目标** | 实现比赛 SOTA |
| **版本** | V3 (Phase 1-4 修复版) |
| **模型架构** | LLaMA (768d) |
| **Tokenizer** | SentencePiece BPE (32000 vocab) |
| **训练数据** | babylm-zho-100M |
| **GPU** | 4x GPU (Accelerate DD) |
| **WandB** | ✅ 已启用 |

---

## 🔧 Phase 1-4 修复详解

### Phase 1: Tokenizer 升级
- **修复**: 从 HuggingFace BERT tokenizer 升级到 SentencePiece BPE
- **原因**: BPE tokenizer 对中文更友好，词表利用率更高
- **效果**: 减少 UNK token，提高训练效率
- **文件**: `spm_tokenizer.py`, `train_tokenizer_v3.py`

### Phase 2: 模型架构优化
- **修复**: 采用标准 LLaMA 架构 (768 hidden dim)
- **参数**: 
  - hidden_size: 768
  - num_hidden_layers: 12
  - num_attention_heads: 12
  - intermediate_size: 2048
  - max_position_embeddings: 1024
- **特性**: RoPE, SwiGLU, RMSNorm, Gradient Checkpointing

### Phase 3: 训练策略优化
- **修复**: 完善学习率调度和数据加载
- **调度器**: Linear warmup (500 steps) + Cosine decay
- **最大学习率**: 6e-4
- **Warmup 起始 LR**: 2.1e-5
- **Batch size**: 64 (per GPU: 16, 4 GPUs)
- **Block size**: 1024 tokens

### Phase 4: 监控与稳定性
- **修复**: 集成 WandB 实时监控
- **特性**: 
  - 训练/验证 loss 曲线
  - 学习率追踪
  - 最佳模型自动保存 (基于 val loss)
  - Early stopping (patience=3)
  - Perplexity 计算

---

## 📊 训练配置

### 超参数

```yaml
model:
  architecture: LLaMA
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  intermediate_size: 2048
  max_position_embeddings: 1024
  vocab_size: 32000

training:
  epochs: 100 (early stopping)
  batch_size: 64
  block_size: 1024
  learning_rate: 6e-4
  warmup_steps: 500
  scheduler: cosine
  weight_decay: 0.1
  gradient_checkpointing: true
  
data:
  train_samples: 73,041
  val_samples: 3,922
  steps_per_epoch: 1,141
  val_steps: 62
  
accelerate:
  config: accelerate_config_v2.yaml
  num_gpus: 4
  mixed_precision: "no" (fp32)
```

---

## 📈 训练进度

### Epoch 1 ✅ 完成

| 指标 | 值 |
|------|-----|
| **耗时** | 14分17秒 |
| **训练速度** | ~1.33 it/s |
| **最终 Train Loss** | 7.2147 |
| **最终 LR** | 4.63e-04 |
| **Val Loss** | 7.3004 |
| **Val PPL** | 1480.82 |
| **最佳模型** | ✅ 已保存 |

#### Loss 下降轨迹 (Epoch 1)

| Step | Loss | LR | 备注 |
|------|------|-----|------|
| 49 | 9.9395 | 2.10e-05 | 初始高位 |
| 99 | 9.2308 | 4.21e-05 | 快速下降 |
| 149 | 8.6245 | 6.31e-05 | |
| 199 | 8.4506 | 8.42e-05 | |
| 249 | 8.2747 | 1.05e-04 | |
| 299 | 8.1909 | 1.26e-04 | |
| 349 | 8.1132 | 1.47e-04 | |
| 399 | 7.9140 | 1.68e-04 | |
| 449 | 7.7945 | 1.89e-04 | |
| 499 | 7.8390 | 2.10e-04 | warmup 中段 |
| 549 | 7.7640 | 2.31e-04 | |
| 599 | 7.6783 | 2.52e-04 | |
| 649 | 7.5754 | 2.73e-04 | |
| 699 | 7.7094 | 2.95e-04 | |
| 749 | 7.5133 | 3.16e-04 | |
| 799 | 7.4429 | 3.37e-04 | |
| 849 | 7.7420 | 3.58e-04 | |
| 899 | 7.2011 | 3.79e-04 | |
| 949 | 7.6798 | 4.00e-04 | |
| 999 | 7.4541 | 4.21e-04 | |
| 1049 | 7.3665 | 4.42e-04 | |
| 1099 | 7.2147 | 4.63e-04 | Epoch 1 结束 |

### Epoch 2 🔄 进行中

| Step | Loss | LR | 备注 |
|------|------|-----|------|
| 8 | 7.2385 | 4.84e-04 | 起始 |
| 58 | 7.3208 | 5.05e-04 | |
| 108 | 7.2374 | 5.26e-04 | |
| 158 | 7.2719 | 5.47e-04 | |
| 208 | 7.0576 | 5.68e-04 | 突破 7.1! |
| 258 | 7.0143 | 5.89e-04 | |
| 308 | 7.0760 | 6.00e-04 | LR 达峰值 |
| 358 | 7.1155 | 6.00e-04 | |
| 408 | 7.0390 | 6.00e-04 | |
| 458 | 7.0728 | 6.00e-04 | 当前 (~42%) |

---

## ⏱️ 时间估算

| 项目 | 时间 |
|------|------|
| **每 Epoch 训练** | ~14分17秒 |
| **每 Epoch 验证** | ~12秒 |
| **模型保存** | ~3秒 |
| **每 Epoch 总计** | ~14.5 分钟 |
| **Early Stopping patience** | 3 epochs |
| **预计总 Epoch 数** | 5-10 |
| **已用时间** | ~20 分钟 |
| **预计剩余** | 50-120 分钟 |
| **预计完成** | 12:40 - 13:50 (北京时间) |

---

## 🔗 WandB 监控

| 项目 | 详情 |
|------|------|
| **Project** | chinese-babylm |
| **Run Name** | llama-v3-spm-768d-4gpu |
| **Run ID** | 5jn3qo90 |
| **Dashboard** | [🔗 查看训练曲线](https://wandb.ai/c5galaxies-sjtu-hpc-center/chinese-babylm/runs/5jn3qo90) |
| **User** | c5galaxies |

### 监控指标
- `train/loss` - 训练 loss (每 50 steps)
- `train/learning_rate` - 学习率变化
- `val/loss` - 验证 loss (每 epoch)
- `val/perplexity` - 验证困惑度 (每 epoch)

---

## 📁 输出文件

```
../../output/babylm-llama-v3/
├── best_model/          # 最佳模型 (Epoch 1: val_loss=7.3004)
│   ├── config.json
│   └── model.safetensors
└── train_v3.log         # 训练日志
```

---

## 🎯 SOTA 目标

| 指标 | 当前 | 目标 |
|------|------|------|
| Val Loss | 7.3004 (Epoch 1) | < 5.0 |
| Val PPL | 1480.82 (Epoch 1) | < 150 |
| Epoch | 1/100 | 收敛为止 |

> **备注**: Epoch 2 训练中 loss 已降至 ~7.04，持续下降趋势良好。预计 5-8 epoch 后达到较优结果。