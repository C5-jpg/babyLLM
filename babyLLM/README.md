# BabyLM — 从头预训练中文语言模型

[![Competition](https://img.shields.io/badge/NLPCC%202026-ChineseBabyLM-blue)](https://chinese-babylm.github.io/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> 首届 ChineseBabyLM 挑战赛参赛项目 — 基于 LLaMA 架构从零预训练中文语言模型
>
> **15 个版本迭代 · PPL 597 → 38.68 · SOTA 性能**

---

## 目录

- [项目概述](#项目概述)
- [核心成果](#核心成果)
- [性能基准测试](#性能基准测试)
- [技术架构](#技术架构)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [版本历史](#版本历史)
- [常见问题](#常见问题)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [参考文献](#参考文献)

---

## 项目概述

本项目参与首届 **ChineseBabyLM 挑战赛**（NLPCC 2026），目标是在约 1 亿中文字符的儿童语料上从零预训练一个高性能的小型中文语言模型。

### 挑战赛信息

| 项目 | 详情 |
|------|------|
| 挑战赛 | NLPCC 2026 ChineseBabyLM Challenge |
| 主页 | <https://chinese-babylm.github.io/> |
| 训练数据 | [babylm-zho-100M](https://huggingface.co/datasets/chinese-babylm-org/babylm-zho-100M) |
| 评测代码 | [evaluation-pipeline-2025](https://github.com/SiyuanSong2004/evaluation-pipeline-2025) |
| 数据规模 | ~100M 中文字符 (~2M 行文本) |
| 约束条件 | 从零预训练，不可使用外部预训练模型，Token 数 ≤100M (Jieba) |
| 硬件 | 4 × NVIDIA RTX A6000 (48GB VRAM) |

### 核心成果

| 指标 | 数值 |
|------|------|
| **最佳 PPL** | **38.68** (V13, SOTA) |
| 最佳参数效率 | V12: 54M params, PPL=38.84 |
| 版本迭代 | 15 个版本 (V1–V15), 历时 ~3 周 |
| PPL 改善 | 597 → 38.68 (降幅 **93.5%**) |
| 官方评测 ZhoBLiMP | 63.5% |
| 官方评测 AFQMC | 69.0% |
| 官方评测 OCNLI | 64.0% |
| 训练精度 | bf16 混合精度 |
| 并行策略 | Accelerate DDP, 4×GPU |

---

## 性能基准测试

### 版本演进时间线

![Version Evolution Timeline](docs/assets/version_timeline.png)

*V1–V15 版本演进时间线，标注每个版本的关键创新与 PPL 指标。*

---

### PPL 演进趋势

![PPL Evolution](docs/assets/ppl_evolution.png)

*各版本最佳 PPL（对数刻度）。V2 的 597 到 V13 的 38.68，降幅达 93.5%。*

---

### 参数效率前沿

![Parameters vs PPL](docs/assets/params_vs_ppl.png)

*参数量与 PPL 的关系。红色虚线为帕累托前沿（最低 PPL/参数量比）。*

---

### 参数效率（PPL per 10M Params）

![Efficiency Frontier](docs/assets/efficiency_frontier.png)

*每个版本的参数效率（PPL / 10M params）。V12 是效率之王。*

---

### 官方评测结果

![Official Eval Radar](docs/assets/official_eval_radar.png)

*V13、V14、V15 在 CLUE 基准任务上的雷达图对比。*

| Version | ZhoBLiMP | Hanzi Structure | Hanzi Pinyin | AFQMC | OCNLI | TNEWS | WSC2020 |
|---------|----------|----------------|-------------|-------|-------|-------|---------|
| **V13** | 63.5 | 64.7 | 49.5 | 69.0 | 64.0 | 53.9 | 63.5 |
| **V14** | 64.3 | 62.4 | 41.9 | 69.0 | 66.0 | 54.1 | 63.5 |
| **V15** | 62.4 | 63.4 | 47.4 | 69.0 | 65.9 | 54.4 | 63.8 |

---

### 分阶段 PPL 改进

![Stage-by-Stage PPL](docs/assets/ppl_by_stage.png)

*V10–V15 每个训练阶段的 PPL 改进。CLM→MNTP 是核心改进阶段，后续阶段边际收益递减。*

---

### 技术贡献分析

![Technique Impact](docs/assets/technique_impact.png)

*各关键技术对 PPL 改善的估计贡献占比。SPM 分词器是单一最大改进来源。*

---

### 完整版本对比表

| 版本 | 架构 | 隐藏维度 | 层数 | 注意力头 | KV 头 | 参数量 | 最佳 PPL | 最佳阶段 | 状态 |
|------|------|----------|------|----------|-------|--------|----------|----------|------|
| V1 | GPT-2 | 768 | 12 | 12 | — | ~110M | ~343 | best_model | 完成 |
| V2 | LLaMA | 768 | 12 | 12 | 4 | ~125M | 597 | best_model | 完成 |
| V3 | LLaMA | 768 | 12 | 12 | 4 | ~125M | 542 | — | **失败** (NCCL) |
| V4 | LLaMA-deep | 1024 | 24 | 16 | 8 | ~350M | N/A | — | **失败** (过大) |
| V5 | LLaMA-small | 512 | 12 | 8 | 4 | ~51M | 525 | best_model | 完成 |
| V6 | LLaMA | 640 | 12 | 10 | 5 | ~75M | N/A | — | **失败** (数据丢失) |
| V7 | LLaMA | 448 | 12 | 8 | 4 | ~30M | 50.8 | best_model | 完成 |
| V8 | LLaMA | 512 | 12 | 8 | 4 | ~35M | 50.8 | stage3_polish | 完成 |
| V9 | LLaMA | 512 | 12 | 8 | 4 | ~35M | 50.8 | polish_probe | 完成 |
| V10 | LLaMA | 512 | 12 | 8 | 4 | 38.7M | 42.9 | stage3_polish | 完成 |
| V11 | LLaMA | 512 | 12 | 8 | 4 | 38.7M | 40.7 | stage5_sd_ema | 完成 |
| V12 | LLaMA | 576 | 14 | 9 | 3 | 54.2M | 38.8 | stage2_mntp_ema | 完成 |
| **V13** | **LLaMA** | **768** | **14** | **12** | **4** | **94.2M** | **38.7** | **stage2_mntp_ema** | **SOTA** |
| V14 | LLaMA | 640 | 12 | 10 | 5 | 59.2M | 41.8 | stage4_sd | 完成 |
| V15 | LLaMA | 640 | 14 | 10 | 5 | 68.2M | 45.1 | stage2_mntp_ema | 完成 |

---

## 技术架构

### 模型架构（以 V13 SOTA 为例）

```
LlamaForCausalLM
├── Tokenizer: SentencePiece Unigram (8K vocab)
├── Embedding: 768d (tied with LM head)
├── Transformer Blocks × 14
│   ├── RMSNorm (eps=1e-5)
│   ├── Self-Attention
│   │   ├── Q: 12 heads × 64d = 768d
│   │   ├── K: 4 heads × 64d = 256d  (GQA, 3:1 ratio)
│   │   ├── V: 4 heads × 64d = 256d
│   │   ├── RoPE (base=10000, max_pos=1024)
│   │   └── SDPA (Scaled Dot-Product Attention)
│   ├── RMSNorm
│   └── FFN (SwiGLU)
│       ├── Gate: 768d → 2048d
│       ├── Up:   768d → 2048d
│       └── Down: 2048d → 768d
├── RMSNorm
└── LM Head (768d → 8K, tied)
```

**关键架构参数：**

| 参数 | V13 (SOTA) | V12 (效率最佳) |
|------|-----------|--------------|
| hidden_size | 768 | 576 |
| num_layers | 14 | 14 |
| num_attention_heads | 12 | 9 |
| num_key_value_heads | 4 | 3 |
| intermediate_size | 2048 | 1536 |
| head_dim | 64 | 64 |
| vocab_size | 8192 | 8192 |
| max_position_embeddings | 1024 | 1024 |
| Total Params | 94.2M | 54.2M |
| tie_word_embeddings | True | True |

---

### 训练流水线架构

![Training Pipeline](docs/assets/training_pipeline.png)

*多阶段训练流水线：数据预处理 → Stage 1 (CLM) → Stage 2 (MNTP) → Stage 3 (Polish, 可选) → 评估。*

#### Stage 1: Causal Language Modeling (CLM)

```bash
python src/v13/train_v13.py \
    --stage clm \
    --d_model 768 --n_layer 14 --n_head 12 --n_kv_heads 4 \
    --lr 6e-4 --epochs 8 --batch_size 16 --grad_accum_steps 2 \
    --scheduler sgdr --focal_loss --focal_gamma 1.5 \
    --use_ema --ema_decay 0.999 \
    --label_smoothing 0.1 --label_smoothing_anneal \
    --attention_dropout 0.1 \
    --data_dir /path/to/data_v13 \
    --output_dir /path/to/output/stage1_clm_sgdr
```

#### Stage 2: Masked Next Token Prediction (MNTP)

```bash
python src/v13/train_v13.py \
    --stage mntp \
    --resume_from /path/to/stage1/best_model_ema \
    --lr 5e-4 --epochs 10 \
    --dynamic_clm_ratio --mask_ratio_start 0.25 --mask_ratio_end 0.1 \
    --scheduler cosine --focal_loss --focal_gamma 1.0 \
    --use_ema --ema_decay 0.999 \
    --label_smoothing 0.05 --label_smoothing_anneal \
    --attention_dropout 0.05
```

#### Stage 3: Polish (可选)

```bash
python src/v13/train_v13.py \
    --stage clm \
    --resume_from /path/to/stage2/best_model_ema \
    --lr 1e-5 --epochs 5 \
    --no_dropblock --no_stochastic_depth
```

---

### 关键训练技术

| 技术 | 说明 | 版本引入 | PPL 改善 |
|------|------|----------|----------|
| **SentencePiece 分词器** | 从 BPE 迁移到 SPM Unigram，8K/32K 词表 | V3 | ~74% |
| **MNTP 混合训练** | CLM + Masked Next Token Prediction 混合训练 | V7 | ~16% |
| **EMA 权重平均** | 指数移动平均 (decay=0.999) | V11 | ~7% |
| **多阶段流水线** | CLM → MNTP → Polish 多阶段递进训练 | V8 | ~8% |
| **Focal Loss** | 解决类别不平衡，gamma=1.5~2.0 | V12 | ~5% |
| **PPL 数据过滤** | 使用已有模型过滤低质量数据 (max_ppl=250) | V13 | ~4% |
| **SGDR 学习率** | 带热重启的余弦退火 (T0=2k, Tmult=2) | V1 (意外), V11 (正式) | ~3% |
| **Label Smoothing** | 标签平滑 (0.1 → 退火至 0) | V10 | ~2% |
| **GQA 注意力** | 分组查询注意力 (3:1 ratio) | V7 | 减少 KV 参数 |
| **Gradient Checkpointing** | 梯度检查点节省显存 | V7 | 训练稳定性 |
| **bf16 混合精度** | Brain Float 16 训练 | V2 | 2× 吞吐量 |

---

## 项目结构

```
babyLLM/
├── README.md                          # 本文档
├── requirements.txt                   # Python 依赖
├── .gitignore
│
├── src/                               # 各版本源代码
│   ├── v1/                            # V1: GPT-2 基线
│   │   ├── train.py, train_tokenizer.py, evaluate_model.py
│   ├── v2/                            # V2: LLaMA 架构迁移
│   │   ├── train_v2.py, train_tokenizer_v2.py, evaluate_v2.py
│   ├── v3/                            # V3: SPM 分词器 (NCCL 失败)
│   │   ├── train_v3.py, spm_tokenizer.py, TRAINING_LOG_V3.md
│   ├── v4/                            # V4: 深层模型 (参数过多)
│   │   ├── train_v4.py, ensemble_checkpoints.py
│   ├── v5/                            # V5: 小模型 + 知识蒸馏
│   │   ├── train_v5.py, generate_teacher_logits.py
│   ├── v6/                            # V6: 3 阶段流水线 (数据丢失)
│   │   ├── train_v6.py, prepare_data_v6.py
│   ├── v7/                            # V7: MNTP 混合训练
│   │   ├── train_v7.py, prepare_data_v7.py, evaluate_v7.py
│   ├── v8/                            # V8: 简化 3 阶段
│   │   ├── train_v8.py, evaluate_v8.py
│   ├── v9/                            # V9: 探针实验
│   │   ├── train_v9.py, evaluate_v9.py
│   ├── v10/                           # V10: 生产管线
│   │   ├── train_v10.py, evaluate_v10.py, notify.py
│   ├── v11/                           # V11: EMA + SGDR + Self-Distill
│   │   ├── train_v11.py, evaluate_v11.py, convert_tokenizer.py, swa_v11.py
│   ├── v12/                           # V12: Focal Loss + 数据清洗
│   │   ├── train_v12.py, evaluate_v12.py, convert_tokenizer.py, clean_data.py
│   ├── v13/                           # V13: PPL 过滤 (SOTA)
│   │   ├── train_v13.py, evaluate_v13.py, convert_tokenizer.py, prepare_data.py
│   ├── v14/                           # V14: 效率版
│   │   ├── train_v14.py, evaluate_v14.py, convert_tokenizer.py, prepare_data.py
│   ├── v15/                           # V15: 优化架构
│   │   ├── train_v15.py, evaluate_v15.py, convert_tokenizer.py, prepare_data.py
│   ├── analyze_versions.py            # 版本分析脚本
│   └── eval_standardized.py           # 标准化评估脚本
│
├── docs/                              # 文档
│   ├── assets/                        # 可视化图表 (PNG)
│   │   ├── ppl_evolution.png
│   │   ├── params_vs_ppl.png
│   │   ├── version_timeline.png
│   │   ├── training_pipeline.png
│   │   ├── official_eval_radar.png
│   │   ├── ppl_by_stage.png
│   │   ├── efficiency_frontier.png
│   │   └── technique_impact.png
│   ├── generate_charts.py             # 图表生成脚本
│   ├── V1_V14_COMPREHENSIVE_ANALYSIS.md
│   ├── V1_V14_TRAINING_EXPERIENCE.md
│   ├── V13_DEEP_ANALYSIS_REPORT.md
│   └── V15_TRAINING_PROTOCOL.md
│
├── data/                              # 数据目录 (gitignored)
│   ├── tokenizer_v7/                  # SPM Unigram 分词器
│   └── processed_v7/                  # 预处理数据
│
├── logs/                              # 训练日志
│   ├── standardized_eval_results.json
│   └── *.jsonl                        # 时间戳日志
│
├── plans/                             # 规划文档
│
├── launch_v10_pipeline.sh             # V10 训练脚本
├── launch_v11_pipeline.sh             # V11 训练脚本
├── launch_v12_pipeline.sh             # V12 训练脚本
├── launch_v13_pipeline.sh             # V13 训练脚本 (SOTA)
├── launch_v14_pipeline.sh             # V14 训练脚本
├── launch_v15_pipeline.sh             # V15 训练脚本
└── accelerate_config_v14.yaml         # Accelerate 配置
```

---

## 环境配置

### 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | 1× NVIDIA GPU (16GB+) | 4× NVIDIA RTX A6000 (48GB) |
| RAM | 32GB | 64GB+ |
| 存储 | 50GB SSD | 200GB SSD + HDD |
| CUDA | 11.8+ | 12.4 |

### 软件依赖

```bash
# Python 依赖
torch>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.0
datasets>=2.14.0
accelerate>=0.21.0
sentencepiece>=0.1.99
jieba>=0.42.1
tqdm>=4.65.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
scipy>=1.11.0
wandb>=0.15.0
```

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/babyLLM.git
cd babyLLM

# 2. 创建 conda 环境
conda create -n babylm python=3.10 -y
conda activate babylm

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 5. 配置 Accelerate (4 GPU DDP)
accelerate config  # 或使用提供的配置文件
accelerate launch --config_file accelerate_config_v14.yaml --help
```

---

## 快速开始

### 1. 下载数据

```bash
# 从 HuggingFace 下载训练数据
python -c "
from datasets import load_dataset
ds = load_dataset('chinese-babylm-org/babylm-zho-100M')
ds['train'].to_json('data/raw/train.jsonl')
"
```

### 2. 准备数据和分词器

```bash
# 使用 V13 的数据准备脚本
python src/v13/prepare_data.py \
    --input_dir data/raw \
    --output_dir data/processed_v13 \
    --tokenizer_dir data/tokenizer_v7
```

### 3. 训练模型（以 V13 SOTA 为例）

```bash
# 一键启动 V13 全阶段训练
bash launch_v13_pipeline.sh

# 或分阶段手动训练
# Stage 1: CLM + SGDR
python src/v13/train_v13.py \
    --stage clm --d_model 768 --n_layer 14 --n_head 12 --n_kv_heads 4 \
    --lr 6e-4 --epochs 8 --scheduler sgdr \
    --focal_loss --focal_gamma 1.5 --use_ema \
    --data_dir data/processed_v13 \
    --output_dir output/babylm-v13/stage1_clm_sgdr

# Stage 2: MNTP
python src/v13/train_v13.py \
    --stage mntp --resume_from output/babylm-v13/stage1_clm_sgdr/best_model_ema \
    --lr 5e-4 --epochs 10 --dynamic_clm_ratio \
    --output_dir output/babylm-v13/stage2_mntp
```

### 4. 评估模型

```bash
python src/v13/evaluate_v13.py \
    --model_path output/babylm-v13/stage2_mntp/best_model_ema \
    --data_path data/processed_v13/val.txt \
    --tokenizer_dir data/tokenizer_v7
```

### 5. 生成图表（可选）

```bash
python docs/generate_charts.py
# 图表将保存到 docs/assets/*.png
```

---

## 详细使用说明

### 训练脚本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--stage` | 训练阶段 (`clm` / `mntp`) | `clm` |
| `--d_model` | 隐藏维度 | 768 |
| `--n_layer` | Transformer 层数 | 14 |
| `--n_head` | 注意力头数 | 12 |
| `--n_kv_heads` | KV 头数 (GQA) | 4 |
| `--lr` | 学习率 | 6e-4 |
| `--epochs` | 训练轮数 | 8 |
| `--batch_size` | 每 GPU 批大小 | 16 |
| `--grad_accum_steps` | 梯度累积步数 | 2 |
| `--max_length` | 最大序列长度 | 1024 |
| `--stride` | 滑动窗口步长 | 512 |
| `--scheduler` | 学习率调度器 (`cosine` / `sgdr`) | `sgdr` |
| `--focal_loss` | 启用 Focal Loss | False |
| `--focal_gamma` | Focal Loss gamma | 2.0 |
| `--use_ema` | 启用 EMA | False |
| `--ema_decay` | EMA 衰减率 | 0.999 |
| `--label_smoothing` | 标签平滑系数 | 0.1 |
| `--label_smoothing_anneal` | 标签平滑退火 | False |
| `--attention_dropout` | 注意力 Dropout | 0.1 |
| `--dynamic_clm_ratio` | 动态 CLM 比例 (MNTP 阶段) | False |
| `--mask_ratio_start` | 起始掩码比例 | 0.25 |
| `--mask_ratio_end` | 终止掩码比例 | 0.1 |
| `--bpe_dropout` | BPE Dropout | 0.1 |
| `--patience` | 早停耐心值 | 3 |
| `--eval_steps` | 评估间隔步数 | 200 |

### 多阶段流水线说明

| 阶段 | 目标 | 学习率 | 调度器 | 关键技术 |
|------|------|--------|--------|----------|
| Stage 1: CLM | 因果语言建模 | 6e-4 | SGDR | Focal Loss, EMA, Label Smoothing |
| Stage 2: MNTP | 掩码下一词预测 | 5e-4 | Cosine | Dynamic CLM ratio, Focal Loss, EMA |
| Stage 3: Polish | 精调 (可选) | 1e-5 | Cosine | 无正则化 |

### 数据预处理流程

1. **原始数据** → HuggingFace `babylm-zho-100M`
2. **清洗** → 去除过短行 (≤2 chars), 中文字符比例过滤
3. **去重** → MinHash 去重
4. **PPL 过滤** (V13+) → 使用已训练模型过滤低质量样本 (max_ppl=250)
5. **Tokenization** → SentencePiece Unigram (8K/32K vocab)
6. **分块** → 滑动窗口 (max_length=1024, stride=512, 50% 重叠)

---

## 版本历史

### 成功版本 (V7–V13)

| 版本 | 日期 | 关键创新 | PPL | 参数量 | 训练时长 |
|------|------|----------|-----|--------|----------|
| **V7** | 2026-04 | MNTP 混合训练, 8K 词表 | 50.8 | 30M | — |
| **V8** | 2026-04 | 简化 3 阶段流水线 | 50.8 | 35M | — |
| **V9** | 2026-04 | 探针实验 (stride, smoothing) | 50.8 | 35M | — |
| **V10** | 2026-04 | 生产管线, SPM Unigram | 42.9 | 38.7M | 2h36m |
| **V11** | 2026-05 | EMA + SGDR + Self-Distillation | 40.7 | 38.7M | — |
| **V12** | 2026-05 | Focal Loss + 数据清洗, 效率之王 | 38.8 | 54.2M | 8h52m |
| **V13** | 2026-05 | PPL 数据过滤, 最大模型, **SOTA** | **38.7** | **94.2M** | 9h34m |

### 效率版本 (V14–V15)

| 版本 | 日期 | 关键创新 | PPL | 参数量 | 训练时长 |
|------|------|----------|-----|--------|----------|
| **V14** | 2026-05 | 效率版, 5 阶段流水线 | 41.8 | 59.2M | 2h42m |
| **V15** | 2026-05 | 深层架构 (14L), Multi-scale EMA | 45.1 | 68.2M | 3h25m |

### 失败版本 (V1–V6)

| 版本 | 失败原因 | 教训 |
|------|----------|------|
| **V1** | GPT-2 架构限制, LR 调度器 bug | 需要迁移到 LLaMA 架构 |
| **V2** | ByteLevel BPE 不适合中文 | 选择合适的分词器至关重要 |
| **V3** | NCCL 通信超时 | 检查网络和 CUDA 版本兼容性 |
| **V4** | 350M 参数过大, tokens/param=0.23 | 遵守 Chinchilla scaling law |
| **V5** | 权重丢失 | 定期备份检查点 |
| **V6** | 过度数据清洗导致 79% 数据丢失 | 数据清洗需保守 |

### 版本关系图

```
V1 (GPT-2) ──→ V2 (LLaMA) ──→ V3 (SPM) ──→ V4 (Deep, failed)
                                    │
                                    ├──→ V5 (Small + KD)
                                    │      │
                                    │      └──→ V6 (3-stage, data lost)
                                    │
                                    └──→ V7 (MNTP) ──→ V8 (3-stage) ──→ V9 (Probes)
                                                              │
                                                              └──→ V10 (Production)
                                                                      │
                                                                      ├──→ V11 (EMA+SGDR)
                                                                      │
                                                                      ├──→ V12 (Focal Loss) ──→ V13 (PPL Filter, SOTA)
                                                                      │
                                                                      └──→ V14 (Efficiency) ──→ V15 (Optimized)
```

---

## 常见问题

### Q: 为什么 V4 的 350M 参数模型表现不好？

V4 违反了 **Chinchilla scaling law**。100M tokens 对于 350M 参数来说太少（tokens/param = 0.23），导致严重欠训练。最佳比例约为 tokens/param ≈ 1.8–2.0×。

### Q: 为什么 V6 数据丢失了 79%？

V6 使用了过于激进的数据清洗策略，导致有效训练数据严重不足。教训：数据清洗应保守，宁可保留一些噪声数据也不要丢失有用数据。

### Q: MNTP 混合训练为什么有效？

CLM 只能看到左侧上下文，而 MNTP 通过随机掩码强制模型利用双向上下文信息。两者结合提供了更全面的语言理解能力。

### Q: EMA 和 SWA 有什么区别？

- **EMA** (Exponential Moving Average): 在线平均权重，每个训练步更新，decay=0.999
- **SWA** (Stochastic Weight Averaging): 离线周期性采样模型权重取平均

EMA 更稳定，SWA 更灵活。V11 两者都尝试了，EMA 效果略好。

### Q: 为什么 V15 表现不佳 (PPL=45.14)？

主要原因：
1. `intermediate_size=1706` 未对齐到 256 的倍数，可能导致权重布局不优化
2. 使用了 V14 的 PPL 过滤数据（比 V13 数据少了 43K 行）
3. 学习率 5e-4 偏低（V12/V13 用 6e-4）

### Q: 如何选择词表大小？

- **8K 词表** (V7–V15): 更高的 token 覆盖率，适合小模型
- **32K 词表**: 更细粒度的 tokenization，适合大模型
- 实验表明 8K 对于 ≤100M 参数模型足够

### Q: 训练过程中 OOM 怎么办？

1. 减小 `--batch_size` (如 16→8)
2. 增加 `--grad_accum_steps` 保持等效批大小
3. 启用 `gradient_checkpointing` (默认已开启)
4. 减小 `--max_length` (如 1024→512)

---

## 贡献指南

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/v16-improvement`)
3. 提交更改 (`git commit -m 'feat(v16): add new technique'`)
4. 推送到远程 (`git push origin feature/v16-improvement`)
5. 创建 Pull Request

### 代码规范

- Python 3.10+, 遵循 PEP 8
- 使用 type hints
- 每个版本的代码放在独立目录 (`src/vN/`)
- 训练脚本必须支持 `--resume_from` 断点续训
- 评估结果必须包含 ISO 8601 时间戳

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---

## 参考文献

- Touvron, H. et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*. arXiv:2302.13971
- Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864
- Shazeer, N. (2020). *GLU Variants Improve Transformer*. arXiv:2002.05202
- Zhang, Z. & Sabuncu, M. (2018). *Focal Loss for Dense Object Detection*. arXiv:1708.02002
- Loshchilov, I. & Hutter, F. (2016). *SGDR: Stochastic Gradient Descent with Warm Restarts*. arXiv:1608.03983
- Kudo, T. & Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. arXiv:1808.06226
- ChineseBabyLM Challenge: <https://chinese-babylm.github.io/>
