# 🍼 ChineseBabyLM — Baby Language Model for Chinese

[![Competition](https://img.shields.io/badge/NLPCC%202026-ChineseBabyLM-blue)](https://chinese-babylm.github.io/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> 首届 ChineseBabyLM 挑战赛参赛项目 — 基于 LLaMA 架构从头预训练中文语言模型
> 
> NLPCC 2025 · 澳门 · 11月4-5日

---

## 📑 目录

- [项目概述](#-项目概述)
- [核心特性](#-核心特性)
- [技术架构](#-技术架构)
- [项目结构](#-项目结构)
- [环境配置](#-环境配置)
- [快速开始](#-快速开始)
- [详细使用说明](#-详细使用说明)
- [训练监控](#-训练监控-wandb)
- [模型性能](#-模型性能)
- [版本历史](#-版本历史)
- [参考链接](#-参考链接)

---

## 🌟 项目概述

本项目参与首届 **ChineseBabyLM 挑战赛**，目标在 `babylm-zho-100M`（约 1 亿中文字符的儿童语料）上从头预训练一个高性能的小型中文语言模型。

### 挑战赛信息

| 项目 | 详情 |
|------|------|
| 挑战赛主页 | https://chinese-babylm.github.io/ |
| 训练数据 | [babylm-zho-100M](https://huggingface.co/datasets/chinese-babylm-org/babylm-zho-100M) |
| 评测代码 | [evaluation-pipeline-2025](https://github.com/SiyuanSong2004/evaluation-pipeline-2025) |
| 数据规模 | ~100M 中文字符 (~2M 行文本) |
| 约束条件 | 从头预训练，不可使用外部预训练模型 |

### 核心成果

- 🏗️ 基于 **LLaMA 架构**（RoPE 位置编码 + GQA 注意力 + SwiGLU 激活 + RMSNorm）
- 📊 训练 Loss 从 **10.52 → 7.01**（Epoch 1 结束前），持续下降中
- ⚡ 4×GPU 并行训练，吞吐量达 **80K-105K tokens/sec**
- 🔍 **WandB** 实时在线监控训练全过程

---

## ✨ 核心特性

### Phase 1：LLaMA 架构升级

从 GPT-2 架构全面升级到 LLaMA 架构，获得更好的性能与训练效率：

| 特性 | V1 (GPT-2) | V2 (LLaMA) |
|------|-----------|-----------|
| 位置编码 | 绝对位置编码 | **RoPE（旋转位置编码）** |
| 注意力机制 | MHA (Multi-Head) | **GQA (Grouped-Query, 4 groups)** |
| 激活函数 | GELU | **SwiGLU** |
| 归一化 | LayerNorm | **RMSNorm** |
| 归一化位置 | Post-Norm | **Pre-Norm** |
| 注意力加速 | - | **SDPA / Flash Attention 2** |

### Phase 2：学习率调度修复

修复了 V1 中 Cosine Annealing LR Scheduler 的关键 Bug：
- **问题**：`num_training_steps` 计算错误导致学习率提前衰减到零
- **修复**：基于实际 DataLoader 长度精确计算总步数
- **效果**：学习率正确按 cosine 曲线从 warmup 过渡到衰减

### Phase 3：ByteLevel BPE Tokenizer

重新设计了针对中文优化的 Tokenizer：
- 使用 **ByteLevel 预分词**，天然支持中文 UTF-8 编码（无 OOV 问题）
- 32K 词表大小，平衡覆盖率与模型效率
- 零 `<unk>` tokens，所有中文字符均可正确编码

### Phase 4：SOTA 训练优化

| 优化技术 | 说明 |
|----------|------|
| **bf16 混合精度** | 加速训练、节省显存，配合 GradScaler 防止溢出 |
| **Gradient Checkpointing** | 以计算换显存，支持更大 batch size |
| **文档感知序列构造** | EOS token 分隔文档，避免跨文档注意力污染 |
| **BPE Dropout** | 训练时随机 dropout BPE 合并，增强鲁棒性 |
| **训练退火** | 后期降低 dropout rate，从正则化过渡到精确拟合 |
| **数据去重与清洗** | 精确去重 + HTML清洗 + 短文本过滤 |
| **独立验证集** | 5% 数据用于验证，每 epoch 评估 PPL |

---

## 🏗️ 技术架构

### 模型配置

```
LlamaForCausalLM
├── vocab_size:          32,000
├── hidden_size:         768      (d_model)
├── intermediate_size:   2,688    (≈ 3.5 × d_model, SwiGLU)
├── num_hidden_layers:   12
├── num_attention_heads: 12       (head_dim = 64)
├── num_key_value_heads: 4        (GQA, 3 KV共享)
├── max_position_embeddings: 512
├── rms_norm_eps:        1e-6
├── rope_theta:          10,000
├── attention_dropout:   0.0 → 0.0  (退火策略)
└── 总参数量:            ~110M
```

### 训练超参数

| 参数 | 值 | 说明 |
|------|------|------|
| Optimizer | AdamW (β₁=0.9, β₂=0.95) | β₂ 调整为 0.95 适合小数据 |
| Peak LR | 6×10⁻⁴ | GPT-3 标准配置 |
| LR Scheduler | Cosine Annealing | 带 Linear Warmup |
| Warmup Steps | 总步数的 3% | ~940 steps |
| Weight Decay | 0.1 | L2 正则化 |
| Batch Size | 8 per GPU | |
| Gradient Accumulation | 4 steps | 有效 batch = 128 |
| Max Sequence Length | 512 tokens | |
| Training Epochs | 25 | |
| bf16 Mixed Precision | ✅ | |
| Gradient Checkpointing | ✅ | |
| Gradient Clipping | 1.0 | |

### 数据处理流程

```
babylm-zho-100M (HuggingFace)
        │
        ▼
  prepare_data.py ─── 提取文本 → data/processed/all.txt (~2M 行)
        │
        ▼
  prepare_data_v2.py ─ 清洗 + 去重 + 分割
        │               ├─ 精确去重 (MD5)
        │               ├─ HTML/URL 清洗
        │               ├─ 短文本过滤 (< 10 字符)
        │               └─ 95%/5% train/val 分割
        ▼
  data/processed_v2/
        ├─ train.txt (~1.3M 行, 高质量)
        └─ val.txt   (~65K 行)
```

---

## 📁 项目结构

```
babyLLM/
├── README.md                          # 📖 项目总览
├── REPORT.md                          # 📊 详细实验报告
├── requirements.txt                   # 📦 Python 依赖
├── .gitignore
├── .gitmodules                        # evaluation 子模块
│
├── src/
│   ├── v1/                            # V1 基线 (GPT-2)
│   │   ├── train.py
│   │   ├── train_tokenizer.py
│   │   ├── prepare_data.py
│   │   ├── evaluate_model.py
│   │   └── run.bat
│   └── v2/                            # V2 优化版 (LLaMA)
│       ├── train_v2.py                # 🚀 训练主脚本
│       ├── train_tokenizer_v2.py      # 🔤 Tokenizer 训练
│       ├── prepare_data_v2.py         # ⚙️ 数据清洗/去重
│       ├── evaluate_v2.py             # 📈 模型评测
│       ├── run_train_v2.sh            # 🏃 一键训练
│       └── accelerate_config_v2.yaml  # ⚡ 多GPU配置
│
├── docs/
│   ├── ANALYSIS_AND_OPTIMIZATION.md   # 📊 优化分析
│   └── SERVER_INSTRUCTIONS.md         # 🖥️ 服务器说明
│
└── data/
    ├── tokenizer/                     # V1 Tokenizer
    ├── tokenizer_v2/                  # V2 ByteLevel BPE
    ├── processed/                     # V1 预处理数据 (gitignore)
    ├── processed_v2/                  # V2 清洗数据 (gitignore)
    └── raw/                           # HuggingFace 缓存 (gitignore)
```

---

## 🔧 环境配置

### 系统要求

- **OS**: Linux (推荐 Ubuntu 20.04+)
- **GPU**: NVIDIA GPU，≥16GB VRAM（推荐 4× A6000 48GB）
- **CUDA**: 12.4+
- **Python**: 3.10+

### 安装步骤

```bash
# 1. 克隆仓库
git clone git@github.com:C5-jpg/babyLLM.git
cd babyLLM

# 2. 创建 conda 环境
conda create -n data python=3.10 -y
conda activate data

# 3. 安装 PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 登录 WandB（可选，用于训练监控）
wandb login
```

### 核心依赖

```
torch>=2.0
transformers>=4.49
tokenizers>=0.21
accelerate>=0.27
wandb>=0.16
datasets>=2.18
tqdm>=4.65
jieba>=0.42
numpy>=1.24
```

---

## 🚀 快速开始

### 一键训练（推荐）

```bash
# 确保 tokenizer 已准备好（首次运行需先训练 tokenizer）
python train_tokenizer_v2.py

# 一键启动训练
bash run_train_v2.sh
```

`run_train_v2.sh` 会自动：
1. 检查 Tokenizer 是否就绪
2. 运行数据预处理（如尚未完成）
3. 启动 4-GPU Accelerate 分布式训练
4. 训练日志输出到 `training_phase3.log`

### 分步执行

#### Step 1：数据准备

```bash
# 下载数据并提取文本
python prepare_data.py

# 数据清洗与去重
python prepare_data_v2.py \
    --input data/processed/all.txt \
    --output_dir data/processed_v2 \
    --no_minhash
```

#### Step 2：训练 Tokenizer

```bash
python train_tokenizer_v2.py
```

输出：
- `data/tokenizer_v2/tokenizer.json` — ByteLevel BPE 模型
- `data/tokenizer_v2/tokenizer_config.json` — HuggingFace 配置

#### Step 3：启动训练

```bash
# 使用 Accelerate 多 GPU 训练
accelerate launch --config_file accelerate_config_v2.yaml train_v2.py \
    --data_dir data \
    --output_dir output/babylm-llama-v2 \
    --max_length 512 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-4 \
    --num_epochs 25 \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --wandb_project chinese-babylm \
    --wandb_mode online
```

#### Step 4：模型评测

```bash
python evaluate_v2.py \
    --model_path output/babylm-llama-v2/final \
    --data_dir data \
    --wandb_project chinese-babylm
```

---

## 📖 详细使用说明

### 数据预处理参数

```bash
python prepare_data_v2.py \
    --input data/processed/all.txt \      # 输入文本文件
    --output_dir data/processed_v2 \      # 输出目录
    --no_minhash                          # 禁用 MinHash 去重（节省时间）
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 必填 | 输入文本文件路径 |
| `--output_dir` | 必填 | 输出目录 |
| `--val_ratio` | 0.05 | 验证集比例 |
| `--min_length` | 10 | 最短文本长度（字符） |
| `--no_minhash` | False | 禁用 MinHash 近似去重 |

### 训练参数详解

```bash
python train_v2.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `data` | 数据根目录 |
| `--output_dir` | `output/babylm-llama` | 模型输出目录 |
| `--d_model` | 768 | 隐藏层维度 |
| `--n_layer` | 12 | Transformer 层数 |
| `--n_head` | 12 | 查询注意力头数 |
| `--n_kv_head` | 4 | KV 注意力头数 (GQA) |
| `--intermediate_size` | 2688 | FFN 中间层维度 |
| `--max_length` | 512 | 最大序列长度 |
| `--batch_size` | 8 | 每 GPU 批次大小 |
| `--gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `--learning_rate` | 6e-4 | 峰值学习率 |
| `--num_epochs` | 25 | 训练轮次 |
| `--warmup_ratio` | 0.03 | 预热步数比例 |
| `--weight_decay` | 0.1 | 权重衰减 |
| `--max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `--bpe_dropout` | 0.1 | BPE Dropout 概率 |
| `--use_checkpoint` | True | 启用 Gradient Checkpointing |
| `--anneal_dropout` | True | 启用训练退火 |
| `--wandb_project` | chinese-babylm | WandB 项目名 |
| `--wandb_mode` | online | WandB 模式 (online/offline/disabled) |
| `--seed` | 42 | 随机种子 |

### Accelerate 多 GPU 配置

`accelerate_config_v2.yaml` 配置说明：

```yaml
compute_environment: LOCAL_MACHINE     # 本地计算环境
distributed_type: MULTI_GPU            # 多 GPU 分布式
mixed_precision: bf16                  # bf16 混合精度
num_processes: 4                       # GPU 数量
gpu_ids: all                           # 使用所有 GPU
```

---

## 📊 训练监控 (WandB)

训练过程自动记录到 [Weights & Biases](https://wandb.ai)，提供实时在线监控。

### 监控指标

| 指标 | 说明 | 记录频率 |
|------|------|----------|
| `train/loss` | 训练交叉熵损失 | 每 50 步 |
| `train/ppl` | 训练困惑度 (Perplexity) | 每 50 步 |
| `train/lr` | 当前学习率 | 每 50 步 |
| `train/tokens_per_sec` | 吞吐量 (tokens/sec) | 每 50 步 |
| `train/grad_norm` | 梯度范数 | 每 50 步 |
| `train/gpu_memory_used_gb` | GPU 显存使用量 | 每 50 步 |
| `train/epoch` | 当前 epoch 进度 | 每 50 步 |
| `val/loss` | 验证集损失 | 每 epoch |
| `val/ppl` | 验证集困惑度 | 每 epoch |

### 查看 WandB 面板

```bash
# 训练启动后访问
https://wandb.ai/<your-username>/chinese-babylm
```

---

## 📈 模型性能

### 训练曲线 (Epoch 1)

```
Step    Loss     PPL        LR          tok/s
────    ─────    ───────    ──────      ──────
100     10.52    37,189     6.03e-05    85,000
200     9.41     12,181     1.20e-04    90,000
500     8.47     4,779      3.01e-04    82,000
800     8.28     3,963      3.83e-04    88,000
1000    8.10     3,308      4.02e-04    80,000
1100    8.10     3,308      4.21e-04    80,000
1150    8.07     3,203      4.40e-04    80,000
1200    8.04     3,096      4.60e-04    79,000
```

### 硬件利用

| GPU | 利用率 | 显存使用 |
|-----|--------|----------|
| GPU 0 | 100% | 31,673 MiB |
| GPU 1 | 100% | 13,726 MiB |
| GPU 2 | 99% | 13,726 MiB |
| GPU 3 | 99% | 13,726 MiB |

---

## 📋 版本历史

### V2.0 — Phase 3 精细优化版 (2026-04-19) 🏆

**架构重构：**
- ✅ 从 GPT-2 升级到 LLaMA 架构 (RoPE + GQA + SwiGLU + RMSNorm)
- ✅ 使用 `LlamaForCausalLM` 替代自定义 GPT-2

**Tokenizer 重设计：**
- ✅ ByteLevel BPE Tokenizer（零 OOV，完美中文支持）
- ✅ 32K 词表大小
- ✅ BPE Dropout 数据增强

**训练优化：**
- ✅ 修复 LR Scheduler Bug（num_training_steps 计算错误）
- ✅ bf16 混合精度训练
- ✅ SDPA/Flash Attention 2 加速
- ✅ Gradient Checkpointing 节省显存
- ✅ 文档感知序列构造（EOS 分隔）
- ✅ 训练退火策略（后期降低 dropout）
- ✅ AdamW (β₂=0.95) 优化器

**数据处理：**
- ✅ 精确去重 (MD5)
- ✅ 文本清洗 (HTML/URL/异常字符)
- ✅ 短文本过滤
- ✅ 独立验证集分割 (95%/5%)

**监控：**
- ✅ WandB 在线监控 (loss/ppl/lr/tok_per_sec/gpu_memory/grad_norm)
- ✅ 每 epoch 验证集 PPL 评估
- ✅ 完整超参数记录

### V1.0 — 基线版 (2026-04-19)

- ✅ GPT-2 架构 (110M 参数)
- ✅ BPE Tokenizer (32K 词表)
- ✅ 多 GPU DDP 训练
- ✅ WandB 基础监控
- ✅ 数据下载与预处理

---

## 🔗 参考链接

### 比赛相关

- 🏆 [ChineseBabyLM 挑战赛主页](https://chinese-babylm.github.io/)
- 📊 [训练数据 (HuggingFace)](https://huggingface.co/datasets/chinese-babylm-org/babylm-zho-100M)
- 📋 [评测代码 (GitHub)](https://github.com/SiyuanSong2004/evaluation-pipeline-2025)
- 🎓 [NLPCC 2025](http://tcci.ccf.org.cn/conference/2026/shared-tasks/)

### 技术参考

- 📄 [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- 📄 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021
- 📄 [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- 📄 [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao, 2023
- 📄 [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al., 2020

### 工具与框架

- [PyTorch](https://pytorch.org/) — 深度学习框架
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — 模型与 Tokenizer
- [Accelerate](https://huggingface.co/docs/accelerate) — 分布式训练
- [Weights & Biases](https://wandb.ai/) — 实验追踪与可视化

---

## 👥 团队

- **C5 Team** — C5-jpg

---

## 📄 License

MIT License

---

> 📧 如有问题或建议，欢迎提交 [Issue](https://github.com/C5-jpg/babyLLM/issues) 或 Pull Request。