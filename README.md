<div align="center">

# 🍼 BabyLLM

### 从零预训练中文语言模型

[![Competition](https://img.shields.io/badge/NLPCC%202026-ChineseBabyLM%20Challenge-blue?style=for-the-badge)](https://chinese-babylm.github.io/)
[![PPL](https://img.shields.io/badge/SOTA%20PPL-38.68-brightgreen?style=for-the-badge)](#-核心成果)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model%20Weights-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/NLP-beginner/babyllm-chinese)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**在约 1 亿中文字符上从零预训练高性能小型语言模型**

历时 **23 天** · 完成 **15 个版本迭代** · PPL 从 **597 → 38.68** (↓93.5%)

[🤗 模型权重](https://huggingface.co/NLP-beginner/babyllm-chinese) · [🚀 快速开始](#-快速开始) · [📊 实验结果](#-官方评测结果) · [🏗️ 技术架构](#-技术架构) · [📜 版本历史](#-版本演进) · [📄 期中报告](babyLLM/docs/midterm_report.pdf)

</div>

---

## 📋 目录

- [🏆 核心成果](#-核心成果)
- [📊 官方评测结果](#-官方评测结果)
- [🏗️ 技术架构](#-技术架构)
- [📈 版本演进](#-版本演进)
- [🔬 关键技术创新](#-关键技术创新)
- [📂 项目结构](#-项目结构)
- [⚡ 快速开始](#-快速开始)
- [📊 详细实验数据](#-详细实验数据)
- [⚠️ 已知问题与后续工作](#️-已知问题与后续工作)
- [❓ FAQ](#-faq)
- [📜 许可证与引用](#-许可证与引用)

---

## 🏆 核心成果

<table>
<tr>
<td width="25%" align="center">

### 🎯 最佳 PPL

# **38.68**

V13 Stage 2 EMA
94.2M 参数

</td>
<td width="25%" align="center">

### 📉 PPL 改善

# **93.5%**

V2: 597 → V13: 38.68
23 天 / 15 版本

</td>
<td width="25%" align="center">

### ⚡ 最佳效率

# **V12**

54.2M 参数
PPL/10M = 7.21

</td>
<td width="25%" align="center">

### 🧪 语法理解

# **63.5%**

ZhoBLiMP 15 维度
语义相似 69.0%

</td>
</tr>
</table>

### 版本演进时间线

![Version Evolution Timeline](babyLLM/docs/assets/version_timeline.png)

*V1–V15 版本演进时间线。蓝色 = 成功，黄色 = SOTA，红色 = 失败。*

---

### PPL 演进趋势

![PPL Evolution](babyLLM/docs/assets/ppl_evolution.png)

*各版本最佳验证集 PPL（对数刻度）。从 V2 的 597 到 V13 的 38.68，整体降幅达 93.5%。*

---

## 📊 官方评测结果

### 雷达图对比

![Official Eval Radar](babyLLM/docs/assets/official_eval_radar.png)

*V13、V14、V15 在 CLUE 基准任务上的雷达图对比。*

### 评测数据表

| 版本 | ZhoBLiMP | 汉字结构 | 汉字拼音 | AFQMC | OCNLI | TNEWS | WSC2020 |
|:----:|:--------:|:--------:|:--------:|:-----:|:-----:|:-----:|:-------:|
| **V13 🏆** | **63.5** | **64.7** | **49.5** | 69.0 | 64.0 | 53.9 | 63.5 |
| V14 | 64.3 | 62.4 | 41.9 | 69.0 | **66.0** | 54.1 | 63.5 |
| V15 | 62.4 | 63.9 | 47.4 | 69.0 | 65.9 | **54.4** | 63.8 |

> **评测配置**: ZhoBLiMP/汉字/拼音为零样本评测; AFQMC/OCNLI/TNEWS/WSC2020 使用 finetune (lr=3e-5, batch=32, epochs=10)

### 分阶段 PPL 改进

![Stage-by-Stage PPL](babyLLM/docs/assets/ppl_by_stage.png)

*V10–V15 每个训练阶段的 PPL 改进。CLM→MNTP 是核心改进阶段。*

---

## 🏗️ 技术架构

### 模型架构（V13 SOTA）

```
LlamaForCausalLM (94.2M params)
├── Tokenizer: SentencePiece Unigram (32K vocab)
├── Embedding: 768d (tied with LM head, 节省 ~24.6M 参数)
├── Transformer Blocks × 14
│   ├── RMSNorm (eps=1e-5, Pre-Norm)
│   ├── Self-Attention (SDPA)
│   │   ├── Q: 12 heads × 64d = 768d
│   │   ├── K:  4 heads × 64d = 256d  (GQA, 3:1 压缩)
│   │   ├── V:  4 heads × 64d = 256d
│   │   └── RoPE (base=10000, max_pos=1024)
│   ├── RMSNorm
│   └── FFN (SwiGLU)
│       ├── Gate: 768d → 2048d
│       ├── Up:   768d → 2048d
│       └── Down: 2048d → 768d
├── RMSNorm (final)
└── LM Head (768d → 32K, tied)
```

### 架构演进：GPT-2 → LLaMA

| 组件 | GPT-2 (V1) | LLaMA (V2+) | 收益 |
|:-----|:-----------|:------------|:-----|
| 位置编码 | Learned Absolute (512) | RoPE (θ=10000) | 零参数, 相对位置, 长度外推 |
| 注意力 | MHA 12 heads | GQA 12Q/4KV | 3× KV 缓存减少 |
| 激活函数 | GELU | SwiGLU | 门控机制, 更好表达力 |
| 归一化 | LayerNorm (Post) | RMSNorm (Pre) | 70% 计算量, 更好梯度流 |
| FFN 维度 | 4×d (3072) | 8/3×d (2048) | SwiGLU 3 投影下相同参数 |

### 训练流水线

![Training Pipeline](babyLLM/docs/assets/training_pipeline.png)

#### 三阶段设计

| 阶段 | 方法 | 学习率 | 调度器 | 核心技术 | PPL 贡献 |
|:----:|:----:|:------:|:------:|:---------|:--------:|
| **Stage 1** | CLM | 6e-4 | SGDR | Focal Loss + EMA + Label Smoothing | 基础→42 |
| **Stage 2** | MNTP | 5e-4 | Cosine | 动态 CLM/Mask 比例 + EMA | 42→**38.68** |
| **Stage 3** | Polish | 2e-5 | Cosine | ⚠️ DropBlock+StochDepth 有害 | 已弃用 |

<details>
<summary>📖 Stage 1 详细 Epoch 数据 (V13)</summary>

| Epoch | Train Loss | Val Loss | Val PPL | Label Smoothing |
|:-----:|:----------:|:--------:|:-------:|:---------------:|
| 1 | 5.3668 | 4.1248 | 61.85 | 0.1000 |
| 4 | 3.8880 | 3.7751 | 43.60 | 0.0786 |
| **7** | **3.7007** | **3.7422** | **42.19** | 0.0571 |
| 8 | 3.5076 | 3.7472 | 42.40 | 0.0500 |

Best: Epoch 7, PPL=42.19 (EMA: PPL=**39.51**, 改进 8.4%)

</details>

<details>
<summary>📖 Stage 2 MNTP 详细数据 (V13)</summary>

| Epoch | Train Loss | Val Loss | Val PPL | CLM Ratio | Mask Ratio |
|:-----:|:----------:|:--------:|:-------:|:---------:|:----------:|
| 1 | 4.3477 | 3.7599 | 42.94 | 25% | 25% |
| 4 | 4.2380 | 3.7148 | 41.05 | 12.5% | 20% |
| **6** | **4.1343** | **3.7070** | **40.73** | 12.5% | 16.7% |
| 10 | 3.9089 | 3.7175 | 41.16 | 6.25% | 10% |

Best: Epoch 6, PPL=40.73 (EMA: PPL=**38.68**, 改进 6.7%)

</details>

---

## 📈 版本演进

### 完整版本对比表

| 版本 | 架构 | d_model | Layers | 参数量 | PPL | 关键创新 | 状态 |
|:----:|:----:|:-------:|:------:|:------:|:---:|:---------|:----:|
| V1 | GPT-2 | 768 | 12 | ~110M | ~343 | 基线模型 | ⚠️ Bug |
| V2 | LLaMA | 768 | 12 | ~125M | 597 | RoPE+GQA+SwiGLU | ❌ Tokenizer |
| V3 | LLaMA | 768 | 12 | ~125M | ~542 | SentencePiece 引入 | ❌ NCCL |
| V4 | LLaMA-deep | 1024 | 24 | ~350M | N/A | 深层架构 | ❌ 过大 |
| V5 | LLaMA-small | 512 | 12 | ~51M | 525 | 知识蒸馏 | ❌ KD 崩溃 |
| V6 | LLaMA | 640 | 12 | ~75M | N/A | 三阶段流水线 | ❌ 数据丢失 |
| V7 | LLaMA | 448 | 12 | ~30M | 50.8 | **MNTP 混合训练** | ✅ |
| V8 | LLaMA | 512 | 12 | ~35M | 50.8 | 简化三阶段 | ✅ |
| V9 | LLaMA | 512 | 12 | ~35M | 50.8 | 探针实验 | ✅ |
| V10 | LLaMA | 512 | 12 | 38.7M | 42.9 | 生产管线+SPM 32K | ✅ |
| V11 | LLaMA | 512 | 12 | 38.7M | 40.7 | **EMA+SGDR+自蒸馏** | ✅ |
| V12 | LLaMA | 576 | 14 | 54.2M | **38.8** | **Focal Loss+数据清洗** | ✅ 效率王 |
| **V13** | **LLaMA** | **768** | **14** | **94.2M** | **38.7** | **PPL 过滤+MinHash** | **🏆 SOTA** |
| V14 | LLaMA | 640 | 12 | 59.2M | 41.8 | 效率优化 | ✅ |
| V15 | LLaMA | 640 | 14 | 68.2M | 45.1 | Multi-scale EMA | ⚠️ 回归 |

### 版本关系图

```
V1 (GPT-2) ──→ V2 (LLaMA) ──→ V3 (SPM) ──→ V4 (Deep, ✗)
                                    │
                                    ├──→ V5 (Small + KD) ──→ V6 (3-stage, ✗)
                                    │
                                    └──→ V7 (MNTP) ──→ V8 ──→ V9
                                                              │
                                                              └──→ V10 (Production)
                                                                      │
                                                                      ├──→ V11 (EMA+SGDR)
                                                                      │
                                                                      ├──→ V12 (Focal Loss) ──→ V13 (🏆 SOTA)
                                                                      │
                                                                      └──→ V14 ──→ V15
```

### 参数效率前沿

![Parameters vs PPL](babyLLM/docs/assets/params_vs_ppl.png)

*参数量与 PPL 的关系。红色虚线为帕累托前沿。V12 (54.2M) 是参数效率最优的模型。*

| 版本 | 参数量 | PPL | PPL/10M 参数 | tokens/param |
|:----:|:------:|:---:|:------------:|:------------:|
| V11 | 38.7M | 40.73 | **10.53** | 2.6× |
| V12 | 54.2M | 38.84 | 7.21 | 1.9× |
| V13 | 94.2M | **38.68** | 4.11 | 1.1× |

> 💡 **Chinchilla Scaling Law**: 最优 tokens/param = 20~200×. 本项目仅 82M tokens, 因此 40~55M 参数是最优范围。

---

## 🔬 关键技术创新

![Technique Impact](babyLLM/docs/assets/technique_impact.png)

### 技术贡献排名

| 排名 | 技术 | 贡献 | 引入版本 | 原理 |
|:----:|:-----|:----:|:--------:|:-----|
| 🥇 | **SentencePiece 分词器** | ~74% PPL ↓ | V3 | 替代 ByteLevel BPE，在字/词级别处理中文 |
| 🥈 | **EMA 权重平均** | 6~8% PPL ↓ | V11 | 指数移动平均平滑训练噪声，decay=0.999 |
| 🥉 | **MNTP 混合训练** | 3~5 PPL ↓ | V7 | GPT-BERT 混合，同时 CLM + Masked 预测 |
| 4 | **SGDR 调度器** | ~2 PPL ↓ | V11 | 周期性重启 LR，探索不同损失景观 |
| 5 | **Focal Loss** | ~1~2 PPL ↓ | V12 | 聚焦困难 token，γ=1.5~2.0 |
| 6 | **PPL 数据过滤** | ~0.5 PPL ↓ | V13 | 用已训练模型过滤低质量数据 |
| 7 | **Label Smoothing 退火** | ~0.3 PPL ↓ | V10 | 软化硬标签，0.1→退火至 0 |

<details>
<summary>📐 核心公式</summary>

**Perplexity (PPL):**
```
PPL = exp( -1/N × Σ log p(x_i | x_{<i}) )
```

**Focal Loss:**
```
FL(p_t) = -(1 - p_t)^γ × log(p_t)    γ = 1.5 (Stage 1), 1.0 (Stage 2)
```

**EMA (Exponential Moving Average):**
```
θ_EMA^(t) = α × θ_EMA^(t-1) + (1-α) × θ^(t)    α = 0.999
```

**MNTP Loss:**
```
L_MNTP = α × L_CLM + (1-α) × L_Mask
动态调整: α ∈ {0.25, 0.125, 0.0625}
```

</details>

### EMA 定量贡献

| 阶段 | Base PPL | EMA PPL | 改进幅度 |
|:----:|:--------:|:-------:|:--------:|
| V13 Stage 1 | 43.15 | 39.51 | **-8.4%** |
| V13 Stage 2 | 41.45 | **38.68** | **-6.7%** |
| V13 Stage 3 | 40.40 | 40.08 | -0.8% |

> 📌 **关键发现**: EMA 在 Stage 1 (高 LR + 高噪声) 效果最显著，后续阶段收益递减。

---

## 📂 项目结构

```
babyllm/
├── README.md                              # 📖 本文档
├── .gitignore
│
├── babyLLM/                               # 🧠 主训练代码
│   ├── README.md                          # 技术文档 (开发者向)
│   ├── REPORT.md                          # V2 实验报告
│   ├── requirements.txt
│   │
│   ├── src/                               # 各版本源代码
│   │   ├── v1/    → V1:  GPT-2 基线
│   │   ├── v2/    → V2:  LLaMA 架构迁移
│   │   ├── v3/    → V3:  SPM 分词器
│   │   ├── v4/    → V4:  深层模型 (失败)
│   │   ├── v5/    → V5:  小模型 + 知识蒸馏
│   │   ├── v6/    → V6:  三阶段流水线
│   │   ├── v7/    → V7:  MNTP 混合训练
│   │   ├── v8/    → V8:  简化三阶段
│   │   ├── v9/    → V9:  探针实验
│   │   ├── v10/   → V10: 生产管线
│   │   ├── v11/   → V11: EMA + SGDR
│   │   ├── v12/   → V12: Focal Loss
│   │   ├── v13/   → V13: PPL 过滤 (SOTA 🏆)
│   │   ├── v14/   → V14: 效率版
│   │   └── v15/   → V15: Multi-scale EMA
│   │
│   ├── docs/                              # 📄 文档与可视化
│   │   ├── assets/                        # 📊 图表 (PNG)
│   │   ├── midterm_report.pdf             # 期中报告
│   │   ├── V1_V14_TRAINING_EXPERIENCE.md  # 训练经验总结
│   │   ├── V13_DEEP_ANALYSIS_REPORT.md    # V13 深度分析
│   │   └── V15_TRAINING_PROTOCOL.md       # V15 训练协议
│   │
│   ├── data/                              # 数据 (gitignored)
│   ├── logs/                              # 训练日志
│   ├── launch_v13_pipeline.sh             # V13 SOTA 训练脚本
│   └── launch_v15_pipeline.sh             # V15 训练脚本
│
├── chinese-babylm-eval-pipeline/          # 📋 官方评测流水线
│   ├── configs/                           # 评测配置
│   └── pipeline.py
│
├── for_chatgpt_ppt/                       # 📦 PPT 素材包
│   ├── project_context.md                 # 技术演进文档
│   ├── instructions_for_chatgpt.txt       # PPT 制作指南
│   └── images/                            # 可视化图片
│
└── docs/                                  # 顶层研究文档
```

---

## ⚡ 快速开始

### 环境配置

```bash
# 1. 克隆仓库
git clone https://github.com/C5-jpg/babyLLM.git
cd babyLLM

# 2. 创建 conda 环境
conda create -n babylm python=3.10 -y
conda activate babylm

# 3. 安装 PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. 安装依赖
cd babyLLM && pip install -r requirements.txt

# 5. 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 训练 V13 SOTA 模型

```bash
# 一键启动全阶段训练
bash launch_v13_pipeline.sh

# 或分阶段手动训练
# Stage 1: CLM + SGDR
python src/v13/train_v13.py \
    --stage clm --d_model 768 --n_layer 14 --n_head 12 --n_kv_heads 4 \
    --lr 6e-4 --epochs 8 --scheduler sgdr \
    --focal_loss --focal_gamma 1.5 --use_ema \
    --output_dir output/babylm-v13/stage1_clm

# Stage 2: MNTP
python src/v13/train_v13.py \
    --stage mntp \
    --resume_from output/babylm-v13/stage1_clm/best_model_ema \
    --lr 5e-4 --epochs 10 --dynamic_clm_ratio \
    --output_dir output/babylm-v13/stage2_mntp
```

### 评测模型

```bash
python src/v13/evaluate_v13.py \
    --model_path output/babylm-v13/stage2_mntp/best_model_ema \
    --data_path data/processed_v13/val.txt
```

<details>
<summary>⚙️ 完整训练参数表</summary>

| 参数 | 说明 | 默认值 |
|:-----|:-----|:-------|
| `--stage` | 训练阶段 (`clm` / `mntp`) | `clm` |
| `--d_model` | 隐藏维度 | 768 |
| `--n_layer` | Transformer 层数 | 14 |
| `--n_head` | 注意力头数 | 12 |
| `--n_kv_heads` | KV 头数 (GQA) | 4 |
| `--lr` | 学习率 | 6e-4 |
| `--epochs` | 训练轮数 | 8 |
| `--batch_size` | 每 GPU 批大小 | 16 |
| `--max_length` | 最大序列长度 | 1024 |
| `--stride` | 滑动窗口步长 | 512 |
| `--scheduler` | 调度器 (`cosine`/`sgdr`) | `sgdr` |
| `--focal_loss` | 启用 Focal Loss | False |
| `--focal_gamma` | Focal Loss gamma | 2.0 |
| `--use_ema` | 启用 EMA | False |
| `--ema_decay` | EMA 衰减率 | 0.999 |
| `--label_smoothing` | 标签平滑 | 0.1 |
| `--dynamic_clm_ratio` | 动态 CLM 比例 | False |
| `--eval_steps` | 评估间隔步数 | 200 |

</details>

---

## 📊 详细实验数据

### ZhoBLiMP 15 维度分析 (V13)

| 维度 | V13 🏆 | V12 | V11 | 随机 |
|:-----|:------:|:---:|:---:|:----:|
| BA (把字句) | **75.33** | 74.36 | 76.33 | 50.0 |
| question | 64.41 | **68.78** | 63.05 | 50.0 |
| nominal_expression | **75.85** | 72.58 | 71.82 | 50.0 |
| classifier | 77.78 | **79.11** | 74.44 | 50.0 |
| npi_licensing | **46.67** | 40.70 | 42.37 | 50.0 |
| topicalization | **63.50** | 54.00 | 60.33 | 50.0 |
| verb_phrase | 75.17 | **77.57** | 79.81 | 50.0 |
| anaphor (照应) ⚠️ | 35.00 | 37.33 | 36.44 | 50.0 |
| passive (被动) ⚠️ | 37.03 | 30.69 | 32.50 | 50.0 |
| argument_structure | **64.05** | 60.67 | 63.19 | 50.0 |
| ellipsis (省略) | 71.00 | **72.11** | 66.56 | 50.0 |
| control_raising | **70.42** | 62.83 | 64.50 | 50.0 |
| relativization | 55.25 | **56.25** | 51.92 | 50.0 |
| fci_licensing | **75.13** | 63.67 | 66.13 | 50.0 |
| quantifiers | 84.67 | **88.00** | 98.17 | 50.0 |
| **平均** | **63.47** | 62.03 | 61.97 | 50.0 |

> ⚠️ 标注的维度低于随机基线，表明模型在深层句法推理上仍有不足。

### 训练效率分析

| 版本 | 总步数 | 训练时长 | PPL 改进/小时 |
|:----:|:------:|:--------:|:-------------:|
| V10 | 20,437 | ~1.1h | ~8.5 |
| V11 | 37,538 | ~4.8h | ~1.0 |
| V12 | 49,520 | ~7.5h | ~0.3 |
| V13 | 64,492 | ~7.7h | ~0.45 |

---

## ⚠️ 已知问题与后续工作

### 已知问题

| # | 问题 | 影响 |
|:-:|:-----|:-----|
| 1 | V15 PPL 回归 (45.1 vs 预期 ~39) | FFN 维度非 256 倍数 + 数据差异 |
| 2 | ZhoBLiMP 弱维度 (anaphor/passive) | 低于随机基线 |
| 3 | CLUEWSC 所有版本相同 (63.49%) | MCC=0.0, 学到浅层启发式 |
| 4 | 贪婪解码产生重复 | 所有权重均存在 |

### 后续工作

| 优先级 | 方向 | 具体措施 |
|:------:|:-----|:---------|
| P1 | 修复 V15.1 | FFN: 1706→1792, 使用 V13 数据, LR=6e-4 |
| P2 | 架构搜索 | 对比 768d/12L vs 576d/14L |
| P3 | 消融实验 | 量化各技术独立贡献 |
| P4 | 数据增强 | 语法模式上采样, PPL 阈值调整 |

---

## ❓ FAQ

<details>
<summary><strong>Q: 为什么 V4 的 350M 参数模型表现不好？</strong></summary>

违反了 **Chinchilla scaling law**。100M tokens 对于 350M 参数太少 (tokens/param = 0.23×)，导致严重欠训练。实验验证最佳比例约为 tokens/param ≈ 1.8–2.5×。

</details>

<details>
<summary><strong>Q: 分词器选择为什么如此重要？</strong></summary>

ByteLevel BPE (V2) 将每个中文字符拆分为 3 个 UTF-8 字节，信息密度降低 3.4×，PPL 恶化 74%。这一负面影响完全抵消了 LLaMA 架构的所有优势。SentencePiece Unigram 在字级别处理中文，是正确选择。

**示例**: "今天天气真好" → ByteLevel BPE: `['ä»Ĭå¤©å¤©æ°Ķ', 'çľŁå¥½', ...]` (8 tokens, 无意义)

</details>

<details>
<summary><strong>Q: MNTP 混合训练为什么有效？</strong></summary>

CLM 只能看到左侧上下文，而 MNTP 通过随机掩码强制模型利用双向上下文信息。两者结合提供了更全面的语言理解能力，这也是 GPT-BERT 2024 冠军的核心技术。

</details>

<details>
<summary><strong>Q: EMA 和 SWA 有什么区别？</strong></summary>

- **EMA** (Exponential Moving Average): 在线平均权重，每步更新，decay=0.999。Stage 1 效果最显著 (8.4% PPL 改善)
- **SWA** (Stochastic Weight Averaging): 离线周期性采样取平均。V11 实验显示收益微弱

</details>

<details>
<summary><strong>Q: 训练过程中 OOM 怎么办？</strong></summary>

1. 减小 `--batch_size` (16→8)
2. 增加 `--grad_accum_steps` 保持等效批大小
3. 启用 `gradient_checkpointing` (默认已开)
4. 减小 `--max_length` (1024→512)

</details>

---

## 📜 许可证与引用

### 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

### 引用格式

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{babylm2026sulab,
  title={BabyLLM: From-Scratch Pre-training of Compact Chinese Language Models},
  author={SULAB Team},
  year={2026},
  howpublished={NLPCC 2026 ChineseBabyLM Challenge},
  url={https://github.com/C5-jpg/babyLLM}
}
```

### 参考文献

| 文献 | 引用 |
|:-----|:-----|
| LLaMA | Touvron et al. (2023). arXiv:2302.13971 |
| RoPE | Su et al. (2021). arXiv:2104.09864 |
| SwiGLU | Shazeer (2020). arXiv:2002.05202 |
| Focal Loss | Lin et al. (2017). arXiv:1708.02002 |
| SGDR | Loshchilov & Hutter (2016). arXiv:1608.03983 |
| SentencePiece | Kudo & Richardson (2018). arXiv:1808.06226 |
| Chinchilla | Hoffmann et al. (2022). arXiv:2203.15556 |

---

### 🔗 相关资源

[![GitHub](https://img.shields.io/badge/GitHub-C5--jpg/babyLLM-181717?style=flat&logo=github)](https://github.com/C5-jpg/babyLLM)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-模型权重-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/NLP-beginner/babyllm-chinese)
[![WandB](https://img.shields.io/badge/WandB-chinese--babylm-FCD000?style=flat)](https://wandb.ai/c5galaxies-sjtu-hpc-center/chinese-babylm)
[![Challenge](https://img.shields.io/badge/ChineseBabyLM-2026-blue?style=flat)](https://chinese-babylm.github.io/)

> 💡 **所有模型权重 (V1~V15.1, 共 22 个检查点) 已上传至 [HuggingFace](https://huggingface.co/NLP-beginner/babyllm-chinese)，可直接通过 `transformers` 加载使用。**

---

<div align="center">

*最后更新: 2026-06-05 · SULAB Team · NLPCC 2026*

</div>
