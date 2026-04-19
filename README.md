# 🍼 BabyLLM — 首届 Chinese BabyLM 挑战赛参赛项目

[![Chinese BabyLM](https://img.shields.io/badge/Chinese%20BabyLM-2026-blue)](https://chinese-babylm.github.io/)
[![NLPCC 2026](https://img.shields.io/badge/NLPCC-2026-green)](http://tcci.ccf.org.cn/conference/2026/shared-tasks/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-c5--jpg/babyLLM-black)](https://github.com/c5-jpg/babyLLM)

> 本项目为 **2026首届 Chinese BabyLM 挑战赛** 的参赛实现。在约 100M 词的中文儿童导向语料上，从零训练一个 GPT-2 架构的语言模型，探索小规模数据下的中文语言模型习得能力。

---

## 📋 目录

- [⚡ 快速开始](#-快速开始)
- [背景介绍](#-背景介绍)
- [挑战赛信息](#-挑战赛信息)
- [项目结构](#-项目结构)
- [环境配置](#-环境配置)
- [数据准备](#-数据准备)
- [模型架构](#-模型架构)
- [训练流程](#-训练流程)
- [评测方法](#-评测方法)
- [实验结果](#-实验结果)
- [服务器部署](#-服务器部署)
- [技术细节](#-技术细节)
- [项目路线图](#-项目路线图)
- [常见问题](#-常见问题)
- [致谢](#-致谢)

---

## ⚡ 快速开始

> 30 秒内跑起训练！

```bash
# 1. 克隆仓库
git clone https://github.com/c5-jpg/babyLLM.git
cd babyLLM

# 2. 创建环境 & 安装依赖
conda create -n data python=3.12 -y
conda activate data
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers tokenizers datasets accelerate jieba tqdm matplotlib scikit-learn scipy

# 3. 下载 & 预处理数据（自动从 HuggingFace 下载 ~100M 词中文语料）
python prepare_data.py

# 4. 开始训练 🚀
python train.py --data_dir data --output_dir output/babylm-gpt2 \
    --d_model 768 --n_layer 12 --n_head 12 --max_length 512 \
    --batch_size 8 --learning_rate 6e-4 --num_epochs 10 \
    --gradient_accumulation_steps 4
```

训练完成后，最佳模型保存在 `output/babylm-gpt2/best_model/`。

---

## 🌟 背景介绍

### Chinese BabyLM 挑战赛

**首届 Chinese BabyLM 挑战赛** 将在 **2026年 NLPCC 会议**（11月4-5日，澳门）上举行。该挑战赛灵感来源于英文 BabyLM Challenge (CoNLL 2023-2024)，旨在研究语言模型在有限数据条件下能多大程度习得人类语言能力。

### 核心问题

> 儿童在约 100M 词的语言输入中就能习得母语。一个在相同规模数据上训练的神经网络，能否达到类似的语言能力？

### 现实意义

- **认知科学**: 理解语言习得的数据效率
- **AI民主化**: 小模型 + 小数据的实用训练方案
- **中文NLP**: 推动中文语言模型的基础研究
- **资源受限场景**: 为低资源语言的模型训练提供参考

---

## 🏆 挑战赛信息

| 项目 | 详情 |
|------|------|
| **挑战赛名称** | 首届 Chinese BabyLM 挑战赛 |
| **举办会议** | NLPCC 2026 (CCF-C类) |
| **时间** | 2026年11月4-5日 |
| **地点** | 澳门 |
| **主页** | https://chinese-babylm.github.io/ |
| **训练数据** | [babylm-zho-100M](https://huggingface.co/datasets/chinese-babylm-org/babylm-zho-100M) |
| **评测代码** | [evaluation-pipeline-2025](https://github.com/SiyuanSong2004/evaluation-pipeline-2025) |

### 挑战赛时间线

| 日期 | 事件 |
|------|------|
| 2026年3月20日 | 任务发布，官网上线，注册开始 |
| 2026年4月15日 | 详细指南发布，训练数据和开放评测任务下发 |
| 2026年4月22日 | Baseline模型发布，HuggingFace排行榜上线 |
| **2026年5月25日** | **注册截止** |
| **2026年6月11日** | **模型提交截止**，隐藏测试集和完整评测流水线发布 |
| 2026年6月20日 | 隐藏测试集最终结果提交截止 |
| 2026年6月30日 | 获奖者公布，最终排行榜发布 |

### 评测赛道

| 赛道 | 任务 | 说明 |
|------|------|------|
| **NLU Track** 📖 | zhOBLiMP | 零样本最小对语法判断（来自CLUE和ZhoBLiMP） |
| **Cognitive Track** 🧠 | MulCogBench | 模型表征与人类认知信号的对比（行为+神经模态） |
| **HANZI Track** 文 | PinyinBench, HanziBench | 汉字拼音和结构属性（最小对比较） |
| **Fine-tune Track** | AFQMC, OCNLI, TNews, CLUEWSC2020 | CLUE中文理解基准微调 |

### 重要规则

- **无架构限制**: Transformer encoder/decoder、encoder-decoder、state-space模型或新设计均可
- **无训练epoch限制**: 可以任意训练轮次
- **数据限制**: 训练语料不得超过100M词预算
- **评测方式**: Phase 1（开放评测）+ Phase 2（隐藏测试集），最终得分取平均
- **可复现性要求**: 各赛道前3名须提交全部代码、训练数据和模型权重
- **模型提交**: 所有最终模型须上传至HuggingFace

### 组织者

- **胡海 (Hai Hu)** — 香港城市大学（挑战赛主席）
- **宋思远 (Siyuan Song)** — 德克萨斯大学奥斯汀分校
- **何林阳 (Linyang He)** — 哥伦比亚大学
- **王少楠 (Shaonan Wang)** — 香港理工大学
- 及来自中科院、上海交大、北师大、清华的研究者

---

## 📁 项目结构

```
babyLLM/
├── README.md                    # 项目说明文档（本文件）
├── SERVER_INSTRUCTIONS.md       # 服务器端 Cline 训练指令
├── train.py                     # 主训练脚本（GPT-2 从头预训练）
├── prepare_data.py              # 数据下载与预处理脚本
├── requirements.txt             # Python 依赖清单
├── run.bat                      # Windows 一键训练脚本
├── .gitignore                   # Git 忽略规则
├── .gitmodules                  # Git 子模块配置（评测代码）
│
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据（从HuggingFace下载，不上传）
│   ├── processed/               # 预处理后的数据（不上传）
│   │   └── train.txt            # 拼接后的训练文本
│   └── tokenizer/               # BPE Tokenizer（已包含在仓库中）
│       ├── tokenizer.json       # Tokenizer 模型文件
│       └── tokenizer_config.json
│
├── output/                      # 模型输出（不上传GitHub）
│   └── babylm-gpt2/
│       ├── best_model/          # 最佳模型权重
│       ├── epoch-*/             # 各 epoch 检查点
│       ├── config.json          # 模型配置
│       └── training_args.json   # 训练参数记录
│
└── evaluation/                  # 评测代码（子模块，来自官方评测仓库）
    ├── configs/config.yaml      # 评测配置文件
    ├── pipeline.py              # 评测流水线主入口
    ├── eval_zero_shot.sh        # 零样本评测脚本
    ├── eval_finetune.sh         # 微调评测脚本
    ├── eval_cogbench.sh         # 认知评测脚本
    └── ...                      # 更多评测工具和工具函数
```

---

## ⚙️ 环境配置

### 方式一：Conda（推荐）

```bash
# 创建虚拟环境
conda create -n data python=3.12 -y
conda activate data

# 安装 PyTorch（CUDA 12.6）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装其他依赖
pip install transformers tokenizers datasets accelerate
pip install jieba tqdm matplotlib wandb
pip install scikit-learn scipy
```

### 方式二：requirements.txt

```bash
pip install -r requirements.txt
```

### 硬件要求

| 配置 | 最低要求 | 推荐 |
|------|----------|------|
| **GPU** | NVIDIA RTX 3090 (24GB) | NVIDIA A100 (40GB+) |
| **内存** | 32 GB | 64 GB+ |
| **硬盘** | 50 GB | 100 GB+ |
| **CUDA** | 11.8+ | 12.0+ |

> 💡 本项目在 **RTX 5060 Ti (16GB)** 上完成初始调试，训练速度约 2.6 it/s。如显存不足，可减小 `--batch_size` 或增大 `--gradient_accumulation_steps`。建议迁移到服务器使用多 GPU 训练。

---

## 📊 数据准备

### 数据集信息

| 属性 | 值 |
|------|-----|
| **数据集名称** | babylm-zho-100M |
| **来源** | [HuggingFace](https://huggingface.co/datasets/chinese-babylm-org/babylm-zho-100M) |
| **样本数** | ~184K 条（过滤后） |
| **Token数** | ~101M tokens（jieba分词） |
| **语言** | 中文（简体为主） |

### 数据来源分布

| 类别 | 说明 | Token数 | 来源 |
|------|------|---------|------|
| 字幕 | 影视字幕，反映日常口语 | 91.3M | WenetSpeech (已过滤1/2) |
| 儿童书籍 | 故事书和阅读理解数据集 | 16.0M | Quangushi, GlotStoryBooks, CFT, CMRC-2019 |
| 儿童导向言语 | 对儿童说的话的转录 | 9.6M | CHILDES, ChildMandarin |
| 教育 | 考题、语法练习、学生作文 | 13.5M | GAOKAO, CK-12, CSQ, FCGEC |
| 儿童可及言语 | 日常生活中儿童可听到的言语转录 | 7.4M | NaturalConv, ChildMandarin |
| 儿童百科 | 适合年龄的非虚构读物 | 25K | WikiJunior, Wikibooks |
| **总计** | | **~137.8M** | |

### 自动下载与预处理

```bash
# 一键完成：下载数据 + 训练Tokenizer + 预处理
python prepare_data.py
```

该脚本会自动完成：
1. 从 HuggingFace 下载 `babylm-zho-100M` 数据集
2. 提取并合并所有文本到 `data/processed/`
3. 从零训练 BPE Tokenizer（32K 词表），保存到 `data/tokenizer/`
4. 转换为 HuggingFace 兼容格式

### 手动步骤

```bash
# 1. 下载原始数据
python -c "
from datasets import load_dataset
ds = load_dataset('chinese-babylm-org/babylm-zho-100M', split='train')
print(f'样本数: {len(ds)}')
"

# 2. 训练 BPE Tokenizer
python -c "
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

ds = load_dataset('chinese-babylm-org/babylm-zho-100M', split='train')

tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=['[PAD]', '[UNK]', '[BOS]', '[EOS]'],
    min_frequency=2, show_progress=True
)

def batch_iterator(batch_size=1000):
    for i in range(0, len(ds), batch_size):
        yield [item['text'] for item in ds[i:i+batch_size]]

tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(ds))
tokenizer.save('data/tokenizer/tokenizer.json')
print('Tokenizer 训练完成！')
"

# 3. 预处理：合并文本
python -c "
from datasets import load_dataset
ds = load_dataset('chinese-babylm-org/babylm-zho-100M', split='train')
with open('data/processed/train.txt', 'w', encoding='utf-8') as f:
    for item in ds:
        f.write(item['text'] + '\n')
print('数据预处理完成！')
"
```

---

## 🧠 模型架构

### GPT-2 配置

我们使用标准 GPT-2 (Small) 架构，参数量约 **110M**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `vocab_size` | 32,000 | BPE 词表大小 |
| `d_model` (hidden_size) | 768 | 隐藏层维度 |
| `n_layer` (num_layers) | 12 | Transformer 层数 |
| `n_head` (num_heads) | 12 | 注意力头数 |
| `d_ff` (intermediate_size) | 3,072 | FFN 中间层维度 |
| `max_length` | 512 | 最大序列长度 |
| `dropout` | 0.1 | Dropout 率 |
| **总参数量** | **~110M** | - |

### 架构图

```
输入 Token IDs
    │
    ▼
┌──────────────┐
│  Token +     │  词嵌入 + 位置嵌入
│  Position    │  (768 维)
│  Embedding   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Transformer │  ×12 层
│  Block       │  ├── Multi-Head Attention (12 heads)
│              │  ├── Layer Norm
│              │  ├── Feed-Forward (768 → 3072 → 768)
│              │  └── Layer Norm + Residual
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Layer Norm  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LM Head     │  线性层 (768 → 32000)
│  (共享权重)   │
└──────┬───────┘
       │
       ▼
   Next Token Prediction
```

### 与标准 GPT-2 的差异

| 特征 | 本项目 | 标准 GPT-2 |
|------|--------|-----------|
| 词表大小 | 32K | 50K |
| 训练数据 | 100M 词中文语料 | 40GB 英文 WebText |
| 词表类型 | 从零训练 BPE | 复用原始词表 |
| 预分词 | WhitespaceSplit + Punctuation | ByteLevel |
| 特殊 Token | `<unk>`, `<s>`, `</s>`, `<pad>`, `<mask>` | `<|endoftext|>` |

---

## 🚀 训练流程

### 快速开始

```bash
# 激活环境
conda activate data

# 单GPU训练
python train.py \
    --data_dir data \
    --output_dir output/babylm-gpt2 \
    --d_model 768 \
    --n_layer 12 \
    --n_head 12 \
    --max_length 512 \
    --batch_size 8 \
    --learning_rate 6e-4 \
    --num_epochs 10 \
    --gradient_accumulation_steps 4
```

### 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **优化器** | AdamW | β₁=0.9, β₂=0.95, weight_decay=0.1 |
| **学习率** | 6×10⁻⁴ | Cosine Annealing |
| **Warmup** | 10% steps | 线性预热 |
| **Batch Size** | 8 × 4 = 32 | 梯度累积模拟 |
| **最大长度** | 512 | 序列截断 |
| **Epoch数** | 10 | 总训练步数 ~45,000 |
| **梯度裁剪** | 1.0 | Max Norm |
| **最小学习率** | 6×10⁻⁵ | Cosine 衰减最小值 |

### 完整参数列表

```bash
python train.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `data` | 数据目录 |
| `--output_dir` | `output/babylm-gpt2` | 模型输出目录 |
| `--d_model` | `768` | 隐藏层维度 |
| `--n_layer` | `12` | Transformer 层数 |
| `--n_head` | `12` | 注意力头数 |
| `--max_length` | `512` | 最大序列长度 |
| `--batch_size` | `8` | 批次大小 |
| `--learning_rate` | `6e-4` | 学习率 |
| `--weight_decay` | `0.1` | 权重衰减 |
| `--num_epochs` | `10` | 训练轮次 |
| `--warmup_ratio` | `0.1` | 预热比例 |
| `--max_grad_norm` | `1.0` | 最大梯度范数 |
| `--gradient_accumulation_steps` | `4` | 梯度累积步数 |
| `--lr_scheduler_type` | `cosine` | 学习率调度器类型 |
| `--logging_steps` | `100` | 日志间隔步数 |
| `--save_steps` | `1000` | 保存间隔步数 |
| `--seed` | `42` | 随机种子 |

### 训练数据统计

```
数据加载完成！
  总字符数: 82,324,004
  训练样本数: 160,790 (block_size=512)
  训练批次: 18,088/epoch (batch_size=8)
```

### 训练曲线

训练在 Epoch 1 期间 Loss 从 ~10.5 快速下降到 ~8.0：

```
Step   100 | Loss: 9.89 | PPL: 19,813
Step   200 | Loss: 9.21 | PPL: 10,082
Step   300 | Loss: 8.77 | PPL:  6,477
Step   500 | Loss: 8.35 | PPL:  4,240
Step   700 | Loss: 8.01 | PPL:  2,935
```

> Loss 持续稳定下降，模型正在有效学习中文语言模式。

### 多GPU训练（服务器）

```bash
# 使用 torchrun 多GPU训练（4卡）
torchrun --nproc_per_node=4 train.py \
    --data_dir data \
    --output_dir output/babylm-gpt2 \
    --d_model 768 --n_layer 12 --n_head 12 \
    --max_length 512 \
    --batch_size 16 \
    --learning_rate 6e-4 \
    --num_epochs 10 \
    --gradient_accumulation_steps 2 \
    --fp16
```

### Windows 一键训练

双击 `run.bat` 或在 CMD 中运行：

```cmd
run.bat
```

该脚本会自动激活 conda 环境、准备数据、并启动训练。

---

## 📏 评测方法

### 评测流水线

```bash
cd evaluation

# 1. 下载评测数据
python pipeline.py download

# 2. 运行完整评测
python pipeline.py eval --config configs/config.yaml

# 3. 单独评测各赛道
## 零样本语法判断 (zhOBLIMP)
bash eval_zero_shot.sh /path/to/model causal

## 微调评测 (CLUE)
bash eval_finetune.sh /path/to/model causal

## 认知评测 (fMRI/MEG/眼动)
bash eval_cogbench.sh /path/to/model causal
```

### 评测配置

编辑 `evaluation/configs/config.yaml`：

```yaml
models:
  - path: /path/to/your/best_model
    backend: causal    # GPT-2 使用 causal

tasks:
  zero_shot:
    - zhoblimp
    - hanzi_structure
    - hanzi_pinyin
  cogbench:
    - word_fmri
    - fmri
  finetune:
    - afqmc
    - ocnli
    - tnews
    - cluewsc2020

finetune_hparams:
  lr: 3.0e-5
  batch_size: 32
  max_epochs: 10
  wsc_epochs: 30
  seed: 42
```

---

## 📈 实验结果

> 训练尚未完成，结果待补充。

| 赛道 | 任务 | 得分 | 备注 |
|------|------|------|------|
| NLU | zhOBLiMP | - | 待评测 |
| Cognitive | MulCogBench | - | 待评测 |
| HANZI | PinyinBench | - | 待评测 |
| HANZI | HanziBench | - | 待评测 |
| Fine-tune | AFQMC | - | 待评测 |
| Fine-tune | OCNLI | - | 待评测 |
| Fine-tune | TNews | - | 待评测 |
| Fine-tune | CLUEWSC2020 | - | 待评测 |

---

## 🖥️ 服务器部署

### 快速部署

```bash
# 1. 克隆仓库
git clone https://github.com/c5-jpg/babyLLM.git
cd babyLLM

# 2. 创建环境
conda create -n data python=3.12 -y
conda activate data
pip install -r requirements.txt

# 3. 准备数据
python prepare_data.py

# 4. 单GPU训练
python train.py --data_dir data --output_dir output/babylm-gpt2 \
    --batch_size 16 --gradient_accumulation_steps 2 --fp16

# 5. 多GPU训练（4卡）
torchrun --nproc_per_node=4 train.py \
    --data_dir data --output_dir output/babylm-gpt2 \
    --batch_size 16 --gradient_accumulation_steps 2 --fp16

# 6. 运行评测
cd evaluation
python pipeline.py download
python pipeline.py eval --config configs/config.yaml
```

详细的 Cline 服务器部署指令请参见 [SERVER_INSTRUCTIONS.md](SERVER_INSTRUCTIONS.md)。

---

## 🔬 技术细节

### 训练策略

| 策略 | 说明 |
|------|------|
| **数据流水线** | 采用流式读取，不一次性加载全部数据到内存 |
| **梯度累积** | 模拟更大 batch size，减少显存占用 |
| **混合精度** | 支持 FP16/BF16 训练（服务器推荐开启） |
| **学习率调度** | Cosine Annealing with Warmup |
| **权重衰减** | 0.1，防止过拟合 |
| **梯度裁剪** | Max Norm = 1.0，防止梯度爆炸 |

### 数据处理流程

```
HuggingFace 数据集
       │
       ▼
  逐行提取文本
       │
       ▼
  合并为 train.txt
       │
       ▼
  BPE Tokenizer 编码
       │
       ▼
  按 block_size=512 切分
       │
       ▼
  90/10 随机划分为训练/验证集
       │
       ▼
  DataLoader 加载训练
```

### 内存优化

- **流式数据加载**: 不一次性加载所有数据
- **梯度累积**: 减少显存峰值
- **数据预处理缓存**: Token 化后的数据保存为 `.pt` 文件（可选）

---

## 🗺️ 项目路线图

- [x] 项目初始化与仓库搭建
- [x] 数据下载与预处理脚本
- [x] GPT-2 模型训练脚本
- [x] BPE Tokenizer 训练
- [x] README 文档编写
- [ ] 完成全量 10 epoch 训练
- [ ] 运行完整评测流水线（4个赛道）
- [ ] 记录并分析评测结果
- [ ] 调参优化（学习率、batch size、模型大小等）
- [ ] 尝试不同架构（BERT、RoPE、Flash Attention）
- [ ] 数据增强实验
- [ ] 模型上传 HuggingFace
- [ ] 撰写参赛论文

---

## ❓ 常见问题

### Q: 显存不足怎么办？

减小 `batch_size` 并增大 `gradient_accumulation_steps`，保持有效 batch size 不变：

```bash
# 原始配置 (需要 ~16GB)
--batch_size 8 --gradient_accumulation_steps 4

# 节省显存 (需要 ~8GB)
--batch_size 4 --gradient_accumulation_steps 8

# 极省显存 (需要 ~6GB)
--batch_size 2 --gradient_accumulation_steps 16
```

### Q: 数据下载慢怎么办？

`prepare_data.py` 已内置 HuggingFace 镜像（`hf-mirror.com`）。如仍有问题，可手动设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 如何恢复训练？

修改 `train.py` 加载已有的 checkpoint，或使用最新的 epoch 目录：

```python
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("output/babylm-gpt2/epoch-5")
```

### Q: 如何在 Windows 上训练？

直接双击 `run.bat`，或在 CMD 中手动执行命令。注意 Windows 下 `num_workers` 建议设为 0。

### Q: Tokenizer 词表大小怎么选？

当前使用 32K，是平衡词表覆盖率和模型参数量的选择。可以尝试 16K（更小更快）或 50K（更大覆盖更好）。

---

## 📚 参考文献

1. **BabyLM Challenge**: Warstadt et al., *Findings of the BabyLM Challenge at the First Conference on Language Modeling*, CoNLL 2023.
2. **GPT-2**: Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI 2019.
3. **Chinese BabyLM**: https://chinese-babylm.github.io/
4. **CLUE Benchmark**: Xu et al., *CLUE: A Chinese Language Understanding Evaluation Benchmark*, COLING 2020.
5. **BPE Tokenization**: Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016.

---

## 🙏 致谢

- [Chinese BabyLM 挑战赛](https://chinese-babylm.github.io/) 组织委员会
- [NLPCC 2026](http://tcci.ccf.org.cn/conference/2026/) 会议
- [HuggingFace](https://huggingface.co/) 提供数据集和工具
- [evaluation-pipeline-2025](https://github.com/SiyuanSong2004/evaluation-pipeline-2025) 官方评测代码

---

## 📄 License

MIT License

---

> 📧 如有任何问题，欢迎提 [Issue](https://github.com/c5-jpg/babyLLM/issues) 或 PR。
>
> ⭐ 如果这个项目对你有帮助，欢迎 Star！