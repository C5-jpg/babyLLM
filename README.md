# 🍼 BabyLLM — 首届 Chinese BabyLM 挑战赛参赛项目

[![Chinese BabyLM](https://img.shields.io/badge/Chinese%20BabyLM-2026-blue)](https://chinese-babylm.github.io/)
[![NLPCC 2026](https://img.shields.io/badge/NLPCC-2026-green)](http://tcci.ccf.org.cn/conference/2026/shared-tasks/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-ee4c2c)](https://pytorch.org/)

> 本项目为 **2026首届 Chinese BabyLM 挑战赛** 的参赛实现。在约 100M 词的中文儿童导向语料上，从零训练一个 GPT-2 架构的语言模型，探索小规模数据下的中文语言模型习得能力。

---

## 📋 目录

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
- [致谢](#-致谢)

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
├── README.md                    # 项目说明文档
├── train.py                     # 主训练脚本
├── prepare_data.py              # 数据下载与预处理脚本
├── requirements.txt             # Python 依赖
├── .gitignore                   # Git 忽略规则
│
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据（从HuggingFace下载）
│   ├── processed/               # 预处理后的数据
│   │   └── train.txt            # 拼接后的训练文本
│   └── tokenizer/               # BPE Tokenizer
│       ├── tokenizer.json       # Tokenizer 模型文件
│       ├── tokenizer_config.json
│       └── vocab.txt            # 词表文件
│
├── output/                      # 模型输出（不上传GitHub）
│   └── babylm-gpt2/
│       ├── best_model/          # 最佳模型权重
│       ├── checkpoint_epoch_*/  # 各 epoch 检查点
│       ├── config.json          # 模型配置
│       └── training_args.json   # 训练参数记录
│
└── evaluation/                  # 评测代码（子模块）
    ├── configs/
    │   └── config.yaml          # 评测配置文件
    ├── pipeline.py              # 评测流水线
    ├── eval_zero_shot.py        # 零样本评测
    ├── eval_finetune.py         # 微调评测
    └── ...
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

> 本项目在 **RTX 5060 Ti (16GB)** 上完成初始调试，训练速度约 2.6 it/s。建议迁移到服务器使用多 GPU 训练。

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

# 准备语料
ds = load_dataset('chinese-babylm-org/babylm-zho-100M', split='train')

# 训练 BPE
tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=['[PAD]', '[UNK]', '[BOS]', '[EOS]'],
    min_frequency=2,
    show_progress=True
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

1. **词表大小**: 32K (标准 GPT-2 为 50K)，专为中文优化
2. **训练数据**: 100M 词中文语料（标准 GPT-2 为 40GB 英文）
3. **词表类型**: 从零训练的 BPE，而非复用 GPT-2 原始词表
4. **预分词**: ByteLevel 预分词器，适合中文字符

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
| **Warmup** | 2000 steps | 线性预热 |
| **Batch Size** | 8 × 4 = 32 | 梯度累积 |
| **最大长度** | 512 | 序列截断 |
| **Epoch数** | 10 | 总训练步数 ~45,000 |
| **梯度裁剪** | 1.0 | Max Norm |
| **最小学习率** | 6×10⁻⁵ | Cosine 衰减最小值 |

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
Step   200 | Loss: 9.21 | PPL: 10,082  (估算)
Step   300 | Loss: 8.77 | PPL:  6,477
Step   500 | Loss: 8.35 | PPL:  4,240
Step   700 | Loss: 8.01 | PPL:  2,935
```

> Loss 持续稳定下降，模型正在有效学习中文语言模式。

### 多GPU训练（服务器）

```bash
# 使用 torchrun 多GPU训练
torchrun --nproc_per_node=4 train.py \
    --data_dir data \
    --output_dir output/babylm-gpt2 \
    --d_model 768 \
    --n_layer 12 \
    --n_head 12 \
    --max_length 512 \
    --batch_size 16 \
    --learning_rate 6e-4 \
    --num_epochs 10 \
    --gradient_accumulation_steps 2 \
    --fp16
```

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

## 认知评测 (fMRI)
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

---

## 🖥️ 服务器部署

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/babyLLM.git
cd babyLLM
```

### 2. 创建环境

```bash
conda create -n data python=3.12 -y
conda activate data
pip install -r requirements.txt
```

### 3. 准备数据

```bash
python prepare_data.py
```

### 4. 启动训练

```bash
# 单GPU
python train.py --data_dir data --output_dir output/babylm-gpt2 \
    --d_model 768 --n_layer 12 --n_head 12 --max_length 512 \
    --batch_size 16 --learning_rate 6e-4 --num_epochs 10 \
    --gradient_accumulation_steps 2 --fp16

# 多GPU (4卡)
torchrun --nproc_per_node=4 train.py \
    --data_dir data --output_dir output/babylm-gpt2 \
    --d_model 768 --n_layer 12 --n_head 12 --max_length 512 \
    --batch_size 16 --learning_rate 6e-4 --num_epochs 10 \
    --gradient_accumulation_steps 2 --fp16
```

### 5. 运行评测

```bash
cd evaluation
python pipeline.py download
python pipeline.py eval --config configs/config.yaml
```

---

## 🔬 技术细节

### 训练策略

1. **数据流水线**: 采用流式读取，不一次性加载全部数据到内存
2. **梯度累积**: 模拟更大 batch size，减少显存占用
3. **混合精度**: 支持 FP16/BF16 训练（服务器推荐开启）
4. **学习率调度**: Cosine Annealing with Warmup
5. **权重衰减**: 0.1，防止过拟合
6. **梯度裁剪**: Max Norm = 1.0，防止梯度爆炸

### 数据处理

1. 从 HuggingFace 下载 babylm-zho-100M 数据集
2. 将所有文本样本拼接为单个文件（用换行符分隔）
3. 从零训练 BPE Tokenizer（32K 词表）
4. 将文本 tokenize 为 ID 序列
5. 按 block_size=512 切分为训练样本
6. 随机打乱后用于训练

### 内存优化

- **流式数据加载**: 不一次性加载所有数据
- **梯度累积**: 减少显存峰值
- **数据预处理缓存**: Token 化后的数据保存为 `.pt` 文件

---

## 📝 TODO

- [ ] 完成全量 10 epoch 训练
- [ ] 运行完整评测流水线
- [ ] 记录并分析评测结果
- [ ] 调参优化（学习率、batch size、模型大小等）
- [ ] 尝试不同架构（BERT、RoPE、Flash Attention）
- [ ] 数据增强实验
- [ ] 撰写参赛论文

---

## 📚 参考文献

1. **BabyLM Challenge**: Warstadt et al., *Findings of the BabyLM Challenge at the First Conference on Language Modeling*, CoNLL 2023.
2. **GPT-2**: Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI 2019.
3. **Chinese BabyLM**: https://chinese-babylm.github.io/
4. **CLUE Benchmark**: Xu et al., *CLUE: A Chinese Language Understanding Evaluation Benchmark*, COLING 2020.

---

## 🙏 致谢

- [Chinese BabyLM 挑战赛](https://chinese-babylm.github.io/) 组织委员会
- [NLPCC 2026](http://tcci.ccf.org.cn/conference/2026/) 会议
- [HuggingFace](https://huggingface.co/) 提供数据集和工具
- [evaluation-pipeline-2025](https://github.com/SiyuanSong2004/evaluation-pipeline-2025) 评测代码

---

## 📄 License

MIT License

---

> 📧 如有任何问题，欢迎提 Issue 或 PR。