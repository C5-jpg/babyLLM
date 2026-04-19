# ChineseBabyLM 挑战赛训练项目

> 首届 ChineseBabyLM 挑战赛 (NLPCC 2026, 澳门, 11月4-5日)

## 项目概述

本项目用于参加首届 ChineseBabyLM 挑战赛，使用 GPT-2 架构在 `babylm-zho-100M` 中文儿童语料上从头预训练语言模型。

- **挑战赛主页**: https://chinese-babylm.github.io/
- **训练数据**: https://huggingface.co/datasets/chinese-babylm-org/babylm-zho-100M
- **评测代码**: https://github.com/SiyuanSong2004/evaluation-pipeline-2025

## 项目结构

```
babyLLM/
├── prepare_data.py       # 数据下载与预处理
├── train_tokenizer.py    # BPE Tokenizer 训练（独立脚本）
├── train.py              # GPT-2 多GPU训练（支持DDP + WandB）
├── requirements.txt      # Python 依赖
├── data/
│   ├── tokenizer/        # 训练好的 BPE Tokenizer (32K词表)
│   ├── processed/        # 预处理后的文本数据
│   └── raw/              # HuggingFace 缓存数据
├── output/               # 模型输出（checkpoints）
└── evaluation/           # 评测代码（子模块）
```

## 环境配置

```bash
# 创建 conda 环境
conda create -n data python=3.10 -y
conda activate data

# 安装依赖
pip install -r requirements.txt
```

### 依赖列表

- `torch>=2.0` + CUDA 12.4
- `transformers>=4.49`
- `tokenizers>=0.21`
- `datasets`
- `accelerate`
- `wandb`
- `tqdm`
- `jieba`

## 使用方法

### 1. 数据准备

```bash
python prepare_data.py
```

自动完成：
- 从 HuggingFace 下载 `babylm-zho-100M` 数据集
- 提取文本到 `data/processed/train.txt`
- 训练 32K 词表的 BPE Tokenizer

### 2. 单独训练 Tokenizer（可选）

```bash
python train_tokenizer.py
```

### 3. 启动训练

#### 单卡训练
```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
    --data_dir data \
    --output_dir output/babylm-gpt2 \
    --d_model 768 --n_layer 12 --n_head 12 \
    --max_length 512 --batch_size 16 \
    --learning_rate 6e-4 --num_epochs 10
```

#### 多卡 DDP 训练（推荐）
```bash
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train.py \
    --data_dir data \
    --output_dir output/babylm-gpt2 \
    --d_model 768 --n_layer 12 --n_head 12 \
    --max_length 512 --batch_size 16 \
    --learning_rate 6e-4 --num_epochs 10 \
    --gradient_accumulation_steps 2 \
    --wandb_project chinese-babylm \
    --wandb_mode online
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--d_model` | 768 | GPT-2 隐藏层维度 |
| `--n_layer` | 12 | Transformer 层数 |
| `--n_head` | 12 | 注意力头数 |
| `--max_length` | 512 | 最大序列长度 |
| `--batch_size` | 16 | 每 GPU 批次大小 |
| `--learning_rate` | 6e-4 | 峰值学习率 |
| `--num_epochs` | 10 | 训练轮次 |
| `--gradient_accumulation_steps` | 2 | 梯度累积步数 |
| `--lr_scheduler_type` | cosine | 学习率调度器 |
| `--warmup_ratio` | 0.1 | 预热比例 |
| `--wandb_project` | chinese-babylm | WandB 项目名 |
| `--wandb_mode` | online | WandB 模式 |

### 模型配置

- **架构**: GPT-2 Small (~110M 参数)
- **词表**: 32K BPE
- **序列长度**: 512 tokens
- **有效 Batch Size**: 96 (16 × 3 GPU × 2 grad_accum)

## WandB 监控

训练过程自动记录到 WandB，包括：
- 训练 Loss / Perplexity（每100步）
- 学习率曲线
- 验证 Loss / PPL（每 epoch）
- 完整超参数配置

访问 https://wandb.ai 查看训练面板。

## 硬件环境

- GPU: 4 × NVIDIA RTX A6000 (49GB VRAM)
- 训练使用 3 卡 DDP (GPU 1-3)
- 预计训练时间: ~4 小时 (10 epochs)

## 版本历史

### v0.2.0 (2026-04-19)
- ✅ 添加 HuggingFace Accelerate 多 GPU DDP 支持
- ✅ 集成 WandB 训练监控
- ✅ 修复 torchvision 兼容性问题
- ✅ 完善日志和 checkpoint 保存

### v0.1.0 (2026-04-19)
- ✅ 数据下载与预处理
- ✅ BPE Tokenizer 训练 (32K 词表)
- ✅ GPT-2 单卡训练脚本
- ✅ 基础训练循环

## 参考链接

- [NLPCC 2026](http://tcci.ccf.org.cn/conference/2026/shared-tasks/)
- [ChineseBabyLM 挑战赛](https://chinese-babylm.github.io/)
- [训练数据](https://huggingface.co/datasets/chinese-babylm-org/babylm-zho-100M)
- [评测代码](https://github.com/SiyuanSong2004/evaluation-pipeline-2025)