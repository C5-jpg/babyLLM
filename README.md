# BabyLLM - 中文语言模型训练项目

> NLPCC 2026 ChineseBabyLM 挑战赛 · SULAB 分支
>
> 最后更新: 2026-04-26

---

## 目录

- [项目概述](#项目概述)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [版本历史](#版本历史)
- [接续训练](#接续训练)
- [团队协作](#团队协作)
- [技术文档](#技术文档)
- [硬件要求](#硬件要求)
- [故障排除](#故障排除)

---

## 项目概述

本项目参与 **NLPCC 2026 ChineseBabyLM 挑战赛**，目标是在 `babylm-zho-100M` 数据集（约 1 亿中文字符）上从头预训练一个高性能的小型中文语言模型。

### 挑战赛详情

| 项目 | 详情 |
|------|------|
| 比赛 | [ChineseBabyLM Challenge](https://chinese-babylm.github.io/) |
| 数据集 | [babylm-zho-100M](https://huggingface.co/datasets/chinese-babylm-org/babylm-zho-100M) |
| 评测流水线 | [evaluation-pipeline-2025](https://github.com/SiyuanSong2004/evaluation-pipeline-2025) |
| 约束条件 | 从头预训练，不可使用外部预训练模型 |
| Token 限制 | ≤100M Jieba tokens |

### 项目组件

- **`babyLLM/`**: 主训练代码，包含多个实验版本（V1-V10）
- **`chinese-babylm-eval-pipeline/`**: 官方评测流水线
- **`docs/`**: SOTA 技术研究文档

---

## 目录结构

```
babyllm/
├── babyLLM/                          # 主训练代码
│   ├── src/                          # 按版本组织的训练脚本
│   │   ├── v1/                      # V1: GPT-2 基线
│   │   ├── v2/                      # V2: LLaMA 架构 + ByteLevel BPE
│   │   ├── v3/                      # V3: SentencePiece Tokenizer + WSD 调度
│   │   ├── v4/                      # V4: 集成检查点
│   │   ├── v5/                      # V5: 知识蒸馏
│   │   ├── v6/                      # V6: 高级数据处理
│   │   ├── v7/                      # V7: SOTA 训练技术
│   │   ├── v8/                      # V8: 优化训练
│   │   ├── v9/                      # V9: 探测实验
│   │   └── v10/                     # V10: 最新训练流水线
│   ├── data/                        # 数据和 tokenizer
│   │   ├── processed/               # V1 处理数据
│   │   ├── processed_v2/            # V2 清洗数据
│   │   ├── processed_v3-v7/        # V3-V7 处理数据
│   │   ├── tokenizer/              # V1 tokenizer
│   │   ├── tokenizer_v2-v7/         # V2-V7 tokenizers
│   │   └── raw/                     # HuggingFace 缓存
│   ├── output/                      # 模型检查点
│   ├── plans/                       # 训练计划和分析
│   │   ├── V1_V5_DEEP_TECHNICAL_ANALYSIS.md
│   │   ├── V5_TRAINING_PLAN.md
│   │   ├── V7_SOTA_TRAINING_PLAN.md
│   │   └── POST_MORTEM_ANALYSIS_V1_V4.md
│   ├── docs/                        # 文档
│   ├── README.md                    # 详细 babyLLM README
│   ├── REPORT.md                    # 实验报告
│   ├── requirements.txt             # Python 依赖
│   └── launch_*.sh                  # 训练启动脚本
│
├── chinese-babylm-eval-pipeline/    # 官方评测流水线
│   ├── evaluation_pipeline/         # 评测脚本
│   ├── eval_*.sh                   # 评测 shell 脚本
│   ├── pipeline.py                 # 集成评测流水线
│   ├── prepare_chinese_data.py     # 数据准备
│   └── README.md                   # 评测流水线文档
│
├── docs/                            # 研究文档
│   └── SOTA_TECHNIQUES_RESEARCH.md # SOTA 技术研究
│
└── README.md                        # 本文件
```

---

## 快速开始

### 环境要求

- **操作系统**: Linux（推荐 Ubuntu 20.04+）
- **GPU**: NVIDIA GPU，≥16GB VRAM（推荐 4× A6000 48GB）
- **CUDA**: 12.4+
- **Python**: 3.10+

### 安装步骤

```bash
# 克隆仓库
git clone -b sulab https://github.com/C5-jpg/babyLLM.git babyllm
cd babyllm

# 创建 conda 环境
conda create -n babylm python=3.10 -y
conda activate babylm

# 安装 PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装依赖
cd babyLLM
pip install -r requirements.txt

# 登录 WandB（可选，用于训练监控）
wandb login
```

### 数据准备

```bash
cd babyLLM

# 下载并准备数据（如果尚未完成）
python src/v2/prepare_data_v2.py \
    --input data/processed/all.txt \
    --output_dir data/processed_v2 \
    --no_minhash

# 训练 tokenizer（如果尚未完成）
python src/v2/train_tokenizer_v2.py
```

### 开始训练

```bash
# 方式 1: 使用启动脚本（推荐）
bash launch_v10_pipeline.sh

# 方式 2: 手动使用 accelerate 训练
accelerate launch --config_file accelerate_config_v2.yaml src/v10/train_v10.py \
    --data_dir data \
    --output_dir output/babylm-v10 \
    --max_length 512 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-4 \
    --num_epochs 25 \
    --wandb_project chinese-babylm \
    --wandb_mode online
```

---

## 版本历史

### V1 - GPT-2 基线 (2026-04-19)
- **架构**: GPT-2 (110M 参数)
- **Tokenizer**: BPE (32K 词表)
- **训练**: 多 GPU DDP
- **状态**: ✅ 已完成
- **关键文件**: `src/v1/`, `launch_v4.sh`

### V2 - LLaMA 架构 (2026-04-19)
- **架构**: LLaMA (RoPE + GQA + SwiGLU + RMSNorm)
- **Tokenizer**: ByteLevel BPE (32K 词表，零 OOV)
- **优化**: bf16 混合精度、梯度检查点、SDPA/Flash Attention 2
- **数据**: 精确去重（MD5）、HTML 清洗、短文本过滤
- **状态**: ✅ 已完成（25/25 epochs，约 7.5 小时）
- **关键文件**: `src/v2/`, `launch_v5.sh`, `launch_v5_kd.sh`

### V3 - SentencePiece + WSD 调度器 (2026-04-20)
- **Tokenizer**: SentencePiece (SPM)
- **LR 调度器**: Warmup-then-Stable-Decay (WSD)
- **状态**: ✅ 已完成
- **关键文件**: `src/v3/`, `launch_v6_pipeline.sh`

### V4 - 集成检查点 (2026-04-20)
- **策略**: 集成多个检查点以获得更好性能
- **状态**: ✅ 已完成
- **关键文件**: `src/v4/`, `launch_v7.sh`

### V5 - 知识蒸馏 (2026-04-21)
- **技术**: 从更大的教师模型进行知识蒸馏
- **状态**: ✅ 已完成
- **关键文件**: `src/v5/`, `launch_v8_pipeline.sh`

### V6 - 高级数据处理 (2026-04-22)
- **改进**: 增强的数据清洗和预处理
- **状态**: ✅ 已完成
- **关键文件**: `src/v6/`, `launch_v9_probe.sh`

### V7 - SOTA 训练技术 (2026-04-23)
- **技术**: BabyLM 2024 获奖者的最新 SOTA 技术
- **状态**: ✅ 已完成
- **关键文件**: `src/v7/`, `launch_v9_polish_probe.sh`

### V8 - 优化训练 (2026-04-24)
- **重点**: 训练效率和优化
- **状态**: ✅ 已完成
- **关键文件**: `src/v8/`

### V9 - 探测实验 (2026-04-25)
- **目的**: 探测实验以理解模型行为
- **状态**: ✅ 已完成
- **关键文件**: `src/v9/`

### V10 - 最新训练流水线 (2026-04-26)
- **状态**: 🚧 开发中
- **关键文件**: `src/v10/`, `launch_v10_pipeline.sh`, `auto_start_v6_after_kd.sh`

---

## 接续训练

### 从现有检查点接续训练

如果要从现有检查点接续训练：

```bash
cd babyLLM

# 查看可用检查点
ls output/babylm-*/checkpoints/

# 恢复训练
accelerate launch --config_file accelerate_config_v2.yaml src/v10/train_v10.py \
    --data_dir data \
    --output_dir output/babylm-v10 \
    --resume_from_checkpoint output/babylm-v10/checkpoints/checkpoint-XXXX \
    --max_length 512 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-4 \
    --num_epochs 25 \
    --wandb_project chinese-babylm
```

### 接续训练的关键参数

- `--resume_from_checkpoint`: 检查点目录路径
- `--num_epochs`: 总 epoch 数（训练将持续到此数量）
- 优化器状态和训练进度将自动恢复

### 检查点管理

检查点保存在 `output/<model_name>/checkpoints/`：
- `checkpoint-XXXX`: 按步数命名
- `final`: 训练完成后的最终检查点
- 每个检查点包含：`model.safetensors`, `optimizer.pt`, `scheduler.pt`, `trainer_state.json`

---

## 团队协作

### 分支策略

- **`main`**: 稳定的生产代码
- **`sulab`**: SULAB 团队当前开发分支
- **功能分支**: 为实验创建新分支（如 `v11-experiment`）

### 工作流程

1. **拉取最新更改**：
   ```bash
   git checkout sulab
   git pull origin sulab
   ```

2. **创建功能分支**：
   ```bash
   git checkout -b v11-your-experiment
   ```

3. **进行更改并提交**：
   ```bash
   git add .
   git commit -m "feat: 你的更改描述"
   ```

4. **推送并创建 PR**：
   ```bash
   git push origin v11-your-experiment
   # 在 GitHub 上创建 PR 以合并到 sulab
   ```

### 代码规范

- 遵循现有代码结构和命名约定
- 为复杂逻辑添加注释
- 添加新功能时更新文档
- 提交前测试更改
- 使用描述性提交消息（conventional commits 格式）

### 文档记录

- 在 `plans/` 目录中记录实验结果
- 用重要发现更新 `REPORT.md`

---

## 技术文档

### 核心文档

- **`babyLLM/README.md`**: 训练代码的详细技术文档
- **`babyLLM/REPORT.md`**: 包含结果的综合实验报告
- **`docs/SOTA_TECHNIQUES_RESEARCH.md`**: 小型语言模型 SOTA 技术研究

### 训练计划

- **`plans/V1_V5_DEEP_TECHNICAL_ANALYSIS.md`**: V1-V5 深度技术分析
- **`plans/V5_TRAINING_PLAN.md`**: V5 训练计划
- **`plans/V5_TRAINING_STATUS_REPORT.md`**: V5 训练状态
- **`plans/V7_SOTA_TRAINING_PLAN.md`**: V7 SOTA 训练计划
- **`plans/POST_MORTEM_ANALYSIS_V1_V4.md`**: V1-V4 事后分析

### 评测流水线

- **`chinese-babylm-eval-pipeline/README.md`**: 官方评测流水线文档
- 参见评测流水线 README 获取详细评测说明

---

## 硬件要求

### 最低配置

- **GPU**: 1× NVIDIA GPU (16GB VRAM)
- **CPU**: 8+ 核
- **RAM**: 32GB
- **存储**: 50GB 可用空间

### 推荐配置

- **GPU**: 4× NVIDIA A6000 (每块 48GB)
- **CPU**: 32+ 核
- **RAM**: 128GB
- **存储**: 500GB SSD

### 性能指标

使用 4× A6000 GPU：
- **吞吐量**: 80K-105K tokens/sec
- **训练时间**: 约 7.5 小时完成 25 epochs (V2)
- **GPU 利用率**: 99-100%
- **内存使用**: 每个 GPU 13-32GB

---

## 故障排除

### 常见问题

#### 内存不足 (OOM)

- 减小 `--batch_size` 或 `--gradient_accumulation_steps`
- 启用梯度检查点：`--use_checkpoint True`
- 使用更小的模型配置

#### 训练卡住

- 使用 `nvidia-smi` 检查 GPU 利用率
- 验证数据加载是否正常
- 检查 WandB 日志中的错误

#### 检查点加载问题

- 确保检查点路径正确
- 验证模型架构与检查点匹配
- 检查所有检查点文件是否存在

#### 数据问题

- 验证数据文件存在于 `data/processed_v2/`
- 检查 tokenizer 是否已训练：`data/tokenizer_v2/tokenizer.json`
- 如需要，重新运行数据准备

### 获取帮助

- 查看 GitHub 仓库中的现有 issue
- 查看 `docs/` 和 `plans/` 中的技术文档
- 参考 `babyLLM/README.md` 获取详细代码文档

---

## 当前状态

### 最新训练 (V10)

- **状态**: 🚧 开发中
- **分支**: `sulab`
- **最后更新**: 2026-04-26
- **关键文件**: `src/v10/`, `launch_v10_pipeline.sh`

### 可用检查点

查看 `babyLLM/output/` 获取之前版本的可用模型检查点。

---

## 许可证

MIT License
