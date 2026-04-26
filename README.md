# 🍼 BabyLLM - Chinese Language Model Training Project

> NLPCC 2026 ChineseBabyLM Challenge · Team C5 · SULAB Branch
> 
> Last Updated: 2026-04-26

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Directory Structure](#-directory-structure)
- [Quick Start](#-quick-start)
- [Version History](#-version-history)
- [Continue Training](#-continue-training)
- [Team Collaboration](#-team-collaboration)
- [Technical Documentation](#-technical-documentation)
- [Hardware Requirements](#-hardware-requirements)
- [Troubleshooting](#-troubleshooting)

---

## 🌟 Project Overview

This project participates in the **NLPCC 2026 ChineseBabyLM Challenge**, aiming to pre-train a high-performance small Chinese language model from scratch on the `babylm-zho-100M` dataset (~100M Chinese characters).

### Challenge Details

| Item | Details |
|------|---------|
| Competition | [ChineseBabyLM Challenge](https://chinese-babylm.github.io/) |
| Dataset | [babylm-zho-100M](https://huggingface.co/datasets/chinese-babylm-org/babylm-zho-100M) |
| Evaluation Pipeline | [evaluation-pipeline-2025](https://github.com/SiyuanSong2004/evaluation-pipeline-2025) |
| Constraint | Train from scratch, no external pre-trained models |
| Token Limit | ≤100M Jieba tokens |

### Project Components

- **`babyLLM/`**: Main training code with multiple experimental versions (V1-V10)
- **`chinese-babylm-eval-pipeline/`**: Official evaluation pipeline for the competition
- **`docs/`**: Research documentation on SOTA techniques

---

## 📁 Directory Structure

```
babyllm/
├── babyLLM/                          # Main training code
│   ├── src/                          # Training scripts by version
│   │   ├── v1/                      # V1: GPT-2 baseline
│   │   ├── v2/                      # V2: LLaMA architecture + ByteLevel BPE
│   │   ├── v3/                      # V3: SentencePiece Tokenizer + WSD scheduler
│   │   ├── v4/                      # V4: Ensemble checkpoints
│   │   ├── v5/                      # V5: Knowledge distillation
│   │   ├── v6/                      # V6: Advanced data processing
│   │   ├── v7/                      # V7: SOTA training techniques
│   │   ├── v8/                      # V8: Optimized training
│   │   ├── v9/                      # V9: Probe experiments
│   │   └── v10/                     # V10: Latest training pipeline
│   ├── data/                        # Data and tokenizers
│   │   ├── processed/               # V1 processed data
│   │   ├── processed_v2/            # V2 cleaned data
│   │   ├── processed_v3-v7/        # V3-V7 processed data
│   │   ├── tokenizer/              # V1 tokenizer
│   │   ├── tokenizer_v2-v7/         # V2-V7 tokenizers
│   │   └── raw/                     # HuggingFace cache
│   ├── output/                      # Model checkpoints
│   ├── plans/                       # Training plans and analysis
│   │   ├── V1_V5_DEEP_TECHNICAL_ANALYSIS.md
│   │   ├── V5_TRAINING_PLAN.md
│   │   ├── V7_SOTA_TRAINING_PLAN.md
│   │   └── POST_MORTEM_ANALYSIS_V1_V4.md
│   ├── docs/                        # Documentation
│   ├── README.md                    # Detailed babyLLM README
│   ├── REPORT.md                    # Experiment report
│   ├── requirements.txt             # Python dependencies
│   └── launch_*.sh                  # Training launch scripts
│
├── chinese-babylm-eval-pipeline/    # Official evaluation pipeline
│   ├── evaluation_pipeline/         # Evaluation scripts
│   ├── eval_*.sh                   # Evaluation shell scripts
│   ├── pipeline.py                 # Integrated evaluation pipeline
│   ├── prepare_chinese_data.py     # Data preparation
│   └── README.md                   # Evaluation pipeline documentation
│
├── docs/                            # Research documentation
│   └── SOTA_TECHNIQUES_RESEARCH.md # SOTA techniques research
│
└── README.md                        # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU, ≥16GB VRAM (4× A6000 48GB recommended)
- **CUDA**: 12.4+
- **Python**: 3.10+

### Installation

```bash
# Clone the repository
git clone -b sulab https://github.com/C5-jpg/babyLLM.git babyllm
cd babyllm

# Create conda environment
conda create -n babylm python=3.10 -y
conda activate babylm

# Install PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
cd babyLLM
pip install -r requirements.txt

# Login to WandB (optional, for training monitoring)
wandb login
```

### Data Preparation

```bash
cd babyLLM

# Download and prepare data (if not already done)
python src/v2/prepare_data_v2.py \
    --input data/processed/all.txt \
    --output_dir data/processed_v2 \
    --no_minhash

# Train tokenizer (if not already done)
python src/v2/train_tokenizer_v2.py
```

### Start Training

```bash
# Option 1: Use launch script (recommended)
bash launch_v10_pipeline.sh

# Option 2: Manual training with accelerate
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

## 📊 Version History

### V1 - GPT-2 Baseline (2026-04-19)
- **Architecture**: GPT-2 (110M parameters)
- **Tokenizer**: BPE (32K vocab)
- **Training**: Multi-GPU DDP
- **Status**: ✅ Completed
- **Key Files**: `src/v1/`, `launch_v4.sh`

### V2 - LLaMA Architecture (2026-04-19)
- **Architecture**: LLaMA (RoPE + GQA + SwiGLU + RMSNorm)
- **Tokenizer**: ByteLevel BPE (32K vocab, zero OOV)
- **Optimizations**: bf16 mixed precision, Gradient Checkpointing, SDPA/Flash Attention 2
- **Data**: Precise deduplication (MD5), HTML cleaning, short text filtering
- **Status**: ✅ Completed (25/25 epochs in ~7.5 hours)
- **Key Files**: `src/v2/`, `launch_v5.sh`, `launch_v5_kd.sh`

### V3 - SentencePiece + WSD Scheduler (2026-04-20)
- **Tokenizer**: SentencePiece (SPM)
- **LR Scheduler**: Warmup-then-Stable-Decay (WSD)
- **Status**: ✅ Completed
- **Key Files**: `src/v3/`, `launch_v6_pipeline.sh`

### V4 - Ensemble Checkpoints (2026-04-20)
- **Strategy**: Ensemble multiple checkpoints for better performance
- **Status**: ✅ Completed
- **Key Files**: `src/v4/`, `launch_v7.sh`

### V5 - Knowledge Distillation (2026-04-21)
- **Technique**: Knowledge distillation from larger teacher model
- **Status**: ✅ Completed
- **Key Files**: `src/v5/`, `launch_v8_pipeline.sh`

### V6 - Advanced Data Processing (2026-04-22)
- **Improvement**: Enhanced data cleaning and preprocessing
- **Status**: ✅ Completed
- **Key Files**: `src/v6/`, `launch_v9_probe.sh`

### V7 - SOTA Training Techniques (2026-04-23)
- **Techniques**: Latest SOTA techniques from BabyLM 2024 winners
- **Status**: ✅ Completed
- **Key Files**: `src/v7/`, `launch_v9_polish_probe.sh`

### V8 - Optimized Training (2026-04-24)
- **Focus**: Training efficiency and optimization
- **Status**: ✅ Completed
- **Key Files**: `src/v8/`

### V9 - Probe Experiments (2026-04-25)
- **Purpose**: Probing experiments to understand model behavior
- **Status**: ✅ Completed
- **Key Files**: `src/v9/`

### V10 - Latest Training Pipeline (2026-04-26)
- **Status**: 🚧 Active development
- **Key Files**: `src/v10/`, `launch_v10_pipeline.sh`, `auto_start_v6_after_kd.sh`

---

## 🔄 Continue Training

### From Existing Checkpoint

If you want to continue training from an existing checkpoint:

```bash
cd babyLLM

# Check available checkpoints
ls output/babylm-*/checkpoints/

# Resume training
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

### Key Parameters for Resuming

- `--resume_from_checkpoint`: Path to checkpoint directory
- `--num_epochs`: Total epochs (training will continue until this number)
- The optimizer state and training progress will be restored automatically

### Checkpoint Management

Checkpoints are saved in `output/<model_name>/checkpoints/`:
- `checkpoint-XXXX`: Named by step number
- `final`: Final checkpoint after training completes
- Each checkpoint contains: `model.safetensors`, `optimizer.pt`, `scheduler.pt`, `trainer_state.json`

---

## 👥 Team Collaboration

### Branch Strategy

- **`main`**: Stable production code
- **`sulab`**: Current development branch for SULAB team
- **Feature branches**: Create new branches for experiments (e.g., `v11-experiment`)

### Workflow

1. **Pull latest changes**:
   ```bash
   git checkout sulab
   git pull origin sulab
   ```

2. **Create feature branch**:
   ```bash
   git checkout -b v11-your-experiment
   ```

3. **Make changes and commit**:
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

4. **Push and create PR**:
   ```bash
   git push origin v11-your-experiment
   # Create PR on GitHub to merge into sulab
   ```

### Code Guidelines

- Follow existing code structure and naming conventions
- Add comments for complex logic
- Update documentation when adding new features
- Test changes before committing
- Use descriptive commit messages (conventional commits format)

### Communication

- Use GitHub Issues for bug reports and feature requests
- Document experimental results in `plans/` directory
- Update `REPORT.md` with significant findings

---

## 📚 Technical Documentation

### Core Documentation

- **`babyLLM/README.md`**: Detailed technical documentation for the training code
- **`babyLLM/REPORT.md`**: Comprehensive experiment report with results
- **`docs/SOTA_TECHNIQUES_RESEARCH.md`**: Research on SOTA techniques for small language models

### Training Plans

- **`plans/V1_V5_DEEP_TECHNICAL_ANALYSIS.md`**: Deep technical analysis of V1-V5
- **`plans/V5_TRAINING_PLAN.md`**: V5 training plan
- **`plans/V5_TRAINING_STATUS_REPORT.md`**: V5 training status
- **`plans/V7_SOTA_TRAINING_PLAN.md`**: V7 SOTA training plan
- **`plans/POST_MORTEM_ANALYSIS_V1_V4.md`**: Post-mortem analysis of V1-V4

### Evaluation Pipeline

- **`chinese-babylm-eval-pipeline/README.md`**: Official evaluation pipeline documentation
- See evaluation pipeline README for detailed evaluation instructions

---

## 💻 Hardware Requirements

### Minimum Configuration

- **GPU**: 1× NVIDIA GPU (16GB VRAM)
- **CPU**: 8+ cores
- **RAM**: 32GB
- **Storage**: 50GB free space

### Recommended Configuration

- **GPU**: 4× NVIDIA A6000 (48GB each)
- **CPU**: 32+ cores
- **RAM**: 128GB
- **Storage**: 500GB SSD

### Performance Metrics

With 4× A6000 GPUs:
- **Throughput**: 80K-105K tokens/sec
- **Training Time**: ~7.5 hours for 25 epochs (V2)
- **GPU Utilization**: 99-100%
- **Memory Usage**: 13-32GB per GPU

---

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory (OOM)

- Reduce `--batch_size` or `--gradient_accumulation_steps`
- Enable gradient checkpointing: `--use_checkpoint True`
- Use smaller model configuration

#### Training Stuck

- Check GPU utilization with `nvidia-smi`
- Verify data loading is working
- Check WandB logs for errors

#### Checkpoint Loading Issues

- Ensure checkpoint path is correct
- Verify model architecture matches checkpoint
- Check if all checkpoint files are present

#### Data Issues

- Verify data files exist in `data/processed_v2/`
- Check tokenizer is trained: `data/tokenizer_v2/tokenizer.json`
- Re-run data preparation if needed

### Getting Help

- Check existing issues in GitHub repository
- Review technical documentation in `docs/` and `plans/`
- Consult `babyLLM/README.md` for detailed code documentation
- Contact team members through project communication channels

---

## 📊 Current Status

### Latest Training (V10)

- **Status**: 🚧 In development
- **Branch**: `sulab`
- **Last Updated**: 2026-04-26
- **Key Files**: `src/v10/`, `launch_v10_pipeline.sh`

### Available Checkpoints

Check `babyLLM/output/` for available model checkpoints from previous versions.

---

## 📄 License

MIT License

---

## 👤 Team

**C5 Team - SULAB Branch**

- **GitHub**: https://github.com/C5-jpg/babyLLM
- **Branch**: `sulab`
- **Competition**: NLPCC 2026 ChineseBabyLM Challenge

---

## 📞 Contact

For questions or issues:
- Open an issue on GitHub
- Contact team members directly
- Check documentation in `docs/` and `plans/` directories

---

> 🎯 **Goal**: Build a high-performance Chinese language model that excels on the ChineseBabyLM challenge evaluation tasks.
