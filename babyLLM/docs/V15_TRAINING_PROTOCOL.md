# V15 Training Protocol

Generated: 2026-05-12

## Overview

V15 is the culmination of V1-V14 analysis, targeting SOTA for compact Chinese language models.

- **Target**: PPL < 38.0, ZhoBLiMP > 65%, ~58M params
- **Architecture**: LLaMA, 640d, 14L, 10Q/5KV GQA
- **Pipeline**: 2-stage (CLM → MNTP)
- **Dataset**: babylm-zho-100M (PPL-filtered, MinHash-deduped)

## Prerequisites

- 4× NVIDIA RTX A6000 (48GB VRAM each)
- CUDA 12.4+
- Python 3.10+ with conda environment `data`
- BabyLM repo cloned at `/home/kehe/babyllm`

## Quick Start

```bash
cd /home/kehe/babyllm/babyLLM
bash launch_v15_pipeline.sh
```

## Stage-by-Stage Protocol

### Stage 1: CLM + SGDR + Focal Loss (10 epochs)

| Parameter | Value |
|-----------|-------|
| Learning Rate | 5e-4 |
| Scheduler | SGDR (T_0=1 epoch, T_mult=2) |
| Focal Loss γ | 2.0 |
| Label Smoothing | 0.1 → anneal |
| BPE Dropout | 0.1 |
| Attention Dropout | 0.1 |
| EMA Decay | 0.999 + 0.9999 |
| Batch Size | 16×4×2 = 128 |
| Max Seq Length | 1024 |
| Stride | 512 |
| Early Stop | patience=5, min_delta=1e-4 |
| Eval Steps | 200 |
| Save Steps | 1000 |

**Expected Duration**: ~2-3 hours
**Expected PPL**: ~42-45

### Stage 2: MNTP + Dynamic CLM (12 epochs)

| Parameter | Value |
|-----------|-------|
| Learning Rate | 4e-4 |
| Scheduler | Cosine |
| Focal Loss γ | 1.5 |
| Label Smoothing | 0.05 → anneal |
| BPE Dropout | 0.1 |
| Attention Dropout | 0.05 |
| CLM Ratio | 0.25→0.125→0.0625 (dynamic) |
| Mask Ratio | 0.25→0.10 (anneal) |
| EMA Decay | 0.999 + 0.9999 |
| Early Stop | patience=10, min_delta=1e-4 |
| Eval Steps | 200 |

**Input**: Stage 1 best EMA model
**Expected Duration**: ~3-4 hours
**Expected PPL**: ~37-38

## New V15 Features

### Multi-Scale EMA
Tracks exponential moving averages at two decay rates simultaneously:
- decay=0.999 (standard, good for noisy gradients)
- decay=0.9999 (slower, preserves more training history)

Both are evaluated at each checkpoint to determine which performs better.

### Gradient Norm Monitoring
Per-layer gradient norms logged every 250 steps. Useful for:
- Detecting training instabilities
- Identifying layers with vanishing/exploding gradients
- Tuning learning rates per layer group

### Finer Eval Cadence
Evaluation every 200 steps (vs V14's 500). Provides:
- Earlier detection of convergence
- Better checkpoint selection
- More granular training curves

### Gradient Spike Detection
Automatically detects gradient norm spikes (>10× previous norm) and logs warnings. Can be extended to auto-reduce LR on spike detection.

## Monitoring

### Weights & Biases
- Project: `chinese-babylm`
- Run names: `babylm-v15-stage1-clm-sgdr`, `babylm-v15-stage2-mntp-dynamic`
- Metrics: train/loss, val/loss, val/ppl, grad_norm/*, train/lr, train/gpu_mem_gb

### Checkpoints
- Location: `/mnt/sda/kehe/babyllm_output/babylm-v15/stage{1,2}_*/`
- `best_model/`: Best non-EMA model
- `best_model_ema/`: Best EMA model (decay=0.999)
- `best_model_ema_0.9999/`: Best EMA model (decay=0.9999)
- `latest_checkpoint/`: Resume point

### Logs
- Pipeline log: `/mnt/sda/kehe/babyllm_output/babylm-v15/pipeline_v15.log`
- Stage logs: Timestamped in `logs/` directory

## Troubleshooting

### OOM Errors
The pipeline auto-reduces batch size on OOM. If persistent:
```bash
# Reduce batch size manually
bash launch_v15_pipeline.sh --batch_size 8
```

### Resume from Checkpoint
```bash
# Pipeline auto-resumes from latest_checkpoint
bash launch_v15_pipeline.sh --skip-data --skip-stage1
```

### Re-run Evaluation Only
```bash
bash launch_v15_pipeline.sh --skip-data --skip-stage1 --skip-stage2
```

## Post-Training

After training completes:
1. Check `eval_stage2_ema.json` for best PPL
2. Run official evaluation pipeline for ZhoBLiMP + fine-tuning tasks
3. Convert tokenizer for HF compatibility
4. Submit to ChineseBabyLM challenge

## Expected Results

| Metric | V13 (Current SOTA) | V15 Target |
|--------|-------------------|------------|
| PPL | 38.68 | < 38.0 |
| Params | 94.2M | ~58M |
| PPL/10M | 4.1 | < 6.6 |
| Tokens/Param | 1.1× | 1.7× |
| ZhoBLiMP | 63.47% | > 65% |
| AFQMC | 69.0% | > 70% |
