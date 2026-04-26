#!/bin/bash
# ============================================================
# ChineseBabyLM V5 - Phase 1 标准预训练启动脚本 (Round 2)
# ~51M 参数小模型 (d_model=512, 12层, 8头)
# 3x A6000 GPU, bf16 混合精度
# 输出目录已软链接到机械硬盘 /mnt/sda/kehe/babyllm_output/
# ============================================================

set -e

cd /home/kehe/babyllm/babyLLM/src/v5

# 激活 conda 环境
source /home/kehe/anaconda3/etc/profile.d/conda.sh
conda activate data 2>/dev/null || true

# ============================================================
# NCCL 稳定性配置 - 防止分布式训练超时
# ============================================================
export NCCL_TIMEOUT=1800000          # 30 分钟超时 (ms), 默认 10 分钟
export NCCL_IB_DISABLE=1             # 单机不需要 InfiniBand
export NCCL_P2P_LEVEL=SYS            # 跨 GPU 通信
export NCCL_DEBUG=WARN               # 只显示警告
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 异步错误处理
export NCCL_SOCKET_IFNAME=^docker0   # 排除 docker 接口

# 验证输出目录是软链接到 HDD
OUTPUT_DIR=/home/kehe/babyllm/babyLLM/output/babylm-llama-v5
echo "输出目录: $OUTPUT_DIR"
echo "实际路径: $(readlink -f $OUTPUT_DIR)"
echo "磁盘空间: $(df -h $(readlink -f $OUTPUT_DIR) | tail -1)"

echo "============================================================"
echo "ChineseBabyLM V5 - Phase 1: 标准预训练 (Round 2)"
echo "模型: ~51M 参数 (d_model=512, 12层, 8Q/4KV)"
echo "GPU: 3x A6000 (48GB) - GPU 1,2,3"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Phase 1: 标准预训练 (GPU 0 被占用，使用 GPU 1-3)
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num_processes=3 \
    --mixed_precision=bf16 \
    train_v5.py \
    --phase pretrain \
    --data_dir /home/kehe/babyllm/babyLLM/data \
    --tokenizer_dir /home/kehe/babyllm/babyLLM/data/tokenizer_v3 \
    --output_dir /home/kehe/babyllm/babyLLM/output/babylm-llama-v5 \
    --d_model 512 \
    --n_layer 12 \
    --n_head 8 \
    --n_kv_heads 4 \
    --max_length 1024 \
    --batch_size 32 \
    --learning_rate 6e-4 \
    --weight_decay 0.1 \
    --num_epochs 15 \
    --warmup_ratio 0.05 \
    --gradient_accumulation_steps 1 \
    --attention_dropout 0.05 \
    --bpe_dropout 0.1 \
    --patience 5 \
    --rope_theta 10000.0 \
    --logging_steps 50 \
    --save_steps 5000 \
    --save_total_limit 2 \
    --wandb_project chinese-babylm \
    --wandb_run_name llama-v5-512d-12l-pretrain-4gpu \
    2>&1 | tee /home/kehe/babyllm/babyLLM/output/train_v5_phase1.log

echo ""
echo "============================================================"
echo "Phase 1 训练完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
