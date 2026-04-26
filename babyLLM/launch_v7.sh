#!/bin/bash
# ============================================================
# ChineseBabyLM V7 - SOTA Sprint Launch Script
# ~35M params, 8K vocab, CLM+MNTP hybrid training
# 4x A6000 GPU, bf16 mixed precision
# ============================================================

set -e

cd /home/kehe/babyllm/babyLLM

source /home/kehe/anaconda3/etc/profile.d/conda.sh
conda activate data 2>/dev/null || true

export NCCL_TIMEOUT=1800000
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^docker0

OUTPUT_DIR="/home/kehe/babyllm/babyLLM/output/babylm-v7"
echo "Output dir: $OUTPUT_DIR"
echo "Actual path: $(readlink -f $OUTPUT_DIR)"
echo "Disk space: $(df -h $(readlink -f $OUTPUT_DIR) | tail -1)"

echo "============================================================"
echo "ChineseBabyLM V7 - CLM+MNTP Hybrid Training"
echo "Model: ~35M params (448d, 12L, 7Q/4KV GQA)"
echo "Tokenizer: 8K SPM with <mask> token"
echo "Data: ~91.5M tokens (V7 cleaned)"
echo "GPU: 4x A6000 (48GB)"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    src/v7/train_v7.py \
    --data_dir /home/kehe/babyllm/babyLLM/data/processed_v7 \
    --tokenizer_dir /home/kehe/babyllm/babyLLM/data/tokenizer_v7 \
    --output_dir "$OUTPUT_DIR" \
    --d_model 448 \
    --n_layer 12 \
    --n_head 8 \
    --n_kv_heads 4 \
    --max_length 1024 \
    --batch_size 32 \
    --learning_rate 6e-4 \
    --weight_decay 0.1 \
    --num_epochs 10 \
    --warmup_ratio 0.05 \
    --gradient_accumulation_steps 1 \
    --label_smoothing 0.1 \
    --attention_dropout 0.1 \
    --bpe_dropout 0.1 \
    --gradient_checkpointing \
    --clm_ratio 0.125 \
    --mask_ratio_start 0.30 \
    --mask_ratio_end 0.15 \
    --patience 8 \
    --rope_theta 10000.0 \
    --logging_steps 50 \
    --save_steps 2000 \
    --save_total_limit 3 \
    --wandb_project chinese-babylm \
    --wandb_run_name babylm-v7-448d-12l-clm-mntp-8kvocab-4gpu-8q4kv \
    2>&1 | tee "$OUTPUT_DIR/train_v7_full.log"

echo ""
echo "============================================================"
echo "V7 Training Complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Best model: $OUTPUT_DIR/best_model"
echo "============================================================"
