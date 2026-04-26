#!/bin/bash
# ChineseBabyLM V4 - 完整训练 pipeline (SOTA 冠军配置)
# 用法: bash run_pipeline_v4.sh
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/output/babylm-llama-v4"
SRC_DIR="${PROJECT_ROOT}/src/v4"
TOKENIZER_DIR="${DATA_DIR}/tokenizer_v4"
TRAIN_FILE="${DATA_DIR}/processed_v3/train.txt"
VAL_FILE="${DATA_DIR}/processed_v3/val.txt"

echo "============================================================"
echo "ChineseBabyLM V4 - SOTA 训练 Pipeline"
echo "============================================================"
echo "项目根目录: ${PROJECT_ROOT}"
echo "数据目录:   ${DATA_DIR}"
echo "输出目录:   ${OUTPUT_DIR}"

# ============================================================
# Step 1: 训练 Tokenizer (如果不存在)
# ============================================================
if [ ! -f "${TOKENIZER_DIR}/tokenizer.json" ]; then
    echo ""
    echo "============================================================"
    echo "Step 1: 训练 V4 Tokenizer"
    echo "============================================================"
    cd "${SRC_DIR}"
    python train_tokenizer_v4.py \
        --input_file "${TRAIN_FILE}" \
        --output_dir "${TOKENIZER_DIR}" \
        --vocab_size 32000 \
        --model_type bpe \
        --character_coverage 0.9995
else
    echo ""
    echo "Step 1: V4 Tokenizer 已存在，跳过训练。"
fi

# ============================================================
# Step 2: 训练模型 (SOTA 冠军配置: 1024d × 24层 × 350M参数)
# ============================================================
echo ""
echo "============================================================"
echo "Step 2: 训练 V4 模型 (SOTA 冠军配置)"
echo "============================================================"
echo "  模型规模: 1024d × 24层 × 16头 × 8KV头 ≈ 350M参数"
echo "  训练策略: Cosine LR + BPE Dropout + Dropout退火"
echo "  数据增强: 50% 滑动窗口（数据量翻倍）"
echo "  RoPE base: 50000（扩展长上下文能力）"
echo "============================================================"
cd "${SRC_DIR}"

# 自动检测 GPU 数量
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
echo "检测到 GPU 数量: ${NUM_GPUS}"

if [ "${NUM_GPUS}" -ge 2 ]; then
    # 多 GPU 训练
    CUDA_VISIBLE_DEVICES=$(python3 -c "print(','.join(str(i) for i in range(${NUM_GPUS})))" 2>/dev/null || echo "0") \
    accelerate launch \
        --num_processes=${NUM_GPUS} \
        --mixed_precision=bf16 \
        train_v4.py \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --d_model 1024 \
        --n_layer 24 \
        --n_head 16 \
        --n_kv_heads 8 \
        --max_length 1024 \
        --batch_size 16 \
        --learning_rate 3e-4 \
        --weight_decay 0.1 \
        --num_epochs 50 \
        --warmup_ratio 0.03 \
        --gradient_accumulation_steps 2 \
        --encode_batch_size 4096 \
        --attention_dropout 0.1 \
        --bpe_dropout 0.1 \
        --patience 10 \
        --rope_theta 50000.0 \
        --logging_steps 50 \
        --save_steps 5000 \
        --wandb_project chinese-babylm \
        --wandb_run_name "llama-v4-1024d-24l-${NUM_GPUS}gpu-sota"
else
    # 单 GPU 训练 (调整 batch_size 适配显存)
    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --num_processes=1 \
        --mixed_precision=bf16 \
        train_v4.py \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --d_model 1024 \
        --n_layer 24 \
        --n_head 16 \
        --n_kv_heads 8 \
        --max_length 1024 \
        --batch_size 8 \
        --learning_rate 3e-4 \
        --weight_decay 0.1 \
        --num_epochs 50 \
        --warmup_ratio 0.03 \
        --gradient_accumulation_steps 4 \
        --attention_dropout 0.1 \
        --bpe_dropout 0.1 \
        --patience 10 \
        --rope_theta 50000.0 \
        --logging_steps 50 \
        --save_steps 5000 \
        --wandb_project chinese-babylm \
        --wandb_run_name "llama-v4-1024d-24l-1gpu-sota"
fi

# ============================================================
# Step 3: PPL 评测
# ============================================================
echo ""
echo "============================================================"
echo "Step 3: 评测 V4 模型 (PPL + 文本生成)"
echo "============================================================"
cd "${SRC_DIR}"

python evaluate_v4.py \
    --model_dir "${OUTPUT_DIR}/best_model" \
    --val_file "${VAL_FILE}" \
    --block_size 1024 \
    --batch_size 8

# ============================================================
# Step 4: BLiMP 竞赛评测
# ============================================================
echo ""
echo "============================================================"
echo "Step 4: BLiMP 语法可接受性评测"
echo "============================================================"
cd "${SRC_DIR}"

if [ -f "evaluate_blimp_v4.py" ]; then
    # 使用内置示例数据测试
    python evaluate_blimp_v4.py \
        --model_dir "${OUTPUT_DIR}/best_model" \
        --use_demo_data \
        --output_file "${OUTPUT_DIR}/blimp_results.json"
else
    echo "⚠️  evaluate_blimp_v4.py 不存在，跳过 BLiMP 评测"
fi

echo ""
echo "============================================================"
echo "V4 Pipeline 完成!"
echo "  模型保存: ${OUTPUT_DIR}/best_model"
echo "  BLiMP结果: ${OUTPUT_DIR}/blimp_results.json"
echo "============================================================"
