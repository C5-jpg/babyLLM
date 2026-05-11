#!/bin/bash
# preflight_check.sh — V14 数据预处理校验
# 在启动训练前验证数据完整性、新鲜度和规模
set -e

DATA_DIR="/mnt/sda/kehe/babyllm_output/babylm-v14/data_v14"
RAW_DATA="/home/kehe/babyllm/babyLLM/data/processed_v7"
TRAIN_FILE="$DATA_DIR/train.txt"
VAL_FILE="$DATA_DIR/val.txt"
PROJECT_DIR="/home/kehe/babyllm/babyLLM"
TOKENIZER_DIR="$PROJECT_DIR/data/tokenizer_v7"
V13_MODEL="/mnt/sda/kehe/babyllm_output/babylm-v13/stage2_mntp/best_model_ema"

echo "=========================================="
echo "  V14 Pre-flight Data Check"
echo "=========================================="

# Check 1: 文件存在性
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "[FAIL] data_v14 训练文件不存在"
    echo "  自动执行数据预处理..."
    
    # 检查 V13 PPL 过滤模型是否存在
    PPL_MODEL_ARG=""
    if [ -d "$V13_MODEL" ]; then
        PPL_MODEL_ARG="--model_path $V13_MODEL --tokenizer_dir $TOKENIZER_DIR"
        echo "  使用 V13 EMA 模型进行 PPL 过滤"
    else
        PPL_MODEL_ARG="--skip_ppl_filter"
        echo "  WARNING: V13 模型不存在，跳过 PPL 过滤"
    fi
    
    python "$PROJECT_DIR/src/v14/prepare_data.py" \
        --input_dir "$RAW_DATA" \
        --output_dir "$DATA_DIR" \
        $PPL_MODEL_ARG \
        --max_ppl 250 \
        --min_ppl 3 \
        --hard_upsample_factor 2
    
    echo "  数据预处理完成"
fi

# Check 2: 文件非空
TRAIN_SIZE=$(stat -c%s "$TRAIN_FILE" 2>/dev/null || echo "0")
VAL_SIZE=$(stat -c%s "$VAL_FILE" 2>/dev/null || echo "0")

if [ "$TRAIN_SIZE" -lt 1000000 ]; then
    echo "[FAIL] train.txt 仅 ${TRAIN_SIZE} bytes，数据不完整"
    exit 1
fi

if [ "$VAL_SIZE" -lt 100000 ]; then
    echo "[FAIL] val.txt 仅 ${VAL_SIZE} bytes，数据不完整"
    exit 1
fi

# Check 3: 时间戳新鲜度
RAW_MTIME=$(stat -c%Y "$RAW_DATA/train.txt" 2>/dev/null || echo "0")
V14_MTIME=$(stat -c%Y "$TRAIN_FILE" 2>/dev/null || echo "0")

if [ "$V14_MTIME" -le "$RAW_MTIME" ]; then
    echo "[WARN] data_v14 数据陈旧于原始数据，建议重新预处理"
fi

# Check 4: 行数和大小报告
TRAIN_LINES=$(wc -l < "$TRAIN_FILE")
VAL_LINES=$(wc -l < "$VAL_FILE")
TRAIN_MB=$((TRAIN_SIZE / 1048576))
VAL_MB=$((VAL_SIZE / 1048576))

echo ""
echo "=========================================="
echo "  Pre-flight Check PASSED"
echo "=========================================="
echo "  train.txt: ${TRAIN_LINES} 行, ${TRAIN_MB} MB"
echo "  val.txt:   ${VAL_LINES} 行, ${VAL_MB} MB"
echo "  Tokenizer: $TOKENIZER_DIR"
echo "=========================================="
