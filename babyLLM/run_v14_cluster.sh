#!/bin/bash
# run_v14_cluster.sh — V14 自动化部署脚本
# 替代 SLURM，在单节点 4×A6000 上启动训练
set -e

PROJECT_DIR="/home/kehe/babyllm/babyLLM"
OUTPUT_DIR="/mnt/sda/kehe/babyllm_output/babylm-v14"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  V14 Training Deployment"
echo "=========================================="

# Step 1: 检查 accelerate 是否安装
if ! python -c "import accelerate" 2>/dev/null; then
    echo "[INFO] Installing accelerate..."
    pip install accelerate -q
fi
echo "[OK] accelerate installed"

# Step 2: Pre-flight 数据校验
echo "[INFO] Running preflight data check..."
if ! bash "$PROJECT_DIR/preflight_check.sh"; then
    echo "[FAIL] Preflight check failed. Fix data issues before training."
    exit 1
fi

# Step 3: 检查 GPU 状态
echo "[INFO] GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

# Step 4: 后台启动训练
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/v14_pipeline_${TIMESTAMP}.log"

echo "[INFO] Starting V14 pipeline..."
echo "  Log: $LOG_FILE"
echo "  PID file: $OUTPUT_DIR/training.pid"

nohup bash "$PROJECT_DIR/launch_v14_pipeline.sh" \
    > "$LOG_FILE" 2>&1 &
PID=$!

echo "$PID" > "$OUTPUT_DIR/training.pid"

echo ""
echo "=========================================="
echo "  V14 Training Started"
echo "=========================================="
echo "  PID: $PID"
echo "  Log: $LOG_FILE"
echo ""
echo "  Monitor: tail -f $LOG_FILE"
echo "  WandB: https://wandb.ai (project: chinese-babylm)"
echo "  Stop: kill $PID"
echo "=========================================="
