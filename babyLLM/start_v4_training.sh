#!/bin/bash
# ChineseBabyLM V4 - tmux 训练启动脚本
# 用法: bash start_v4_training.sh

SESSION="babylm_v4"
PROJECT_ROOT="/home/kehe/babyllm/babyLLM"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/output/babylm-llama-v4"
SRC_DIR="${PROJECT_ROOT}/src/v4"
TOKENIZER_DIR="${DATA_DIR}/tokenizer_v4"
TRAIN_FILE="${DATA_DIR}/processed_v3/train.txt"
VAL_FILE="${DATA_DIR}/processed_v3/val.txt"
LOG_FILE="${PROJECT_ROOT}/output/train_v4.log"

# 检查 session 是否已存在
if tmux has-session -t ${SESSION} 2>/dev/null; then
    echo "Session '${SESSION}' 已存在，先杀死旧 session..."
    tmux kill-session -t ${SESSION}
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 自动检测 GPU 数量
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 4)
echo "检测到 GPU 数量: ${NUM_GPUS}"

echo "启动 tmux session: ${SESSION}"
tmux new-session -d -s ${SESSION} -x 220 -y 50

# 构建训练命令
TRAIN_CMD="cd ${SRC_DIR} && conda activate data && \\
CUDA_VISIBLE_DEVICES=0,1,2,3 \\
accelerate launch \\
    --num_processes=4 \\
    --mixed_precision=bf16 \\
    train_v4.py \\
    --data_dir ${DATA_DIR} \\
    --output_dir ${OUTPUT_DIR} \\
    --d_model 1024 \\
    --n_layer 24 \\
    --n_head 16 \\
    --n_kv_heads 8 \\
    --max_length 1024 \\
    --batch_size 16 \\
    --learning_rate 3e-4 \\
    --weight_decay 0.1 \\
    --num_epochs 50 \\
    --warmup_ratio 0.03 \\
    --gradient_accumulation_steps 2 \\
    --attention_dropout 0.1 \\
    --bpe_dropout 0.1 \\
    --patience 10 \\
    --rope_theta 50000.0 \\
    --logging_steps 50 \\
    --save_steps 5000 \\
    --wandb_project chinese-babylm \\
    --wandb_run_name \"llama-v4-1024d-24l-cosine-bpe_dropout-dropout_anneal-4gpu\" \\
    2>&1 | tee ${LOG_FILE}"

tmux send-keys -t ${SESSION} "${TRAIN_CMD}" Enter

echo ""
echo "====================================================="
echo "V4 训练已在 tmux session '${SESSION}' 中启动!"
echo "日志文件: ${LOG_FILE}"
echo ""
echo "查看训练状态:"
echo "  tmux attach -t ${SESSION}"
echo "  tail -f ${LOG_FILE}"
echo "====================================================="
