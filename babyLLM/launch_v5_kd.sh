#!/bin/bash
# ============================================================
# ChineseBabyLM V5 - Phase 2 知识蒸馏启动脚本
# 使用 V2 best_model 作为教师，对 V5 学生模型进行 KD 微调
# ============================================================

set -e

cd /home/kehe/babyllm/babyLLM/src/v5

source /home/kehe/anaconda3/etc/profile.d/conda.sh
conda activate data 2>/dev/null || true

TEACHER_MODEL="/home/kehe/babyllm/babyLLM/output/babylm-llama-v2/best_model"
STUDENT_MODEL="/home/kehe/babyllm/babyLLM/output/babylm-llama-v5/best_model"
TEACHER_LOGITS_DIR="/home/kehe/babyllm/babyLLM/output/teacher_logits_v2"
KD_OUTPUT="/home/kehe/babyllm/babyLLM/output/babylm-llama-v5-kd"

echo "============================================================"
echo "ChineseBabyLM V5 - Phase 2: 知识蒸馏"
echo "教师模型: V2 best_model (125M)"
echo "学生模型: V5 best_model (~60M)"
echo "GPU: 4x A6000 (48GB)"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Step 1: 检查教师模型是否存在
if [ ! -d "$TEACHER_MODEL" ]; then
    echo "错误: 教师模型不存在: $TEACHER_MODEL"
    echo "请先确认 V2 best_model 已训练完成"
    exit 1
fi

# Step 2: 检查学生模型是否存在
if [ ! -d "$STUDENT_MODEL" ]; then
    echo "错误: 学生模型不存在: $STUDENT_MODEL"
    echo "请先运行 Phase 1 预训练: bash launch_v5.sh"
    exit 1
fi

# Step 3: 生成教师 logits (如果尚未生成)
if [ ! -f "$TEACHER_LOGITS_DIR/teacher_logits.npy" ]; then
    echo ""
    echo "--- Step 3a: 生成教师 logits ---"
    mkdir -p "$TEACHER_LOGITS_DIR"
    
    python generate_teacher_logits.py \
        --teacher_model_path "$TEACHER_MODEL" \
        --tokenizer_dir /home/kehe/babyllm/babyLLM/data/tokenizer_v3 \
        --data_file /home/kehe/babyllm/babyLLM/data/processed_v3/train.txt \
        --output_dir "$TEACHER_LOGITS_DIR" \
        --block_size 1024 \
        --batch_size 16 \
        --top_k 10
    
    echo "教师 logits 生成完成"
else
    echo "教师 logits 已存在，跳过生成步骤"
fi

# Step 4: Phase 2 KD 训练
echo ""
echo "--- Step 3b: Phase 2 知识蒸馏训练 ---"
mkdir -p "$KD_OUTPUT"

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num_processes=3 \
    --mixed_precision=bf16 \
    train_v5.py \
    --phase kd \
    --data_dir /home/kehe/babyllm/babyLLM/data \
    --tokenizer_dir /home/kehe/babyllm/babyLLM/data/tokenizer_v3 \
    --output_dir "$KD_OUTPUT" \
    --student_model_path "$STUDENT_MODEL" \
    --teacher_logits_dir "$TEACHER_LOGITS_DIR" \
    --d_model 512 \
    --n_layer 12 \
    --n_head 8 \
    --n_kv_heads 4 \
    --max_length 1024 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --num_epochs 10 \
    --warmup_ratio 0.03 \
    --gradient_accumulation_steps 1 \
    --attention_dropout 0.05 \
    --bpe_dropout 0.0 \
    --patience 3 \
    --rope_theta 10000.0 \
    --lambda_ce 0.3 \
    --lambda_kd 0.7 \
    --temperature 2.0 \
    --top_k 10 \
    --logging_steps 50 \
    --save_steps 5000 \
    --save_total_limit 2 \
    --wandb_project chinese-babylm \
    --wandb_run_name llama-v5-512d-12l-kd-v2teacher-4gpu \
    2>&1 | tee /home/kehe/babyllm/babyLLM/output/train_v5_phase2_kd.log

echo ""
echo "============================================================"
echo "Phase 2 知识蒸馏完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "最终模型: $KD_OUTPUT/best_model"
echo "============================================================"
