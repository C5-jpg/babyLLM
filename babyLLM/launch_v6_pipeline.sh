#!/bin/bash
# ============================================================
# ChineseBabyLM V6 自动化训练流水线
# 在 V5 Phase 2 KD 完成后自动执行
# V6 三阶段: CLM -> CLM+MLM -> Reverse KL KD
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

V6_DATA="/home/kehe/babyllm/babyLLM/data/processed_v6"
V6_OUTPUT="/home/kehe/babyllm/babyLLM/output/babylm-llama-v6"
TOKENIZER="/home/kehe/babyllm/babyLLM/data/tokenizer_v3"
V3_TRAIN="/home/kehe/babyllm/babyLLM/data/processed_v3/train.txt"
V3_VAL="/home/kehe/babyllm/babyLLM/data/processed_v3/val.txt"
V5_KD_BEST="/home/kehe/babyllm/babyLLM/output/babylm-llama-v5-kd/best_model"
TEACHER_LOGITS_V5KD="/home/kehe/babyllm/babyLLM/output/teacher_logits_v5kd"
GPU_IDS="0,1,2,3"
NUM_GPU=4

PIPELINE_LOG="/home/kehe/babyllm/babyLLM/output/v6_pipeline.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo ""
echo "============================================================"
echo "ChineseBabyLM V6 自动化训练流水线"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: ${NUM_GPU}x A6000 (GPU ${GPU_IDS})"
echo "============================================================"

# ============================================================
# Step 0: 准备 V6 数据（如果尚未准备）
# ============================================================
if [ ! -f "$V6_DATA/train.txt" ]; then
    echo ""
    echo "--- Step 0: V6 数据清洗 ---"
    python src/v6/prepare_data_v6.py \
        --input "$V3_TRAIN" \
        --val_input "$V3_VAL" \
        --output_dir "$V6_DATA" \
        --min_length 15 \
        --max_length 300 \
        --max_special_ratio 0.3
    echo "V6 数据准备完成: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo ""
    echo "V6 数据已存在，跳过数据准备"
    echo "  train.txt: $(wc -l < $V6_DATA/train.txt) lines"
    echo "  val.txt: $(wc -l < $V6_DATA/val.txt) lines"
fi

# ============================================================
# Step 1: V6 Stage 1 - CLM 预训练
# ============================================================
STAGE1_OUTPUT="${V6_OUTPUT}-stage1-clm"

echo ""
echo "============================================================"
echo "--- Step 1: V6 Stage 1 - CLM 预训练 ---"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes=$NUM_GPU \
    --mixed_precision=bf16 \
    src/v6/train_v6.py \
    --stage clm \
    --data_dir /home/kehe/babyllm/babyLLM/data \
    --output_dir "$STAGE1_OUTPUT" \
    --tokenizer_dir "$TOKENIZER" \
    --d_model 640 \
    --n_layer 12 \
    --n_head 10 \
    --n_kv_heads 5 \
    --max_length 1024 \
    --batch_size 24 \
    --learning_rate 6e-4 \
    --weight_decay 0.1 \
    --num_epochs 4 \
    --warmup_ratio 0.05 \
    --label_smoothing 0.1 \
    --attention_dropout 0.1 \
    --bpe_dropout 0.1 \
    --patience 5 \
    --rope_theta 10000.0 \
    --gradient_checkpointing \
    --logging_steps 50 \
    --save_steps 5000 \
    --save_total_limit 2 \
    --wandb_project chinese-babylm \
    --wandb_run_name llama-v6-640d-12l-stage1-clm-4gpu

echo "Step 1 完成: $(date '+%Y-%m-%d %H:%M:%S')"

# ============================================================
# Step 2: V6 Stage 2 - CLM+MLM 混合训练
# ============================================================
STAGE2_OUTPUT="${V6_OUTPUT}-stage2-clm-mlm"

echo ""
echo "============================================================"
echo "--- Step 2: V6 Stage 2 - CLM+MLM 混合训练 ---"
echo "从 Stage 1 最佳模型继续训练"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes=$NUM_GPU \
    --mixed_precision=bf16 \
    src/v6/train_v6.py \
    --stage clm_mlm \
    --data_dir /home/kehe/babyllm/babyLLM/data \
    --output_dir "$STAGE2_OUTPUT" \
    --tokenizer_dir "$TOKENIZER" \
    --resume_from "${STAGE1_OUTPUT}/best_model" \
    --d_model 640 \
    --n_layer 12 \
    --n_head 10 \
    --n_kv_heads 5 \
    --max_length 1024 \
    --batch_size 24 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --num_epochs 4 \
    --warmup_ratio 0.03 \
    --label_smoothing 0.1 \
    --attention_dropout 0.05 \
    --bpe_dropout 0.05 \
    --clm_ratio 0.125 \
    --mask_ratio_start 0.30 \
    --mask_ratio_end 0.15 \
    --patience 5 \
    --rope_theta 10000.0 \
    --gradient_checkpointing \
    --logging_steps 50 \
    --save_steps 5000 \
    --save_total_limit 2 \
    --wandb_project chinese-babylm \
    --wandb_run_name llama-v6-640d-12l-stage2-clm-mlm-4gpu

echo "Step 2 完成: $(date '+%Y-%m-%d %H:%M:%S')"

# ============================================================
# Step 3: 生成 V5-KD Teacher Logits（用于 V6 Stage 3 KD）
# ============================================================
echo ""
echo "============================================================"
echo "--- Step 3: 生成 Teacher Logits (V5 KD best model -> V6) ---"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# 决定用哪个模型作为教师：优先 V5 KD best_model，否则 V5 Phase 1 best_model
if [ -d "$V5_KD_BEST" ] && [ -f "$V5_KD_BEST/model.safetensors" ]; then
    TEACHER_MODEL="$V5_KD_BEST"
    echo "教师模型: V5 KD best_model"
else
    TEACHER_MODEL="/home/kehe/babyllm/babyLLM/output/babylm-llama-v5/best_model"
    echo "教师模型: V5 Phase 1 best_model (V5 KD best_model 不可用)"
fi

mkdir -p "$TEACHER_LOGITS_V5KD"

if [ ! -f "$TEACHER_LOGITS_V5KD/teacher_logits.npy" ]; then
    DATA_FILE="$V6_DATA/train.txt"
    if [ ! -f "$DATA_FILE" ]; then
        DATA_FILE="$V3_TRAIN"
    fi

    CUDA_VISIBLE_DEVICES=0 python src/v5/generate_teacher_logits.py \
        --teacher_model_path "$TEACHER_MODEL" \
        --tokenizer_dir "$TOKENIZER" \
        --data_file "$DATA_FILE" \
        --output_dir "$TEACHER_LOGITS_V5KD" \
        --block_size 1024 \
        --batch_size 16 \
        --top_k 10
    echo "Teacher logits 生成完成: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "Teacher logits 已存在，跳过生成"
fi

# ============================================================
# Step 4: V6 Stage 3 - Reverse KL 知识蒸馏
# ============================================================
STAGE3_OUTPUT="${V6_OUTPUT}-stage3-kd"

echo ""
echo "============================================================"
echo "--- Step 4: V6 Stage 3 - Reverse KL 知识蒸馏 ---"
echo "从 Stage 2 最佳模型继续训练"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes=$NUM_GPU \
    --mixed_precision=bf16 \
    src/v6/train_v6.py \
    --stage kd \
    --data_dir /home/kehe/babyllm/babyLLM/data \
    --output_dir "$STAGE3_OUTPUT" \
    --tokenizer_dir "$TOKENIZER" \
    --resume_from "${STAGE2_OUTPUT}/best_model" \
    --teacher_logits_dir "$TEACHER_LOGITS_V5KD" \
    --d_model 640 \
    --n_layer 12 \
    --n_head 10 \
    --n_kv_heads 5 \
    --max_length 1024 \
    --batch_size 24 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --num_epochs 4 \
    --warmup_ratio 0.03 \
    --label_smoothing 0.1 \
    --attention_dropout 0.0 \
    --bpe_dropout 0.0 \
    --lambda_ce 0.5 \
    --lambda_kd 0.5 \
    --temperature 3.0 \
    --top_k 10 \
    --use_reverse_kl \
    --patience 3 \
    --rope_theta 10000.0 \
    --gradient_checkpointing \
    --logging_steps 50 \
    --save_steps 5000 \
    --save_total_limit 2 \
    --wandb_project chinese-babylm \
    --wandb_run_name llama-v6-640d-12l-stage3-reverse-kl-kd-4gpu

echo "Step 4 完成: $(date '+%Y-%m-%d %H:%M:%S')"

# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================================"
echo "V6 自动化训练流水线全部完成!"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "模型输出:"
echo "  Stage 1 (CLM):          $STAGE1_OUTPUT/best_model"
echo "  Stage 2 (CLM+MLM):      $STAGE2_OUTPUT/best_model"
echo "  Stage 3 (Reverse KL KD): $STAGE3_OUTPUT/best_model"
echo "============================================================"
