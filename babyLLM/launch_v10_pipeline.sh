#!/bin/bash
set -euo pipefail

# ============================================================
# ChineseBabyLM V10 — SOTA 三阶段全自动流水线
# ============================================================
#
# 全自动: Stage1 -> eval -> Stage2 -> eval -> Stage3 -> eval -> notify
# 通知: wandb.alert() 自动发送到 Slack + Email (在 wandb.ai/settings 配置)
#
# 用法:
#   bash launch_v10_pipeline.sh                    # 完整流水线
#   bash launch_v10_pipeline.sh --skip-stage1      # 跳过已完成的 Stage 1
#   bash launch_v10_pipeline.sh --skip-stage1 --skip-stage2
#
# 后台运行:
#   nohup bash launch_v10_pipeline.sh > /mnt/sda/kehe/babyllm_output/babylm-v10/pipeline.log 2>&1 &
# ============================================================

source /home/kehe/anaconda3/etc/profile.d/conda.sh
conda activate data

export NCCL_TIMEOUT=1800000
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^docker0
export OMP_NUM_THREADS=8

PROJECT_DIR="/home/kehe/babyllm/babyLLM"
DATA_DIR="$PROJECT_DIR/data/processed_v7"
TOKENIZER_DIR="$PROJECT_DIR/data/tokenizer_v7"
HDD_OUT="/mnt/sda/kehe/babyllm_output/babylm-v10"
LOG_FILE="$HDD_OUT/pipeline_v10.log"

GPUS="0,1,2,3"
NUM_GPUS=4
BATCH=32
MAX_LENGTH=1024
STRIDE=512

SKIP_STAGE1=false
SKIP_STAGE2=false
SKIP_STAGE3=false
for arg in "$@"; do
    case $arg in
        --skip-stage1) SKIP_STAGE1=true ;;
        --skip-stage2) SKIP_STAGE2=true ;;
        --skip-stage3) SKIP_STAGE3=true ;;
    esac
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

eval_stage() {
    local model_path="$1"
    local eval_out="$2"
    local stage_name="$3"
    log "Evaluating $stage_name ..."
    python "$PROJECT_DIR/src/v10/evaluate_v10.py" \
        --model_path "$model_path" \
        --val_file "$DATA_DIR/val.txt" \
        --output_json "$eval_out" 2>&1 | tee -a "$LOG_FILE" || true
    if [ -f "$eval_out" ]; then
        local result
        result=$(python -c "import json; d=json.load(open('$eval_out')); print(f\"loss={d['loss']:.4f} ppl={d['ppl']:.2f}\")")
        log "$stage_name eval: $result"
    fi
}

mkdir -p "$HDD_OUT"
cd "$PROJECT_DIR"

PIPELINE_START=$(date +%s)

log "============================================================"
log "ChineseBabyLM V10 — SOTA 三阶段全自动流水线"
log "============================================================"
log "  Data:      $DATA_DIR"
log "  Tokenizer: $TOKENIZER_DIR"
log "  Output:    $HDD_OUT"
log "  GPUs:      4× A6000 48GB (CUDA $GPUS)"
log "  Batch:     $BATCH/GPU (eff. $((BATCH * NUM_GPUS)))"
log "  Stride:    $STRIDE (50% overlap)"
log "  Notify:    wandb.alert → Slack + Email"
log "  Log:       $LOG_FILE"
log "============================================================"

# ============================================================
# Stage 1: CLM 预训练
# ============================================================
S1_OUT="$HDD_OUT/stage1_clm"

if [ "$SKIP_STAGE1" = false ] && [ ! -d "$S1_OUT/best_model" ]; then
    log ""
    log "=== Stage 1: CLM 预训练 (3 epochs, lr=6e-4, bpe_dropout=0.1) ==="
    S1_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v10/train_v10.py \
        --stage clm \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S1_OUT" \
        --lr 6e-4 \
        --epochs 3 \
        --batch_size $BATCH \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --bpe_dropout 0.1 \
        --label_smoothing 0.05 \
        --attention_dropout 0.1 \
        --patience 0 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v10-stage1-clm" \
        2>&1 | tee -a "$LOG_FILE"

    S1_DUR=$(( ($(date +%s) - S1_START) / 60 ))
    log "Stage 1 done in ${S1_DUR} min"
    eval_stage "$S1_OUT/best_model" "$HDD_OUT/eval_stage1.json" "Stage1-CLM"
elif [ -d "$S1_OUT/best_model" ]; then
    log "Stage 1 already exists, skipping"
fi

# ============================================================
# Stage 2: MNTP 训练
# ============================================================
S2_OUT="$HDD_OUT/stage2_mntp"
S1_BEST="$S1_OUT/best_model"

if [ "$SKIP_STAGE2" = false ] && [ ! -d "$S2_OUT/best_model" ]; then
    log ""
    log "=== Stage 2: MNTP 训练 (8 epochs, lr=5e-4, patience=8) ==="
    S2_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v10/train_v10.py \
        --stage mntp \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S2_OUT" \
        --resume_from "$S1_BEST" \
        --lr 5e-4 \
        --epochs 8 \
        --batch_size $BATCH \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --clm_ratio 0.125 \
        --mask_ratio_start 0.25 \
        --mask_ratio_end 0.12 \
        --bpe_dropout 0.1 \
        --label_smoothing 0.05 \
        --attention_dropout 0.1 \
        --patience 8 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v10-stage2-mntp" \
        2>&1 | tee -a "$LOG_FILE"

    S2_DUR=$(( ($(date +%s) - S2_START) / 60 ))
    log "Stage 2 done in ${S2_DUR} min"
    eval_stage "$S2_OUT/best_model" "$HDD_OUT/eval_stage2.json" "Stage2-MNTP"
elif [ -d "$S2_OUT/best_model" ]; then
    log "Stage 2 already exists, skipping"
fi

# ============================================================
# Stage 3: CLM Polish
# ============================================================
S3_OUT="$HDD_OUT/stage3_polish"
S2_BEST="$S2_OUT/best_model"

if [ "$SKIP_STAGE3" = false ] && [ ! -d "$S3_OUT/best_model" ]; then
    log ""
    log "=== Stage 3: CLM Polish (2 epochs, lr=1e-4) ==="
    S3_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v10/train_v10.py \
        --stage clm \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S3_OUT" \
        --resume_from "$S2_BEST" \
        --lr 1e-4 \
        --epochs 2 \
        --batch_size $BATCH \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --bpe_dropout 0.0 \
        --label_smoothing 0.02 \
        --attention_dropout 0.0 \
        --patience 0 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v10-stage3-polish" \
        2>&1 | tee -a "$LOG_FILE"

    S3_DUR=$(( ($(date +%s) - S3_START) / 60 ))
    log "Stage 3 done in ${S3_DUR} min"
    eval_stage "$S3_OUT/best_model" "$HDD_OUT/eval_stage3.json" "Stage3-Polish"
elif [ -d "$S3_OUT/best_model" ]; then
    log "Stage 3 already exists, skipping"
fi

# ============================================================
# Final Evaluation
# ============================================================
S3_BEST="$S3_OUT/best_model"
EVAL_OUT="$HDD_OUT/eval_final.json"

log ""
log "=== Final Evaluation ==="
eval_stage "$S3_BEST" "$EVAL_OUT" "Final"

# ============================================================
# Summary
# ============================================================
TOTAL_DUR=$(( ($(date +%s) - PIPELINE_START) / 60 ))

log ""
log "============================================================"
log "V10 Pipeline Complete! Total: ${TOTAL_DUR} min ($(( TOTAL_DUR / 60 ))h $(( TOTAL_DUR % 60 ))m)"
log "============================================================"

for stage_eval in "$HDD_OUT"/eval_stage*.json "$EVAL_OUT"; do
    if [ -f "$stage_eval" ]; then
        name=$(basename "$stage_eval" .json)
        python -c "
import json
d = json.load(open('$stage_eval'))
print(f'  $name: loss={d[\"loss\"]:.4f}, ppl={d[\"ppl\"]:.2f}')
" 2>/dev/null | tee -a "$LOG_FILE" || true
    fi
done

log "============================================================"
log "Logs: $LOG_FILE"
log "============================================================"
