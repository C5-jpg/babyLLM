#!/bin/bash
set -uo pipefail

# ============================================================
# ChineseBabyLM V14 — Optimized Efficiency Build (~52M params)
# Based on V1-V13 Deep Analysis + BabyLM 2024/2025 Research
#
# Key changes vs V13:
#   - Architecture: 640d, 12L, 10Q/5KV GQA (~52M params, was 94.2M)
#   - Better tokens/param ratio: 1.9x (was 1.1x in V13)
#   - 5-stage pipeline: CLM→MNTP→Polish→Self-Distill→Annealing
#   - Stage 3: NO DropBlock/StochDepth (V13 lesson: negative optimization)
#   - Effective batch: 16x4x2=128 (reduced to prevent OOM crashes)
#   - Enhanced Focal Loss: gamma=2.0→1.5 (was 1.5→1.0)
#   - MNTP mask_ratio_end=0.15 (more conservative, per GPT-BERT findings)
#   - Self-distillation as Stage 4 (T=4.0, λ=0.7)
#   - LR annealing + data replay as Stage 5
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
export WANDB_PROJECT="chinese-babylm"

PROJECT_DIR="/home/kehe/babyllm/babyLLM"
TOKENIZER_DIR="$PROJECT_DIR/data/tokenizer_v7"
RAW_DATA="$PROJECT_DIR/data/processed_v7"
HDD_OUT="/mnt/sda/kehe/babyllm_output/babylm-v14"
EVAL_DIR="/home/kehe/babyllm/chinese-babylm-eval-pipeline"
LOG_FILE="$HDD_OUT/pipeline_v14.log"

V13_BEST="/mnt/sda/kehe/babyllm_output/babylm-v13/stage2_mntp/best_model_ema"
V12_BEST="/mnt/sda/kehe/babyllm_output/babylm-v12/stage2_mntp/best_model_ema"

GPUS="0,1,2,3"
NUM_GPUS=4
BATCH=16
ACCUM=2
MAX_LENGTH=1024
STRIDE=512

D_MODEL=640
N_LAYER=12
N_HEAD=10
N_KV=5

STAGE1_FAILED=false
STAGE2_FAILED=false
STAGE3_FAILED=false
STAGE4_FAILED=false
STAGE5_FAILED=false

for arg in "$@"; do
    case $arg in
        --skip-data) SKIP_DATA=true ;;
        --skip-stage1) SKIP_STAGE1=true ;;
        --skip-stage2) SKIP_STAGE2=true ;;
        --skip-stage3) SKIP_STAGE3=true ;;
        --skip-stage4) SKIP_STAGE4=true ;;
        --skip-stage5) SKIP_STAGE5=true ;;
        --skip-eval) SKIP_EVAL=true ;;
    esac
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# OOM-aware stage runner: auto-reduces batch_size on OOM, keeps effective batch constant
run_stage_with_oom_retry() {
    local stage_name="$1"
    shift
    local current_batch=$BATCH
    local current_accum=$ACCUM
    local max_retries=3

    for attempt in $(seq 1 $max_retries); do
        log "  Attempt $attempt/$max_retries (batch=$current_batch, accum=$current_accum)"
        
        if CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
            --num_processes=$NUM_GPUS --mixed_precision=bf16 \
            "$@" --batch_size $current_batch --grad_accum_steps $current_accum \
            2>&1 | tee -a "$LOG_FILE"; then
            return 0
        fi
        
        exit_code=$?
        if grep -q "CUDA out of memory" "$LOG_FILE" 2>/dev/null; then
            log "  OOM detected at attempt $attempt! Halving batch_size..."
            current_batch=$((current_batch / 2))
            current_accum=$((current_accum * 2))
            if [ "$current_batch" -lt 4 ]; then
                log "  FATAL: batch_size < 4 after OOM retries, cannot continue"
                return 1
            fi
            sleep 10  # 等待 GPU 显存释放
        else
            log "  Stage $stage_name failed with exit code $exit_code (not OOM)"
            return $exit_code
        fi
    done
    log "  FAILED after $max_retries OOM retries"
    return 1
}

eval_stage() {
    local model_path="$1"
    local eval_out="$2"
    local stage_name="$3"
    log "Evaluating $stage_name ..."
    local ema_args=""
    if [ -d "${model_path}_ema" ]; then
        ema_args="--use_ema --ema_path ${model_path}_ema/ema_best.pt"
    fi
    python "$PROJECT_DIR/src/v14/evaluate_v14.py" \
        --model_path "$model_path" \
        --val_file "$DATA_DIR/val.txt" \
        --output_json "$eval_out" \
        $ema_args \
        2>&1 | tee -a "$LOG_FILE" || true
    if [ -f "$eval_out" ]; then
        local result
        result=$(python -c "import json; d=json.load(open('$eval_out')); print(f\"loss={d['loss']:.4f} ppl={d['ppl']:.2f}\")" 2>/dev/null)
        log "$stage_name eval: $result"
    fi
}

mkdir -p "$HDD_OUT"
cd "$PROJECT_DIR"

PIPELINE_START=$(date +%s)

log "============================================================"
log "ChineseBabyLM V14 — Optimized Efficiency Pipeline (~52M params)"
log "============================================================"
log "  Model: ${D_MODEL}d, ${N_LAYER}L, ${N_HEAD}Q/${N_KV}KV GQA"
log "  Batch: ${BATCH}x${NUM_GPUS}x${ACCUM} = $((BATCH*NUM_GPUS*ACCUM)) effective"
log "  Pipeline: CLM(10ep) → MNTP(12ep) → Polish(5ep) → SD(4ep) → Anneal(3ep)"
log "  Log:   $LOG_FILE"
log "============================================================"

# ============================================================
# Phase 1: Data Preparation (reuse V13 data if available)
# ============================================================
DATA_DIR="$HDD_OUT/data_v14"

if [ "${SKIP_DATA:-false}" = false ]; then
    # Skip if data already exists and is non-empty
    if [ -f "$DATA_DIR/train.txt" ] && [ -f "$DATA_DIR/val.txt" ] && [ "$(stat -c%s "$DATA_DIR/train.txt")" -gt 1000000 ]; then
        TRAIN_LINES=$(wc -l < "$DATA_DIR/train.txt")
        log "  Data already exists ($TRAIN_LINES lines), skipping preparation"
    else
        log ""
        log "=== Phase 1: Data Preparation ==="

        PPL_MODEL=""
        if [ -d "$V13_BEST" ]; then
        PPL_MODEL="$V13_BEST"
        log "  Using V13 best EMA model for PPL filtering"
    elif [ -d "$V12_BEST" ]; then
        PPL_MODEL="$V12_BEST"
        log "  Using V12 best EMA model for PPL filtering"
    fi

    if [ -n "$PPL_MODEL" ]; then
        python "$PROJECT_DIR/src/v14/prepare_data.py" \
            --input_dir "$RAW_DATA" \
            --output_dir "$DATA_DIR" \
            --model_path "$PPL_MODEL" \
            --tokenizer_dir "$TOKENIZER_DIR" \
            --max_ppl 250 \
            --min_ppl 3 \
            --hard_upsample_factor 2 \
            2>&1 | tee -a "$LOG_FILE"
    else
        log "  No PPL filter model found, preparing without PPL filter"
        python "$PROJECT_DIR/src/v14/prepare_data.py" \
            --input_dir "$RAW_DATA" \
            --output_dir "$DATA_DIR" \
            --skip_ppl_filter \
            --hard_upsample_factor 1 \
            2>&1 | tee -a "$LOG_FILE"
    fi
    log "Phase 1 done"
    fi
else
    log "Phase 1 skipped (--skip-data)"
fi

if [ ! -f "$DATA_DIR/train.txt" ]; then
    log "ERROR: $DATA_DIR/train.txt not found!"
    log "Phase 1 failed, cannot continue training stages"
    exit 1
fi

TRAIN_LINES=$(wc -l < "$DATA_DIR/train.txt")
log "  Training data: ${TRAIN_LINES} lines"

# ============================================================
# Stage 1: CLM Pretraining (SGDR + Focal, 10 epochs)
# ============================================================
S1_OUT="$HDD_OUT/stage1_clm_sgdr"

if [ "${SKIP_STAGE1:-false}" = false ] && [ ! -d "$S1_OUT/best_model" ]; then
    log ""
    log "=== Stage 1: CLM+SGDR+Focal (10 epochs, lr=5e-4, gamma=2.0) ==="
    S1_START=$(date +%s)

    RESUME_ARG=""
    if [ -d "$S1_OUT/latest_checkpoint" ] && [ -f "$S1_OUT/latest_checkpoint/trainer_state.json" ]; then
        log "  Found checkpoint to resume from: $S1_OUT/latest_checkpoint"
        RESUME_ARG="--resume_from_checkpoint $S1_OUT/latest_checkpoint"
    fi

    if ! CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v14/train_v14.py \
        --stage clm \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S1_OUT" \
        --d_model $D_MODEL --n_layer $N_LAYER --n_head $N_HEAD --n_kv_heads $N_KV \
        --scheduler sgdr \
        --lr 5e-4 \
        --epochs 10 \
        --batch_size $BATCH --grad_accum_steps $ACCUM \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --bpe_dropout 0.1 \
        --label_smoothing 0.1 \
        --label_smoothing_anneal \
        --attention_dropout 0.1 \
        --focal_loss --focal_gamma 2.0 \
        --use_ema --ema_decay 0.999 \
        --patience 5 \
        --eval_steps 500 --early_stop_patience 3 --early_stop_min_delta 1e-4 \
        --save_steps 500 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v14-stage1-clm-sgdr" \
        $RESUME_ARG \
        2>&1 | tee -a "$LOG_FILE"; then
        log "WARNING: Stage 1 training FAILED (exit code $?)"
        STAGE1_FAILED=true
    fi

    S1_DUR=$(( ($(date +%s) - S1_START) / 60 ))
    log "Stage 1 finished in ${S1_DUR} min"
    eval_stage "$S1_OUT/best_model" "$HDD_OUT/eval_stage1.json" "Stage1-CLM-SGDR"
    if [ -d "$S1_OUT/best_model_ema" ]; then
        eval_stage "$S1_OUT/best_model_ema" "$HDD_OUT/eval_stage1_ema.json" "Stage1-CLM-SGDR-EMA"
    fi
elif [ -d "$S1_OUT/best_model" ]; then
    log "Stage 1 already completed (best_model exists), skipping"
fi

# ============================================================
# Stage 2: MNTP (Dynamic CLM, 12 epochs)
# ============================================================
S2_OUT="$HDD_OUT/stage2_mntp"
S1_BEST="$S1_OUT/best_model"
if [ -d "$S1_OUT/best_model_ema" ]; then
    S1_BEST="$S1_OUT/best_model_ema"
    log "  Using EMA model from Stage 1 for Stage 2"
fi

if [ "${SKIP_STAGE2:-false}" = false ] && [ ! -d "$S2_OUT/best_model" ]; then
    if [ ! -d "$S1_BEST" ]; then
        log "WARNING: Stage 1 best model not found at $S1_BEST, cannot start Stage 2"
        STAGE2_FAILED=true
    else
        log ""
        log "=== Stage 2: MNTP+Dynamic CLM (12 epochs, lr=4e-4, gamma=1.5) ==="
        S2_START=$(date +%s)

        RESUME_ARG=""
        if [ -d "$S2_OUT/latest_checkpoint" ] && [ -f "$S2_OUT/latest_checkpoint/trainer_state.json" ]; then
            log "  Found checkpoint to resume from: $S2_OUT/latest_checkpoint"
            RESUME_ARG="--resume_from_checkpoint $S2_OUT/latest_checkpoint"
        fi

        if ! CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
            --num_processes=$NUM_GPUS --mixed_precision=bf16 \
            src/v14/train_v14.py \
            --stage mntp \
            --data_dir "$DATA_DIR" \
            --tokenizer_dir "$TOKENIZER_DIR" \
            --output_dir "$S2_OUT" \
            --resume_from "$S1_BEST" \
            --d_model $D_MODEL --n_layer $N_LAYER --n_head $N_HEAD --n_kv_heads $N_KV \
            --scheduler cosine \
            --lr 4e-4 \
            --epochs 12 \
            --batch_size $BATCH --grad_accum_steps $ACCUM \
            --max_length $MAX_LENGTH \
            --stride $STRIDE \
            --clm_ratio 0.125 \
            --dynamic_clm_ratio \
            --mask_ratio_start 0.25 \
            --mask_ratio_end 0.15 \
            --bpe_dropout 0.1 \
            --label_smoothing 0.05 \
            --label_smoothing_anneal \
            --attention_dropout 0.05 \
            --focal_loss --focal_gamma 1.5 \
            --use_ema --ema_decay 0.999 \
            --patience 10 \
            --eval_steps 500 --early_stop_patience 3 --early_stop_min_delta 1e-4 \
            --save_steps 500 \
            --save_total_limit 3 \
            --logging_steps 50 \
            --wandb_run_name "babylm-v14-stage2-mntp-dynamic" \
            $RESUME_ARG \
            2>&1 | tee -a "$LOG_FILE"; then
            log "WARNING: Stage 2 training FAILED (exit code $?)"
            STAGE2_FAILED=true
        fi

        S2_DUR=$(( ($(date +%s) - S2_START) / 60 ))
        log "Stage 2 finished in ${S2_DUR} min"
        eval_stage "$S2_OUT/best_model" "$HDD_OUT/eval_stage2.json" "Stage2-MNTP"
        if [ -d "$S2_OUT/best_model_ema" ]; then
            eval_stage "$S2_OUT/best_model_ema" "$HDD_OUT/eval_stage2_ema.json" "Stage2-MNTP-EMA"
        fi
    fi
elif [ -d "$S2_OUT/best_model" ]; then
    log "Stage 2 already completed (best_model exists), skipping"
fi

# ============================================================
# Stage 3: Polish (5 epochs, lr=1e-5, pure CLM, no extra regularization)
# ============================================================
S3_OUT="$HDD_OUT/stage3_polish"
S2_BEST="$S2_OUT/best_model"
if [ -d "$S2_OUT/best_model_ema" ]; then
    S2_BEST="$S2_OUT/best_model_ema"
    log "  Using EMA model from Stage 2 for Stage 3"
fi

if [ "${SKIP_STAGE3:-false}" = false ] && [ ! -d "$S3_OUT/best_model" ]; then
    if [ ! -d "$S2_BEST" ]; then
        log "WARNING: Stage 2 best model not found at $S2_BEST, cannot start Stage 3"
        STAGE3_FAILED=true
    else
        log ""
        log "=== Stage 3: Polish (5 epochs, lr=1e-5, no DropBlock/StochDepth) ==="
        S3_START=$(date +%s)

        RESUME_ARG=""
        if [ -d "$S3_OUT/latest_checkpoint" ] && [ -f "$S3_OUT/latest_checkpoint/trainer_state.json" ]; then
            log "  Found checkpoint to resume from: $S3_OUT/latest_checkpoint"
            RESUME_ARG="--resume_from_checkpoint $S3_OUT/latest_checkpoint"
        fi

        if ! CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
            --num_processes=$NUM_GPUS --mixed_precision=bf16 \
            src/v14/train_v14.py \
            --stage clm \
            --data_dir "$DATA_DIR" \
            --tokenizer_dir "$TOKENIZER_DIR" \
            --output_dir "$S3_OUT" \
            --resume_from "$S2_BEST" \
            --d_model $D_MODEL --n_layer $N_LAYER --n_head $N_HEAD --n_kv_heads $N_KV \
            --scheduler cosine \
            --lr 1e-5 \
            --epochs 5 \
            --batch_size $BATCH --grad_accum_steps $ACCUM \
            --max_length $MAX_LENGTH \
            --stride $STRIDE \
            --bpe_dropout 0.0 \
            --label_smoothing 0.01 \
            --attention_dropout 0.0 \
            --use_ema --ema_decay 0.999 \
            --patience 3 \
            --eval_steps 500 --early_stop_patience 3 --early_stop_min_delta 1e-4 \
            --save_steps 500 \
            --save_total_limit 3 \
            --logging_steps 50 \
            --wandb_run_name "babylm-v14-stage3-polish-clean" \
            $RESUME_ARG \
            2>&1 | tee -a "$LOG_FILE"; then
            log "WARNING: Stage 3 training FAILED (exit code $?)"
            STAGE3_FAILED=true
        fi

        S3_DUR=$(( ($(date +%s) - S3_START) / 60 ))
        log "Stage 3 finished in ${S3_DUR} min"
        eval_stage "$S3_OUT/best_model" "$HDD_OUT/eval_stage3.json" "Stage3-Polish"
        if [ -d "$S3_OUT/best_model_ema" ]; then
            eval_stage "$S3_OUT/best_model_ema" "$HDD_OUT/eval_stage3_ema.json" "Stage3-Polish-EMA"
        fi
    fi
elif [ -d "$S3_OUT/best_model" ]; then
    log "Stage 3 already completed (best_model exists), skipping"
fi

# ============================================================
# Stage 4: Self-Distillation (4 epochs, lr=3e-5)
# ============================================================
S4_OUT="$HDD_OUT/stage4_self_distill"
S3_BEST="$S3_OUT/best_model"
if [ -d "$S3_OUT/best_model_ema" ]; then
    S3_BEST="$S3_OUT/best_model_ema"
    log "  Using EMA model from Stage 3 for Stage 4"
fi

if [ "${SKIP_STAGE4:-false}" = false ] && [ ! -d "$S4_OUT/best_model" ]; then
    if [ ! -d "$S3_BEST" ]; then
        log "WARNING: Stage 3 best model not found at $S3_BEST, cannot start Stage 4"
        STAGE4_FAILED=true
    else
        log ""
        log "=== Stage 4: Self-Distillation (4 epochs, lr=3e-5, T=4.0, lambda=0.7) ==="
        S4_START=$(date +%s)

        RESUME_ARG=""
        if [ -d "$S4_OUT/latest_checkpoint" ] && [ -f "$S4_OUT/latest_checkpoint/trainer_state.json" ]; then
            log "  Found checkpoint to resume from: $S4_OUT/latest_checkpoint"
            RESUME_ARG="--resume_from_checkpoint $S4_OUT/latest_checkpoint"
        fi

        if ! CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
            --num_processes=$NUM_GPUS --mixed_precision=bf16 \
            src/v14/train_v14.py \
            --stage clm \
            --data_dir "$DATA_DIR" \
            --tokenizer_dir "$TOKENIZER_DIR" \
            --output_dir "$S4_OUT" \
            --resume_from "$S3_BEST" \
            --d_model $D_MODEL --n_layer $N_LAYER --n_head $N_HEAD --n_kv_heads $N_KV \
            --scheduler cosine \
            --lr 3e-5 \
            --epochs 4 \
            --batch_size $BATCH --grad_accum_steps $ACCUM \
            --max_length $MAX_LENGTH \
            --stride $STRIDE \
            --bpe_dropout 0.0 \
            --label_smoothing 0.0 \
            --attention_dropout 0.0 \
            --self_distill --sd_temperature 4.0 --sd_lambda 0.7 --sd_update_freq 100 \
            --use_ema --ema_decay 0.999 \
            --patience 3 \
            --eval_steps 500 --early_stop_patience 3 --early_stop_min_delta 1e-4 \
            --save_steps 500 \
            --save_total_limit 3 \
            --logging_steps 50 \
            --wandb_run_name "babylm-v14-stage4-self-distill" \
            $RESUME_ARG \
            2>&1 | tee -a "$LOG_FILE"; then
            log "WARNING: Stage 4 training FAILED (exit code $?)"
            STAGE4_FAILED=true
        fi

        S4_DUR=$(( ($(date +%s) - S4_START) / 60 ))
        log "Stage 4 finished in ${S4_DUR} min"
        eval_stage "$S4_OUT/best_model" "$HDD_OUT/eval_stage4.json" "Stage4-SelfDistill"
        if [ -d "$S4_OUT/best_model_ema" ]; then
            eval_stage "$S4_OUT/best_model_ema" "$HDD_OUT/eval_stage4_ema.json" "Stage4-SelfDistill-EMA"
        fi
    fi
elif [ -d "$S4_OUT/best_model" ]; then
    log "Stage 4 already completed (best_model exists), skipping"
fi

# ============================================================
# Stage 5: LR Annealing (3 epochs, lr=5e-6)
# ============================================================
S5_OUT="$HDD_OUT/stage5_annealing"
S4_BEST="$S4_OUT/best_model"
if [ -d "$S4_OUT/best_model_ema" ]; then
    S4_BEST="$S4_OUT/best_model_ema"
    log "  Using EMA model from Stage 4 for Stage 5"
fi

if [ "${SKIP_STAGE5:-false}" = false ] && [ ! -d "$S5_OUT/best_model" ]; then
    if [ ! -d "$S4_BEST" ]; then
        log "WARNING: Stage 4 best model not found at $S4_BEST, cannot start Stage 5"
        STAGE5_FAILED=true
    else
        log ""
        log "=== Stage 5: LR Annealing (3 epochs, lr=5e-6) ==="
        S5_START=$(date +%s)

        RESUME_ARG=""
        if [ -d "$S5_OUT/latest_checkpoint" ] && [ -f "$S5_OUT/latest_checkpoint/trainer_state.json" ]; then
            log "  Found checkpoint to resume from: $S5_OUT/latest_checkpoint"
            RESUME_ARG="--resume_from_checkpoint $S5_OUT/latest_checkpoint"
        fi

        if ! CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
            --num_processes=$NUM_GPUS --mixed_precision=bf16 \
            src/v14/train_v14.py \
            --stage clm \
            --data_dir "$DATA_DIR" \
            --tokenizer_dir "$TOKENIZER_DIR" \
            --output_dir "$S5_OUT" \
            --resume_from "$S4_BEST" \
            --d_model $D_MODEL --n_layer $N_LAYER --n_head $N_HEAD --n_kv_heads $N_KV \
            --scheduler cosine \
            --lr 5e-6 \
            --epochs 3 \
            --batch_size $BATCH --grad_accum_steps $ACCUM \
            --max_length $MAX_LENGTH \
            --stride $STRIDE \
            --bpe_dropout 0.0 \
            --label_smoothing 0.0 \
            --attention_dropout 0.0 \
            --use_ema --ema_decay 0.999 \
            --patience 3 \
            --eval_steps 500 --early_stop_patience 3 --early_stop_min_delta 1e-4 \
            --save_steps 500 \
            --save_total_limit 3 \
            --logging_steps 50 \
            --wandb_run_name "babylm-v14-stage5-annealing" \
            $RESUME_ARG \
            2>&1 | tee -a "$LOG_FILE"; then
            log "WARNING: Stage 5 training FAILED (exit code $?)"
            STAGE5_FAILED=true
        fi

        S5_DUR=$(( ($(date +%s) - S5_START) / 60 ))
        log "Stage 5 finished in ${S5_DUR} min"
        eval_stage "$S5_OUT/best_model" "$HDD_OUT/eval_stage5.json" "Stage5-Annealing"
        if [ -d "$S5_OUT/best_model_ema" ]; then
            eval_stage "$S5_OUT/best_model_ema" "$HDD_OUT/eval_stage5_ema.json" "Stage5-Annealing-EMA"
        fi
    fi
elif [ -d "$S5_OUT/best_model" ]; then
    log "Stage 5 already completed (best_model exists), skipping"
fi

# ============================================================
# Stage 6: Official Evaluation
# ============================================================
if [ "${SKIP_EVAL:-false}" = false ]; then
    FINAL_MODEL=""

    FINAL_MODEL="$S5_OUT/best_model_ema"
    if [ ! -d "$FINAL_MODEL" ]; then
        FINAL_MODEL="$S5_OUT/best_model"
    fi

    for stage_dir in "$S4_OUT" "$S3_OUT" "$S2_OUT"; do
        if [ ! -d "$FINAL_MODEL" ]; then
            FINAL_MODEL="${stage_dir}/best_model_ema"
            if [ ! -d "$FINAL_MODEL" ]; then
                FINAL_MODEL="${stage_dir}/best_model"
            fi
        fi
    done

    BEST_PPL=999
    BEST_MODEL=""
    for stage_num in 5 4 3 2 1; do
        EMA_JSON="$HDD_OUT/eval_stage${stage_num}_ema.json"
        BASE_JSON="$HDD_OUT/eval_stage${stage_num}.json"
        for json_file in "$EMA_JSON" "$BASE_JSON"; do
            if [ -f "$json_file" ]; then
                PPL=$(python -c "import json; d=json.load(open('$json_file')); print(d['ppl'])" 2>/dev/null || echo "999")
                if python -c "exit(0 if float('$PPL') < float('$BEST_PPL') else 1)" 2>/dev/null; then
                    BEST_PPL=$PPL
                    if [ "$json_file" = "$EMA_JSON" ]; then
                        BEST_MODEL="$HDD_OUT/stage${stage_num}_*/best_model_ema"
                    else
                        BEST_MODEL="$HDD_OUT/stage${stage_num}_*/best_model"
                    fi
                fi
            fi
        done
    done

    if [ -n "$BEST_MODEL" ]; then
        FINAL_MODEL=$(ls -d $BEST_MODEL 2>/dev/null | head -1)
    fi

    if [ ! -d "$FINAL_MODEL" ]; then
        log "WARNING: No model found for official evaluation, skipping"
    else
        log ""
        log "=== Stage 6: Official Evaluation ==="
        log "  Best model: $FINAL_MODEL (PPL: $BEST_PPL)"
        S6_START=$(date +%s)

        log "  Converting tokenizer for HF compatibility ..."
        python "$PROJECT_DIR/src/v14/convert_tokenizer.py" \
            --spm_model "$TOKENIZER_DIR/spm.model" \
            --output_dir "$FINAL_MODEL" \
            2>&1 | tee -a "$LOG_FILE" || true

        V14_CONFIG="$EVAL_DIR/configs/config_v14.yaml"
        mkdir -p "$EVAL_DIR/configs"
        cat > "$V14_CONFIG" << YAML
models:
  - path: $FINAL_MODEL
    backend: causal

tasks:
  zero_shot:
    - zhoblimp
    - hanzi_structure
    - hanzi_pinyin
  finetune:
    - afqmc
    - ocnli
    - tnews
    - cluewsc2020

eval_dir: $EVAL_DIR/evaluation_data
results_dir: $HDD_OUT/official_eval

finetune_hparams:
  lr: 3.0e-5
  batch_size: 32
  max_epochs: 10
  wsc_epochs: 30
  seed: 42
YAML

        log "  Running official eval on $FINAL_MODEL ..."
        cd "$EVAL_DIR"
        python pipeline.py eval \
            --config configs/config_v14.yaml \
            --results_dir "$HDD_OUT/official_eval" \
            2>&1 | tee -a "$LOG_FILE" || true
        cd "$PROJECT_DIR"

        S6_DUR=$(( ($(date +%s) - S6_START) / 60 ))
        log "Stage 6 done in ${S6_DUR} min"
    fi
fi

# ============================================================
# Final Summary
# ============================================================
TOTAL_DUR=$(( ($(date +%s) - PIPELINE_START) / 60 ))

log ""
log "============================================================"
log "V14 Pipeline Complete! Total: ${TOTAL_DUR} min ($(( TOTAL_DUR / 60 ))h $(( TOTAL_DUR % 60 ))m)"
log "============================================================"

if [ "$STAGE1_FAILED" = true ]; then log "  WARNING: Stage 1 FAILED"; fi
if [ "$STAGE2_FAILED" = true ]; then log "  WARNING: Stage 2 FAILED"; fi
if [ "$STAGE3_FAILED" = true ]; then log "  WARNING: Stage 3 FAILED"; fi
if [ "$STAGE4_FAILED" = true ]; then log "  WARNING: Stage 4 FAILED"; fi
if [ "$STAGE5_FAILED" = true ]; then log "  WARNING: Stage 5 FAILED"; fi

for stage_eval in "$HDD_OUT"/eval_stage*.json; do
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
log "Model: $HDD_OUT"
log "Logs:  $LOG_FILE"
log "============================================================"
