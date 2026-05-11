#!/bin/bash
set -uo pipefail

# ============================================================
# ChineseBabyLM V13 — SOTA Competition Pipeline (94.2M params)
# 3-Stage Training + Official Eval
# Robust: error isolation per stage, auto-resume from checkpoints
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
HDD_OUT="/mnt/sda/kehe/babyllm_output/babylm-v13"
EVAL_DIR="/home/kehe/babyllm/chinese-babylm-eval-pipeline"
LOG_FILE="$HDD_OUT/pipeline_v13.log"

V12_BEST="/mnt/sda/kehe/babyllm_output/babylm-v12/stage2_mntp/best_model_ema"

GPUS="0,1,2,3"
NUM_GPUS=4
BATCH=16
ACCUM=2
MAX_LENGTH=1024
STRIDE=512

D_MODEL=768
N_LAYER=14
N_HEAD=12
N_KV=4

STAGE1_FAILED=false
STAGE2_FAILED=false
STAGE3_FAILED=false

for arg in "$@"; do
    case $arg in
        --skip-data) SKIP_DATA=true ;;
        --skip-stage1) SKIP_STAGE1=true ;;
        --skip-stage2) SKIP_STAGE2=true ;;
        --skip-stage3) SKIP_STAGE3=true ;;
        --skip-eval) SKIP_EVAL=true ;;
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
    local ema_args=""
    if [ -d "${model_path}_ema" ]; then
        ema_args="--use_ema --ema_path ${model_path}_ema/ema_best.pt"
    fi
    python "$PROJECT_DIR/src/v13/evaluate_v13.py" \
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
log "ChineseBabyLM V13 — SOTA Pipeline (94.2M params, 768d/14L)"
log "============================================================"
log "  Model: ${D_MODEL}d, ${N_LAYER}L, ${N_HEAD}Q/${N_KV}KV GQA"
log "  Batch: ${BATCH}x${NUM_GPUS}x${ACCUM} = $((BATCH*NUM_GPUS*ACCUM)) effective"
log "  Log:   $LOG_FILE"
log "============================================================"

# ============================================================
# Phase 1: Data Preparation
# ============================================================
DATA_DIR="$HDD_OUT/data_v13"

if [ "${SKIP_DATA:-false}" = false ]; then
    log ""
    log "=== Phase 1: Data Preparation ==="

    if [ -d "$V12_BEST" ]; then
        log "  Using V12 best model for PPL filtering: $V12_BEST"
        python "$PROJECT_DIR/src/v13/prepare_data.py" \
            --input_dir "$RAW_DATA" \
            --output_dir "$DATA_DIR" \
            --model_path "$V12_BEST" \
            --tokenizer_dir "$TOKENIZER_DIR" \
            --max_ppl 200 \
            --min_ppl 5 \
            --hard_upsample_factor 2 \
            2>&1 | tee -a "$LOG_FILE"
    else
        log "  V12 model not found, skipping PPL filter"
        python "$PROJECT_DIR/src/v13/prepare_data.py" \
            --input_dir "$RAW_DATA" \
            --output_dir "$DATA_DIR" \
            --skip_ppl_filter \
            --hard_upsample_factor 1 \
            2>&1 | tee -a "$LOG_FILE"
    fi
    log "Phase 1 done"
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
# Stage 1: CLM Pretraining (SGDR + Focal, 8 epochs)
# ============================================================
S1_OUT="$HDD_OUT/stage1_clm_sgdr"

if [ "${SKIP_STAGE1:-false}" = false ] && [ ! -d "$S1_OUT/best_model" ]; then
    log ""
    log "=== Stage 1: CLM+SGDR+Focal (8 epochs, lr=6e-4, gamma=1.5) ==="
    S1_START=$(date +%s)

    RESUME_ARG=""
    if [ -d "$S1_OUT/latest_checkpoint" ] && [ -f "$S1_OUT/latest_checkpoint/trainer_state.json" ]; then
        log "  Found checkpoint to resume from: $S1_OUT/latest_checkpoint"
        RESUME_ARG="--resume_from_checkpoint $S1_OUT/latest_checkpoint"
    fi

    if ! CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v13/train_v13.py \
        --stage clm \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S1_OUT" \
        --d_model $D_MODEL --n_layer $N_LAYER --n_head $N_HEAD --n_kv_heads $N_KV \
        --scheduler sgdr \
        --lr 6e-4 \
        --epochs 8 \
        --batch_size $BATCH --grad_accum_steps $ACCUM \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --bpe_dropout 0.1 \
        --label_smoothing 0.1 \
        --label_smoothing_anneal \
        --attention_dropout 0.1 \
        --focal_loss --focal_gamma 1.5 \
        --use_ema --ema_decay 0.999 \
        --patience 3 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v13-stage1-clm-sgdr" \
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
# Stage 2: MNTP (Dynamic CLM, 10 epochs)
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
        log "=== Stage 2: MNTP+Dynamic CLM (10 epochs, lr=5e-4, gamma=1.0) ==="
        S2_START=$(date +%s)

        RESUME_ARG=""
        if [ -d "$S2_OUT/latest_checkpoint" ] && [ -f "$S2_OUT/latest_checkpoint/trainer_state.json" ]; then
            log "  Found checkpoint to resume from: $S2_OUT/latest_checkpoint"
            RESUME_ARG="--resume_from_checkpoint $S2_OUT/latest_checkpoint"
        fi

        if ! CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
            --num_processes=$NUM_GPUS --mixed_precision=bf16 \
            src/v13/train_v13.py \
            --stage mntp \
            --data_dir "$DATA_DIR" \
            --tokenizer_dir "$TOKENIZER_DIR" \
            --output_dir "$S2_OUT" \
            --resume_from "$S1_BEST" \
            --d_model $D_MODEL --n_layer $N_LAYER --n_head $N_HEAD --n_kv_heads $N_KV \
            --scheduler cosine \
            --lr 5e-4 \
            --epochs 10 \
            --batch_size $BATCH --grad_accum_steps $ACCUM \
            --max_length $MAX_LENGTH \
            --stride $STRIDE \
            --clm_ratio 0.125 \
            --dynamic_clm_ratio \
            --mask_ratio_start 0.25 \
            --mask_ratio_end 0.10 \
            --bpe_dropout 0.1 \
            --label_smoothing 0.05 \
            --label_smoothing_anneal \
            --attention_dropout 0.05 \
            --focal_loss --focal_gamma 1.0 \
            --use_ema --ema_decay 0.999 \
            --patience 8 \
            --save_steps 2000 \
            --save_total_limit 3 \
            --logging_steps 50 \
            --wandb_run_name "babylm-v13-stage2-mntp-dynamic" \
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
# Stage 3: CLM Polish (5 epochs, lr=2e-5, DropBlock+StochDepth)
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
        log "=== Stage 3: Polish+DropBlock+StochDepth (5 epochs, lr=2e-5) ==="
        S3_START=$(date +%s)

        RESUME_ARG=""
        if [ -d "$S3_OUT/latest_checkpoint" ] && [ -f "$S3_OUT/latest_checkpoint/trainer_state.json" ]; then
            log "  Found checkpoint to resume from: $S3_OUT/latest_checkpoint"
            RESUME_ARG="--resume_from_checkpoint $S3_OUT/latest_checkpoint"
        fi

        if ! CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
            --num_processes=$NUM_GPUS --mixed_precision=bf16 \
            src/v13/train_v13.py \
            --stage clm \
            --data_dir "$DATA_DIR" \
            --tokenizer_dir "$TOKENIZER_DIR" \
            --output_dir "$S3_OUT" \
            --resume_from "$S2_BEST" \
            --d_model $D_MODEL --n_layer $N_LAYER --n_head $N_HEAD --n_kv_heads $N_KV \
            --scheduler cosine \
            --lr 2e-5 \
            --epochs 5 \
            --batch_size $BATCH --grad_accum_steps $ACCUM \
            --max_length $MAX_LENGTH \
            --stride $STRIDE \
            --bpe_dropout 0.0 \
            --label_smoothing 0.01 \
            --attention_dropout 0.0 \
            --dropblock 0.1 --dropblock_size 3 \
            --stochastic_depth 0.05 \
            --use_ema --ema_decay 0.999 \
            --patience 5 \
            --save_steps 2000 \
            --save_total_limit 3 \
            --logging_steps 50 \
            --wandb_run_name "babylm-v13-stage3-polish-dropblock" \
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
# Stage 4: Official Evaluation
# ============================================================
if [ "${SKIP_EVAL:-false}" = false ]; then
    FINAL_MODEL=""

    FINAL_MODEL="$S3_OUT/best_model_ema"
    if [ ! -d "$FINAL_MODEL" ]; then
        FINAL_MODEL="$S3_OUT/best_model"
    fi

    if [ ! -d "$FINAL_MODEL" ]; then
        FINAL_MODEL="$S2_OUT/best_model_ema"
        if [ ! -d "$FINAL_MODEL" ]; then
            FINAL_MODEL="$S2_OUT/best_model"
        fi
    fi

    if [ -d "$S2_OUT/best_model_ema" ] && [ -f "$HDD_OUT/eval_stage2_ema.json" ] && [ -f "$HDD_OUT/eval_stage3_ema.json" ]; then
        S2_PPL=$(python -c "import json; d=json.load(open('$HDD_OUT/eval_stage2_ema.json')); print(d['ppl'])" 2>/dev/null || echo "999")
        S3_PPL=$(python -c "import json; d=json.load(open('$HDD_OUT/eval_stage3_ema.json')); print(d['ppl'])" 2>/dev/null || echo "999")
        log "  Stage 2 EMA PPL: $S2_PPL, Stage 3 EMA PPL: $S3_PPL"
        if python -c "exit(0 if float('$S2_PPL') < float('$S3_PPL') else 1)" 2>/dev/null; then
            FINAL_MODEL="$S2_OUT/best_model_ema"
            log "  Stage 2 EMA is better, using it for official eval"
        fi
    fi

    if [ ! -d "$FINAL_MODEL" ]; then
        log "WARNING: No model found for official evaluation, skipping"
    else
        log ""
        log "=== Stage 4: Official Evaluation ==="
        S4_START=$(date +%s)

        log "  Converting tokenizer for HF compatibility ..."
        python "$PROJECT_DIR/src/v13/convert_tokenizer.py" \
            --spm_model "$TOKENIZER_DIR/spm.model" \
            --output_dir "$FINAL_MODEL" \
            2>&1 | tee -a "$LOG_FILE" || true

        V13_CONFIG="$EVAL_DIR/configs/config_v13.yaml"
        mkdir -p "$EVAL_DIR/configs"
        cat > "$V13_CONFIG" << YAML
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
            --config configs/config_v13.yaml \
            --results_dir "$HDD_OUT/official_eval" \
            2>&1 | tee -a "$LOG_FILE" || true
        cd "$PROJECT_DIR"

        S4_DUR=$(( ($(date +%s) - S4_START) / 60 ))
        log "Stage 4 done in ${S4_DUR} min"
    fi
fi

# ============================================================
# Final Summary
# ============================================================
TOTAL_DUR=$(( ($(date +%s) - PIPELINE_START) / 60 ))

log ""
log "============================================================"
log "V13 Pipeline Complete! Total: ${TOTAL_DUR} min ($(( TOTAL_DUR / 60 ))h $(( TOTAL_DUR % 60 ))m)"
log "============================================================"

if [ "$STAGE1_FAILED" = true ]; then log "  WARNING: Stage 1 FAILED"; fi
if [ "$STAGE2_FAILED" = true ]; then log "  WARNING: Stage 2 FAILED"; fi
if [ "$STAGE3_FAILED" = true ]; then log "  WARNING: Stage 3 FAILED"; fi

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
