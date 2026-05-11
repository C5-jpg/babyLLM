#!/bin/bash
set -uo pipefail

# ============================================================
# ChineseBabyLM V15 — SOTA Compact Model (~58M params)
# Based on V1-V14 Deep Analysis + BabyLM Research
#
# Architecture: 640d, 14L, 10Q/5KV GQA (~58M params)
# Pipeline: 2-stage (CLM→MNTP), V13 proved Polish ineffective
#
# Inherited best practices:
#   - SentencePiece 32K tokenizer
#   - CLM+MNTP hybrid training (GPT-BERT 2024 winner technique)
#   - EMA (multi-scale: 0.999 + 0.9999)
#   - SGDR scheduler (Stage 1)
#   - Focal Loss (γ=2.0 Stage 1, γ=1.5 Stage 2)
#   - Label smoothing annealing
#   - BPE dropout 0.1
#   - Dynamic CLM ratio
#   - PPL-filtered data with MinHash dedup
#
# New V15 features:
#   - Multi-scale EMA (decay=0.999 and 0.9999)
#   - Per-layer gradient norm monitoring
#   - Eval every 200 steps (finer cadence)
#   - Gradient norm spike detection
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
HDD_OUT="/mnt/sda/kehe/babyllm_output/babylm-v15"
EVAL_DIR="/home/kehe/babyllm/chinese-babylm-eval-pipeline"
LOG_FILE="$HDD_OUT/pipeline_v15.log"

V13_BEST="/mnt/sda/kehe/babyllm_output/babylm-v13/stage2_mntp/best_model_ema"
V12_BEST="/mnt/sda/kehe/babyllm_output/babylm-v12/stage2_mntp/best_model_ema"

GPUS="0,1,2,3"
NUM_GPUS=4
BATCH=16
ACCUM=2
MAX_LENGTH=1024
STRIDE=512

D_MODEL=640
N_LAYER=14
N_HEAD=10
N_KV=5

for arg in "$@"; do
    case $arg in
        --skip-data) SKIP_DATA=true ;;
        --skip-stage1) SKIP_STAGE1=true ;;
        --skip-stage2) SKIP_STAGE2=true ;;
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
    python "$PROJECT_DIR/src/v15/evaluate_v15.py" \
        --model_path "$model_path" \
        --val_file "$DATA_DIR/val.txt" \
        --output_json "$eval_out" \
        --compute_accuracy \
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
log "ChineseBabyLM V15 — SOTA Compact Model Pipeline (~58M params)"
log "============================================================"
log "  Model: ${D_MODEL}d, ${N_LAYER}L, ${N_HEAD}Q/${N_KV}KV GQA"
log "  Batch: ${BATCH}x${NUM_GPUS}x${ACCUM} = $((BATCH*NUM_GPUS*ACCUM)) effective"
log "  Pipeline: CLM(10ep) → MNTP(12ep)"
log "  Log:   $LOG_FILE"
log "============================================================"

# ============================================================
# Phase 1: Data Preparation
# ============================================================
DATA_DIR="$HDD_OUT/data_v15"

if [ "${SKIP_DATA:-false}" = false ]; then
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
            python "$PROJECT_DIR/src/v15/prepare_data.py" \
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
            python "$PROJECT_DIR/src/v15/prepare_data.py" \
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
    # Try reusing V14 data
    V14_DATA="/mnt/sda/kehe/babyllm_output/babylm-v14/data_v14"
    if [ -f "$V14_DATA/train.txt" ]; then
        log "  Reusing V14 data from $V14_DATA"
        mkdir -p "$DATA_DIR"
        ln -sf "$V14_DATA/train.txt" "$DATA_DIR/train.txt"
        ln -sf "$V14_DATA/val.txt" "$DATA_DIR/val.txt"
    else
        log "ERROR: No training data found!"
        exit 1
    fi
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
        src/v15/train_v15.py \
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
        --eval_steps 200 --early_stop_patience 5 --early_stop_min_delta 1e-4 \
        --save_steps 1000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --log_grad_norms \
        --wandb_run_name "babylm-v15-stage1-clm-sgdr" \
        $RESUME_ARG \
        2>&1 | tee -a "$LOG_FILE"; then
        log "WARNING: Stage 1 training FAILED (exit code $?)"
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
            src/v15/train_v15.py \
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
            --mask_ratio_end 0.10 \
            --bpe_dropout 0.1 \
            --label_smoothing 0.05 \
            --label_smoothing_anneal \
            --attention_dropout 0.05 \
            --focal_loss --focal_gamma 1.5 \
            --use_ema --ema_decay 0.999 \
            --patience 10 \
            --eval_steps 200 --early_stop_patience 5 --early_stop_min_delta 1e-4 \
            --save_steps 1000 \
            --save_total_limit 3 \
            --logging_steps 50 \
            --log_grad_norms \
            --wandb_run_name "babylm-v15-stage2-mntp-dynamic" \
            $RESUME_ARG \
            2>&1 | tee -a "$LOG_FILE"; then
            log "WARNING: Stage 2 training FAILED (exit code $?)"
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
# Stage 3: Official Evaluation
# ============================================================
if [ "${SKIP_EVAL:-false}" = false ]; then
    BEST_PPL=999
    BEST_MODEL=""
    for stage_num in 2 1; do
        EMA_JSON="$HDD_OUT/eval_stage${stage_num}_ema.json"
        BASE_JSON="$HDD_OUT/eval_stage${stage_num}.json"
        for json_file in "$EMA_JSON" "$BASE_JSON"; do
            if [ -f "$json_file" ]; then
                PPL=$(python -c "import json; d=json.load(open('$json_file')); print(d['ppl'])" 2>/dev/null || echo "999")
                if python -c "exit(0 if float('$PPL') < float('$BEST_PPL') else 1)" 2>/dev/null; then
                    BEST_PPL=$PPL
                    BEST_MODEL="$json_file"
                fi
            fi
        done
    done

    if [ -n "$BEST_MODEL" ]; then
        log ""
        log "=== Stage 3: Official Evaluation ==="
        log "  Best model PPL: $BEST_PPL (from $BEST_MODEL)"

        FINAL_MODEL=$(python -c "import json; print(json.load(open('$BEST_MODEL'))['model_path'])" 2>/dev/null)
        if [ -n "$FINAL_MODEL" ] && [ -d "$FINAL_MODEL" ]; then
            log "  Converting tokenizer for HF compatibility ..."
            python "$PROJECT_DIR/src/v15/convert_tokenizer.py" \
                --spm_model "$TOKENIZER_DIR/spm.model" \
                --output_dir "$FINAL_MODEL" \
                2>&1 | tee -a "$LOG_FILE" || true

            V15_CONFIG="$EVAL_DIR/configs/config_v15.yaml"
            mkdir -p "$EVAL_DIR/configs"
            cat > "$V15_CONFIG" << YAML
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
                --config configs/config_v15.yaml \
                --results_dir "$HDD_OUT/official_eval" \
                2>&1 | tee -a "$LOG_FILE" || true
            cd "$PROJECT_DIR"
        fi
    fi
fi

# ============================================================
# Final Summary
# ============================================================
TOTAL_DUR=$(( ($(date +%s) - PIPELINE_START) / 60 ))

log ""
log "============================================================"
log "V15 Pipeline Complete! Total: ${TOTAL_DUR} min ($(( TOTAL_DUR / 60 ))h $(( TOTAL_DUR % 60 ))m)"
log "============================================================"

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
