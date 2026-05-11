#!/bin/bash
set -euo pipefail

# ============================================================
# ChineseBabyLM V12 — 5-Stage Automated Pipeline
# ============================================================
# 54.2M params, 576d, 14L, 9Q/3KV GQA
# Stage 1: CLM+SGDR+Focal (8 epochs)
# Stage 2: MNTP+Dynamic CLM (10 epochs)
# Stage 3: CLM Polish (8 epochs, patience=8)
# Stage 4: Self-Distill (6 epochs, T=4.0, step-level teacher)
# Stage 5: Annealing (3 epochs)
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
HDD_OUT="/mnt/sda/kehe/babyllm_output/babylm-v12"
EVAL_DIR="/home/kehe/babyllm/chinese-babylm-eval-pipeline"
LOG_FILE="$HDD_OUT/pipeline_v12.log"

GPUS="0,1,2,3"
NUM_GPUS=4
BATCH=32
MAX_LENGTH=1024
STRIDE=512

D_MODEL=576
N_LAYER=14
N_HEAD=9
N_KV=3

SKIP_STAGE0=false
SKIP_STAGE1=false
SKIP_STAGE2=false
SKIP_STAGE3=false
SKIP_STAGE4=false
SKIP_STAGE5=false
SKIP_OFFICIAL=false
for arg in "$@"; do
    case $arg in
        --skip-stage0) SKIP_STAGE0=true ;;
        --skip-stage1) SKIP_STAGE1=true ;;
        --skip-stage2) SKIP_STAGE2=true ;;
        --skip-stage3) SKIP_STAGE3=true ;;
        --skip-stage4) SKIP_STAGE4=true ;;
        --skip-stage5) SKIP_STAGE5=true ;;
        --skip-official) SKIP_OFFICIAL=true ;;
    esac
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

eval_stage() {
    local model_path="$1"
    local eval_out="$2"
    local stage_name="$3"
    local use_ema="${4:-false}"
    local ema_path="${5:-}"
    log "Evaluating $stage_name ..."
    local ema_args=""
    if [ "$use_ema" = "true" ] && [ -f "$ema_path" ]; then
        ema_args="--use_ema --ema_path $ema_path"
    fi
    python "$PROJECT_DIR/src/v12/evaluate_v12.py" \
        --model_path "$model_path" \
        --val_file "$DATA_DIR/val.txt" \
        --output_json "$eval_out" \
        $ema_args \
        2>&1 | tee -a "$LOG_FILE" || true
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
log "ChineseBabyLM V12 — 5-Stage Pipeline (54.2M params)"
log "============================================================"
log "  Model: ${D_MODEL}d, ${N_LAYER}L, ${N_HEAD}Q/${N_KV}KV GQA"
log "  GPUs:  4× A6000 48GB"
log "  Log:   $LOG_FILE"
log "============================================================"

# ============================================================
# Stage 0: Data Cleaning
# ============================================================
DATA_DIR="$HDD_OUT/data_cleaned"

if [ "$SKIP_STAGE0" = false ] && [ ! -d "$DATA_DIR" ]; then
    log ""
    log "=== Stage 0: Data Cleaning (dedup + quality filter) ==="
    python "$PROJECT_DIR/src/v12/clean_data.py" \
        --input_dir "$RAW_DATA" \
        --output_dir "$DATA_DIR" \
        --min_chars 5 \
        --max_repeat_ratio 0.5 \
        2>&1 | tee -a "$LOG_FILE"
    log "Stage 0 done"
elif [ -d "$DATA_DIR" ]; then
    log "Stage 0 already done, skipping"
    DATA_DIR="$DATA_DIR"
fi

# ============================================================
# Stage 1: CLM Pretraining (SGDR + Focal Loss, 8 epochs)
# ============================================================
S1_OUT="$HDD_OUT/stage1_clm_sgdr"

if [ "$SKIP_STAGE1" = false ] && [ ! -d "$S1_OUT/best_model" ]; then
    log ""
    log "=== Stage 1: CLM+SGDR+Focal (8 epochs, lr=6e-4) ==="
    S1_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v12/train_v12.py \
        --stage clm \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S1_OUT" \
        --d_model $D_MODEL --n_layer $N_LAYER --n_head $N_HEAD --n_kv_heads $N_KV \
        --scheduler sgdr \
        --lr 6e-4 \
        --epochs 8 \
        --batch_size $BATCH \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --bpe_dropout 0.1 \
        --label_smoothing 0.1 \
        --label_smoothing_anneal \
        --attention_dropout 0.1 \
        --focal_loss --focal_gamma 2.0 \
        --use_ema --ema_decay 0.999 \
        --patience 3 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v12-stage1-clm-sgdr-focal" \
        2>&1 | tee -a "$LOG_FILE"

    S1_DUR=$(( ($(date +%s) - S1_START) / 60 ))
    log "Stage 1 done in ${S1_DUR} min"
    eval_stage "$S1_OUT/best_model" "$HDD_OUT/eval_stage1.json" "Stage1-CLM-SGDR"
    if [ -d "$S1_OUT/best_model_ema" ]; then
        eval_stage "$S1_OUT/best_model_ema" "$HDD_OUT/eval_stage1_ema.json" "Stage1-CLM-SGDR-EMA"
    fi
elif [ -d "$S1_OUT/best_model" ]; then
    log "Stage 1 already exists, skipping"
fi

# ============================================================
# Stage 2: MNTP (Dynamic CLM ratio, 10 epochs)
# ============================================================
S2_OUT="$HDD_OUT/stage2_mntp"
S1_BEST="$S1_OUT/best_model"
if [ -d "$S1_OUT/best_model_ema" ]; then
    S1_BEST="$S1_OUT/best_model_ema"
    log "  Using EMA model from Stage 1 for Stage 2"
fi

if [ "$SKIP_STAGE2" = false ] && [ ! -d "$S2_OUT/best_model" ]; then
    log ""
    log "=== Stage 2: MNTP+Dynamic CLM (10 epochs, lr=5e-4) ==="
    S2_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v12/train_v12.py \
        --stage mntp \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S2_OUT" \
        --resume_from "$S1_BEST" \
        --d_model $D_MODEL --n_layer $N_LAYER --n_head $N_HEAD --n_kv_heads $N_KV \
        --scheduler cosine \
        --lr 5e-4 \
        --epochs 10 \
        --batch_size $BATCH \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --clm_ratio 0.125 \
        --dynamic_clm_ratio \
        --mask_ratio_start 0.25 \
        --mask_ratio_end 0.10 \
        --bpe_dropout 0.1 \
        --label_smoothing 0.05 \
        --label_smoothing_anneal \
        --attention_dropout 0.1 \
        --focal_loss --focal_gamma 2.0 \
        --use_ema --ema_decay 0.999 \
        --patience 8 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v12-stage2-mntp-dynamic" \
        2>&1 | tee -a "$LOG_FILE"

    S2_DUR=$(( ($(date +%s) - S2_START) / 60 ))
    log "Stage 2 done in ${S2_DUR} min"
    eval_stage "$S2_OUT/best_model" "$HDD_OUT/eval_stage2.json" "Stage2-MNTP"
    if [ -d "$S2_OUT/best_model_ema" ]; then
        eval_stage "$S2_OUT/best_model_ema" "$HDD_OUT/eval_stage2_ema.json" "Stage2-MNTP-EMA"
    fi
elif [ -d "$S2_OUT/best_model" ]; then
    log "Stage 2 already exists, skipping"
fi

# ============================================================
# Stage 3: CLM Polish (8 epochs, lr=3e-5, patience=8)
# ============================================================
S3_OUT="$HDD_OUT/stage3_polish"
S2_BEST="$S2_OUT/best_model"
if [ -d "$S2_OUT/best_model_ema" ]; then
    S2_BEST="$S2_OUT/best_model_ema"
    log "  Using EMA model from Stage 2 for Stage 3"
fi

if [ "$SKIP_STAGE3" = false ] && [ ! -d "$S3_OUT/best_model" ]; then
    log ""
    log "=== Stage 3: CLM Polish (8 epochs, lr=3e-5, patience=8) ==="
    S3_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v12/train_v12.py \
        --stage clm \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S3_OUT" \
        --resume_from "$S2_BEST" \
        --d_model $D_MODEL --n_layer $N_LAYER --n_head $N_HEAD --n_kv_heads $N_KV \
        --scheduler cosine \
        --lr 3e-5 \
        --epochs 8 \
        --batch_size $BATCH \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --bpe_dropout 0.0 \
        --label_smoothing 0.02 \
        --label_smoothing_anneal \
        --attention_dropout 0.0 \
        --use_ema --ema_decay 0.999 \
        --patience 8 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v12-stage3-polish" \
        2>&1 | tee -a "$LOG_FILE"

    S3_DUR=$(( ($(date +%s) - S3_START) / 60 ))
    log "Stage 3 done in ${S3_DUR} min"
    eval_stage "$S3_OUT/best_model" "$HDD_OUT/eval_stage3.json" "Stage3-Polish"
    if [ -d "$S3_OUT/best_model_ema" ]; then
        eval_stage "$S3_OUT/best_model_ema" "$HDD_OUT/eval_stage3_ema.json" "Stage3-Polish-EMA"
    fi
elif [ -d "$S3_OUT/best_model" ]; then
    log "Stage 3 already exists, skipping"
fi

# ============================================================
# Stage 4: Self-Distillation (6 epochs, T=4.0, sd_lambda=0.7)
# ============================================================
S4_OUT="$HDD_OUT/stage4_self_distill"
S3_BEST="$S3_OUT/best_model"
if [ -d "$S3_OUT/best_model_ema" ]; then
    S3_BEST="$S3_OUT/best_model_ema"
    log "  Using EMA model from Stage 3 for Stage 4"
fi

if [ "$SKIP_STAGE4" = false ] && [ ! -d "$S4_OUT/best_model" ]; then
    log ""
    log "=== Stage 4: Self-Distillation (6 epochs, lr=3e-5, T=4.0, lambda=0.7) ==="
    S4_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v12/train_v12.py \
        --stage clm \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S4_OUT" \
        --resume_from "$S3_BEST" \
        --d_model $D_MODEL --n_layer $N_LAYER --n_head $N_HEAD --n_kv_heads $N_KV \
        --scheduler cosine \
        --lr 3e-5 \
        --epochs 6 \
        --batch_size $BATCH \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --bpe_dropout 0.0 \
        --label_smoothing 0.0 \
        --attention_dropout 0.0 \
        --use_ema --ema_decay 0.999 \
        --self_distill --sd_temperature 4.0 --sd_lambda 0.7 --sd_update_freq 100 \
        --patience 5 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v12-stage4-self-distill" \
        2>&1 | tee -a "$LOG_FILE"

    S4_DUR=$(( ($(date +%s) - S4_START) / 60 ))
    log "Stage 4 done in ${S4_DUR} min"
    eval_stage "$S4_OUT/best_model" "$HDD_OUT/eval_stage4.json" "Stage4-SelfDistill"
    if [ -d "$S4_OUT/best_model_ema" ]; then
        eval_stage "$S4_OUT/best_model_ema" "$HDD_OUT/eval_stage4_ema.json" "Stage4-SelfDistill-EMA"
    fi
elif [ -d "$S4_OUT/best_model" ]; then
    log "Stage 4 already exists, skipping"
fi

# ============================================================
# Stage 5: Annealing (3 epochs, lr=5e-6)
# ============================================================
S5_OUT="$HDD_OUT/stage5_annealing"
S4_BEST="$S4_OUT/best_model"
if [ -d "$S4_OUT/best_model_ema" ]; then
    S4_BEST="$S4_OUT/best_model_ema"
    log "  Using EMA model from Stage 4 for Stage 5"
fi

if [ "$SKIP_STAGE5" = false ] && [ ! -d "$S5_OUT/best_model" ]; then
    log ""
    log "=== Stage 5: Annealing (3 epochs, lr=5e-6) ==="
    S5_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v12/train_v12.py \
        --stage clm \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S5_OUT" \
        --resume_from "$S4_BEST" \
        --d_model $D_MODEL --n_layer $N_LAYER --n_head $N_HEAD --n_kv_heads $N_KV \
        --scheduler cosine \
        --lr 5e-6 \
        --epochs 3 \
        --batch_size $BATCH \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --bpe_dropout 0.0 \
        --label_smoothing 0.0 \
        --attention_dropout 0.0 \
        --use_ema --ema_decay 0.999 \
        --patience 3 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v12-stage5-annealing" \
        2>&1 | tee -a "$LOG_FILE"

    S5_DUR=$(( ($(date +%s) - S5_START) / 60 ))
    log "Stage 5 done in ${S5_DUR} min"
    eval_stage "$S5_OUT/best_model" "$HDD_OUT/eval_stage5.json" "Stage5-Annealing"
    if [ -d "$S5_OUT/best_model_ema" ]; then
        eval_stage "$S5_OUT/best_model_ema" "$HDD_OUT/eval_stage5_ema.json" "Stage5-Annealing-EMA"
    fi
elif [ -d "$S5_OUT/best_model" ]; then
    log "Stage 5 already exists, skipping"
fi

# ============================================================
# Stage 6: Official Evaluation
# ============================================================
if [ "$SKIP_OFFICIAL" = false ]; then
    log ""
    log "=== Stage 6: Official Evaluation Pipeline ==="
    S6_START=$(date +%s)

    FINAL_MODEL="$S5_OUT/best_model_ema"
    if [ ! -d "$FINAL_MODEL" ]; then
        FINAL_MODEL="$S5_OUT/best_model"
    fi

    log "  Converting tokenizer for HF compatibility ..."
    python "$PROJECT_DIR/src/v12/convert_tokenizer.py" \
        --spm_model "$TOKENIZER_DIR/spm.model" \
        --output_dir "$FINAL_MODEL" \
        2>&1 | tee -a "$LOG_FILE" || true

    V12_CONFIG="$EVAL_DIR/configs/config_v12.yaml"
    mkdir -p "$EVAL_DIR/configs"
    cat > "$V12_CONFIG" << YAML
models:
  - path: $FINAL_MODEL
    backend: causal

tasks:
  zero_shot:
    - zhoblimp
    - hanzi_structure
    - hanzi_pinyin
  cogbench:
    - word_fmri
    - fmri
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

    log "  Running official eval pipeline on $FINAL_MODEL ..."
    cd "$EVAL_DIR"
    python pipeline.py eval \
        --config configs/config_v12.yaml \
        --results_dir "$HDD_OUT/official_eval" \
        2>&1 | tee -a "$LOG_FILE" || true
    cd "$PROJECT_DIR"

    S6_DUR=$(( ($(date +%s) - S6_START) / 60 ))
    log "Stage 6 done in ${S6_DUR} min"
fi

# ============================================================
# Final Summary
# ============================================================
TOTAL_DUR=$(( ($(date +%s) - PIPELINE_START) / 60 ))

log ""
log "============================================================"
log "V12 Pipeline Complete! Total: ${TOTAL_DUR} min ($(( TOTAL_DUR / 60 ))h $(( TOTAL_DUR % 60 ))m)"
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
