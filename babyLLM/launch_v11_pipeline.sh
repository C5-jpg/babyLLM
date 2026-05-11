#!/bin/bash
set -euo pipefail

# ============================================================
# ChineseBabyLM V11 — SOTA 七阶段全自动流水线
# ============================================================
#
# 全自动: Stage1 → eval → Stage2 → eval → Stage3 → eval
#         → Stage4 → eval → Stage5 → eval → SWA → eval → Official Eval
#
# 所有输出到机械硬盘 /mnt/sda/ 避免空间不足
#
# 用法:
#   bash launch_v11_pipeline.sh
#   bash launch_v11_pipeline.sh --skip-stage1
#   bash launch_v11_pipeline.sh --skip-stage1 --skip-stage2
#
# 后台运行:
#   nohup bash launch_v11_pipeline.sh > /mnt/sda/kehe/babyllm_output/babylm-v11/pipeline_v11.log 2>&1 &
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
DATA_DIR="$PROJECT_DIR/data/processed_v7"
TOKENIZER_DIR="$PROJECT_DIR/data/tokenizer_v7"
HDD_OUT="/mnt/sda/kehe/babyllm_output/babylm-v11"
LOG_FILE="$HDD_OUT/pipeline_v11.log"
EVAL_DIR="/home/kehe/babyllm/chinese-babylm-eval-pipeline"

GPUS="0,1,2,3"
NUM_GPUS=4
BATCH=32
MAX_LENGTH=1024
STRIDE=512

SKIP_STAGE1=false
SKIP_STAGE2=false
SKIP_STAGE3=false
SKIP_STAGE4=false
SKIP_STAGE5=false
SKIP_SWA=false
SKIP_OFFICIAL=false
for arg in "$@"; do
    case $arg in
        --skip-stage1) SKIP_STAGE1=true ;;
        --skip-stage2) SKIP_STAGE2=true ;;
        --skip-stage3) SKIP_STAGE3=true ;;
        --skip-stage4) SKIP_STAGE4=true ;;
        --skip-stage5) SKIP_STAGE5=true ;;
        --skip-swa)    SKIP_SWA=true ;;
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
    python "$PROJECT_DIR/src/v11/evaluate_v11.py" \
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
log "ChineseBabyLM V11 — SOTA 七阶段全自动流水线"
log "============================================================"
log "  Data:      $DATA_DIR"
log "  Tokenizer: $TOKENIZER_DIR"
log "  Output:    $HDD_OUT (HDD)"
log "  GPUs:      4× A6000 48GB (CUDA $GPUS)"
log "  Batch:     $BATCH/GPU (eff. $((BATCH * NUM_GPUS)))"
log "  Stride:    $STRIDE (50% overlap)"
log "  Log:       $LOG_FILE"
log "============================================================"

# ============================================================
# Stage 1: CLM 预训练 (SGDR, 6 epochs)
# ============================================================
S1_OUT="$HDD_OUT/stage1_clm_sgdr"

if [ "$SKIP_STAGE1" = false ] && [ ! -d "$S1_OUT/best_model" ]; then
    log ""
    log "=== Stage 1: CLM 预训练 (SGDR, 6 epochs, lr=6e-4) ==="
    S1_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v11/train_v11.py \
        --stage clm \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S1_OUT" \
        --scheduler sgdr \
        --lr 6e-4 \
        --epochs 6 \
        --batch_size $BATCH \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --bpe_dropout 0.1 \
        --label_smoothing 0.1 \
        --label_smoothing_anneal \
        --attention_dropout 0.1 \
        --use_ema --ema_decay 0.999 \
        --patience 3 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v11-stage1-clm-sgdr" \
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
# Stage 2: MNTP (动态 CLM ratio, 8 epochs)
# ============================================================
S2_OUT="$HDD_OUT/stage2_mntp"
S1_BEST="$S1_OUT/best_model"
if [ -d "$S1_OUT/best_model_ema" ]; then
    S1_BEST="$S1_OUT/best_model_ema"
    log "  Using EMA model from Stage 1 for Stage 2"
fi

if [ "$SKIP_STAGE2" = false ] && [ ! -d "$S2_OUT/best_model" ]; then
    log ""
    log "=== Stage 2: MNTP (动态 CLM ratio, 8 epochs, lr=5e-4) ==="
    S2_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v11/train_v11.py \
        --stage mntp \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S2_OUT" \
        --resume_from "$S1_BEST" \
        --scheduler cosine \
        --lr 5e-4 \
        --epochs 8 \
        --batch_size $BATCH \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --clm_ratio 0.125 \
        --dynamic_clm_ratio \
        --mask_ratio_start 0.25 \
        --mask_ratio_end 0.12 \
        --bpe_dropout 0.1 \
        --label_smoothing 0.05 \
        --label_smoothing_anneal \
        --attention_dropout 0.1 \
        --use_ema --ema_decay 0.999 \
        --patience 8 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v11-stage2-mntp-dynamic" \
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
# Stage 3: CLM Polish (5 epochs, lr=5e-5)
# ============================================================
S3_OUT="$HDD_OUT/stage3_polish"
S2_BEST="$S2_OUT/best_model"
if [ -d "$S2_OUT/best_model_ema" ]; then
    S2_BEST="$S2_OUT/best_model_ema"
    log "  Using EMA model from Stage 2 for Stage 3"
fi

if [ "$SKIP_STAGE3" = false ] && [ ! -d "$S3_OUT/best_model" ]; then
    log ""
    log "=== Stage 3: CLM Polish (5 epochs, lr=5e-5) ==="
    S3_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v11/train_v11.py \
        --stage clm \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S3_OUT" \
        --resume_from "$S2_BEST" \
        --scheduler cosine \
        --lr 5e-5 \
        --epochs 5 \
        --batch_size $BATCH \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --bpe_dropout 0.0 \
        --label_smoothing 0.02 \
        --label_smoothing_anneal \
        --attention_dropout 0.0 \
        --use_ema --ema_decay 0.999 \
        --patience 5 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v11-stage3-polish" \
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
# Stage 4: Annealing (3 epochs, lr=1e-5)
# ============================================================
S4_OUT="$HDD_OUT/stage4_annealing"
S3_BEST="$S3_OUT/best_model"
if [ -d "$S3_OUT/best_model_ema" ]; then
    S3_BEST="$S3_OUT/best_model_ema"
    log "  Using EMA model from Stage 3 for Stage 4"
fi

if [ "$SKIP_STAGE4" = false ] && [ ! -d "$S4_OUT/best_model" ]; then
    log ""
    log "=== Stage 4: Annealing (3 epochs, lr=1e-5) ==="
    S4_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v11/train_v11.py \
        --stage clm \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S4_OUT" \
        --resume_from "$S3_BEST" \
        --scheduler cosine \
        --lr 1e-5 \
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
        --wandb_run_name "babylm-v11-stage4-annealing" \
        2>&1 | tee -a "$LOG_FILE"

    S4_DUR=$(( ($(date +%s) - S4_START) / 60 ))
    log "Stage 4 done in ${S4_DUR} min"
    eval_stage "$S4_OUT/best_model" "$HDD_OUT/eval_stage4.json" "Stage4-Annealing"
    if [ -d "$S4_OUT/best_model_ema" ]; then
        eval_stage "$S4_OUT/best_model_ema" "$HDD_OUT/eval_stage4_ema.json" "Stage4-Annealing-EMA"
    fi
elif [ -d "$S4_OUT/best_model" ]; then
    log "Stage 4 already exists, skipping"
fi

# ============================================================
# Stage 5: Self-Distillation (4 epochs, lr=5e-5)
# ============================================================
S5_OUT="$HDD_OUT/stage5_self_distill"
S4_BEST="$S4_OUT/best_model"
if [ -d "$S4_OUT/best_model_ema" ]; then
    S4_BEST="$S4_OUT/best_model_ema"
    log "  Using EMA model from Stage 4 for Stage 5"
fi

if [ "$SKIP_STAGE5" = false ] && [ ! -d "$S5_OUT/best_model" ]; then
    log ""
    log "=== Stage 5: Self-Distillation (4 epochs, lr=5e-5) ==="
    S5_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=$NUM_GPUS --mixed_precision=bf16 \
        src/v11/train_v11.py \
        --stage clm \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$S5_OUT" \
        --resume_from "$S4_BEST" \
        --scheduler cosine \
        --lr 5e-5 \
        --epochs 4 \
        --batch_size $BATCH \
        --max_length $MAX_LENGTH \
        --stride $STRIDE \
        --bpe_dropout 0.0 \
        --label_smoothing 0.0 \
        --attention_dropout 0.0 \
        --use_ema --ema_decay 0.999 \
        --self_distill --sd_temperature 2.0 --sd_lambda 0.5 \
        --patience 3 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --wandb_run_name "babylm-v11-stage5-self-distill" \
        2>&1 | tee -a "$LOG_FILE"

    S5_DUR=$(( ($(date +%s) - S5_START) / 60 ))
    log "Stage 5 done in ${S5_DUR} min"
    eval_stage "$S5_OUT/best_model" "$HDD_OUT/eval_stage5.json" "Stage5-SelfDistill"
    if [ -d "$S5_OUT/best_model_ema" ]; then
        eval_stage "$S5_OUT/best_model_ema" "$HDD_OUT/eval_stage5_ema.json" "Stage5-SelfDistill-EMA"
    fi
elif [ -d "$S5_OUT/best_model" ]; then
    log "Stage 5 already exists, skipping"
fi

# ============================================================
# Stage 6: SWA (Stochastic Weight Averaging)
# ============================================================
S6_OUT="$HDD_OUT/stage6_swa"

if [ "$SKIP_SWA" = false ] && [ ! -d "$S6_OUT/model" ]; then
    log ""
    log "=== Stage 6: SWA Weight Averaging ==="
    S6_START=$(date +%s)

    SWA_DIRS=()
    for s in stage3_polish stage4_annealing stage5_self_distill; do
        for sub in best_model_ema best_model; do
            d="$HDD_OUT/$s/$sub"
            if [ -d "$d" ]; then
                SWA_DIRS+=("$d")
                break
            fi
        done
    done

    if [ ${#SWA_DIRS[@]} -ge 2 ]; then
        log "  Averaging ${#SWA_DIRS[@]} checkpoints:"
        for d in "${SWA_DIRS[@]}"; do
            log "    $d"
        done

        python "$PROJECT_DIR/src/v11/swa_v11.py" \
            --checkpoint_dirs "${SWA_DIRS[@]}" \
            --output_dir "$S6_OUT/model" \
            --tokenizer_dir "$TOKENIZER_DIR" \
            2>&1 | tee -a "$LOG_FILE"

        S6_DUR=$(( ($(date +%s) - S6_START) / 60 ))
        log "Stage 6 done in ${S6_DUR} min"
        eval_stage "$S6_OUT/model" "$HDD_OUT/eval_stage6_swa.json" "Stage6-SWA"
    else
        log "  Not enough checkpoints for SWA (${#SWA_DIRS[@]} found, need >= 2)"
        if [ ${#SWA_DIRS[@]} -eq 1 ]; then
            log "  Copying single best checkpoint as SWA model"
            mkdir -p "$S6_OUT/model"
            cp -r "${SWA_DIRS[0]}"/* "$S6_OUT/model/"
            eval_stage "$S6_OUT/model" "$HDD_OUT/eval_stage6_swa.json" "Stage6-SWA"
        fi
    fi
elif [ -d "$S6_OUT/model" ]; then
    log "Stage 6 already exists, skipping"
fi

# ============================================================
# Stage 7: 官方评测流水线
# ============================================================
if [ "$SKIP_OFFICIAL" = false ]; then
    log ""
    log "=== Stage 7: Official Evaluation Pipeline ==="
    S7_START=$(date +%s)

    FINAL_MODEL="$S6_OUT/model"
    if [ ! -d "$FINAL_MODEL" ]; then
        S5_BEST="$S5_OUT/best_model"
        if [ -d "$S5_OUT/best_model_ema" ]; then
            S5_BEST="$S5_OUT/best_model_ema"
        fi
        FINAL_MODEL="$S5_BEST"
    fi

    log "  Converting tokenizer for HF compatibility ..."
    python "$PROJECT_DIR/src/v11/convert_tokenizer.py" \
        --spm_model "$TOKENIZER_DIR/spm.model" \
        --output_dir "$FINAL_MODEL" \
        2>&1 | tee -a "$LOG_FILE" || true

    V11_CONFIG="$EVAL_DIR/configs/config_v11.yaml"
    if [ -f "$V11_CONFIG" ]; then
        log "  Running official eval pipeline on $FINAL_MODEL ..."
        cd "$EVAL_DIR"
        python pipeline.py eval \
            --config configs/config_v11.yaml \
            --results_dir "$HDD_OUT/official_eval" \
            2>&1 | tee -a "$LOG_FILE" || true
        cd "$PROJECT_DIR"
    else
        log "  Config $V11_CONFIG not found, creating ..."
        mkdir -p "$EVAL_DIR/configs"
        cat > "$V11_CONFIG" << YAML
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
            --config configs/config_v11.yaml \
            --results_dir "$HDD_OUT/official_eval" \
            2>&1 | tee -a "$LOG_FILE" || true
        cd "$PROJECT_DIR"
    fi

    S7_DUR=$(( ($(date +%s) - S7_START) / 60 ))
    log "Stage 7 done in ${S7_DUR} min"
fi

# ============================================================
# Final Summary
# ============================================================
TOTAL_DUR=$(( ($(date +%s) - PIPELINE_START) / 60 ))

log ""
log "============================================================"
log "V11 Pipeline Complete! Total: ${TOTAL_DUR} min ($(( TOTAL_DUR / 60 ))h $(( TOTAL_DUR % 60 ))m)"
log "============================================================"

for stage_eval in "$HDD_OUT"/eval_stage*.json "$HDD_OUT"/eval_final*.json; do
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
log "Model: $HDD_OUT/stage6_swa/model"
log "Logs:  $LOG_FILE"
log "============================================================"
