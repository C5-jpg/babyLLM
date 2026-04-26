#!/bin/bash
set -euo pipefail

source /home/kehe/anaconda3/etc/profile.d/conda.sh
conda activate data

export NCCL_TIMEOUT=1800000
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^docker0

PROJECT_DIR="/home/kehe/babyllm/babyLLM"
DATA_DIR="$PROJECT_DIR/data/processed_v7"
TOKENIZER_DIR="$PROJECT_DIR/data/tokenizer_v7"
HDD_OUT="/mnt/sda/kehe/babyllm_output/babylm-v8"

mkdir -p "$HDD_OUT"
cd "$PROJECT_DIR"

echo "=== Stage 1: CLM ==="
S1_OUT="$HDD_OUT/stage1_clm"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 --mixed_precision=bf16 src/v8/train_v8.py \
    --stage clm --data_dir "$DATA_DIR" --tokenizer_dir "$TOKENIZER_DIR" \
    --output_dir "$S1_OUT" --lr 6e-4 --epochs 2

echo "=== Stage 2: MNTP ==="
S2_OUT="$HDD_OUT/stage2_mntp"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 --mixed_precision=bf16 src/v8/train_v8.py \
    --stage mntp --data_dir "$DATA_DIR" --tokenizer_dir "$TOKENIZER_DIR" \
    --output_dir "$S2_OUT" --resume_from "$S1_OUT/best_model" \
    --lr 5e-4 --epochs 6

echo "=== Stage 3: Polish ==="
S3_OUT="$HDD_OUT/stage3_polish"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 --mixed_precision=bf16 src/v8/train_v8.py \
    --stage clm --data_dir "$DATA_DIR" --tokenizer_dir "$TOKENIZER_DIR" \
    --output_dir "$S3_OUT" --resume_from "$S2_OUT/best_model" \
    --lr 1e-4 --epochs 2

echo "=== Evaluation ==="
python src/v8/evaluate_v8.py --model_path "$S3_OUT/best_model" --val_file "$DATA_DIR/val.txt" > "$HDD_OUT/eval_final.json"
