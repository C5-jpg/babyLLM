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
V8_BEST="/mnt/sda/kehe/babyllm_output/babylm-v8/stage3_polish/best_model"
HDD_OUT="/mnt/sda/kehe/babyllm_output/babylm-v9"
mkdir -p "$HDD_OUT"
cd "$PROJECT_DIR"
echo "=== V9 Polish Probe: conservative CLM from v8 best ==="
S1_OUT="$HDD_OUT/probe_clm_polish_lr5e-5"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 --mixed_precision=bf16 src/v9/train_v9.py \
    --stage clm --data_dir "$DATA_DIR" --tokenizer_dir "$TOKENIZER_DIR" \
    --output_dir "$S1_OUT" --resume_from "$V8_BEST" \
    --lr 5e-5 --epochs 1 --batch_size 32 --stride 1024 \
    --label_smoothing 0.0 --max_steps 700
echo "=== V9 Polish Probe Evaluation ==="
python src/v9/evaluate_v9.py --model_path "$S1_OUT/best_model" --val_file "$DATA_DIR/val.txt" > "$HDD_OUT/eval_polish_probe.json"
cat "$HDD_OUT/eval_polish_probe.json"
