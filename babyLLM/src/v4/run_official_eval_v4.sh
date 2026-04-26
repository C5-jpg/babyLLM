#!/bin/bash
set -euo pipefail

MODEL_PATH=${1:-"/home/kehe/babyllm/babyLLM/output/babylm-llama-v4/best_model"}
EVAL_PIPELINE_DIR=${2:-"/home/kehe/babyllm/chinese-babylm-eval-pipeline"}
OUTPUT_DIR=${3:-"/home/kehe/babyllm/babyLLM/output/v4_official_eval"}
BACKEND=${BACKEND:-"causal"}

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Chinese BabyLM V4 Official Evaluation"
echo "=============================================="
echo "MODEL_PATH=${MODEL_PATH}"
echo "EVAL_PIPELINE_DIR=${EVAL_PIPELINE_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "BACKEND=${BACKEND}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "[ERROR] model path not found: ${MODEL_PATH}"
  exit 1
fi

if [ ! -d "${EVAL_PIPELINE_DIR}" ]; then
  echo "[ERROR] eval pipeline dir not found: ${EVAL_PIPELINE_DIR}"
  exit 1
fi

cd "${EVAL_PIPELINE_DIR}"

# NLU + Hanzi zero-shot
bash eval_zero_shot.sh "${MODEL_PATH}" "${BACKEND}" "evaluation_data/full_eval"

# NLU fine-tuning (CLUE)
bash eval_finetuning.sh "${MODEL_PATH}" "${BACKEND}" 3e-5 64 5 5 42

# Cog track
bash eval_cogbench_fast.sh \
  --model_path "${MODEL_PATH}" \
  --backend "${BACKEND}" \
  --task word_fmri,fmri \
  --eval_dir "evaluation_data/cogbench-fmri-0415" \
  --output_dir "${OUTPUT_DIR}"

echo "Official evaluation completed."
