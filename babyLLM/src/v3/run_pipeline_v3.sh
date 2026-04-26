#!/bin/bash
# 一键运行 V3 训练管线

set -e

echo "======================================"
echo "🚀 启动 ChineseBabyLM V3 训练管线"
echo "======================================"

# 1. 准备数据
if [ ! -d "data/processed_v3" ]; then
    echo "步骤 1: 数据预处理 (增强清洗版)..."
    python src/v3/prepare_data_v3.py \
        --input data/processed/all.txt \
        --output_dir data/processed_v3
else
    echo "步骤 1: V3数据已存在，跳过预处理。"
fi

# 2. 训练 Tokenizer
if [ ! -d "data/tokenizer_v3" ]; then
    echo "步骤 2: 训练 SentencePiece Tokenizer..."
    python src/v3/train_tokenizer_v3.py \
        --input data/processed_v3/train.txt \
        --output_dir data/tokenizer_v3 \
        --vocab_size 32000
else
    echo "步骤 2: V3 Tokenizer已存在，跳过训练。"
fi

# 3. 运行模型训练
echo "步骤 3: 启动 Accelerate 多 GPU 训练 (LLaMA V3)..."
# 使用与 V2 相同的 accelerate 配置，或者回退到默认
CONFIG="src/v2/accelerate_config_v2.yaml"
if [ ! -f "$CONFIG" ]; then
    CONFIG="default"
fi

# 开始训练
if [ "$CONFIG" = "default" ]; then
    accelerate launch src/v3/train_v3.py \
        --data_dir data \
        --output_dir output/babylm-llama-v3 \
        --wandb_run_name "llama-v3-spm-wsd"
else
    accelerate launch --config_file $CONFIG src/v3/train_v3.py \
        --data_dir data \
        --output_dir output/babylm-llama-v3 \
        --wandb_run_name "llama-v3-spm-wsd"
fi

# 4. 评测模型
echo "步骤 4: 运行模型评测..."
python src/v3/evaluate_v3.py \
    --model_path output/babylm-llama-v3/best_model \
    --val_file data/processed_v3/val.txt

echo "======================================"
echo "✅ V3 训练管线全部完成!"
echo "======================================"
