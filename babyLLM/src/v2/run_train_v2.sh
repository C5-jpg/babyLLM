#!/bin/bash
# ChineseBabyLM V2 - Phase 3 精细优化训练启动脚本
# 使用 4× A6000 GPU, bf16 混合精度, Flash Attention 2
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "ChineseBabyLM V2 - Phase 3 精细优化训练"
echo "============================================================"
echo "启动时间: $(date)"
echo ""

PYTHON=/home/kehe/anaconda3/envs/data/bin/python
ACCELERATE=/home/kehe/anaconda3/envs/data/bin/accelerate

# Step 1: 检查 tokenizer
if [ ! -f "data/tokenizer_v2/tokenizer.json" ]; then
    echo "❌ Tokenizer 未找到! 请先运行: python train_tokenizer_v2.py"
    exit 1
fi
echo "✅ Tokenizer 就绪"

# Step 2: 数据预处理（如果尚未处理）
if [ ! -f "data/processed_v2/train.txt" ]; then
    echo ""
    echo "⚙️ 运行数据预处理..."
    $PYTHON prepare_data_v2.py \
        --input data/processed/all.txt \
        --output_dir data/processed_v2 \
        --no_minhash
    echo "✅ 数据预处理完成"
else
    echo "✅ 预处理数据就绪"
fi

# Step 3: 检查 GPU
echo ""
echo "GPU 状态:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "无法获取 GPU 信息"
echo ""

# Step 4: 检查依赖
echo "检查 PyTorch SDPA..."
$PYTHON -c "import torch; print(f'✅ PyTorch {torch.__version__}, SDPA available')" 2>/dev/null

echo ""
echo "============================================================"
echo "🚀 启动 4×GPU DDP 训练 (Phase 3 精细优化)"
echo "============================================================"
echo ""

# 训练参数说明:
# - d_model=768, n_layer=12, n_head=12, n_kv_heads=4 → ~125M 参数
# - batch_size=16/GPU × grad_accum=2 × 4 GPU = 有效 batch 128
# - learning_rate=6e-4, cosine decay with 5% warmup
# - max_length=1024 (RoPE 支持长序列)
# - 25 epochs (充分利用数据)
# - Flash Attention 2 + Gradient Checkpointing
# - BPE Dropout=0.1 数据增强
# - WandB online 监控

$ACCELERATE launch \
    --config_file accelerate_config_v2.yaml \
    train_v2.py \
    --data_dir data \
    --output_dir output/babylm-llama-v2 \
    --d_model 768 \
    --n_layer 12 \
    --n_head 12 \
    --n_kv_heads 4 \
    --max_length 1024 \
    --batch_size 16 \
    --learning_rate 6e-4 \
    --weight_decay 0.1 \
    --num_epochs 25 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 2 \
    --use_flash_attention \
    --gradient_checkpointing \
    --bpe_dropout 0.1 \
    --dropout_anneal \
    --logging_steps 50 \
    --save_steps 2000 \
    --seed 42 \
    --wandb_project chinese-babylm \
    --wandb_run_name "llama-v2-phase3-768d-12l-gqa4-fa2-bpedrop" \
    --wandb_mode online

echo ""
echo "============================================================"
echo "🎉 训练完成时间: $(date)"
echo "============================================================"

# Step 5: 运行评测
echo ""
echo "📊 运行模型评测..."
$PYTHON evaluate_v2.py \
    --model_path output/babylm-llama-v2/best_model \
    --val_file data/processed_v2/val.txt \
    --wandb_project chinese-babylm \
    --wandb_run_name "llama-v2-phase3-768d-12l-gqa4-fa2-bpedrop"

echo ""
echo "============================================================"
echo "✅ 全部完成!"
echo "============================================================"