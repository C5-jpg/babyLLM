#!/bin/bash
cd /home/kehe/babyllm/babyLLM/src/v4
conda activate data 2>/dev/null || source activate data 2>/dev/null
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    train_v4.py \
    --data_dir /home/kehe/babyllm/babyLLM/data \
    --output_dir /home/kehe/babyllm/babyLLM/output/babylm-llama-v4 \
    --d_model 1024 \
    --n_layer 24 \
    --n_head 16 \
    --n_kv_heads 8 \
    --max_length 1024 \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --num_epochs 50 \
    --warmup_ratio 0.03 \
    --gradient_accumulation_steps 2 \
    --attention_dropout 0.1 \
    --bpe_dropout 0.1 \
    --patience 10 \
    --rope_theta 50000.0 \
    --logging_steps 50 \
    --save_steps 5000 \
    --wandb_project chinese-babylm \
    --wandb_run_name llama-v4-1024d-24l-cosine-bpe_dropout-dropout_anneal-4gpu \
    2>&1 | tee /home/kehe/babyllm/babyLLM/output/train_v4.log
