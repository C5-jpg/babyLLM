@echo off
REM ChineseBabyLM 训练运行脚本
REM 使用 data conda 虚拟环境

echo ============================================================
echo ChineseBabyLM 挑战赛训练
echo ============================================================

REM 激活 conda data 环境
call D:\anaconda3\Scripts\activate.bat data

REM 切换到项目目录
cd /d e:\reps\babyLLM

echo.
echo [Step 1/3] 数据准备: 下载 + 预处理 + 训练 Tokenizer
echo ------------------------------------------------------------
python prepare_data.py --save_dir data --vocab_size 32000

echo.
echo [Step 2/3] 训练模型
echo ------------------------------------------------------------
python train.py ^
    --data_dir data ^
    --output_dir output/babylm-gpt2 ^
    --d_model 768 ^
    --n_layer 12 ^
    --n_head 12 ^
    --max_length 512 ^
    --batch_size 8 ^
    --learning_rate 6e-4 ^
    --num_epochs 10 ^
    --gradient_accumulation_steps 4 ^
    --lr_scheduler_type cosine ^
    --warmup_ratio 0.1 ^
    --logging_steps 100 ^
    --save_steps 1000 ^
    --seed 42

echo.
echo [Step 3/3] 完成!
echo ------------------------------------------------------------
echo 模型保存在: output\babylm-gpt2
echo ============================================================

pause