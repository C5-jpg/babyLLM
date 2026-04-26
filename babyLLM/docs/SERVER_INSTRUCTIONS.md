# 服务器端 Cline 训练指令

将以下内容完整复制粘贴给服务器上的 Cline：

---

## 指令内容（复制以下全部内容）

```
请帮我完成以下任务：克隆 BabyLLM 仓库并训练一个中文 GPT-2 语言模型。

### 第一步：克隆仓库并安装环境

1. 克隆代码仓库：
```bash
git clone https://github.com/c5-jpg/babyLLM.git
cd babyLLM
```

如果 GitHub 仓库还没准备好，请手动创建以下目录结构：
```bash
mkdir -p babyLLM && cd babyLLM
mkdir -p data/tokenizer data/processed output
```

2. 创建 Python 虚拟环境并安装依赖：
```bash
conda create -n data python=3.12 -y
conda activate data
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers tokenizers datasets accelerate jieba tqdm matplotlib scikit-learn scipy
```

### 第二步：数据准备

创建并运行 `prepare_data.py` 脚本来下载和预处理数据：

```python
"""
prepare_data.py - 下载 babylm-zho-100M 数据并训练 BPE Tokenizer
"""
import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm

def main():
    # 创建目录
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/tokenizer", exist_ok=True)
    
    # 1. 下载数据集
    print("正在下载 babylm-zho-100M 数据集...")
    ds = load_dataset("chinese-babylm-org/babylm-zho-100M", split="train")
    print(f"数据集样本数: {len(ds):,}")
    
    # 2. 合并所有文本
    print("合并文本数据...")
    output_file = "data/processed/train.txt"
    total_chars = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(ds, desc="写入文本"):
            text = item["text"].strip()
            if text:
                f.write(text + "\n")
                total_chars += len(text)
    print(f"总字符数: {total_chars:,}")
    print(f"训练文本已保存到: {output_file}")
    
    # 3. 训练 BPE Tokenizer
    print("\n训练 BPE Tokenizer (32K 词表)...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        min_frequency=2,
        show_progress=True,
    )
    
    def batch_iterator(batch_size=1000):
        for i in range(0, len(ds), batch_size):
            yield [item["text"] for item in ds[i : i + batch_size]]
    
    tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(ds))
    tokenizer.save("data/tokenizer/tokenizer.json")
    
    # 4. 创建 tokenizer_config.json
    import json
    config = {
        "model_type": "gpt2",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    with open("data/tokenizer/tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n数据准备完成！")
    print(f"  Tokenizer: data/tokenizer/tokenizer.json")
    print(f"  训练数据: {output_file}")

if __name__ == "__main__":
    main()
```

运行：
```bash
conda activate data
python prepare_data.py
```

### 第三步：开始训练

单GPU训练（适合 24GB+ VRAM 的 GPU）：
```bash
conda activate data
python train.py \
    --data_dir data \
    --output_dir output/babylm-gpt2 \
    --d_model 768 \
    --n_layer 12 \
    --n_head 12 \
    --max_length 512 \
    --batch_size 16 \
    --learning_rate 6e-4 \
    --num_epochs 10 \
    --gradient_accumulation_steps 2 \
    --logging_steps 100 \
    --save_steps 2000 \
    --seed 42
```

多GPU训练（4卡，推荐）：
```bash
conda activate data
torchrun --nproc_per_node=4 train.py \
    --data_dir data \
    --output_dir output/babylm-gpt2 \
    --d_model 768 \
    --n_layer 12 \
    --n_head 12 \
    --max_length 512 \
    --batch_size 16 \
    --learning_rate 6e-4 \
    --num_epochs 10 \
    --gradient_accumulation_steps 2 \
    --logging_steps 100 \
    --save_steps 2000 \
    --seed 42
```

如果显存不足（如16GB），减小batch_size：
```bash
python train.py \
    --data_dir data \
    --output_dir output/babylm-gpt2 \
    --d_model 768 \
    --n_layer 12 \
    --n_head 12 \
    --max_length 512 \
    --batch_size 8 \
    --learning_rate 6e-4 \
    --num_epochs 10 \
    --gradient_accumulation_steps 4 \
    --logging_steps 100 \
    --save_steps 2000
```

### 注意事项

1. train.py 脚本需要从仓库获取。如果仓库已克隆，它就在根目录下
2. 训练数据约 100M 中文词，从 HuggingFace 自动下载
3. 模型是 GPT-2 Small 架构（~110M参数）
4. 训练大约需要 45,000 步（10 epochs）
5. 最佳模型保存在 output/babylm-gpt2/best_model/
6. 每个 epoch 的检查点保存在 output/babylm-gpt2/epoch-N/
7. 训练日志同时输出到终端和 training.log 文件

请先确认 GPU 信息（nvidia-smi），然后按步骤执行。
```

---

## 使用方法

1. 在服务器上打开 VS Code + Cline
2. 将上面 ``` 代码块内的全部内容复制粘贴给 Cline
3. Cline 会自动按步骤执行