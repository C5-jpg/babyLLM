"""
ChineseBabyLM 数据准备脚本
从 HuggingFace 下载 babylm-zho-100M 数据集并进行预处理
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# 设置 HuggingFace 镜像（解决SSL问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_dataset(save_dir="data/raw"):
    """从 HuggingFace 下载 babylm-zho-100M 数据集"""
    from datasets import load_dataset
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("下载 ChineseBabyLM 数据集: babylm-zho-100M")
    print("=" * 60)
    
    dataset = load_dataset(
        "chinese-babylm-org/babylm-zho-100M",
        cache_dir=save_dir
    )
    
    print(f"\n数据集结构: {dataset}")
    print(f"数据集特征: {dataset['train'].features if 'train' in dataset else 'N/A'}")
    
    # 查看数据集大小
    for split in dataset:
        print(f"  {split}: {len(dataset[split])} 条样本")
        if len(dataset[split]) > 0:
            sample = dataset[split][0]
            print(f"  样本键: {sample.keys()}")
            for key, val in sample.items():
                if isinstance(val, str):
                    print(f"    {key}: {val[:100]}...")
                else:
                    print(f"    {key}: {val}")
    
    return dataset


def extract_texts(dataset, output_dir="data/processed"):
    """从数据集中提取所有文本，保存为纯文本文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_texts = []
    
    for split in dataset:
        print(f"\n处理 {split} 分割...")
        texts = []
        for item in tqdm(dataset[split], desc=f"提取 {split}"):
            # 尝试不同的字段名
            text = item.get("text", "") or item.get("content", "") or item.get("sentence", "")
            if text:
                texts.append(text.strip())
        
        all_texts.extend(texts)
        
        # 保存分割文本
        split_file = os.path.join(output_dir, f"{split}.txt")
        with open(split_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")
        print(f"  保存 {len(texts)} 条文本到 {split_file}")
    
    # 保存合并文本
    all_file = os.path.join(output_dir, "all.txt")
    with open(all_file, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text + "\n")
    
    # 统计信息
    total_chars = sum(len(t) for t in all_texts)
    print(f"\n统计信息:")
    print(f"  总文本数: {len(all_texts)}")
    print(f"  总字符数: {total_chars:,}")
    print(f"  平均文本长度: {total_chars / len(all_texts):.1f} 字符")
    
    return all_texts


def train_tokenizer(texts, vocab_size=32000, save_dir="data/tokenizer"):
    """从头训练 BPE tokenizer"""
    from tokenizers import (
        Tokenizer,
        models,
        trainers,
        pre_tokenizers,
        decoders,
        processors
    )
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"训练 BPE Tokenizer (词表大小: {vocab_size})")
    print("=" * 60)
    
    # 初始化 BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # 使用中文字符级别的预分词
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation(behavior="isolated"),
    ])
    
    tokenizer.decoder = decoders.BPEDecoder()
    
    # 特殊token
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
    
    # 训练器配置
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )
    
    # 训练
    # 写入临时文件用于训练
    temp_file = os.path.join(save_dir, "temp_train.txt")
    print("写入临时训练文件...")
    with open(temp_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    
    print("开始训练 tokenizer...")
    tokenizer.train([temp_file], trainer=trainer)
    
    # 保存
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer 保存到: {tokenizer_path}")
    
    # 测试
    test_text = "这是一个测试句子，用来验证tokenizer是否正常工作。"
    encoding = tokenizer.encode(test_text)
    print(f"\nTokenizer 测试:")
    print(f"  输入: {test_text}")
    print(f"  Token IDs: {encoding.ids}")
    print(f"  Tokens: {encoding.tokens}")
    print(f"  词表大小: {tokenizer.get_vocab_size()}")
    
    # 清理临时文件
    os.remove(temp_file)
    
    return tokenizer


def create_hf_tokenizer(tokenizer_path, save_dir="data/tokenizer"):
    """将 tokenizers 库的 tokenizer 转换为 HuggingFace 格式"""
    from transformers import PreTrainedTokenizerFast
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    
    tokenizer.save_pretrained(save_dir)
    print(f"HuggingFace tokenizer 保存到: {save_dir}")
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="ChineseBabyLM 数据准备")
    parser.add_argument("--save_dir", type=str, default="data", help="数据保存目录")
    parser.add_argument("--vocab_size", type=int, default=32000, help="词表大小")
    args = parser.parse_args()
    
    raw_dir = os.path.join(args.save_dir, "raw")
    processed_dir = os.path.join(args.save_dir, "processed")
    tokenizer_dir = os.path.join(args.save_dir, "tokenizer")
    
    # Step 1: 下载数据
    dataset = download_dataset(raw_dir)
    
    # Step 2: 提取文本
    texts = extract_texts(dataset, processed_dir)
    
    # Step 3: 训练 tokenizer
    tokenizer = train_tokenizer(texts, args.vocab_size, tokenizer_dir)
    
    # Step 4: 转换为 HF 格式
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
    hf_tokenizer = create_hf_tokenizer(tokenizer_path, tokenizer_dir)
    
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print(f"  原始数据: {raw_dir}")
    print(f"  处理后数据: {processed_dir}")
    print(f"  Tokenizer: {tokenizer_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()