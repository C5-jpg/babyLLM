"""
ChineseBabyLM V3 - SentencePiece Tokenizer 训练脚本
修复: 使用 SentencePiece 替换 ByteLevel BPE，确保中文字词完整，解决乱码和过度拆分问题。
"""
import os
import argparse
import sentencepiece as spm
from transformers import LlamaTokenizerFast

def train_sentencepiece(input_file, output_dir, vocab_size=32000):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"训练 SentencePiece Tokenizer (BPE 模型, 词表: {vocab_size})")
    print(f"训练数据: {input_file}")
    print("=" * 60)
    
    model_prefix = os.path.join(output_dir, "spm")
    
    # 训练 SentencePiece 模型
    # 使用 BPE 算法，特别适合处理中英文混合
    # character_coverage 设为 0.9995 以覆盖绝大多数中文字符
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='bpe',
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        train_extremely_large_corpus=True
    )
    print(f"\nSentencePiece 模型训练完成: {model_prefix}.model")
    
    # 将 SentencePiece 模型转换为 HuggingFace 兼容的 Tokenizer
    print("\n转换并保存为 HuggingFace 格式...")
    tokenizer = LlamaTokenizerFast(
        vocab_file=f"{model_prefix}.model",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(output_dir)
    print(f"HuggingFace Tokenizer 保存到: {output_dir}")
    
    return tokenizer

def test_tokenizer(tokenizer_dir):
    tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_dir)
    
    test_texts = [
        "这是一个测试句子，用来验证tokenizer是否正常工作。",
        "今天天气真好，我想出去玩。",
        "小猫在阳光下睡觉。",
        "我喜欢吃苹果和香蕉。",
        "中国的首都是北京。",
        "The quick brown fox jumps over the lazy dog.",
        "人工智能 (AI) 正在改变世界。"
    ]
    
    print("\n" + "=" * 60)
    print("Tokenizer 编码测试")
    print("=" * 60)
    
    total_chars = 0
    total_tokens = 0
    for text in test_texts:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(encoding)
        
        total_chars += len(text)
        total_tokens += len(encoding)
        
        has_unk = "<unk>" in tokens
        flag = " ⚠️ HAS UNK!" if has_unk else " ✅"
        print(f"\n  文本: {text}")
        print(f"  Tokens ({len(encoding)}){flag}: {tokens}")
        print(f"  解码回退: {tokenizer.decode(encoding)}")

    print(f"\n统计:")
    print(f"  平均 Token/字符比: {total_tokens / total_chars:.3f}")
    print(f"  词表大小: {tokenizer.vocab_size}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed_v3/train.txt")
    parser.add_argument("--output_dir", type=str, default="data/tokenizer_v3")
    parser.add_argument("--vocab_size", type=int, default=32000)
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 找不到训练数据文件 {args.input}")
        return
        
    train_sentencepiece(args.input, args.output_dir, args.vocab_size)
    test_tokenizer(args.output_dir)

if __name__ == "__main__":
    main()
