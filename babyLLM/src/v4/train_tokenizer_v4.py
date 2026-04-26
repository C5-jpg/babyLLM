"""
ChineseBabyLM V4 - SentencePiece Tokenizer 训练脚本
改进:
1. byte_fallback=True 确保零 UNK
2. split_digits=True 数字单独分割
3. 转换为标准 LlamaTokenizerFast 格式，兼容 HF 评测 pipeline
"""
import os
import sys
import argparse
import sentencepiece as spm
from transformers import LlamaTokenizer, LlamaTokenizerFast


def train_tokenizer(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model_prefix = os.path.join(args.output_dir, "spiece")

    print(f"训练 SentencePiece tokenizer...")
    print(f"  输入: {args.input_file}")
    print(f"  词表大小: {args.vocab_size}")
    print(f"  模型类型: {args.model_type}")
    print(f"  输出: {model_prefix}")
    print(f"  线程数: {args.num_threads}")
    print(f"  input_sentence_size: {args.input_sentence_size}")
    print(f"  shuffle_input_sentence: {args.shuffle_input_sentence}")
    print(f"  train_extremely_large_corpus: {args.train_extremely_large_corpus}")

    if not args.skip_spm_train:
        spm.SentencePieceTrainer.train(
            input=args.input_file,
            model_prefix=model_prefix,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            model_type=args.model_type,
            byte_fallback=True,
            split_digits=True,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            num_threads=args.num_threads,
            max_sentence_length=16384,
            shuffle_input_sentence=args.shuffle_input_sentence,
            input_sentence_size=args.input_sentence_size,
            train_extremely_large_corpus=args.train_extremely_large_corpus,
            num_sub_iterations=args.num_sub_iterations,
        )
        print("SentencePiece 模型训练完成。")
    else:
        print("跳过 SentencePiece 训练，使用现有 spiece.model 做 HF 转换。")

    # 转换为 HuggingFace 标准格式（先保存 slow tokenizer，确保兼容）
    print("转换为 HuggingFace LlamaTokenizer 格式...")
    tokenizer = LlamaTokenizer(
        vocab_file=model_prefix + ".model",
        legacy=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(args.output_dir)
    print(f"HF tokenizer 已保存到: {args.output_dir}")

    # 验证
    test_tokenizer(args.output_dir)


def test_tokenizer(tokenizer_dir):
    print("\n" + "=" * 60)
    print("Tokenizer 验证")
    print("=" * 60)

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir)
    print(f"词表大小(slow): {tokenizer.vocab_size}")
    print(f"特殊 token(slow): bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}, "
          f"pad={tokenizer.pad_token_id}, unk={tokenizer.unk_token_id}")

    fast_tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_dir)
    print(f"词表大小(fast): {fast_tokenizer.vocab_size}")
    print(f"特殊 token(fast): bos={fast_tokenizer.bos_token_id}, eos={fast_tokenizer.eos_token_id}, "
          f"pad={fast_tokenizer.pad_token_id}, unk={fast_tokenizer.unk_token_id}")

    test_texts = [
        "这是一个测试句子。",
        "今天天气真好，我想出去玩。",
        "小猫在阳光下睡觉。",
        "中国的首都是北京。",
        "The quick brown fox jumps over the lazy dog.",
        "我喜欢吃苹果和香蕉。",
        "123456789",
        "2026年4月21日",
    ]

    has_unk = False
    total_chars = 0
    total_tokens = 0

    for text in test_texts:
        ids = fast_tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        decoded = fast_tokenizer.decode(ids)
        n_tokens = len(ids)
        n_chars = len(text)
        ratio = n_tokens / n_chars

        unk_count = sum(1 for t in tokens if t == "<unk>")
        unk_flag = " ⚠️ UNK!" if unk_count > 0 else ""
        if unk_count > 0:
            has_unk = True

        total_chars += n_chars
        total_tokens += n_tokens

        print(f"\n  输入: {text}")
        print(f"  Tokens ({n_tokens}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"  解码: {decoded}")
        print(f"  Token/Char: {ratio:.3f}{unk_flag}")

    avg_ratio = total_tokens / total_chars
    print(f"\n  平均 Token/Char 比: {avg_ratio:.3f}")

    if has_unk:
        print("  ⚠️ 警告: 存在 UNK token，请检查 tokenizer 训练参数。")
    else:
        print("  ✅ 无 UNK token，tokenizer 质量良好。")

    # 验证 roundtrip
    roundtrip_text = "今天天气真好，我想出去玩。Hello World! 123"
    ids = fast_tokenizer.encode(roundtrip_text, add_special_tokens=False)
    decoded = fast_tokenizer.decode(ids)
    if decoded.strip() == roundtrip_text.strip():
        print("  ✅ Roundtrip 验证通过。")
    else:
        print(f"  ⚠️ Roundtrip 失败: '{decoded}' != '{roundtrip_text}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChineseBabyLM V4 - Tokenizer 训练")
    parser.add_argument("--input_file", type=str, default="../../data/processed_v3/train.txt")
    parser.add_argument("--output_dir", type=str, default="../../data/tokenizer_v4")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--model_type", type=str, default="bpe", choices=["bpe", "unigram"])
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--num_threads", type=int, default=min(64, os.cpu_count() or 16))
    parser.add_argument("--input_sentence_size", type=int, default=10000000)
    parser.add_argument("--num_sub_iterations", type=int, default=2)
    parser.add_argument("--shuffle_input_sentence", action="store_true", default=True)
    parser.add_argument("--no_shuffle_input_sentence", action="store_true")
    parser.add_argument("--train_extremely_large_corpus", action="store_true", default=True)
    parser.add_argument("--no_train_extremely_large_corpus", action="store_true")
    parser.add_argument("--skip_spm_train", action="store_true", default=False)
    args = parser.parse_args()

    if args.no_shuffle_input_sentence:
        args.shuffle_input_sentence = False
    if args.no_train_extremely_large_corpus:
        args.train_extremely_large_corpus = False

    train_tokenizer(args)
