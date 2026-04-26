"""
ChineseBabyLM V7 - Train 8K SPM Tokenizer

Small vocabulary (8000) optimized for ~70M token Chinese dataset.
Includes <mask> token for MNTP training.
"""

import os
import argparse
import sentencepiece as spm


def train_tokenizer(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Training SPM tokenizer...")
    print(f"  Input: {args.input_file}")
    print(f"  Vocab size: {args.vocab_size}")
    print(f"  Output: {args.output_dir}")

    model_prefix = os.path.join(args.output_dir, "spm")

    spm.SentencePieceTrainer.train(
        input=args.input_file,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=["<mask>"],
        split_by_unicode_script=True,
        split_by_number=True,
        split_by_whitespace=True,
        treat_whitespace_as_suffix=False,
        allow_whitespace_only_pieces=True,
        max_sentence_length=2048,
        num_threads=8,
        input_sentence_size=500000,
        shuffle_input_sentence=True,
    )

    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    print(f"\nTokenizer trained successfully!")
    print(f"  Vocab size: {sp.get_piece_size()}")
    print(f"  <pad> id: {sp.pad_id()}")
    print(f"  <unk> id: {sp.unk_id()}")
    print(f"  <s> id: {sp.bos_id()}")
    print(f"  </s> id: {sp.eos_id()}")
    print(f"  <mask> id: {sp.piece_to_id('<mask>')}")

    test_sentences = [
        "今天天气真好",
        "我喜欢学习中文",
        "人工智能改变了世界",
        "这个方法的效果非常好",
    ]
    print(f"\nTest tokenization:")
    for sent in test_sentences:
        ids = sp.encode(sent)
        pieces = sp.encode(sent, out_type=str)
        print(f"  {sent} → {len(ids)} tokens: {' '.join(pieces[:20])}")

    from transformers import LlamaTokenizerFast
    import json

    tokenizer_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
        "model_type": "llama",
        "tokenizer_class": "LlamaTokenizerFast",
    }
    with open(os.path.join(args.output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    print(f"\nFiles saved to {args.output_dir}:")
    for fname in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, fname)
        print(f"  {fname}: {os.path.getsize(fpath) / 1024:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/processed_v3/train.txt")
    parser.add_argument("--output_dir", type=str, default="data/tokenizer_v7")
    parser.add_argument("--vocab_size", type=int, default=8000)
    args = parser.parse_args()
    train_tokenizer(args)
