"""
ChineseBabyLM V7 - Evaluation Script
Supports SPM tokenizer + LlamaForCausalLM
"""

import os
import math
import json
import argparse
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
import sentencepiece as spm


class V7Tokenizer:
    def __init__(self, model_dir):
        sp_path = os.path.join(model_dir, "spm.model")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_path)
        self.vocab_size = self.sp.get_piece_size()
        self.pad_token_id = self.sp.pad_id()
        self.eos_token_id = self.sp.eos_id()
        self.bos_token_id = self.sp.bos_id()
        self.unk_token_id = self.sp.unk_id()

        hf_tok_path = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(hf_tok_path):
            self.hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        else:
            self.hf_tokenizer = None

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp.decode(ids)

    def encode_for_model(self, text, return_tensors="pt"):
        ids = self.encode(text)
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids


def compute_ppl(
    model, tokenizer, val_file, block_size=1024, device="cuda", max_lines=None
):
    print(f"\nComputing PPL (block_size={block_size})...")
    eos_id = tokenizer.eos_token_id
    all_tokens = []
    line_count = 0

    with open(val_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line)
            all_tokens.extend(ids)
            all_tokens.append(eos_id)
            line_count += 1
            if max_lines and line_count >= max_lines:
                break

    print(f"  Total tokens: {len(all_tokens):,}, Lines: {line_count:,}")

    total_loss = 0.0
    total_tokens = 0
    num_chunks = 0

    model.eval()
    with torch.no_grad():
        for i in tqdm(
            range(0, len(all_tokens) - block_size, block_size), desc="Evaluating"
        ):
            chunk = all_tokens[i : i + block_size]
            if len(chunk) < block_size:
                break
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item() * block_size
            total_tokens += block_size
            num_chunks += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(min(avg_loss, 20))

    print(f"  Chunks: {num_chunks}, Tokens: {total_tokens:,}")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  PPL: {ppl:.2f}")
    return avg_loss, ppl


def test_generation(model, tokenizer, device, prompts, max_new_tokens=100):
    print("\n" + "=" * 60)
    print("Text Generation Test")
    print("=" * 60)

    use_hf = tokenizer.hf_tokenizer is not None

    for prompt in prompts:
        print(f"\n--- Prompt: {prompt!r} ---")

        if use_hf:
            input_ids = tokenizer.hf_tokenizer.encode(prompt, return_tensors="pt").to(
                device
            )
        else:
            input_ids = tokenizer.encode_for_model(prompt).to(device)

        model.config.pad_token_id = model.config.eos_token_id

        with torch.no_grad():
            greedy_out = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False
            )
        greedy_text = tokenizer.decode(greedy_out[0])
        print(f"  [Greedy]: {greedy_text}")

        with torch.no_grad():
            sample_out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
            )
        sample_text = tokenizer.decode(sample_out[0])
        print(f"  [Sample]: {sample_text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/babylm-v7/best_model")
    parser.add_argument("--val_file", type=str, default="data/processed_v7/val.txt")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--max_lines", type=int, default=None)
    parser.add_argument("--no_generate", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("ChineseBabyLM V7 - Evaluation")
    print("=" * 60)

    config_path = os.path.join(args.model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        print(
            f"  Architecture: LLaMA ({cfg.get('hidden_size')}d, {cfg.get('num_hidden_layers')}L)"
        )
        print(
            f"  Vocab: {cfg.get('vocab_size')}, Heads: {cfg.get('num_attention_heads')}Q/{cfg.get('num_key_value_heads')}KV"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print(f"\nLoading model: {args.model_path}")
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {params:.1f}M")

    tokenizer = V7Tokenizer(args.model_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    if os.path.exists(args.val_file):
        ppl_loss, ppl = compute_ppl(
            model, tokenizer, args.val_file, args.block_size, device, args.max_lines
        )

    if not args.no_generate:
        prompts = [
            "今天",
            "我喜欢",
            "从前有一座山",
            "老师说",
            "春天来了",
            "中国的首都是",
            "人工智能",
            "小猫在阳光下",
        ]
        test_generation(model, tokenizer, device, prompts)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
