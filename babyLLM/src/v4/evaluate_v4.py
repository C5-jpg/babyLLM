"""
ChineseBabyLM V4 - 评测脚本
功能:
1. 计算验证集 PPL
2. 文本生成质量测试
3. 调用官方评测 pipeline (如已安装)
"""
import os
import sys
import math
import argparse
import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast, set_seed
from tqdm import tqdm

set_seed(42)


def compute_ppl(model, tokenizer, file_path, block_size=1024, batch_size=8, device="cuda"):
    """计算验证集 Perplexity"""
    print(f"\n计算 PPL: {file_path}")
    print(f"  block_size={block_size}, batch_size={batch_size}")

    # Tokenize
    all_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)

    print(f"  总 tokens: {len(all_ids):,}")

    # 分块
    chunks = []
    for i in range(0, len(all_ids) - block_size, block_size):
        chunks.append(all_ids[i:i + block_size])
    print(f"  评测 chunks: {len(chunks):,}")

    # 计算 loss
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(chunks), batch_size), desc="Computing PPL"):
            batch_chunks = chunks[i:i + batch_size]
            input_ids = torch.tensor(batch_chunks, dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            bs, seq_len = input_ids.shape
            total_loss += outputs.loss.item() * bs * seq_len
            total_tokens += bs * seq_len

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))
    print(f"\n  Loss: {avg_loss:.4f}")
    print(f"  PPL: {ppl:.2f}")
    return avg_loss, ppl


def test_generation(model, tokenizer, device="cuda"):
    """文本生成质量测试"""
    print("\n" + "=" * 60)
    print("文本生成测试")
    print("=" * 60)

    prompts = [
        "今天",
        "中国的首都是",
        "小猫在",
        "从前有一个",
        "学习语言",
    ]

    model.eval()
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        print(f"\n  Prompt: {prompt}")

        # Greedy
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=50,
                do_sample=False, repetition_penalty=1.2,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"  Greedy: {text}")

        # Sampling
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=50,
                do_sample=True, temperature=0.8,
                top_k=50, top_p=0.9,
                repetition_penalty=1.2,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"  Sample: {text}")


def main():
    parser = argparse.ArgumentParser(description="ChineseBabyLM V4 - 评测")
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录")
    parser.add_argument("--val_file", type=str, default=None, help="验证集文件")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    args = parser.parse_args()

    print(f"加载模型: {args.model_dir}")
    tokenizer = LlamaTokenizerFast.from_pretrained(args.model_dir)
    model = LlamaForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,} ({total_params / 1e6:.1f}M)")

    if not args.skip_ppl and args.val_file:
        compute_ppl(model, tokenizer, args.val_file,
                    args.block_size, args.batch_size, args.device)

    if not args.skip_generation:
        test_generation(model, tokenizer, args.device)


if __name__ == "__main__":
    main()
