"""
ChineseBabyLM V11 — Evaluation Script

Evaluates a trained model:
  - PPL computation with non-overlapping chunks
  - Text generation samples (greedy + sampling)
  - Supports both regular and EMA models
"""

import argparse
import json
import math
import os

import sentencepiece as spm
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import LlamaForCausalLM


def compute_ppl(model, sp, val_file, device, max_length=1024):
    model.eval()
    with open(val_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    all_tokens = []
    for line in lines:
        all_tokens.extend(sp.encode(line, out_type=int))
        all_tokens.append(sp.eos_id())

    print(f"  Total lines: {len(lines):,}")
    print(f"  Total tokens: {len(all_tokens):,}")

    chunks = 0
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(all_tokens) - max_length, max_length), desc="Evaluating"):
            chunk = all_tokens[i : i + max_length + 1]
            if len(chunk) < max_length + 1:
                continue
            input_ids = torch.tensor([chunk[:-1]], device=device)
            labels = torch.tensor([chunk[1:]], device=device)
            outputs = model(input_ids=input_ids)
            loss = F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1)
            )
            total_loss += loss.item() * max_length
            total_tokens += max_length
            chunks += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl, chunks, total_tokens


def generate_samples(model, sp, device, prompts, max_new_tokens=64):
    model.eval()
    results = []
    for prompt in prompts:
        input_ids = sp.encode(prompt, out_type=int)
        input_tensor = torch.tensor([input_ids], device=device)

        greedy_ids = model.generate(
            input_tensor, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=sp.eos_id(),
        )
        greedy_text = sp.decode(greedy_ids[0].tolist())

        sample_ids = model.generate(
            input_tensor, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.8, top_p=0.9,
            pad_token_id=sp.eos_id(),
        )
        sample_text = sp.decode(sample_ids[0].tolist())

        results.append({"prompt": prompt, "greedy": greedy_text, "sampled": sample_text})
    return results


def main():
    parser = argparse.ArgumentParser(description="ChineseBabyLM V11 Evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_path", default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("ChineseBabyLM V11 - Evaluation")
    print("=" * 60)

    print(f"  Loading model: {args.model_path}")
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to(device)

    if args.use_ema and args.ema_path and os.path.exists(args.ema_path):
        print(f"  Loading EMA state: {args.ema_path}")
        ema_state = torch.load(args.ema_path, map_location="cpu", weights_only=True)
        model_sd = model.state_dict()
        for k in ema_state:
            if k in model_sd:
                model_sd[k] = ema_state[k].to(model_sd[k].dtype)
        model.load_state_dict(model_sd, strict=False)
        print("  EMA weights applied")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,} ({total_params / 1e6:.1f}M)")

    sp = spm.SentencePieceProcessor()
    sp_model_path = os.path.join(args.model_path, "spm.model")
    if not os.path.exists(sp_model_path):
        sp_model_path = os.path.join(args.model_path, "tokenizer.model")
    sp.load(sp_model_path)
    print(f"  Vocab: {sp.get_piece_size():,}")
    print(f"  Mask token ID: {sp.piece_to_id('<mask>')}")

    print("\nComputing PPL...")
    avg_loss, ppl, chunks, total_tokens = compute_ppl(
        model, sp, args.val_file, device, args.max_length
    )
    print(f"\n  Chunks: {chunks:,}, Tokens: {total_tokens:,}")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  PPL: {ppl:.2f}")

    print("\n" + "=" * 60)
    print("Text Generation Test")
    print("=" * 60)
    prompts = [
        "今天", "我喜欢", "小明和小红", "在一个很远的",
        "春天来了，", "从前有一座山", "老师说", "中国的首都是",
        "小猫在阳光下", "人工智能正在",
    ]
    results = generate_samples(model, sp, device, prompts)
    for r in results:
        print(f"\n--- Prompt: '{r['prompt']}' ---")
        print(f"  [Greedy]: {r['greedy']}")
        print(f"  [Sample]: {r['sampled']}")

    result_data = {
        "loss": avg_loss, "ppl": ppl, "chunks": chunks,
        "total_tokens": total_tokens, "params": total_params,
        "generation": results, "model_path": args.model_path,
        "use_ema": args.use_ema,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_json}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
