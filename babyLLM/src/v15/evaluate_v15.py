"""
ChineseBabyLM V15 — Evaluation Script

Evaluates a trained model:
  - PPL computation with non-overlapping chunks
  - Text generation samples (greedy + sampling)
  - Supports both regular and multi-scale EMA models
  - Results logged with ISO 8601 timestamps
"""

import argparse
import json
import math
import os
from datetime import datetime, timezone

import sentencepiece as spm
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import LlamaForCausalLM


STANDARD_PROMPTS = [
    "今天",
    "我喜欢",
    "小明和小红",
    "在一个很远的",
    "春天来了，",
    "从前有一座山",
    "老师说",
    "中国的首都是",
    "小猫在阳光下",
    "人工智能正在",
]


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


def compute_token_accuracy(model, sp, val_file, device, max_length=1024, max_batches=100):
    model.eval()
    with open(val_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    all_tokens = []
    for line in lines:
        all_tokens.extend(sp.encode(line, out_type=int))
        all_tokens.append(sp.eos_id())

    correct = 0
    total = 0
    batches = 0

    with torch.no_grad():
        for i in range(0, len(all_tokens) - max_length, max_length):
            chunk = all_tokens[i : i + max_length + 1]
            if len(chunk) < max_length + 1:
                continue
            input_ids = torch.tensor([chunk[:-1]], device=device)
            labels = torch.tensor([chunk[1:]], device=device)
            outputs = model(input_ids=input_ids)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            batches += 1
            if batches >= max_batches:
                break

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


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
    parser = argparse.ArgumentParser(description="ChineseBabyLM V15 Evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_path", default="")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute_accuracy", action="store_true", help="Compute token-level accuracy")
    args = parser.parse_args()

    device = torch.device(args.device)
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"  Loading model from: {args.model_path}")
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())

    tokenizer_dir = args.model_path
    sp = spm.SentencePieceProcessor()
    spm_path = os.path.join(tokenizer_dir, "spm.model")
    if not os.path.exists(spm_path):
        for parent in [os.path.dirname(args.model_path), args.model_path]:
            candidate = os.path.join(parent, "spm.model")
            if os.path.exists(candidate):
                spm_path = candidate
                break
    sp.load(spm_path)

    if args.use_ema and args.ema_path and os.path.exists(args.ema_path):
        print(f"  Loading EMA weights from: {args.ema_path}")
        ema_state = torch.load(args.ema_path, map_location="cpu", weights_only=True)
        if isinstance(ema_state, dict) and "0.999" in ema_state:
            ema_weights = ema_state["0.999"]
        else:
            ema_weights = ema_state
        model.load_state_dict(ema_weights, strict=False)

    print("  Computing PPL...")
    loss, ppl, chunks, total_tokens = compute_ppl(model, sp, args.val_file, device, args.max_length)
    print(f"  Loss: {loss:.4f}, PPL: {ppl:.2f}, Chunks: {chunks}, Tokens: {total_tokens:,}")

    accuracy = None
    if args.compute_accuracy:
        print("  Computing token accuracy...")
        accuracy, correct, total = compute_token_accuracy(model, sp, args.val_file, device, args.max_length)
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

    print("  Generating samples...")
    generation = generate_samples(model, sp, device, STANDARD_PROMPTS)

    result = {
        "timestamp": timestamp,
        "version": "v15",
        "loss": loss,
        "ppl": ppl,
        "chunks": chunks,
        "total_tokens": total_tokens,
        "params": params,
        "model_path": args.model_path,
        "use_ema": args.use_ema,
        "generation": generation,
    }
    if accuracy is not None:
        result["token_accuracy"] = accuracy

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
