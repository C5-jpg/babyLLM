"""
ChineseBabyLM — Standardized Evaluation Script (V1-V14)

Evaluates ALL versions on the same held-out validation set with consistent metrics.
Handles architecture differences across versions:
  - V1 (GPT-2): Uses HuggingFace GPT2LMHeadModel
  - V2-V14 (LLaMA): Uses LlamaForCausalLM with SPM/LlamaTokenizer

Metrics: PPL, loss, token-level accuracy, generation quality
All results logged with ISO 8601 timestamps.

Usage:
  python eval_standardized.py --val_file /path/to/val.txt --output results.json
  python eval_standardized.py --val_file /path/to/val.txt --output results.json --model_paths '{"v13": "/path/to/v13/model"}'
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from tqdm import tqdm


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

VERSION_CONFIGS = {
    "v1": {"arch": "gpt2", "output_dir": "/mnt/sda/kehe/babyllm_output/babylm-gpt2", "best_subdir": "best_model"},
    "v2": {"arch": "llama", "output_dir": "/mnt/sda/kehe/babyllm_output/babylm-llama-v2", "best_subdir": "best_model", "tokenizer_dir": "/home/kehe/babyllm/babyLLM/data/tokenizer"},
    "v3": {"arch": "failed", "reason": "vocab_size mismatch crash, no checkpoints produced"},
    "v4": {"arch": "llama", "output_dir": "/mnt/sda/kehe/babyllm_output/babylm-llama-v4", "best_subdir": "best_model", "tokenizer_dir": "/home/kehe/babyllm/babyLLM/data/tokenizer_v3"},
    "v5": {"arch": "llama", "output_dir": "/mnt/sda/kehe/babyllm_output/babylm-llama-v5", "best_subdir": "best_model", "tokenizer_dir": "/home/kehe/babyllm/babyLLM/data/tokenizer_v2"},
    "v6": {"arch": "llama", "output_dir": "/mnt/sda/kehe/babyllm_output/babylm-llama-v6-stage3-kd", "best_subdir": "best_model", "tokenizer_dir": "/home/kehe/babyllm/babyLLM/data/tokenizer_v2"},
    "v7": {"arch": "llama", "output_dir": "/mnt/sda/kehe/babyllm_output/babylm-v7", "best_subdir": "best_model", "tokenizer_dir": "/home/kehe/babyllm/babyLLM/data/tokenizer_v7"},
    "v8": {"arch": "llama", "output_dir": "/mnt/sda/kehe/babyllm_output/babylm-v8/stage3_polish", "best_subdir": "best_model", "tokenizer_dir": "/home/kehe/babyllm/babyLLM/data/tokenizer_v7"},
    "v9": {"arch": "llama", "output_dir": "/mnt/sda/kehe/babyllm_output/babylm-v9/probe_clm_polish_lr5e-5", "best_subdir": "best_model", "tokenizer_dir": "/home/kehe/babyllm/babyLLM/data/tokenizer_v7"},
    "v10": {"arch": "llama", "output_dir": "/mnt/sda/kehe/babyllm_output/babylm-v10/stage3_polish", "best_subdir": "best_model", "tokenizer_dir": "/home/kehe/babyllm/babyLLM/data/tokenizer_v7"},
    "v11": {"arch": "llama", "output_dir": "/mnt/sda/kehe/babyllm_output/babylm-v11/stage5_self_distill", "best_subdir": "best_model_ema", "tokenizer_dir": "/home/kehe/babyllm/babyLLM/data/tokenizer_v7"},
    "v12": {"arch": "llama", "output_dir": "/mnt/sda/kehe/babyllm_output/babylm-v12/stage2_mntp", "best_subdir": "best_model_ema", "tokenizer_dir": "/home/kehe/babyllm/babyLLM/data/tokenizer_v7"},
    "v13": {"arch": "llama", "output_dir": "/mnt/sda/kehe/babyllm_output/babylm-v13/stage2_mntp", "best_subdir": "best_model_ema", "tokenizer_dir": "/home/kehe/babyllm/babyLLM/data/tokenizer_v7"},
    "v14": {"arch": "llama", "output_dir": "/mnt/sda/kehe/babyllm_output/babylm-v14/stage2_mntp", "best_subdir": "best_model_ema", "tokenizer_dir": "/home/kehe/babyllm/babyLLM/data/tokenizer_v7"},
}


def load_gpt2_model(model_path, device):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_llama_model(model_path, tokenizer_dir, device):
    from transformers import LlamaForCausalLM
    import sentencepiece as spm

    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    sp = spm.SentencePieceProcessor()
    spm_path = os.path.join(model_path, "spm.model")
    if not os.path.exists(spm_path):
        spm_path = os.path.join(tokenizer_dir, "spm.model")
    sp.load(spm_path)

    return model, sp


def compute_ppl_gpt2(model, tokenizer, val_file, device, max_length=1024):
    with open(val_file, "r", encoding="utf-8") as f:
        text = f.read()

    encoded = tokenizer(text, return_tensors="pt", truncation=False)
    all_tokens = encoded["input_ids"][0].tolist()

    chunks = 0
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(all_tokens) - max_length, max_length), desc="Evaluating GPT-2"):
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


def compute_ppl_llama(model, sp, val_file, device, max_length=1024):
    with open(val_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    all_tokens = []
    for line in lines:
        all_tokens.extend(sp.encode(line, out_type=int))
        all_tokens.append(sp.eos_id())

    chunks = 0
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(all_tokens) - max_length, max_length), desc="Evaluating LLaMA"):
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


def generate_samples_gpt2(model, tokenizer, device, prompts, max_new_tokens=64):
    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        greedy_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
        greedy_text = tokenizer.decode(greedy_ids[0])

        sample_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.8, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        sample_text = tokenizer.decode(sample_ids[0])
        results.append({"prompt": prompt, "greedy": greedy_text, "sampled": sample_text})
    return results


def generate_samples_llama(model, sp, device, prompts, max_new_tokens=64):
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


def evaluate_version(version, config, val_file, device, max_length=1024):
    timestamp = datetime.now(timezone.utc).isoformat()

    if config["arch"] == "failed":
        return {
            "version": version,
            "status": "failed",
            "reason": config.get("reason", "unknown"),
            "timestamp": timestamp,
        }

    model_path = os.path.join(config["output_dir"], config["best_subdir"])
    if not os.path.isdir(model_path):
        return {
            "version": version,
            "status": "missing",
            "reason": f"Model directory not found: {model_path}",
            "timestamp": timestamp,
        }

    print(f"\n{'='*60}")
    print(f"  Evaluating {version}: {model_path}")
    print(f"{'='*60}")

    try:
        if config["arch"] == "gpt2":
            model, tokenizer = load_gpt2_model(model_path, device)
            params = sum(p.numel() for p in model.parameters())
            loss, ppl, chunks, total_tokens = compute_ppl_gpt2(model, tokenizer, val_file, device, max_length)
            generation = generate_samples_gpt2(model, tokenizer, device, STANDARD_PROMPTS)
        else:
            model, sp = load_llama_model(model_path, config.get("tokenizer_dir", ""), device)
            params = sum(p.numel() for p in model.parameters())
            loss, ppl, chunks, total_tokens = compute_ppl_llama(model, sp, val_file, device, max_length)
            generation = generate_samples_llama(model, sp, device, STANDARD_PROMPTS)

        result = {
            "version": version,
            "status": "success",
            "timestamp": timestamp,
            "arch": config["arch"],
            "params": params,
            "loss": loss,
            "ppl": ppl,
            "chunks": chunks,
            "total_tokens": total_tokens,
            "model_path": model_path,
            "generation": generation,
        }

        print(f"  {version}: loss={loss:.4f}, ppl={ppl:.2f}, params={params:,}")
        return result

    except Exception as e:
        return {
            "version": version,
            "status": "error",
            "reason": str(e),
            "timestamp": timestamp,
        }


def main():
    parser = argparse.ArgumentParser(description="Standardized evaluation across all BabyLM versions")
    parser.add_argument("--val_file", required=True, help="Held-out validation set")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--versions", default="all", help="Comma-separated versions to eval (default: all)")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    timestamp = datetime.now(timezone.utc).isoformat()

    if args.versions == "all":
        versions = list(VERSION_CONFIGS.keys())
    else:
        versions = [v.strip() for v in args.versions.split(",")]

    print(f"Standardized evaluation starting at {timestamp}")
    print(f"Validation file: {args.val_file}")
    print(f"Versions to evaluate: {versions}")

    results = {
        "metadata": {
            "timestamp": timestamp,
            "val_file": args.val_file,
            "max_length": args.max_length,
            "device": str(device),
            "versions_evaluated": versions,
        },
        "results": {},
    }

    for version in versions:
        if version not in VERSION_CONFIGS:
            print(f"WARNING: Unknown version '{version}', skipping")
            continue
        result = evaluate_version(version, VERSION_CONFIGS[version], args.val_file, device, args.max_length)
        results["results"][version] = result

    end_timestamp = datetime.now(timezone.utc).isoformat()
    results["metadata"]["completed_at"] = end_timestamp

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Evaluation complete. Results saved to: {args.output}")
    print(f"{'='*60}")

    print("\nSummary:")
    print(f"{'Version':<8} {'Status':<10} {'PPL':<10} {'Loss':<10} {'Params':<12}")
    print("-" * 50)
    for version in versions:
        r = results["results"].get(version, {})
        status = r.get("status", "N/A")
        ppl = f"{r['ppl']:.2f}" if "ppl" in r else "N/A"
        loss = f"{r['loss']:.4f}" if "loss" in r else "N/A"
        params = f"{r['params']:,}" if "params" in r else "N/A"
        print(f"{version:<8} {status:<10} {ppl:<10} {loss:<10} {params:<12}")


if __name__ == "__main__":
    main()
