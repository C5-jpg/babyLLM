"""
V14 Data Preparation Pipeline:
  1. PPL filtering (using V13 best model, max_ppl=250, min_ppl=3)
  2. MinHash dedup
  3. Quality filtering
  4. Hard example upsampling
"""
import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict

import sentencepiece as spm
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM


def compute_ppl_batch(model, input_ids, device):
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        loss = loss.view(shift_labels.size(0), -1)
        mask = shift_labels != 0
        ppl_per_sample = []
        for i in range(loss.size(0)):
            valid = mask[i].sum().item()
            if valid > 0:
                avg = loss[i][mask[i]].mean().item()
                ppl_per_sample.append(math.exp(min(avg, 20)))
            else:
                ppl_per_sample.append(float("inf"))
        return ppl_per_sample


def ppl_filter(lines, model_path, tokenizer_path, max_ppl=200, min_ppl=5, batch_size=32, max_length=512):
    print(f"  Loading model for PPL filtering: {model_path}")
    device = "cuda:0"
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(tokenizer_path, "spm.model"))
    eos_id = sp.eos_id()

    print(f"  Computing PPL for {len(lines):,} lines...")
    results = []
    for i in tqdm(range(0, len(lines), batch_size), desc="  PPL filter"):
        batch_lines = lines[i : i + batch_size]
        batch_ids = []
        for line in batch_lines:
            ids = sp.encode(line, out_type=int)
            ids = ids[: max_length - 1]
            ids.append(eos_id)
            if len(ids) < 4:
                ids = [0, eos_id, 0, eos_id]
            batch_ids.append(ids)

        max_len = max(len(x) for x in batch_ids)
        padded = []
        for ids in batch_ids:
            padded.append(ids + [0] * (max_len - len(ids)))

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        ppls = compute_ppl_batch(model, input_ids, device)

        for line, ppl in zip(batch_lines, ppls):
            results.append((line, ppl))

    kept = []
    removed = 0
    ppl_dist = defaultdict(int)
    for line, ppl in results:
        bucket = int(ppl) // 10 * 10
        ppl_dist[bucket] += 1
        if ppl <= max_ppl and ppl >= min_ppl:
            kept.append((line, ppl))
        else:
            removed += 1

    print(f"  PPL filter: {len(lines):,} → {len(kept):,} (removed {removed:,}, {100*removed/len(lines):.1f}%)")
    p95 = sorted([p for _, p in results])
    if p95:
        print(f"  PPL distribution: median={p95[len(p95)//2]:.1f}, p95={p95[int(len(p95)*0.95)]:.1f}, max={p95[-1]:.1f}")
    return kept


def minhash_dedup(lines, num_perm=128, threshold=0.7, ngrams=3):
    print(f"  MinHash dedup: {len(lines):,} lines, ngrams={ngrams}, threshold={threshold}")
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  datasketch not installed, using exact dedup only")
        return lines

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    seen = set()
    kept = []
    duplicates = 0

    for i, line in enumerate(tqdm(lines, desc="  MinHash")):
        tokens = line.lower().split()
        if len(tokens) < ngrams:
            kept.append(line)
            continue
        mh = MinHash(num_perm=num_perm)
        for j in range(len(tokens) - ngrams + 1):
            ng = " ".join(tokens[j : j + ngrams])
            mh.update(ng.encode("utf-8"))
        key = f"line_{i}"
        if lsh.query(mh):
            duplicates += 1
        else:
            lsh.insert(key, mh)
            kept.append(line)

    print(f"  MinHash dedup: {len(lines):,} → {len(kept):,} (removed {duplicates:,} near-duplicates)")
    return kept


def quality_filter(lines, min_chars=5, max_repeat_ratio=0.5, min_unique_ratio=0.1):
    kept = []
    removed = 0
    for line in lines:
        n = len(line)
        if n < min_chars:
            removed += 1
            continue
        char_counts = defaultdict(int)
        for c in line:
            char_counts[c] += 1
        max_count = max(char_counts.values())
        if max_count / n > max_repeat_ratio:
            removed += 1
            continue
        unique_ratio = len(char_counts) / n
        if unique_ratio < min_unique_ratio:
            removed += 1
            continue
        cn_chars = sum(1 for c in line if "\u4e00" <= c <= "\u9fff")
        if cn_chars / max(n, 1) < 0.3:
            removed += 1
            continue
        kept.append(line)
    print(f"  Quality filter: {len(lines):,} → {len(kept):,} (removed {removed:,})")
    return kept


def hard_upsample(lines_with_ppl, easy_thresh=30, hard_thresh=80, hard_factor=2):
    easy = []
    medium = []
    hard = []
    for line, ppl in lines_with_ppl:
        if ppl < easy_thresh:
            easy.append(line)
        elif ppl > hard_thresh:
            hard.append(line)
        else:
            medium.append(line)

    hard_upsampled = hard * hard_factor
    all_lines = easy + medium + hard_upsampled
    print(f"  Hard upsample: easy={len(easy):,}, medium={len(medium):,}, hard={len(hard):,} (×{hard_factor}) → total={len(all_lines):,}")
    return all_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="", help="Path to model for PPL filtering")
    parser.add_argument("--tokenizer_dir", default="")
    parser.add_argument("--max_ppl", type=float, default=200)
    parser.add_argument("--min_ppl", type=float, default=5)
    parser.add_argument("--skip_ppl_filter", action="store_true")
    parser.add_argument("--hard_upsample_factor", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "val"]:
        inpath = os.path.join(args.input_dir, f"{split}.txt")
        outpath = os.path.join(args.output_dir, f"{split}.txt")
        if not os.path.exists(inpath):
            print(f"  Skip {split} (not found)")
            continue

        with open(inpath, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"\n=== Processing {split}: {len(lines):,} lines ===")

        lines_with_ppl = None

        if split == "train" and not args.skip_ppl_filter and args.model_path and os.path.exists(args.model_path):
            lines_with_ppl = ppl_filter(
                lines, args.model_path, args.tokenizer_dir,
                max_ppl=args.max_ppl, min_ppl=args.min_ppl,
            )
            filtered_lines = [l for l, _ in lines_with_ppl]
        else:
            print(f"  Skipping PPL filter for {split}")
            filtered_lines = lines
            lines_with_ppl = None

        filtered_lines = quality_filter(filtered_lines)

        filtered_lines = minhash_dedup(filtered_lines)

        if split == "train" and lines_with_ppl is not None and args.hard_upsample_factor > 1:
            ppl_map = {l: p for l, p in lines_with_ppl}
            lines_ppl = [(l, ppl_map.get(l, 50)) for l in filtered_lines]
            final_lines = hard_upsample(lines_ppl, hard_factor=args.hard_upsample_factor)
        else:
            final_lines = filtered_lines

        import random
        random.seed(42)
        random.shuffle(final_lines)

        with open(outpath, "w", encoding="utf-8") as f:
            for l in final_lines:
                f.write(l + "\n")
        print(f"  Final {split}: {len(final_lines):,} lines → {outpath}")

    tokenizer_src = args.tokenizer_dir or os.path.join(args.input_dir, "..", "tokenizer_v7")
    for fname in os.listdir(args.input_dir):
        src = os.path.join(args.input_dir, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst) and fname.endswith((".model", ".json", ".txt")):
            import shutil
            if os.path.isdir(src):
                continue
            shutil.copy2(src, dst)

    print("\n=== Data preparation complete ===")


if __name__ == "__main__":
    main()
