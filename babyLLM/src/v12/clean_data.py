"""
V12 Data Cleaning: dedup + quality filter
"""
import argparse
import hashlib
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_chars", type=int, default=5)
    parser.add_argument("--max_chars", type=int, default=10000)
    parser.add_argument("--max_repeat_ratio", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "val"]:
        inpath = os.path.join(args.input_dir, f"{split}.txt")
        outpath = os.path.join(args.output_dir, f"{split}.txt")
        if not os.path.exists(inpath):
            print(f"  Skip {inpath} (not found)")
            continue

        seen_hashes = set()
        total = 0
        kept = 0
        out_lines = []

        with open(inpath, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                text = line.strip()
                n = len(text)

                if n < args.min_chars or n > args.max_chars:
                    continue

                char_counts = {}
                for c in text:
                    char_counts[c] = char_counts.get(c, 0) + 1
                max_count = max(char_counts.values())
                if max_count / n > args.max_repeat_ratio:
                    continue

                h = hashlib.md5(text.encode()).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                out_lines.append(text)
                kept += 1

        with open(outpath, "w", encoding="utf-8") as f:
            for l in out_lines:
                f.write(l + "\n")

        print(f"  {split}: {total:,} → {kept:,} (removed {total - kept:,}, {100*(total-kept)/total:.1f}%)")

    for fname in os.listdir(args.input_dir):
        src = os.path.join(args.input_dir, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")


if __name__ == "__main__":
    main()
