"""
ChineseBabyLM V6 - Enhanced Data Cleaning Pipeline

Incremental improvements over V3 processed data:
1. Remove LaTeX/math formulas (key noise source)
2. Remove URLs and email addresses
3. Deduplicate consecutive punctuation
4. Stricter length filtering (15-300 chars)
5. Remove lines with excessive special characters
"""

import os
import re
import argparse
import hashlib
import logging
from collections import Counter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def remove_latex(text):
    text = re.sub(r"\$\$[^$]+\$\$", " ", text)
    text = re.sub(r"\$[^$]+\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\b", " ", text)
    text = re.sub(r"[{}\\]", " ", text)
    return text


def remove_urls_emails(text):
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    return text


def dedup_punctuation(text):
    text = re.sub(r"([，。、！？；：,\.!?;:])\1+", r"\1", text)
    text = re.sub(r"([。！？])\1{2,}", r"\1\1", text)
    text = re.sub(r"[\-=_~]{4,}", " ", text)
    text = re.sub(r"(\.\s*){4,}", " ", text)
    return text


def clean_control_chars(text):
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r" {3,}", " ", text)
    return text


def compute_special_char_ratio(line):
    if len(line) == 0:
        return 1.0
    special = sum(
        1
        for c in line
        if not (
            "\u4e00" <= c <= "\u9fff"
            or "\u3000" <= c <= "\u303f"
            or c.isalnum()
            or c in ' ，。、！？；：""（）【】《》—…· \n.,!?;:\'"()-[]<>'
        )
    )
    return special / len(line)


def is_quality_line(line, min_length=15, max_length=300, max_special_ratio=0.3):
    line = line.strip()
    if len(line) < min_length or len(line) > max_length:
        return False
    if compute_special_char_ratio(line) > max_special_ratio:
        return False
    chinese_chars = sum(1 for c in line if "\u4e00" <= c <= "\u9fff")
    if chinese_chars < min_length * 0.3:
        return False
    return True


def exact_dedup(lines):
    seen = set()
    deduped = []
    for line in lines:
        h = hashlib.md5(line.strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(line)
    return deduped


def clean_line(line):
    line = remove_latex(line)
    line = remove_urls_emails(line)
    line = dedup_punctuation(line)
    line = clean_control_chars(line)
    return line.strip()


def process_file(
    input_path,
    output_path,
    min_length=15,
    max_length=300,
    max_special_ratio=0.3,
    dedup=True,
):
    logger.info(f"Reading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    logger.info(f"  Raw lines: {len(raw_lines):,}")

    cleaned = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        cleaned_line = clean_line(line)
        if is_quality_line(cleaned_line, min_length, max_length, max_special_ratio):
            cleaned.append(cleaned_line)
    logger.info(f"  After cleaning & filtering: {len(cleaned):,}")

    if dedup:
        before = len(cleaned)
        cleaned = exact_dedup(cleaned)
        logger.info(
            f"  After exact dedup: {len(cleaned):,} (removed {before - len(cleaned):,})"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for line in cleaned:
            f.write(line + "\n")
    logger.info(f"  Written to: {output_path} ({len(cleaned):,} lines)")

    stats = {
        "raw": len(raw_lines),
        "cleaned": len(cleaned),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="V6 Enhanced Data Cleaning")
    parser.add_argument("--input", required=True, help="Input train.txt")
    parser.add_argument("--val_input", required=True, help="Input val.txt")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--min_length", type=int, default=15)
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--max_special_ratio", type=float, default=0.3)
    parser.add_argument("--no_dedup", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_out = os.path.join(args.output_dir, "train.txt")
    val_out = os.path.join(args.output_dir, "val.txt")

    train_stats = process_file(
        args.input,
        train_out,
        min_length=args.min_length,
        max_length=args.max_length,
        max_special_ratio=args.max_special_ratio,
        dedup=not args.no_dedup,
    )
    val_stats = process_file(
        args.val_input,
        val_out,
        min_length=args.min_length,
        max_length=args.max_length,
        max_special_ratio=args.max_special_ratio,
        dedup=False,
    )

    logger.info("=" * 60)
    logger.info("V6 Data Cleaning Summary:")
    logger.info(f"  Train: {train_stats['raw']:,} -> {train_stats['cleaned']:,} lines")
    logger.info(f"  Val:   {val_stats['raw']:,} -> {val_stats['cleaned']:,} lines")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
