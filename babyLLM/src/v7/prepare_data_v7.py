"""
ChineseBabyLM V7 - Lightweight Data Cleaning
Preserves maximum data volume. Only removes CHILDES prefixes and empty lines.
"""

import os
import re
import argparse
import hashlib
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CHILDES_PREFIX = re.compile(
    r"^(TARGET_CHILD|MOTHER|FATHER|TEACHER|INVESTIGATOR|CHILD|INVEST|INV)"
    r"(?:_\w+)?:\s*"
)
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
MULTI_SPACE = re.compile(r" {3,}")


def clean_line(line):
    line = line.strip()
    if not line or len(line) < 3:
        return None
    line = CHILDES_PREFIX.sub("", line)
    line = CONTROL_CHARS.sub("", line)
    line = MULTI_SPACE.sub(" ", line)
    line = line.strip()
    if not line or len(line) < 3:
        return None
    return line


def process_file(input_path, output_path, dedup=True):
    logger.info(f"Reading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    logger.info(f"  Raw lines: {len(raw_lines):,}")

    cleaned = []
    for line in raw_lines:
        c = clean_line(line)
        if c:
            cleaned.append(c)
    logger.info(f"  After cleaning: {len(cleaned):,}")

    if dedup:
        seen = set()
        deduped = []
        for line in cleaned:
            h = hashlib.md5(line.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                deduped.append(line)
        logger.info(
            f"  After dedup: {len(deduped):,} (removed {len(cleaned) - len(deduped):,})"
        )
        cleaned = deduped

    with open(output_path, "w", encoding="utf-8") as f:
        for line in cleaned:
            f.write(line + "\n")
    logger.info(f"  Written: {output_path} ({len(cleaned):,} lines)")
    return len(cleaned)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--val_input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--no_dedup", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_n = process_file(
        args.input, os.path.join(args.output_dir, "train.txt"), dedup=not args.no_dedup
    )
    val_n = process_file(
        args.val_input, os.path.join(args.output_dir, "val.txt"), dedup=False
    )
    logger.info(f"Summary: train={train_n:,} val={val_n:,}")


if __name__ == "__main__":
    main()
