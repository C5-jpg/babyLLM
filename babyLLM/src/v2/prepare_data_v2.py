"""
ChineseBabyLM V2 - 数据预处理脚本
功能:
1. 去重（精确匹配 + MinHash 近似去重）
2. 文本清洗（HTML标签、异常字符、过短/过长行）
3. 创建独立 train/val 分割 (95%/5%)
4. 统计信息输出
"""
import os
import re
import hashlib
import argparse
from collections import defaultdict
from tqdm import tqdm


def clean_text(text):
    """清洗单行文本"""
    # 去除首尾空白
    text = text.strip()
    if not text:
        return None
    
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 去除 URL
    text = re.sub(r'https?://\S+', '', text)
    
    # 去除过多连续空白
    text = re.sub(r'\s+', ' ', text)
    
    # 去除过多连续标点（超过3个相同标点）
    text = re.sub(r'([。，！？、；：""''（）\.\,\!\?\;\:\"\'\(\)])\1{3,}', r'\1\1', text)
    
    # 去除控制字符（保留换行和制表符）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # 再次 strip
    text = text.strip()
    
    # 过滤过短文本（少于5个字符）
    if len(text) < 5:
        return None
    
    # 过滤过长文本（超过10000个字符，可能是异常数据）
    if len(text) > 10000:
        return None
    
    # 过滤几乎全是标点的文本
    alpha_chars = len(re.sub(r'[\s\W]+', '', text))
    if alpha_chars < len(text) * 0.3:
        return None
    
    return text


def exact_dedup(lines):
    """精确去重"""
    seen = set()
    unique_lines = []
    dup_count = 0
    for line in lines:
        # 使用 normalized version 作为 key
        key = line.strip().lower()
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        unique_lines.append(line)
    return unique_lines, dup_count


def minhash_dedup(lines, num_perm=128, threshold=0.8, seed=42):
    """MinHash 近似去重（无需额外依赖）"""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("⚠️ datasketch 未安装，跳过 MinHash 去重。安装: pip install datasketch")
        return lines, 0
    
    # 使用字符级 3-gram
    def get_ngrams(text, n=3):
        text = text.lower()
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_lines = []
    dup_count = 0
    
    for i, line in enumerate(tqdm(lines, desc="MinHash 去重")):
        ngrams = get_ngrams(line)
        if len(ngrams) == 0:
            continue
        
        mh = MinHash(num_perm=num_perm, seed=seed)
        for ng in ngrams:
            mh.update(ng.encode('utf-8'))
        
        # 检查是否与已有文本过于相似
        key = f"doc_{i}"
        result = lsh.query(mh)
        
        if result:
            dup_count += 1
        else:
            lsh.insert(key, mh)
            unique_lines.append(line)
    
    return unique_lines, dup_count


def process_data(input_file, output_dir, use_minhash=True):
    """完整的数据处理流程"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ChineseBabyLM V2 - 数据预处理")
    print("=" * 60)
    
    # 读取原始数据
    print(f"\n📥 读取数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    print(f"   原始行数: {len(raw_lines):,}")
    
    # Step 1: 文本清洗
    print("\n🧹 清洗文本...")
    cleaned_lines = []
    removed_clean = 0
    for line in tqdm(raw_lines, desc="清洗中"):
        cleaned = clean_text(line)
        if cleaned is not None:
            cleaned_lines.append(cleaned)
        else:
            removed_clean += 1
    print(f"   清洗后: {len(cleaned_lines):,} 行 (移除 {removed_clean:,})")
    
    # Step 2: 精确去重
    print("\n🔍 精确去重...")
    unique_lines, exact_dups = exact_dedup(cleaned_lines)
    print(f"   去重后: {len(unique_lines):,} 行 (精确重复: {exact_dups:,})")
    
    # Step 3: MinHash 近似去重
    if use_minhash:
        print("\n🔍 MinHash 近似去重...")
        unique_lines, approx_dups = minhash_dedup(unique_lines, threshold=0.85)
        print(f"   去重后: {len(unique_lines):,} 行 (近似重复: {approx_dups:,})")
    
    # Step 4: 打乱数据
    import random
    random.seed(42)
    random.shuffle(unique_lines)
    
    # Step 5: Train/Val 分割
    total = len(unique_lines)
    val_size = max(int(total * 0.005), 1000)  # 0.5% 作为验证集，至少1000行
    train_size = total - val_size
    
    train_lines = unique_lines[:train_size]
    val_lines = unique_lines[train_size:]
    
    print(f"\n📊 数据分割:")
    print(f"   训练集: {train_size:,} 行")
    print(f"   验证集: {val_size:,} 行")
    
    # 保存
    train_file = os.path.join(output_dir, "train.txt")
    val_file = os.path.join(output_dir, "val.txt")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for line in val_lines:
            f.write(line + '\n')
    
    # 统计信息
    train_chars = sum(len(l) for l in train_lines)
    val_chars = sum(len(l) for l in val_lines)
    
    print(f"\n📈 最终统计:")
    print(f"   训练集字符数: {train_chars:,}")
    print(f"   验证集字符数: {val_chars:,}")
    print(f"   总字符数: {(train_chars + val_chars):,}")
    print(f"   平均行长度: {train_chars / train_size:.1f} 字符")
    
    print(f"\n💾 保存完成:")
    print(f"   训练集: {train_file}")
    print(f"   验证集: {val_file}")
    print("=" * 60)
    
    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(description="ChineseBabyLM V2 数据预处理")
    parser.add_argument("--input", type=str, default="data/processed/all.txt",
                        help="输入文本文件")
    parser.add_argument("--output_dir", type=str, default="data/processed_v2",
                        help="输出目录")
    parser.add_argument("--no_minhash", action="store_true",
                        help="跳过 MinHash 去重（较慢）")
    args = parser.parse_args()
    
    process_data(args.input, args.output_dir, use_minhash=not args.no_minhash)


if __name__ == "__main__":
    main()