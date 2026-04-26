"""
离线生成教师模型 Top-K Logits

使用 V2 best_model (或任何同词表模型) 作为教师，
对训练数据做前向推理，保存每个位置的 top-K logits 和 token indices。

参考: DistilQwen2.5 白盒蒸馏方案
"""
import os
import sys
import math
import argparse
import logging
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizerFast

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "v3"))
if V3_DIR not in sys.path:
    sys.path.append(V3_DIR)
from spm_tokenizer import SPMTokenizer
from train_v5 import DocumentAwareDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_tokenizer(tokenizer_dir):
    """加载 tokenizer"""
    try:
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_dir)
        test_ids = tokenizer.encode("今天天气真好", add_special_tokens=False)
        if tokenizer.vocab_size <= 1000 or len(test_ids) == 0:
            raise ValueError("Invalid tokenizer")
        return tokenizer
    except Exception:
        spm_path = os.path.join(tokenizer_dir, "spiece.model")
        if not os.path.exists(spm_path):
            spm_path = os.path.join(tokenizer_dir, "spm.model")
        return SPMTokenizer(spm_path)


def generate_teacher_logits(args):
    """生成教师模型的 top-K logits"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {mem:.1f} GB")

    # 加载 tokenizer
    tokenizer = load_tokenizer(args.tokenizer_dir)
    vocab_size = tokenizer.vocab_size
    logger.info(f"词表大小: {vocab_size}")

    # 加载教师模型
    logger.info(f"加载教师模型: {args.teacher_model_path}")
    teacher_model = LlamaForCausalLM.from_pretrained(
        args.teacher_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    teacher_model.eval()
    total_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info(f"教师模型参数量: {total_params:,} ({total_params / 1e6:.1f}M)")

    # 加载数据集
    sp_model_path = os.path.join(args.tokenizer_dir, "spiece.model")
    if not os.path.exists(sp_model_path):
        sp_model_path = os.path.join(args.tokenizer_dir, "spm.model")

    dataset = DocumentAwareDataset(
        tokenizer=tokenizer,
        file_path=args.data_file,
        block_size=args.block_size,
        bpe_dropout=0.0,
        sp_model_path=sp_model_path,
    )
    logger.info(f"数据集样本数: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 生成 logits
    all_topk_logits = []
    all_topk_indices = []
    total_samples = 0

    logger.info(f"开始生成教师 logits (top-{args.top_k})...")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Block size: {args.block_size}")
    logger.info(f"  输出目录: {args.output_dir}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="生成 logits")):
            input_ids = batch["input_ids"].to(device)

            # 前向传播
            outputs = teacher_model(input_ids=input_ids)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

            # 获取 top-K
            topk_values, topk_indices = torch.topk(
                logits.float(), k=args.top_k, dim=-1
            )  # [batch, seq_len, K]

            # 转为 float16 节省空间
            topk_values = topk_values.cpu().half().numpy()
            topk_indices = topk_indices.cpu().numpy()

            all_topk_logits.append(topk_values)
            all_topk_indices.append(topk_indices)
            total_samples += input_ids.shape[0]

            # 每 100 batch 保存一次中间结果
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"  已处理 {total_samples} 样本...")

            # 清理 GPU 缓存
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 合并所有结果
    all_topk_logits = np.concatenate(all_topk_logits, axis=0)
    all_topk_indices = np.concatenate(all_topk_indices, axis=0)

    logger.info(f"总样本数: {all_topk_logits.shape[0]:,}")
    logger.info(f"Logits 形状: {all_topk_logits.shape}")

    # 保存
    logits_path = os.path.join(args.output_dir, "teacher_logits.npy")
    indices_path = os.path.join(args.output_dir, "teacher_indices.npy")

    np.save(logits_path, all_topk_logits)
    np.save(indices_path, all_topk_indices)

    # 保存元信息
    meta = {
        "teacher_model": args.teacher_model_path,
        "data_file": args.data_file,
        "block_size": args.block_size,
        "top_k": args.top_k,
        "num_samples": all_topk_logits.shape[0],
        "seq_len": all_topk_logits.shape[1],
        "dtype": "float16",
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)

    # 文件大小
    logits_size = os.path.getsize(logits_path) / 1024**3
    indices_size = os.path.getsize(indices_path) / 1024**3
    logger.info(f"保存完成:")
    logger.info(f"  teacher_logits.npy: {logits_size:.2f} GB")
    logger.info(f"  teacher_indices.npy: {indices_size:.2f} GB")
    logger.info(f"  总计: {logits_size + indices_size:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="生成教师模型 Top-K Logits")
    parser.add_argument("--teacher_model_path", type=str, required=True,
                        help="教师模型路径 (如 output/babylm-llama-v2/best_model)")
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                        help="Tokenizer 目录")
    parser.add_argument("--data_file", type=str, required=True,
                        help="训练数据文件 (train.txt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    generate_teacher_logits(args)


if __name__ == "__main__":
    main()
