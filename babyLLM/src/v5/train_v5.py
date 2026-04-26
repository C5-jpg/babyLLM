"""
ChineseBabyLM V5 - 小模型 + 知识蒸馏训练脚本

核心改进:
1. 参数量匹配数据规模: ~60M 参数 (d_model=512, 12层)
2. 两阶段训练: Phase 1 标准 CE 预训练 → Phase 2 白盒 KD 微调
3. 白盒知识蒸馏: 使用教师模型 top-K logits 指导学生模型
4. 复用 V3 SentencePiece tokenizer (32K BPE)
5. RMSNorm eps=1e-5 提升数值稳定性
6. rope_theta=10000 回归标准值
7. Dropout 退火: 0.05 → 0

参考: DistilQwen2.5 (Wang et al., ACL 2025)
"""
import os
import sys
import math
import time
import json
import logging
import argparse
import shutil
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator
import wandb
from tqdm import tqdm
import sentencepiece as spm
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "v3"))
if V3_DIR not in sys.path:
    sys.path.append(V3_DIR)
from spm_tokenizer import SPMTokenizer

# ============================================================
# 日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据集 - 文档感知序列构造 + BPE Dropout
# ============================================================
class DocumentAwareDataset(Dataset):
    """文档感知序列构造 + SentencePiece BPE Dropout"""

    def __init__(self, tokenizer, file_path, block_size=1024,
                 bpe_dropout=0.0, sp_model_path=None, encode_batch_size=4096):
        assert os.path.isfile(file_path), f"文件不存在: {file_path}"
        self.block_size = block_size
        self.bpe_dropout = bpe_dropout
        self.sp_model_path = sp_model_path
        self.sp = None
        self.encode_batch_size = encode_batch_size

        if bpe_dropout > 0 and sp_model_path and os.path.exists(sp_model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(sp_model_path)
            logger.info(f"BPE Dropout 已启用: alpha={bpe_dropout}")

        logger.info(f"读取数据: {file_path}, block_size={block_size}")

        self.lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.lines.append(line)

        logger.info(f"文档数: {len(self.lines):,}")

        self.tokenizer = tokenizer
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id or 0
        self.examples = []
        self._tokenize(use_dropout=False)
        self._fixed_length = len(self.examples)
        logger.info(f"训练样本数: {len(self.examples):,}")

    def _tokenize(self, use_dropout=False):
        all_token_ids = []

        if use_dropout and self.sp is not None:
            for line in self.lines:
                ids = self.sp.encode(
                    line,
                    enable_sampling=True,
                    alpha=self.bpe_dropout,
                    nbest_size=-1,
                )
                all_token_ids.extend(ids)
                all_token_ids.append(self.eos_id)
        else:
            if hasattr(self.tokenizer, "__call__"):
                for i in range(0, len(self.lines), self.encode_batch_size):
                    batch_lines = self.lines[i:i + self.encode_batch_size]
                    batch_encoded = self.tokenizer(
                        batch_lines,
                        add_special_tokens=False,
                        truncation=False,
                    )["input_ids"]
                    for ids in batch_encoded:
                        all_token_ids.extend(ids)
                        all_token_ids.append(self.eos_id)
            else:
                for line in self.lines:
                    ids = self.tokenizer.encode(line, add_special_tokens=False)
                    all_token_ids.extend(ids)
                    all_token_ids.append(self.eos_id)

        # 50% 重叠滑动窗口
        stride = self.block_size // 2
        new_examples = []
        i = 0
        while i + self.block_size + 1 <= len(all_token_ids):
            new_examples.append(all_token_ids[i:i + self.block_size + 1])
            i += stride

        if len(new_examples) == 0 or (i < len(all_token_ids) - 1 and len(all_token_ids) - i > 1):
            remaining = all_token_ids[i:]
            if len(remaining) > 1:
                padded = remaining + [self.pad_id] * (self.block_size + 1 - len(remaining))
                new_examples.append(padded[:self.block_size + 1])

        self.examples = new_examples

    def retokenize(self):
        if self.sp is None:
            return
        self._tokenize(use_dropout=True)
        if len(self.examples) < self._fixed_length:
            pad = self.examples[-1] if self.examples else [0] * (self.block_size + 1)
            self.examples.extend([pad] * (self._fixed_length - len(self.examples)))
        elif len(self.examples) > self._fixed_length:
            self.examples = self.examples[:self._fixed_length]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


# ============================================================
# 安全保存工具函数
# ============================================================
def check_disk_space(path, min_free_gb=5.0):
    """检查磁盘空间是否充足"""
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        logger.warning(
            f"⚠️ 磁盘空间不足! 可用: {free_gb:.1f}GB, 最低需要: {min_free_gb}GB. "
            f"请将输出目录软链接到机械硬盘."
        )
        return False
    logger.info(f"磁盘空间检查通过: {free_gb:.1f}GB 可用 (路径: {path})")
    return True


def safe_save_model(model, save_dir, accelerator, config=None, tokenizer=None):
    """
    安全保存模型，带磁盘空间检查和保存验证。
    如果 save_pretrained() 失败（如磁盘空间不足），使用 torch.save 作为 fallback。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 检查磁盘空间
    total_params = sum(p.numel() for p in model.parameters())
    required_bytes = total_params * 4 * 2  # FP32 × 2 安全系数
    required_gb = required_bytes / (1024**3)
    usage = shutil.disk_usage(save_dir)
    free_gb = usage.free / (1024**3)

    if free_gb < required_gb:
        logger.error(
            f"🔴 磁盘空间不足! 可用: {free_gb:.2f}GB, 估算需要: {required_gb:.2f}GB"
        )
        logger.error(f"保存路径: {save_dir}")
        logger.error("请将输出目录软链接到空间更大的磁盘!")

    # 保存配置和 tokenizer
    if config is not None:
        config.save_pretrained(save_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)

    # 尝试 save_pretrained
    try:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(save_dir)
    except Exception as e:
        logger.warning(f"save_pretrained() 失败: {e}, 尝试 torch.save fallback...")

    # 验证权重文件是否存在
    weight_files = (
        glob.glob(os.path.join(save_dir, "model.safetensors"))
        + glob.glob(os.path.join(save_dir, "model-*.safetensors"))
        + glob.glob(os.path.join(save_dir, "pytorch_model.bin"))
    )

    if not weight_files:
        logger.warning("权重文件未找到，使用 torch.save fallback...")
        try:
            state_dict = accelerator.get_state_dict(model)
            fallback_path = os.path.join(save_dir, "pytorch_model.bin")
            torch.save(state_dict, fallback_path)
            weight_files = [fallback_path]
            logger.info(f"Fallback 保存成功: {fallback_path}")
        except Exception as e:
            logger.error(f"🔴 模型保存完全失败: {e}")
            raise RuntimeError(f"模型保存失败! 请检查磁盘空间: {save_dir}")

    total_size = sum(os.path.getsize(f) for f in weight_files)
    logger.info(
        f"✅ 模型保存验证通过: {len(weight_files)} 个权重文件, "
        f"总大小: {total_size / 1024**2:.1f}MB, 路径: {save_dir}"
    )
    return True


# ============================================================
# KD 数据集 - 加载教师 logits
# ============================================================
class KDDataset(Dataset):
    """知识蒸馏数据集: 同时提供 input_ids, labels 和教师 top-K logits"""

    def __init__(self, base_dataset, teacher_logits_dir, top_k=10):
        self.base_dataset = base_dataset
        self.top_k = top_k
        self.teacher_logits = None
        self.teacher_indices = None

        # 尝试加载预生成的教师 logits
        logits_path = os.path.join(teacher_logits_dir, "teacher_logits.npy")
        indices_path = os.path.join(teacher_logits_dir, "teacher_indices.npy")

        if os.path.exists(logits_path) and os.path.exists(indices_path):
            logger.info(f"加载教师 logits: {logits_path}")
            self.teacher_logits = np.load(logits_path, mmap_mode="r")
            self.teacher_indices = np.load(indices_path, mmap_mode="r")
            logger.info(f"教师 logits 形状: {self.teacher_logits.shape}")
        else:
            logger.warning(f"教师 logits 不存在: {logits_path}")
            logger.warning("将以纯 CE 模式训练（无 KD）")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        result = {
            "input_ids": item["input_ids"],
            "labels": item["labels"],
        }

        if self.teacher_logits is not None and idx < len(self.teacher_logits):
            result["teacher_logits"] = torch.tensor(
                self.teacher_logits[idx], dtype=torch.float16
            )
            result["teacher_indices"] = torch.tensor(
                self.teacher_indices[idx], dtype=torch.long
            )
        else:
            # 如果没有教师 logits，提供空的占位符
            seq_len = item["input_ids"].shape[0]
            result["teacher_logits"] = torch.zeros(seq_len, self.top_k, dtype=torch.float16)
            result["teacher_indices"] = torch.zeros(seq_len, self.top_k, dtype=torch.long)

        return result


# ============================================================
# 评测
# ============================================================
def evaluate(model, val_loader, accelerator):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中", ncols=100,
                          disable=not accelerator.is_main_process):
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            input_ids = batch["input_ids"]
            if input_ids.dim() == 1:
                bs, seq_len = 1, input_ids.shape[0]
            else:
                bs, seq_len = input_ids.shape
            total_loss += outputs.loss.item() * bs * seq_len
            total_tokens += bs * seq_len

    stats = torch.tensor([total_loss, total_tokens],
                         dtype=torch.float32, device=accelerator.device)
    gathered = accelerator.gather(stats).view(-1, 2)
    total_loss_sum = gathered[:, 0].sum().item()
    total_tokens_sum = gathered[:, 1].sum().item()
    avg_loss = total_loss_sum / total_tokens_sum if total_tokens_sum > 0 else float('inf')
    model.train()
    return avg_loss


# ============================================================
# Dropout 退火
# ============================================================
def set_dropout(model, dropout_p):
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_p
            count += 1
    return count


# ============================================================
# KD 损失函数
# ============================================================
def compute_kd_loss(student_logits, teacher_logits_topk, teacher_indices_topk,
                    labels, temperature=2.0, lambda_ce=0.3, lambda_kd=0.7,
                    top_k=10, ignore_index=-100):
    """
    白盒知识蒸馏损失函数 (参考 DistilQwen2.5)

    student_logits: [batch, seq_len, vocab_size]
    teacher_logits_topk: [batch, seq_len, K] - 教师模型 top-K logits
    teacher_indices_topk: [batch, seq_len, K] - 教师模型 top-K token indices
    labels: [batch, seq_len]
    """
    batch_size, seq_len, vocab_size = student_logits.shape

    # 1. 标准 CE 损失
    ce_loss = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=ignore_index,
    )

    # 2. KD 损失
    # 从学生 logits 中提取教师 top-K 位置的 logits
    flat_student = student_logits.view(-1, vocab_size)  # [B*S, V]
    flat_teacher_indices = teacher_indices_topk.view(-1, top_k)  # [B*S, K]

    # Gather student logits at teacher's top-K positions
    student_topk = torch.gather(
        flat_student, dim=1, index=flat_teacher_indices
    )  # [B*S, K]

    # 温度缩放
    teacher_probs = F.softmax(
        teacher_logits_topk.view(-1, top_k).float() / temperature, dim=-1
    )
    student_log_probs = F.log_softmax(
        student_topk / temperature, dim=-1
    )

    # KL 散度 (乘以 T^2 保持梯度量级)
    kd_loss = F.kl_div(
        student_log_probs, teacher_probs, reduction='batchmean'
    ) * (temperature ** 2)

    # 3. 混合损失
    total_loss = lambda_ce * ce_loss + lambda_kd * kd_loss
    return total_loss, {
        'ce_loss': ce_loss.item(),
        'kd_loss': kd_loss.item(),
        'total_loss': total_loss.item(),
    }


# ============================================================
# 训练
# ============================================================
def train(args):
    # Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    phase_name = "Phase 1: 标准预训练" if args.phase == "pretrain" else "Phase 2: 知识蒸馏"
    accelerator.print("=" * 60)
    accelerator.print(f"ChineseBabyLM V5 - {phase_name}")
    accelerator.print("=" * 60)
    accelerator.print(f"设备: {accelerator.device}, 进程数: {accelerator.num_processes}")
    if torch.cuda.is_available():
        accelerator.print(f"GPU: {torch.cuda.get_device_name()}")
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            accelerator.print(f"  GPU {i}: {mem:.1f} GB")

    set_seed(args.seed)

    # ============================================================
    # Tokenizer
    # ============================================================
    tokenizer_dir = args.tokenizer_dir or os.path.join(args.data_dir, "tokenizer_v3")
    accelerator.print(f"加载 Tokenizer: {tokenizer_dir}")

    tokenizer = None
    tokenizer_backend = "hf_fast"
    try:
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_dir)
        test_ids = tokenizer.encode("今天天气真好", add_special_tokens=False)
        if tokenizer.vocab_size <= 1000 or len(test_ids) == 0:
            raise ValueError(f"Invalid HF tokenizer")
    except Exception as e:
        accelerator.print(f"HF tokenizer 加载失败，回退到 SPMTokenizer: {e}")
        tokenizer_backend = "spm_wrapper"
        spm_model_path = os.path.join(tokenizer_dir, "spm.model")
        if not os.path.exists(spm_model_path):
            alt_path = os.path.join(tokenizer_dir, "spiece.model")
            if os.path.exists(alt_path):
                spm_model_path = alt_path
            else:
                raise FileNotFoundError(f"找不到 SentencePiece 模型")
        tokenizer = SPMTokenizer(spm_model_path)

    vocab_size = tokenizer.vocab_size
    accelerator.print(f"Tokenizer backend: {tokenizer_backend}")
    accelerator.print(f"词表大小: {vocab_size}")

    sp_model_path = os.path.join(tokenizer_dir, "spiece.model")
    if not os.path.exists(sp_model_path):
        sp_model_path = os.path.join(tokenizer_dir, "spm.model")

    # ============================================================
    # 模型配置 - V5: ~60M 参数小模型
    # ============================================================
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.d_model,
        intermediate_size=int(args.d_model * 8 / 3),
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        num_key_value_heads=args.n_kv_heads,
        max_position_embeddings=args.max_length,
        rms_norm_eps=1e-5,  # V5: 提升数值稳定性
        rope_theta=args.rope_theta,
        tie_word_embeddings=True,  # V5: 节省参数
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        hidden_act="silu",
        attention_dropout=args.attention_dropout,
        attn_implementation="sdpa" if args.use_flash_attention else "eager",
    )

    if args.phase == "kd" and args.student_model_path:
        # Phase 2: 加载 Phase 1 的最佳模型
        accelerator.print(f"\n加载学生模型 (Phase 1 最佳): {args.student_model_path}")
        model = LlamaForCausalLM.from_pretrained(args.student_model_path)
        accelerator.print(f"  从 checkpoint 加载完成")
    else:
        model = LlamaForCausalLM(config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        accelerator.print("Gradient Checkpointing 已启用")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"\n模型: BabyLLM-V5")
    accelerator.print(f"  隐藏维度: {args.d_model}")
    accelerator.print(f"  层数: {args.n_layer}")
    accelerator.print(f"  注意力头: {args.n_head} (Q) / {args.n_kv_heads} (KV)")
    accelerator.print(f"  FFN 维度: {config.intermediate_size}")
    accelerator.print(f"  序列长度: {args.max_length}")
    accelerator.print(f"  参数量: {total_params:,} ({total_params / 1e6:.1f}M)")
    accelerator.print(f"  可训练参数: {trainable_params:,}")

    # ============================================================
    # 数据集
    # ============================================================
    train_file = os.path.join(args.data_dir, "processed_v3", "train.txt")
    val_file = os.path.join(args.data_dir, "processed_v3", "val.txt")

    if not os.path.exists(train_file):
        train_file = os.path.join(args.data_dir, "processed_v2", "train.txt")
    if not os.path.exists(val_file):
        val_file = os.path.join(args.data_dir, "processed_v2", "val.txt")

    accelerator.print(f"\n训练数据: {train_file}")
    accelerator.print(f"验证数据: {val_file}")

    base_train_dataset = DocumentAwareDataset(
        tokenizer=tokenizer, file_path=train_file,
        block_size=args.max_length, bpe_dropout=args.bpe_dropout,
        sp_model_path=sp_model_path, encode_batch_size=args.encode_batch_size,
    )
    val_dataset = DocumentAwareDataset(
        tokenizer=tokenizer, file_path=val_file,
        block_size=args.max_length, bpe_dropout=0.0,
        encode_batch_size=args.encode_batch_size,
    )

    # Phase 2: 包装为 KD 数据集
    if args.phase == "kd" and args.teacher_logits_dir:
        accelerator.print(f"\n启用知识蒸馏模式")
        accelerator.print(f"  教师 logits 目录: {args.teacher_logits_dir}")
        accelerator.print(f"  Top-K: {args.top_k}")
        accelerator.print(f"  温度: {args.temperature}")
        accelerator.print(f"  λ_ce: {args.lambda_ce}, λ_kd: {args.lambda_kd}")
        train_dataset = KDDataset(base_train_dataset, args.teacher_logits_dir, args.top_k)
    else:
        train_dataset = base_train_dataset

    accelerator.print(f"训练集: {len(train_dataset):,} 样本")
    accelerator.print(f"验证集: {len(val_dataset):,} 样本")

    # DataLoader
    if args.phase == "kd" and isinstance(train_dataset, KDDataset):
        # KD 模式需要自定义 collate
        def kd_collate_fn(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "teacher_logits": torch.stack([b["teacher_logits"] for b in batch]),
                "teacher_indices": torch.stack([b["teacher_indices"] for b in batch]),
            }
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
            persistent_workers=True, collate_fn=kd_collate_fn,
        )
    else:
        train_loader = DataLoader(
            base_train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
            persistent_workers=True,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
        persistent_workers=True,
    )

    # ============================================================
    # 优化器
    # ============================================================
    no_decay = ["bias", "layernorm.weight", "rmsnorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_params, lr=args.learning_rate,
        betas=(0.9, 0.95), eps=1e-8,
    )

    # ============================================================
    # LR Scheduler
    # ============================================================
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    warmup_steps = int(args.warmup_ratio * max_train_steps)

    accelerator.print(f"\nLR Scheduler: Cosine with Warmup")
    accelerator.print(f"  每 epoch 更新步数: {num_update_steps_per_epoch}")
    accelerator.print(f"  总训练步数: {max_train_steps}")
    accelerator.print(f"  Warmup 步数: {warmup_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Accelerate prepare
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    effective_batch = args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    tokens_per_step = effective_batch * args.max_length

    # ============================================================
    # WandB
    # ============================================================
    if accelerator.is_main_process:
        run_name = args.wandb_run_name or f"llama-v5-{args.d_model}d-{args.n_layer}l-{args.phase}"
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=run_name, mode=args.wandb_mode,
            config={
                "model": "BabyLLM-V5", "phase": args.phase,
                "d_model": args.d_model, "n_layer": args.n_layer,
                "n_head": args.n_head, "n_kv_heads": args.n_kv_heads,
                "max_length": args.max_length, "vocab_size": vocab_size,
                "total_params": total_params, "batch_size_per_gpu": args.batch_size,
                "effective_batch_size": effective_batch,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "num_epochs": args.num_epochs, "warmup_ratio": args.warmup_ratio,
                "lr_scheduler": "cosine_with_warmup",
                "max_grad_norm": args.max_grad_norm,
                "mixed_precision": "bf16", "num_gpus": accelerator.num_processes,
                "seed": args.seed, "dataset": "babylm-zho-100M",
                "bpe_dropout": args.bpe_dropout,
                "dropout_anneal": args.dropout_anneal,
                "attention_dropout": args.attention_dropout,
                "patience": args.patience,
                "rms_norm_eps": 1e-5,
                "rope_theta": args.rope_theta,
                # KD params
                "lambda_ce": args.lambda_ce, "lambda_kd": args.lambda_kd,
                "temperature": args.temperature, "top_k": args.top_k,
            },
        )

    # Output directory
    output_dir = args.output_dir
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        # 磁盘空间检查
        check_disk_space(output_dir, min_free_gb=2.0)

    accelerator.print(f"\n{'=' * 60}")
    accelerator.print(f"训练配置: {phase_name}")
    accelerator.print(f"  GPU: {accelerator.num_processes} x {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    accelerator.print(f"  Batch/GPU: {args.batch_size}")
    accelerator.print(f"  有效 Batch: {effective_batch}")
    accelerator.print(f"  Tokens/Step: {tokens_per_step:,}")
    accelerator.print(f"  LR: {args.learning_rate} -> cosine -> 0")
    accelerator.print(f"  Epochs: {args.num_epochs}")
    accelerator.print(f"  总步数: {max_train_steps}")
    accelerator.print(f"  BPE Dropout: {args.bpe_dropout}")
    accelerator.print(f"  Dropout 退火: {args.dropout_anneal}")
    accelerator.print(f"  Early Stop Patience: {args.patience}")
    if args.phase == "kd":
        accelerator.print(f"  KD 温度: {args.temperature}")
        accelerator.print(f"  KD lambda_ce: {args.lambda_ce}, lambda_kd: {args.lambda_kd}")
    accelerator.print(f"{'=' * 60}\n")

    # ============================================================
    # 训练循环
    # ============================================================
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()
    is_kd = args.phase == "kd"

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_ce_loss = 0.0
        epoch_kd_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad()

        # BPE Dropout: 每 epoch 重新 tokenize
        if args.bpe_dropout > 0 and epoch > 0:
            accelerator.print(f"Epoch {epoch + 1}: BPE Dropout 重新 tokenize...")
            base_train_dataset.retokenize()

        # Dropout 退火: 0.05 -> 0 (后 30% 训练)
        if args.dropout_anneal and args.attention_dropout > 0:
            progress = epoch / args.num_epochs
            if progress > 0.7:
                new_dropout = args.attention_dropout * max(0.0, 1.0 - (progress - 0.7) / 0.3)
                count = set_dropout(accelerator.unwrap_model(model), new_dropout)
                if accelerator.is_main_process:
                    accelerator.print(f"  Dropout 退火: {new_dropout:.4f} ({count} layers)")

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            ncols=140, disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            step_start = time.time()

            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                # 前向传播
                outputs = model(input_ids=input_ids, labels=labels)

                if is_kd and "teacher_logits" in batch:
                    # KD 模式: 混合损失
                    total_loss, loss_dict = compute_kd_loss(
                        student_logits=outputs.logits,
                        teacher_logits_topk=batch["teacher_logits"],
                        teacher_indices_topk=batch["teacher_indices"],
                        labels=labels,
                        temperature=args.temperature,
                        lambda_ce=args.lambda_ce,
                        lambda_kd=args.lambda_kd,
                        top_k=args.top_k,
                    )
                    ce_loss_val = loss_dict['ce_loss']
                    kd_loss_val = loss_dict['kd_loss']
                else:
                    # 标准 CE 模式
                    total_loss = outputs.loss
                    ce_loss_val = total_loss.item()
                    kd_loss_val = 0.0

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # 统计
            bs, seq_len = input_ids.shape
            epoch_loss += total_loss.item() * bs * seq_len
            epoch_tokens += bs * seq_len
            epoch_ce_loss += ce_loss_val * bs * seq_len
            epoch_kd_loss += kd_loss_val * bs * seq_len
            current_lr = lr_scheduler.get_last_lr()[0]
            raw_loss = total_loss.item()
            ppl = math.exp(min(raw_loss, 20))
            step_time = time.time() - step_start
            tokens_per_sec = bs * seq_len / step_time * accelerator.num_processes

            progress_bar.set_postfix(
                loss=f"{raw_loss:.4f}", ppl=f"{ppl:.1f}",
                lr=f"{current_lr:.2e}", step=global_step,
                tps=f"{tokens_per_sec:.0f}",
            )

            # WandB 日志
            if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                log_dict = {
                    "train/loss": raw_loss,
                    "train/ppl": ppl,
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                    "train/tokens_per_sec": tokens_per_sec,
                }
                if is_kd:
                    log_dict["train/ce_loss"] = ce_loss_val
                    log_dict["train/kd_loss"] = kd_loss_val
                if accelerator.sync_gradients:
                    log_dict["train/grad_norm"] = (
                        grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
                    )
                if torch.cuda.is_available():
                    log_dict["system/gpu_memory_gb"] = torch.cuda.memory_allocated() / 1024**3
                wandb.log(log_dict, step=global_step)

            # 中间 checkpoint
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                safe_save_model(model, ckpt_dir, accelerator, config, tokenizer)
                logger.info(f"保存 checkpoint: {ckpt_dir}")
                if args.save_total_limit > 0:
                    import glob
                    all_ckpts = sorted(
                        glob.glob(os.path.join(output_dir, "checkpoint-*")),
                        key=lambda x: int(x.split("-")[-1])
                    )
                    for old_ckpt in all_ckpts[:-args.save_total_limit]:
                        import shutil
                        shutil.rmtree(old_ckpt, ignore_errors=True)

        # ============================================================
        # Epoch 评测
        # ============================================================
        avg_train_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
        epoch_time = time.time() - epoch_start

        val_loss = evaluate(model, val_loader, accelerator)
        val_ppl = math.exp(min(val_loss, 20))

        accelerator.print(f"\n{'=' * 60}")
        kd_info = ""
        if is_kd:
            avg_ce = epoch_ce_loss / epoch_tokens if epoch_tokens > 0 else 0
            avg_kd = epoch_kd_loss / epoch_tokens if epoch_tokens > 0 else 0
            kd_info = f" | CE: {avg_ce:.4f} | KD: {avg_kd:.4f}"
        accelerator.print(
            f"Epoch {epoch + 1} ({epoch_time / 60:.1f}min) | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | "
            f"LR: {current_lr:.2e}{kd_info}"
        )

        if accelerator.is_main_process:
            epoch_log = {
                "epoch/train_loss": avg_train_loss,
                "epoch/val_loss": val_loss,
                "epoch/val_ppl": val_ppl,
                "epoch/epoch": epoch + 1,
                "epoch/epoch_time_min": epoch_time / 60,
                "epoch/learning_rate": current_lr,
            }
            if is_kd:
                epoch_log["epoch/ce_loss"] = avg_ce
                epoch_log["epoch/kd_loss"] = avg_kd
            wandb.log(epoch_log, step=global_step)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_dir = os.path.join(output_dir, "best_model")
                safe_save_model(model, best_dir, accelerator, config, tokenizer)
                accelerator.print(
                    f"  新最佳模型 (val_loss={val_loss:.4f}, ppl={val_ppl:.2f})"
                )
                wandb.log({
                    "best/val_loss": val_loss, "best/val_ppl": val_ppl,
                    "best/at_epoch": epoch + 1, "best/at_step": global_step,
                }, step=global_step)
            else:
                patience_counter += 1
                accelerator.print(f"  早停计数: {patience_counter} / {args.patience}")

        accelerator.print(f"{'=' * 60}")

        # Early stopping
        if patience_counter >= args.patience:
            accelerator.print(
                f"\n验证集 Loss 连续 {args.patience} 轮未改善，触发早停。"
            )
            break

    # ============================================================
    # 训练结束
    # ============================================================
    total_time = time.time() - start_time
    accelerator.print("\n" + "=" * 60)
    accelerator.print(f"V5 {phase_name} 训练完成!")
    accelerator.print(f"  最佳 Val Loss: {best_val_loss:.4f}")
    accelerator.print(f"  最佳 Val PPL: {math.exp(min(best_val_loss, 20)):.2f}")
    accelerator.print(f"  训练 Epochs: {epoch + 1}")
    accelerator.print(f"  总训练时间: {total_time / 3600:.1f} 小时")
    accelerator.print(f"  模型参数量: {total_params / 1e6:.1f}M")
    accelerator.print(f"  模型保存: {output_dir}")
    accelerator.print("=" * 60)

    if accelerator.is_main_process:
        wandb.finish()
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="ChineseBabyLM V5 - 小模型 + 知识蒸馏")

    # 训练阶段
    parser.add_argument("--phase", type=str, default="pretrain",
                        choices=["pretrain", "kd"],
                        help="训练阶段: pretrain=标准预训练, kd=知识蒸馏")

    # 数据
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output/babylm-llama-v5")
    parser.add_argument("--tokenizer_dir", type=str, default=None)

    # 模型架构 - V5: ~60M 参数
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)

    # KD 参数
    parser.add_argument("--student_model_path", type=str, default=None,
                        help="Phase 2: 学生模型路径 (Phase 1 最佳模型)")
    parser.add_argument("--teacher_logits_dir", type=str, default=None,
                        help="Phase 2: 教师 logits 目录")
    parser.add_argument("--lambda_ce", type=float, default=0.3,
                        help="CE 损失权重")
    parser.add_argument("--lambda_kd", type=float, default=0.7,
                        help="KD 损失权重")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="KD 温度参数")
    parser.add_argument("--top_k", type=int, default=10,
                        help="教师 top-K logits")

    # 训练超参
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # 正则化
    parser.add_argument("--attention_dropout", type=float, default=0.05)
    parser.add_argument("--bpe_dropout", type=float, default=0.1)
    parser.add_argument("--dropout_anneal", action="store_true", default=True)
    parser.add_argument("--no_dropout_anneal", action="store_true")
    parser.add_argument("--encode_batch_size", type=int, default=4096)

    # 优化
    parser.add_argument("--use_flash_attention", action="store_true", default=True)
    parser.add_argument("--no_flash_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")

    # 早停
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save_total_limit", type=int, default=2)

    # RoPE
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # 日志和保存
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    # WandB
    parser.add_argument("--wandb_project", type=str, default="chinese-babylm")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")

    args = parser.parse_args()

    # 处理 flag
    if args.no_flash_attention:
        args.use_flash_attention = False
    if args.no_gradient_checkpointing:
        args.gradient_checkpointing = False
    if args.no_dropout_anneal:
        args.dropout_anneal = False

    # 设置日志文件
    log_file = os.path.join(args.output_dir, f"train_v5_{args.phase}.log")
    os.makedirs(args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info(f"ChineseBabyLM V5 - Phase: {args.phase}")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    train(args)


if __name__ == "__main__":
    main()
