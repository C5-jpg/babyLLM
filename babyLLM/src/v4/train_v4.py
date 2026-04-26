"""
ChineseBabyLM V4 - LLaMA 架构训练脚本 (SOTA 版)
核心改进:
1. 修复 LR 调度器: Cosine Decay with Warmup (替代有 Bug 的 WSD)
2. 更深架构: 16 层 + tie_word_embeddings 节省参数
3. Dropout 退火: 前 70% 训练 dropout=0.1，后 30% 线性衰减到 0
4. BPE Dropout 数据增强: SentencePiece enable_sampling
5. 标准 HF 格式: LlamaTokenizerFast 兼容评测 pipeline
6. 增强早停: patience=7
7. 中间检查点保存
"""
import os
import sys
import math
import time
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
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
        logging.FileHandler("training_v4.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据集 - 支持 BPE Dropout (SentencePiece native)
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

        # 如果启用 BPE dropout，加载 SentencePiece 原始模型
        if bpe_dropout > 0 and sp_model_path and os.path.exists(sp_model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(sp_model_path)
            logger.info(f"BPE Dropout 已启用: alpha={bpe_dropout}")

        logger.info(f"读取数据: {file_path}, block_size={block_size}")

        # 读取所有行（用于 BPE dropout 时重新 tokenize）
        self.lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.lines.append(line)

        logger.info(f"文档数: {len(self.lines):,}")

        # 初始 tokenize
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
            # 使用 SentencePiece 原生 BPE dropout
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

        # 构造固定长度样本 - 使用 50% 重叠滑动窗口，数据量翻倍
        stride = self.block_size // 2  # 512 步长（50% 重叠）
        new_examples = []
        i = 0
        while i + self.block_size + 1 <= len(all_token_ids):
            new_examples.append(all_token_ids[i:i + self.block_size + 1])
            i += stride

        # 处理剩余 tokens（最后一个不完整块用 pad 补齐）
        if len(new_examples) == 0 or (i < len(all_token_ids) - 1 and len(all_token_ids) - i > 1):
            remaining = all_token_ids[i:]
            if len(remaining) > 1:
                padded = remaining + [self.pad_id] * (self.block_size + 1 - len(remaining))
                new_examples.append(padded[:self.block_size + 1])

        self.examples = new_examples

    def retokenize(self):
        """每 epoch 调用，使用 BPE dropout 重新 tokenize"""
        if self.sp is None:
            return
        self._tokenize(use_dropout=True)
        # 保持样本数量不变
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
    """设置模型中所有 Dropout 层的 dropout 概率"""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_p
            count += 1
    return count


# ============================================================
# 训练
# ============================================================
def train(args):
    # Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    accelerator.print("=" * 60)
    accelerator.print("ChineseBabyLM V4 - SOTA 训练")
    accelerator.print("=" * 60)
    accelerator.print(f"设备: {accelerator.device}, 进程数: {accelerator.num_processes}")
    if torch.cuda.is_available():
        accelerator.print(f"GPU: {torch.cuda.get_device_name()}")
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            accelerator.print(f"  GPU {i}: {mem:.1f} GB")

    set_seed(args.seed)

    # ============================================================
    # Tokenizer - 标准 HF LlamaTokenizerFast
    # ============================================================
    tokenizer_dir = args.tokenizer_dir or os.path.join(args.data_dir, "tokenizer_v4")
    accelerator.print(f"加载 Tokenizer: {tokenizer_dir}")

    tokenizer = None
    tokenizer_backend = "hf_fast"
    try:
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_dir)
        test_ids = tokenizer.encode("今天天气真好", add_special_tokens=False)
        if tokenizer.vocab_size <= 1000 or len(test_ids) == 0:
            raise ValueError(f"Invalid HF tokenizer (vocab_size={tokenizer.vocab_size}, test_len={len(test_ids)})")
    except Exception as e:
        accelerator.print(f"⚠️ HF tokenizer 加载失败，回退到 SPMTokenizer: {e}")
        tokenizer_backend = "spm_wrapper"
        spm_model_path = os.path.join(tokenizer_dir, "spm.model")
        if not os.path.exists(spm_model_path):
            alt_path = os.path.join(tokenizer_dir, "spiece.model")
            if os.path.exists(alt_path):
                spm_model_path = alt_path
            else:
                raise FileNotFoundError(f"找不到 SentencePiece 模型: {spm_model_path} 或 {alt_path}")
        tokenizer = SPMTokenizer(spm_model_path)

    vocab_size = tokenizer.vocab_size
    accelerator.print(f"Tokenizer backend: {tokenizer_backend}")
    accelerator.print(f"词表大小: {vocab_size}")

    # SentencePiece 原始模型路径（用于 BPE dropout）
    sp_model_path = os.path.join(tokenizer_dir, "spiece.model")
    if not os.path.exists(sp_model_path):
        sp_model_path = os.path.join(tokenizer_dir, "spm.model")

    # ============================================================
    # 模型配置 - V4: 16 层 + tie_embeddings
    # ============================================================
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.d_model,
        intermediate_size=int(args.d_model * 8 / 3),
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        num_key_value_heads=args.n_kv_heads,
        max_position_embeddings=args.max_length,
        rms_norm_eps=1e-6,
        rope_theta=args.rope_theta,
        tie_word_embeddings=args.tie_embeddings,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        hidden_act="silu",
        attention_dropout=args.attention_dropout,
        attn_implementation="sdpa" if args.use_flash_attention else "eager",
    )

    model = LlamaForCausalLM(config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        accelerator.print("✅ Gradient Checkpointing 已启用")

    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"\n模型: LLaMA-V4")
    accelerator.print(f"  隐藏维度: {args.d_model}")
    accelerator.print(f"  层数: {args.n_layer}")
    accelerator.print(f"  注意力头: {args.n_head} (Q) / {args.n_kv_heads} (KV)")
    accelerator.print(f"  FFN 维度: {config.intermediate_size}")
    accelerator.print(f"  序列长度: {args.max_length}")
    accelerator.print(f"  Tie Embeddings: {args.tie_embeddings}")
    accelerator.print(f"  参数量: {total_params:,} ({total_params / 1e6:.1f}M)")

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

    full_dataset = DocumentAwareDataset(
        tokenizer=tokenizer, file_path=train_file,
        block_size=args.max_length, bpe_dropout=args.bpe_dropout,
        sp_model_path=sp_model_path, encode_batch_size=args.encode_batch_size,
    )
    val_dataset = DocumentAwareDataset(
        tokenizer=tokenizer, file_path=val_file,
        block_size=args.max_length, bpe_dropout=0.0,
        encode_batch_size=args.encode_batch_size,
    )

    accelerator.print(f"训练集: {len(full_dataset):,} 样本")
    accelerator.print(f"验证集: {len(val_dataset):,} 样本")

    train_loader = DataLoader(
        full_dataset, batch_size=args.batch_size, shuffle=True,
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
    # LR Scheduler - 修复: 使用 Cosine Decay with Warmup
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
        run_name = args.wandb_run_name or f"llama-v4-{args.d_model}d-{args.n_layer}l"
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=run_name, mode=args.wandb_mode,
            config={
                "model": "LLaMA-V4", "d_model": args.d_model, "n_layer": args.n_layer,
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
                "tie_embeddings": args.tie_embeddings,
                "bpe_dropout": args.bpe_dropout,
                "dropout_anneal": args.dropout_anneal,
                "attention_dropout": args.attention_dropout,
                "patience": args.patience,
            },
        )

    # Output directory
    output_dir = args.output_dir
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    accelerator.print(f"\n{'=' * 60}")
    accelerator.print(f"训练配置:")
    accelerator.print(f"  GPU: {accelerator.num_processes} × {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    accelerator.print(f"  Batch/GPU: {args.batch_size}")
    accelerator.print(f"  有效 Batch: {effective_batch}")
    accelerator.print(f"  Tokens/Step: {tokens_per_step:,}")
    accelerator.print(f"  LR: {args.learning_rate} → cosine → 0")
    accelerator.print(f"  Epochs: {args.num_epochs}")
    accelerator.print(f"  总步数: {max_train_steps}")
    accelerator.print(f"  BPE Dropout: {args.bpe_dropout}")
    accelerator.print(f"  Dropout 退火: {args.dropout_anneal}")
    accelerator.print(f"  Early Stop Patience: {args.patience}")
    accelerator.print(f"{'=' * 60}\n")

    # ============================================================
    # 训练循环
    # ============================================================
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start = time.time()
        optimizer.zero_grad()

        # BPE Dropout: 每 epoch 重新 tokenize
        if args.bpe_dropout > 0 and epoch > 0:
            accelerator.print(f"Epoch {epoch + 1}: BPE Dropout 重新 tokenize...")
            full_dataset.retokenize()

        # Dropout 退火
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
                outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # 统计
            bs, seq_len = batch["input_ids"].shape
            epoch_loss += loss.item() * bs * seq_len
            epoch_tokens += bs * seq_len
            current_lr = lr_scheduler.get_last_lr()[0]
            raw_loss = loss.item()
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
                avg_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else raw_loss
                log_dict = {
                    "train/loss": raw_loss,
                    "train/avg_loss": avg_loss,
                    "train/ppl": ppl,
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                    "train/tokens_per_sec": tokens_per_sec,
                }
                if accelerator.sync_gradients:
                    log_dict["train/grad_norm"] = (
                        grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
                    )
                if torch.cuda.is_available():
                    log_dict["system/gpu_memory_gb"] = torch.cuda.memory_allocated() / 1024**3
                wandb.log(log_dict, step=global_step)

            # 中间 checkpoint（只保留最新1个，节省磁盘空间）
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                logger.info(f"保存 checkpoint: {ckpt_dir}")
                # 删除旧 checkpoint，只保留最新1个
                if args.save_total_limit > 0:
                    import glob
                    all_ckpts = sorted(
                        glob.glob(os.path.join(output_dir, "checkpoint-*")),
                        key=lambda x: int(x.split("-")[-1])
                    )
                    for old_ckpt in all_ckpts[:-args.save_total_limit]:
                        import shutil
                        shutil.rmtree(old_ckpt, ignore_errors=True)
                        logger.info(f"删除旧 checkpoint: {old_ckpt}")

        # ============================================================
        # Epoch 评测
        # ============================================================
        avg_train_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
        epoch_time = time.time() - epoch_start

        val_loss = evaluate(model, val_loader, accelerator)
        val_ppl = math.exp(min(val_loss, 20))

        accelerator.print(f"\n{'=' * 60}")
        accelerator.print(
            f"Epoch {epoch + 1} ({epoch_time / 60:.1f}min) | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | "
            f"LR: {current_lr:.2e}"
        )

        if accelerator.is_main_process:
            wandb.log({
                "epoch/train_loss": avg_train_loss,
                "epoch/val_loss": val_loss,
                "epoch/val_ppl": val_ppl,
                "epoch/epoch": epoch + 1,
                "epoch/epoch_time_min": epoch_time / 60,
                "epoch/learning_rate": current_lr,
            }, step=global_step)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_dir = os.path.join(output_dir, "best_model")
                os.makedirs(best_dir, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                accelerator.print(
                    f"✅ 新最佳模型 (val_loss={val_loss:.4f}, ppl={val_ppl:.2f})"
                )
                wandb.log({
                    "best/val_loss": val_loss, "best/val_ppl": val_ppl,
                    "best/at_epoch": epoch + 1, "best/at_step": global_step,
                }, step=global_step)
            else:
                patience_counter += 1
                accelerator.print(f"⚠️ 早停计数: {patience_counter} / {args.patience}")

        accelerator.print(f"{'=' * 60}")

        # Early stopping
        if patience_counter >= args.patience:
            accelerator.print(
                f"\n⏹️ 验证集 Loss 连续 {args.patience} 轮未改善，触发早停。"
            )
            break

    # ============================================================
    # 训练结束
    # ============================================================
    total_time = time.time() - start_time
    accelerator.print("\n" + "=" * 60)
    accelerator.print("🎉 V4 训练完成!")
    accelerator.print(f"  最佳 Val Loss: {best_val_loss:.4f}")
    accelerator.print(f"  最佳 Val PPL: {math.exp(min(best_val_loss, 20)):.2f}")
    accelerator.print(f"  训练 Epochs: {epoch + 1}")
    accelerator.print(f"  总训练时间: {total_time / 3600:.1f} 小时")
    accelerator.print(f"  模型保存: {output_dir}")
    accelerator.print("=" * 60)

    if accelerator.is_main_process:
        wandb.finish()
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="ChineseBabyLM V4 - SOTA 训练")

    # 数据
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output/babylm-llama-v4")
    parser.add_argument("--tokenizer_dir", type=str, default=None)

    # 模型架构
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layer", type=int, default=16)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--tie_embeddings", action="store_true", default=True)
    parser.add_argument("--no_tie_embeddings", action="store_true")

    # 训练超参
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # 正则化
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--bpe_dropout", type=float, default=0.1)
    parser.add_argument("--dropout_anneal", action="store_true", default=True)
    parser.add_argument("--no_dropout_anneal", action="store_true")
    parser.add_argument("--encode_batch_size", type=int, default=4096)

    # 优化
    parser.add_argument("--use_flash_attention", action="store_true", default=True)
    parser.add_argument("--no_flash_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")

    # 早停
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="保留最近N个checkpoint，0=保留全部")

    # RoPE 扩展
    parser.add_argument("--rope_theta", type=float, default=50000.0,
                        help="RoPE base frequency，默认50000 (更好的长上下文编码)")

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
    if args.no_tie_embeddings:
        args.tie_embeddings = False
    if args.no_flash_attention:
        args.use_flash_attention = False
    if args.no_gradient_checkpointing:
        args.gradient_checkpointing = False
    if args.no_dropout_anneal:
        args.dropout_anneal = False

    logger.info("=" * 60)
    logger.info("ChineseBabyLM V4 - SOTA 训练")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    train(args)


if __name__ == "__main__":
    main()
