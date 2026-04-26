"""
ChineseBabyLM V2 - LLaMA 架构训练脚本 (Phase 3 精细优化版)
核心改进:
1. LlamaForCausalLM (RoPE + GQA + SwiGLU + RMSNorm)
2. 修复 LR Scheduler Bug
3. ByteLevel BPE Tokenizer
4. bf16 混合精度 + Flash Attention 2
5. 文档感知序列构造（EOS 分隔，避免跨文档污染）
6. BPE Dropout 数据增强
7. Gradient Checkpointing 省显存
8. 训练退火策略（后期降低 dropout）
9. WandB 全面监控（tokens/sec, GPU利用率, 梯度范数等）
10. 独立验证集评测
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
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator
import wandb
from tqdm import tqdm

# ============================================================
# 日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_v2.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据集 - 文档感知序列构造
# ============================================================
class DocumentAwareDataset(Dataset):
    """文档感知的序列构造：用 EOS token 分隔文档，避免跨文档污染"""
    
    def __init__(self, tokenizer, file_path, block_size=1024, bpe_dropout=0.0):
        assert os.path.isfile(file_path), f"文件不存在: {file_path}"
        self.block_size = block_size
        self.bpe_dropout = bpe_dropout
        
        logger.info(f"读取数据: {file_path}, block_size={block_size}, bpe_dropout={bpe_dropout}")
        
        # 先收集所有文档的 token IDs
        all_token_ids = []
        eos_id = tokenizer.eos_token_id
        num_docs = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Tokenizing"):
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line)
                all_token_ids.extend(ids)
                all_token_ids.append(eos_id)  # 文档间用 EOS 分隔
                num_docs += 1
        
        logger.info(f"文档数: {num_docs:,}, 总 tokens (含 EOS): {len(all_token_ids):,}")
        
        # 构造固定长度的训练样本
        self.examples = []
        i = 0
        while i + block_size + 1 <= len(all_token_ids):
            self.examples.append(all_token_ids[i:i + block_size + 1])
            i += block_size + 1
        
        # 处理剩余 tokens（padding）
        if i < len(all_token_ids) - 1:
            remaining = all_token_ids[i:]
            pad_id = tokenizer.pad_token_id or 0
            padded = remaining + [pad_id] * (block_size + 1 - len(remaining))
            self.examples.append(padded[:block_size + 1])
        
        logger.info(f"训练样本数: {len(self.examples):,}")
        
        # 保存原始 token IDs 用于 BPE dropout（如果启用）
        self._tokenizer = tokenizer
        self._file_path = file_path
        self._all_lines = None  # lazy load
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        chunk = self.examples[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


class DocumentAwareDatasetWithBPEDropout(Dataset):
    """支持 BPE Dropout 的数据集 - 每次 epoch 重新 tokenize"""
    
    def __init__(self, tokenizer, file_path, block_size=1024, bpe_dropout=0.1):
        assert os.path.isfile(file_path), f"文件不存在: {file_path}"
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.bpe_dropout = bpe_dropout
        self.file_path = file_path
        
        # 读取所有行
        self.lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.lines.append(line)
        
        logger.info(f"BPE Dropout Dataset: {len(self.lines):,} 行, dropout={bpe_dropout}")
        
        # 初始化：先用标准 tokenize 创建第一版
        self.examples = []
        self._retokenize_initial()
        # 记住初始样本数量，后续 retokenize 保持不变
        self._target_length = len(self.examples)
        logger.info(f"初始样本数（固定）: {self._target_length:,}")
    
    def _retokenize_initial(self):
        """初始 tokenize（不限制数量）"""
        eos_id = self.tokenizer.eos_token_id
        all_token_ids = []
        
        for line in self.lines:
            ids = self.tokenizer.encode(line)
            all_token_ids.extend(ids)
            all_token_ids.append(eos_id)
        
        self.examples = []
        block_size = self.block_size
        i = 0
        while i + block_size + 1 <= len(all_token_ids):
            self.examples.append(all_token_ids[i:i + block_size + 1])
            i += block_size + 1
    
    def _retokenize(self):
        """重新 tokenize 所有数据（使用 BPE dropout）"""
        eos_id = self.tokenizer.eos_token_id
        all_token_ids = []
        
        # 启用 BPE dropout
        if self.bpe_dropout > 0 and hasattr(self.tokenizer, 'backend_tokenizer'):
            self.tokenizer.backend_tokenizer.model.dropout = self.bpe_dropout
        
        for line in self.lines:
            ids = self.tokenizer.encode(line)
            all_token_ids.extend(ids)
            all_token_ids.append(eos_id)
        
        # 恢复 BPE dropout
        if self.bpe_dropout > 0 and hasattr(self.tokenizer, 'backend_tokenizer'):
            self.tokenizer.backend_tokenizer.model.dropout = 0.0
        
        # 构造样本
        new_examples = []
        block_size = self.block_size
        i = 0
        while i + block_size + 1 <= len(all_token_ids):
            new_examples.append(all_token_ids[i:i + block_size + 1])
            i += block_size + 1
        
        # ✅ Bug 修复: 保持样本数量不变，防止 DataLoader 索引越界
        target_len = self._target_length
        if len(new_examples) < target_len:
            # 用最后一个样本填充
            pad = new_examples[-1] if new_examples else [0] * (block_size + 1)
            new_examples.extend([pad] * (target_len - len(new_examples)))
        elif len(new_examples) > target_len:
            new_examples = new_examples[:target_len]
        
        self.examples = new_examples
    
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
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中", ncols=100,
                         disable=not accelerator.is_main_process):
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            # loss 是 per-token 平均
            input_ids = batch["input_ids"]
            if input_ids.dim() == 1:
                bs, seq_len = 1, input_ids.shape[0]
            else:
                bs, seq_len = input_ids.shape
            total_loss += outputs.loss.item() * bs * seq_len  # ✅ Bug fix: loss 是 per-token 平均
            total_tokens += bs * seq_len
            num_batches += 1
    
    # ✅ 修复: accelerator.gather 沿 dim=0 拼接，返回 1D tensor
    stats = torch.tensor([total_loss, total_tokens, num_batches],
                         dtype=torch.float32, device=accelerator.device)
    gathered = accelerator.gather(stats)  # shape: [num_processes * 3]
    gathered = gathered.view(-1, 3)       # reshape 为 [num_processes, 3]
    
    total_loss_sum = gathered[:, 0].sum().item()
    total_tokens_sum = gathered[:, 1].sum().item()
    avg_loss = total_loss_sum / total_tokens_sum if total_tokens_sum > 0 else float('inf')
    model.train()
    return avg_loss


# ============================================================
# 训练
# ============================================================
def train(args):
    # Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    
    accelerator.print(f"设备: {accelerator.device}, 进程数: {accelerator.num_processes}")
    if torch.cuda.is_available():
        accelerator.print(f"GPU: {torch.cuda.get_device_name()}")
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            accelerator.print(f"  GPU {i}: {mem:.1f} GB")
    
    set_seed(args.seed)
    
    # Tokenizer
    tokenizer_dir = os.path.join(args.data_dir, "tokenizer_v2")
    accelerator.print(f"加载 Tokenizer: {tokenizer_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    accelerator.print(f"词表大小: {vocab_size}")
    
    # ✅ LLaMA 模型配置
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.d_model,
        intermediate_size=int(args.d_model * 8 / 3),  # SwiGLU
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        num_key_value_heads=args.n_kv_heads,
        max_position_embeddings=args.max_length,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        hidden_act="silu",
        attn_implementation="sdpa" if args.use_flash_attention else "eager",
    )
    
    model = LlamaForCausalLM(config)
    
    # ✅ Gradient Checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        accelerator.print("✅ Gradient Checkpointing 已启用")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"\n模型: LLaMA-Small")
    accelerator.print(f"  隐藏维度: {args.d_model}")
    accelerator.print(f"  层数: {args.n_layer}")
    accelerator.print(f"  注意力头: {args.n_head} (Q) / {args.n_kv_heads} (KV)")
    accelerator.print(f"  FFN 维度: {config.intermediate_size}")
    accelerator.print(f"  序列长度: {args.max_length}")
    accelerator.print(f"  参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    accelerator.print(f"  Flash Attention: {args.use_flash_attention}")
    
    # ✅ 独立验证集
    train_file = os.path.join(args.data_dir, "processed_v2", "train.txt")
    val_file = os.path.join(args.data_dir, "processed_v2", "val.txt")
    
    # 回退方案
    if not os.path.exists(train_file):
        train_file = os.path.join(args.data_dir, "processed", "train.txt")
    if not os.path.exists(val_file):
        accelerator.print("⚠️ 独立验证集不存在，将从训练集中分割")
        val_file = None
    
    accelerator.print(f"\n训练数据: {train_file}")
    accelerator.print(f"验证数据: {val_file or '从训练集分割'}")
    
    # ✅ 创建数据集（文档感知序列构造）
    if args.bpe_dropout > 0:
        accelerator.print(f"\n使用 BPE Dropout Dataset (dropout={args.bpe_dropout})")
        full_dataset = DocumentAwareDatasetWithBPEDropout(
            tokenizer=tokenizer, file_path=train_file,
            block_size=args.max_length, bpe_dropout=args.bpe_dropout,
        )
    else:
        full_dataset = DocumentAwareDataset(
            tokenizer=tokenizer, file_path=train_file,
            block_size=args.max_length,
        )
    
    if val_file:
        if args.bpe_dropout > 0:
            val_dataset = DocumentAwareDataset(
                tokenizer=tokenizer, file_path=val_file,
                block_size=args.max_length,
            )
        else:
            val_dataset = DocumentAwareDataset(
                tokenizer=tokenizer, file_path=val_file,
                block_size=args.max_length,
            )
    else:
        train_size = int(0.95 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        full_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    accelerator.print(f"训练集: {len(full_dataset):,} 样本, 验证集: {len(val_dataset):,} 样本")
    
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
    
    # 优化器 - 使用分层学习率
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "rmsnorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_params, lr=args.learning_rate,
                      betas=(0.9, 0.95), eps=1e-8)
    
    # LR Scheduler
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    warmup_steps = int(args.warmup_ratio * max_train_steps)
    
    accelerator.print(f"\nLR Scheduler:")
    accelerator.print(f"  每 epoch 步数: {num_update_steps_per_epoch}")
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
    
    # WandB
    if accelerator.is_main_process:
        run_name = args.wandb_run_name or f"llama-v2-{args.d_model}d-{args.n_layer}l-gqa{args.n_kv_heads}"
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=run_name, mode=args.wandb_mode,
            config={
                "model": "LLaMA-Small", "d_model": args.d_model, "n_layer": args.n_layer,
                "n_head": args.n_head, "n_kv_heads": args.n_kv_heads,
                "max_length": args.max_length, "vocab_size": vocab_size,
                "total_params": total_params, "batch_size_per_gpu": args.batch_size,
                "effective_batch_size": effective_batch,
                "tokens_per_step": tokens_per_step,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "num_epochs": args.num_epochs, "warmup_ratio": args.warmup_ratio,
                "lr_scheduler": "cosine_with_warmup", "max_grad_norm": args.max_grad_norm,
                "mixed_precision": "bf16", "num_gpus": accelerator.num_processes,
                "seed": args.seed, "dataset": "babylm-zho-100M",
                "tokenizer": "ByteLevel_BPE_32K",
                "flash_attention": args.use_flash_attention,
                "gradient_checkpointing": args.gradient_checkpointing,
                "bpe_dropout": args.bpe_dropout,
            },
        )
    
    accelerator.print(f"\n{'='*60}")
    accelerator.print(f"训练配置:")
    accelerator.print(f"  GPU: {accelerator.num_processes} × {torch.cuda.get_device_name()}")
    accelerator.print(f"  Batch/GPU: {args.batch_size}")
    accelerator.print(f"  有效 Batch: {effective_batch}")
    accelerator.print(f"  Tokens/Step: {tokens_per_step:,}")
    accelerator.print(f"  LR: {args.learning_rate} → cosine → 0")
    accelerator.print(f"  Epochs: {args.num_epochs}")
    accelerator.print(f"  总步数: {max_train_steps}")
    accelerator.print(f"  BPE Dropout: {args.bpe_dropout}")
    accelerator.print(f"  Flash Attention: {args.use_flash_attention}")
    accelerator.print(f"  Grad Checkpointing: {args.gradient_checkpointing}")
    accelerator.print(f"{'='*60}")
    
    # 输出目录
    output_dir = args.output_dir
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    # ============================================================
    # 训练循环
    # ============================================================
    accelerator.print("\n" + "=" * 60)
    accelerator.print("🚀 开始 LLaMA 训练!")
    accelerator.print("=" * 60)
    
    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()
    epoch_start_time = time.time()
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        optimizer.zero_grad()
        
        # ✅ BPE Dropout: 每 epoch 重新 tokenize
        if args.bpe_dropout > 0 and hasattr(full_dataset, '_retokenize'):
            accelerator.print(f"\n Epoch {epoch+1}: BPE Dropout 重新 tokenize...")
            full_dataset._retokenize()
        
        # ✅ 训练退火：后半段关闭 dropout
        if hasattr(args, 'dropout_anneal') and args.dropout_anneal:
            progress = epoch / args.num_epochs
            if progress > 0.7:
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = max(0.0, module.p * (1 - (progress - 0.7) / 0.3))
        
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}",
            ncols=140, disable=not accelerator.is_main_process,
        )
        
        for step, batch in enumerate(progress_bar):
            step_start = time.time()
            
            with accelerator.accumulate(model):
                outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = outputs.loss
                
                # ✅ 反向传播
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    # 梯度裁剪
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
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
                loss=f"{raw_loss:.4f}", ppl=f"{ppl:.2f}",
                lr=f"{current_lr:.2e}", step=global_step,
                tps=f"{tokens_per_sec:.0f}",
            )
            
            # ✅ WandB 日志（增强版）
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
                    "train/step_time": step_time,
                }
                
                # 记录梯度范数
                if accelerator.sync_gradients:
                    log_dict["train/grad_norm"] = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
                
                # GPU 内存使用
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    log_dict["system/gpu_memory_gb"] = gpu_mem
                    log_dict["system/gpu_memory_reserved_gb"] = gpu_mem_reserved
                
                wandb.log(log_dict, step=global_step)
                
                logger.info(
                    f"Epoch {epoch+1} Step {global_step} | "
                    f"Loss: {avg_loss:.4f} | PPL: {math.exp(min(avg_loss, 20)):.2f} | "
                    f"LR: {current_lr:.2e} | tok/s: {tokens_per_sec:.0f}"
                )
            
            # 保存 checkpoint
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                logger.info(f"保存 checkpoint: {ckpt_dir}")
        
        # Epoch 评测
        avg_train_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
        epoch_time = time.time() - epoch_start_time
        
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"Epoch {epoch+1} 完成 (耗时 {epoch_time/60:.1f} 分钟)")
        
        val_loss = evaluate(model, val_loader, accelerator)
        val_ppl = math.exp(min(val_loss, 20))
        
        accelerator.print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
        )
        accelerator.print(f"{'='*60}")
        
        if accelerator.is_main_process:
            # 总训练 tokens
            total_tokens_trained = global_step * tokens_per_step
            
            wandb.log({
                "epoch/train_loss": avg_train_loss,
                "epoch/val_loss": val_loss,
                "epoch/val_ppl": val_ppl,
                "epoch/epoch": epoch + 1,
                "epoch/total_tokens": total_tokens_trained,
                "epoch/epoch_time_min": epoch_time / 60,
                "epoch/total_time_min": (time.time() - start_time) / 60,
            }, step=global_step)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dir = os.path.join(output_dir, "best_model")
                os.makedirs(best_dir, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                logger.info(
                    f"✅ 新最佳模型 (val_loss={val_loss:.4f}, ppl={val_ppl:.2f}): {best_dir}"
                )
                
                # 记录最佳指标到 WandB
                wandb.log({
                    "best/val_loss": val_loss,
                    "best/val_ppl": val_ppl,
                    "best/at_epoch": epoch + 1,
                    "best/at_step": global_step,
                }, step=global_step)
            
            # 每 epoch 保存
            epoch_dir = os.path.join(output_dir, f"epoch-{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
        
        epoch_start_time = time.time()
    
    # 训练结束
    total_time = time.time() - start_time
    accelerator.print("\n" + "=" * 60)
    accelerator.print("🎉 训练完成!")
    accelerator.print(f"最佳 Val Loss: {best_val_loss:.4f}")
    accelerator.print(f"最佳 Val PPL: {math.exp(min(best_val_loss, 20)):.2f}")
    accelerator.print(f"总训练时间: {total_time/3600:.1f} 小时")
    accelerator.print(f"模型保存: {output_dir}")
    accelerator.print("=" * 60)
    
    if accelerator.is_main_process:
        wandb.finish()
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="ChineseBabyLM V2 - LLaMA 训练 (Phase 3)")
    
    # 数据
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output/babylm-llama-v2")
    
    # 模型架构
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    
    # 训练超参
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    
    # 优化选项
    parser.add_argument("--use_flash_attention", action="store_true", default=True)
    parser.add_argument("--no_flash_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--bpe_dropout", type=float, default=0.1)
    parser.add_argument("--dropout_anneal", action="store_true", default=True)
    parser.add_argument("--no_dropout_anneal", action="store_true")
    
    # 日志和保存
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=2000)
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
    
    logger.info("=" * 60)
    logger.info("ChineseBabyLM V2 - LLaMA 架构训练 (Phase 3 精细优化)")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    
    train(args)


if __name__ == "__main__":
    main()