"""
ChineseBabyLM V3 - LLaMA 架构训练脚本 (终极优化版)
核心改进:
1. Early Stopping 机制
2. WSD (Warmup-Stable-Decay) 学习率调度策略
3. SentencePiece Tokenizer 适配
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
    set_seed,
)
from spm_tokenizer import SPMTokenizer
from accelerate import Accelerator
import wandb
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("training_v3.log", encoding="utf-8")])
logger = logging.getLogger(__name__)

class DocumentAwareDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=1024):
        assert os.path.isfile(file_path), f"文件不存在: {file_path}"
        self.block_size = block_size
        logger.info(f"读取数据: {file_path}, block_size={block_size}")
        
        all_token_ids = []
        eos_id = tokenizer.eos_token_id
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Tokenizing"):
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line, add_special_tokens=False)
                all_token_ids.extend(ids)
                all_token_ids.append(eos_id)
                
        self.examples = []
        i = 0
        while i + block_size + 1 <= len(all_token_ids):
            self.examples.append(all_token_ids[i:i + block_size + 1])
            i += block_size + 1
            
        if i < len(all_token_ids) - 1:
            remaining = all_token_ids[i:]
            pad_id = tokenizer.pad_token_id or 0
            padded = remaining + [pad_id] * (block_size + 1 - len(remaining))
            self.examples.append(padded[:block_size + 1])
            
        logger.info(f"训练样本数: {len(self.examples):,}")
    
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        chunk = self.examples[idx]
        return {"input_ids": torch.tensor(chunk[:-1], dtype=torch.long), "labels": torch.tensor(chunk[1:], dtype=torch.long)}

def evaluate(model, val_loader, accelerator):
    model.eval()
    total_loss, total_tokens, num_batches = 0.0, 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中", disable=not accelerator.is_main_process):
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            input_ids = batch["input_ids"]
            bs, seq_len = (1, input_ids.shape[0]) if input_ids.dim() == 1 else input_ids.shape
            total_loss += outputs.loss.item() * bs * seq_len
            total_tokens += bs * seq_len
            num_batches += 1
            
    stats = torch.tensor([total_loss, total_tokens, num_batches], dtype=torch.float32, device=accelerator.device)
    gathered = accelerator.gather(stats).view(-1, 3)
    avg_loss = gathered[:, 0].sum().item() / gathered[:, 1].sum().item() if gathered[:, 1].sum().item() > 0 else float('inf')
    model.train()
    return avg_loss

def get_wsd_scheduler(optimizer, num_warmup_steps, num_training_steps, stable_ratio=0.8):
    """Warmup-Stable-Decay (WSD) Scheduler"""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        stable_steps = int(num_training_steps * stable_ratio)
        if current_step < stable_steps:
            return 1.0
        # Decay phase (linear to 0.1)
        decay_steps = num_training_steps - stable_steps
        step_in_decay = current_step - stable_steps
        return max(0.1, 1.0 - 0.9 * (float(step_in_decay) / float(max(1, decay_steps))))
        
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)

def train(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="bf16")
    set_seed(args.seed)
    
    # 使用自定义 SPMTokenizer 加载 SentencePiece 模型
    tokenizer_dir = os.path.join(args.data_dir, "tokenizer_v3")
    tokenizer = SPMTokenizer.from_pretrained(tokenizer_dir)
    
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size, hidden_size=args.d_model,
        intermediate_size=int(args.d_model * 8 / 3), num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head, num_key_value_heads=args.n_kv_heads,
        max_position_embeddings=args.max_length, rms_norm_eps=1e-6, rope_theta=10000.0,
        tie_word_embeddings=False, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        hidden_act="silu", attn_implementation="sdpa" if args.use_flash_attention else "eager"
    )
    model = LlamaForCausalLM(config)
    if args.gradient_checkpointing: model.gradient_checkpointing_enable()
    
    train_file = os.path.join(args.data_dir, "processed_v3", "train.txt")
    val_file = os.path.join(args.data_dir, "processed_v3", "val.txt")
    
    full_dataset = DocumentAwareDataset(tokenizer, train_file, args.max_length)
    val_dataset = DocumentAwareDataset(tokenizer, val_file, args.max_length)
    
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "rmsnorm.weight"]
    optimizer_grouped_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_params, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    warmup_steps = int(args.warmup_ratio * max_train_steps)
    
    lr_scheduler = get_wsd_scheduler(optimizer, warmup_steps, max_train_steps)
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, lr_scheduler)
    
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or f"llama-v3-{args.d_model}d", mode=args.wandb_mode)
        os.makedirs(args.output_dir, exist_ok=True)
        config.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
    global_step, best_val_loss, patience_counter = 0, float("inf"), 0
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, epoch_tokens = 0.0, 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients: accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
            bs, seq_len = batch["input_ids"].shape
            epoch_loss += loss.item() * bs * seq_len
            epoch_tokens += bs * seq_len
            
            if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                wandb.log({"train/loss": loss.item(), "train/lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
                
        # Evaluation & Early Stopping
        val_loss = evaluate(model, val_loader, accelerator)
        accelerator.print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val PPL: {math.exp(min(val_loss, 20)):.2f}")
        
        if accelerator.is_main_process:
            wandb.log({"eval/loss": val_loss, "eval/ppl": math.exp(min(val_loss, 20)), "epoch": epoch+1}, step=global_step)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_dir = os.path.join(args.output_dir, "best_model")
                os.makedirs(best_dir, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                accelerator.print(f"✅ 保存新最佳模型: {best_dir}")
            else:
                patience_counter += 1
                accelerator.print(f"⚠️ 早停计数: {patience_counter} / {args.patience}")
                
        if patience_counter >= args.patience:
            accelerator.print(f"\n⏹️ 验证集 Loss 连续 {args.patience} 轮未改善，触发早停机制。")
            break

    if accelerator.is_main_process:
        wandb.finish()
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output/babylm-llama-v3")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--use_flash_attention", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="chinese-babylm")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")
    train(parser.parse_args())
