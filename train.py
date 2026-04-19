"""
ChineseBabyLM 挑战赛训练脚本
使用 GPT-2 架构在 babylm-zho-100M 数据上从头预训练
支持多GPU DDP训练 (通过 HuggingFace Accelerate)
适配硬件: NVIDIA RTX A6000 (49GB VRAM) x3
"""

import os
import math
import logging
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    get_scheduler,
    set_seed,
)
from accelerate import Accelerator
import wandb

from tqdm import tqdm


# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据集类
# ============================================================
class TextDataset(Dataset):
    """从文本文件构建语言模型的dataset（分块处理，节省内存）"""

    def __init__(self, tokenizer, file_path, block_size=512):
        assert os.path.isfile(file_path), f"文件不存在: {file_path}"
        
        self.block_size = block_size
        
        logger.info(f"读取数据文件: {file_path}")
        
        # 分块读取和 tokenize，避免一次性加载全部文本
        chunk_size = block_size * 8  # 每次读取的文本块大小
        
        self.examples = []
        token_buffer = []
        total_tokens = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Tokenizing"):
                line = line.strip()
                if not line:
                    continue
                
                # 逐行 tokenize
                ids = tokenizer.encode(line)
                token_buffer.extend(ids)
                total_tokens += len(ids)
                
                # 当缓冲区足够大时，分割为训练样本
                while len(token_buffer) >= block_size:
                    chunk = token_buffer[:block_size]
                    self.examples.append(chunk)
                    token_buffer = token_buffer[block_size:]
        
        # 处理剩余的 tokens
        if len(token_buffer) > 0:
            # 填充到 block_size
            padding = [tokenizer.pad_token_id or 0] * (block_size - len(token_buffer))
            chunk = token_buffer + padding
            self.examples.append(chunk)
        
        logger.info(f"Token 总数: {total_tokens:,}")
        logger.info(f"创建 {len(self.examples)} 个训练样本 (block_size={block_size})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


# ============================================================
# 训练函数
# ============================================================
def train(args):
    # ============================================================
    # 初始化 Accelerate (自动管理多GPU DDP)
    # ============================================================
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="no",  # A6000 不需要混合精度
    )
    
    # 设备信息
    accelerator.print(f"使用设备: {accelerator.device}")
    accelerator.print(f"进程数: {accelerator.num_processes}")
    if torch.cuda.is_available():
        accelerator.print(f"GPU: {torch.cuda.get_device_name()}")
        accelerator.print(f"GPU 内存: {torch.cuda.get_device_properties(accelerator.local_process_index).total_memory / 1024**3:.1f} GB")
    
    # 随机种子
    set_seed(args.seed)
    
    # ============================================================
    # 加载 Tokenizer
    # ============================================================
    tokenizer_dir = os.path.join(args.data_dir, "tokenizer")
    accelerator.print(f"加载 Tokenizer: {tokenizer_dir}")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    vocab_size = tokenizer.vocab_size
    accelerator.print(f"词表大小: {vocab_size}")
    
    # ============================================================
    # 模型配置 (GPT-2 Small, ~110M 参数)
    # ============================================================
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.max_length,
        n_embd=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        summary_first_dropout=0.1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    if accelerator.is_main_process:
        logger.info(f"\n模型配置:")
        logger.info(f"  词表大小: {config.vocab_size}")
        logger.info(f"  序列长度: {config.n_positions}")
        logger.info(f"  隐藏维度: {config.n_embd}")
        logger.info(f"  层数: {config.n_layer}")
        logger.info(f"  注意力头: {config.n_head}")
    
    # 创建模型
    model = GPT2LMHeadModel(config)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"\n模型参数量:")
    accelerator.print(f"  总参数: {total_params:,} ({total_params/1e6:.1f}M)")
    accelerator.print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    # ============================================================
    # 数据集
    # ============================================================
    train_file = os.path.join(args.data_dir, "processed", "train.txt")
    if not os.path.exists(train_file):
        train_file = os.path.join(args.data_dir, "processed", "all.txt")
    
    accelerator.print(f"\n加载训练数据: {train_file}")
    
    # 只在主进程显示 tokenizing 进度条
    if not accelerator.is_main_process:
        import tqdm as _tqdm
        _tqdm.tqdm.disable = True
    
    full_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=args.max_length,
    )
    
    # 划分训练集和验证集 (90% / 10%)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    accelerator.print(f"训练集: {len(train_dataset)} 样本")
    accelerator.print(f"验证集: {len(val_dataset)} 样本")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    
    # ============================================================
    # 优化器和学习率调度器
    # ============================================================
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # 计算总训练步数 (考虑多GPU)
    num_update_steps_per_epoch = len(train_loader) // accelerator.num_processes // args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * max_train_steps),
        num_training_steps=max_train_steps,
    )
    
    # ============================================================
    # Accelerate 准备 (自动处理DDP, 设备分配等)
    # ============================================================
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    
    # ============================================================
    # WandB 初始化 (只在主进程)
    # ============================================================
    if accelerator.is_main_process:
        run_name = args.wandb_run_name or f"gpt2-{args.d_model}d-{args.n_layer}l-{args.n_head}h-3gpu"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            config={
                "model": "GPT-2",
                "d_model": args.d_model,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "max_length": args.max_length,
                "vocab_size": vocab_size,
                "total_params": total_params,
                "batch_size_per_gpu": args.batch_size,
                "effective_batch_size": effective_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_epochs": args.num_epochs,
                "warmup_ratio": args.warmup_ratio,
                "lr_scheduler": args.lr_scheduler_type,
                "max_grad_norm": args.max_grad_norm,
                "num_gpus": accelerator.num_processes,
                "seed": args.seed,
                "dataset": "babylm-zho-100M",
            },
        )
        accelerator.print(f"WandB 已初始化: project={args.wandb_project}, run={run_name}")
    
    accelerator.print(f"\n训练配置:")
    accelerator.print(f"  GPU 数量: {accelerator.num_processes}")
    accelerator.print(f"  Batch size/GPU: {args.batch_size}")
    accelerator.print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    accelerator.print(f"  有效 batch size: {effective_batch_size}")
    accelerator.print(f"  学习率: {args.learning_rate}")
    accelerator.print(f"  调度器: {args.lr_scheduler_type}")
    accelerator.print(f"  预热比例: {args.warmup_ratio}")
    accelerator.print(f"  Epoch数: {args.num_epochs}")
    accelerator.print(f"  总训练步数: {max_train_steps}")
    
    # ============================================================
    # 输出目录
    # ============================================================
    output_dir = args.output_dir
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    # ============================================================
    # 训练循环
    # ============================================================
    accelerator.print("\n" + "=" * 60)
    accelerator.print("开始训练!")
    accelerator.print("=" * 60)
    
    global_step = 0
    best_val_loss = float("inf")
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            ncols=120,
            disable=not accelerator.is_main_process,
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                
                # 前向传播
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                # 反向传播 (Accelerate 管理)
                accelerator.backward(loss)
                
                # 梯度裁剪
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item()
            
            # 更新进度条
            current_lr = lr_scheduler.get_last_lr()[0]
            raw_loss = loss.item()
            ppl = math.exp(min(raw_loss, 20))
            progress_bar.set_postfix(
                loss=f"{raw_loss:.4f}",
                ppl=f"{ppl:.2f}",
                lr=f"{current_lr:.2e}",
                step=global_step,
            )
            
            # 定期日志 + WandB (只在主进程)
            if global_step % args.logging_steps == 0 and global_step > 0 and accelerator.is_main_process:
                avg_loss = epoch_loss / (step + 1)
                logger.info(
                    f"Epoch {epoch+1} Step {global_step} | "
                    f"Loss: {avg_loss:.4f} | PPL: {math.exp(min(avg_loss, 20)):.2f} | "
                    f"LR: {current_lr:.2e}"
                )
                wandb.log({
                    "train/loss": raw_loss,
                    "train/avg_loss": avg_loss,
                    "train/ppl": ppl,
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch + 1,
                    "train/step": global_step,
                }, step=global_step)
            
            # 定期保存 (只在主进程)
            if global_step % args.save_steps == 0 and global_step > 0 and accelerator.is_main_process:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"保存 checkpoint: {checkpoint_dir}")
        
        # Epoch 结束，验证
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, accelerator)
        val_ppl = math.exp(min(val_loss, 20))
        
        accelerator.print(
            f"\nEpoch {epoch+1} 完成 | "
            f"训练 Loss: {avg_train_loss:.4f} | "
            f"验证 Loss: {val_loss:.4f} | "
            f"验证 PPL: {val_ppl:.2f}"
        )
        
        # WandB 记录 epoch 指标 (只在主进程)
        if accelerator.is_main_process:
            wandb.log({
                "epoch/train_loss": avg_train_loss,
                "epoch/val_loss": val_loss,
                "epoch/val_ppl": val_ppl,
                "epoch/best_val_loss": best_val_loss,
                "epoch/epoch": epoch + 1,
            }, step=global_step)
        
        # 保存最佳模型 (只在主进程)
        if val_loss < best_val_loss and accelerator.is_main_process:
            best_val_loss = val_loss
            best_dir = os.path.join(output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logger.info(f"保存最佳模型 (val_loss={val_loss:.4f}): {best_dir}")
        
        # 每个 epoch 结束保存 (只在主进程)
        if accelerator.is_main_process:
            epoch_dir = os.path.join(output_dir, f"epoch-{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
    
    # ============================================================
    # 训练结束
    # ============================================================
    accelerator.print("\n" + "=" * 60)
    accelerator.print("训练完成!")
    accelerator.print(f"最佳验证 Loss: {best_val_loss:.4f}")
    accelerator.print(f"最佳验证 PPL: {math.exp(min(best_val_loss, 20)):.2f}")
    accelerator.print(f"模型保存在: {output_dir}")
    accelerator.print("=" * 60)
    
    # WandB 结束 (只在主进程)
    if accelerator.is_main_process:
        wandb.finish()
    
    accelerator.end_training()


def evaluate(model, val_loader, accelerator):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中", ncols=100, 
                         disable=not accelerator.is_main_process):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
    
    # 多GPU间同步 loss
    total_loss_tensor = torch.tensor(total_loss, device=accelerator.device)
    num_batches_tensor = torch.tensor(len(val_loader), device=accelerator.device)
    
    # Gather from all processes
    gathered_losses = accelerator.gather(total_loss_tensor)
    gathered_counts = accelerator.gather(num_batches_tensor)
    
    avg_loss = gathered_losses.sum().item() / gathered_counts.sum().item()
    model.train()
    return avg_loss


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ChineseBabyLM 训练脚本")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="output/babylm-gpt2", help="输出目录")
    
    # 模型参数
    parser.add_argument("--d_model", type=int, default=768, help="隐藏层维度")
    parser.add_argument("--n_layer", type=int, default=12, help="Transformer层数")
    parser.add_argument("--n_head", type=int, default=12, help="注意力头数")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16, help="每GPU批次大小")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮次")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="最大梯度范数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器")
    
    # 日志和保存
    parser.add_argument("--logging_steps", type=int, default=100, help="日志间隔步数")
    parser.add_argument("--save_steps", type=int, default=2000, help="保存间隔步数")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # wandb 参数
    parser.add_argument("--wandb_project", type=str, default="chinese-babylm", help="wandb 项目名")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb 实体/团队名")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb 运行名称")
    parser.add_argument("--wandb_mode", type=str, default="online", help="wandb 模式: online/offline/disabled")
    
    args = parser.parse_args()
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("ChineseBabyLM 训练配置 (Accelerate 多GPU + WandB)")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)
    
    train(args)


if __name__ == "__main__":
    main()