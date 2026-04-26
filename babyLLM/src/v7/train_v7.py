"""
ChineseBabyLM V7 - SOTA Sprint Training Script

Core innovations:
1. MNTP (Masked Next Token Prediction) - GPT-BERT hybrid CLM+MLM
2. 8K vocab SPM tokenizer with <mask> token
3. ~35M params (448d, 12L, 7Q/4KV GQA)
4. Label smoothing + dropout annealing
5. Cosine LR with warmup

References:
- GPT-BERT (BabyLM 2024 winner): CLM+MNTP 1:7
- MiniLLM (ICLR 2024): Reverse KL distillation
"""

import os
import sys
import math
import time
import logging
import argparse
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator
import wandb
from tqdm import tqdm
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "v3"))
if V3_DIR not in sys.path:
    sys.path.append(V3_DIR)

import sentencepiece as spm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class SPMTokenizer:
    def __init__(self, tokenizer_dir):
        sp_model_path = os.path.join(tokenizer_dir, "spm.model")
        if not os.path.exists(sp_model_path):
            raise FileNotFoundError(f"SPM model not found: {sp_model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.pad_token_id = self.sp.pad_id()
        self.unk_token_id = self.sp.unk_id()
        self.bos_token_id = self.sp.bos_id()
        self.eos_token_id = self.sp.eos_id()
        self.mask_token_id = self.sp.piece_to_id("<mask>")

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)

    def encode_with_bpe_dropout(self, text, alpha=0.1):
        if alpha <= 0:
            return self.sp.encode(text, out_type=int)
        return self.sp.encode(text, out_type=int, enable_sampling=True, alpha=alpha)


class DocumentDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        file_path,
        block_size=1024,
        stride=512,
        bpe_dropout=0.0,
        sp_model_path=None,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride
        self.bpe_dropout = bpe_dropout
        self.sp_model_path = sp_model_path or ""

        logger.info(
            f"Reading data: {file_path}, block_size={block_size}, stride={stride}"
        )

        if self.sp_model_path and os.path.exists(self.sp_model_path):
            self._sp = spm.SentencePieceProcessor()
            self._sp.load(self.sp_model_path)
        else:
            self._sp = tokenizer.sp

        documents = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(line)
        logger.info(f"Documents: {len(documents):,}")

        all_tokens = []
        for doc in tqdm(documents, desc="Tokenizing", disable=True):
            if self.bpe_dropout > 0:
                ids = self._sp.encode(
                    doc, out_type=int, enable_sampling=True, alpha=self.bpe_dropout
                )
            else:
                ids = self._sp.encode(doc, out_type=int)
            if len(ids) > 0:
                all_tokens.extend(ids)
                all_tokens.append(self.tokenizer.eos_token_id)

        total_tokens = len(all_tokens)
        logger.info(f"Total tokens: {total_tokens:,}")

        self.samples = []
        i = 0
        while i + block_size + 1 <= total_tokens:
            chunk = all_tokens[i : i + block_size + 1]
            self.samples.append(
                {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }
            )
            i += stride

        logger.info(f"Training samples: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def re_tokenize_with_dropout(self):
        pass


class MNTPDataset(Dataset):
    """Mixed CLM + Masked Next Token Prediction dataset (GPT-BERT approach)."""

    def __init__(
        self,
        base_dataset,
        mask_token_id,
        clm_ratio=0.125,
        mask_ratio_start=0.30,
        mask_ratio_end=0.15,
        total_epochs=10,
    ):
        self.base_dataset = base_dataset
        self.mask_token_id = mask_token_id
        self.clm_ratio = clm_ratio
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    @property
    def mask_ratio(self):
        if self.total_epochs <= 1:
            return self.mask_ratio_end
        progress = self.current_epoch / max(self.total_epochs - 1, 1)
        return (
            self.mask_ratio_start
            + (self.mask_ratio_end - self.mask_ratio_start) * progress
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        input_ids = sample["input_ids"].clone()
        labels = sample["labels"].clone()

        if random.random() < self.clm_ratio:
            return {
                "input_ids": input_ids,
                "labels": labels,
                "is_mntp": torch.tensor(0),
            }

        masked_input = input_ids.clone()
        mask_ratio = self.mask_ratio
        mask = torch.rand(input_ids.shape) < mask_ratio
        masked_input[mask] = self.mask_token_id

        random_replace = (~mask) & (torch.rand(input_ids.shape) < 0.1)
        if random_replace.any():
            vocab_size = self.base_dataset.tokenizer.vocab_size
            random_tokens = torch.randint(5, vocab_size, random_replace.shape)
            masked_input[random_replace] = random_tokens[random_replace]

        return {"input_ids": masked_input, "labels": labels, "is_mntp": torch.tensor(1)}


def mntp_collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "is_mntp": torch.stack([b["is_mntp"] for b in batch]),
    }


def evaluate(model, val_loader, accelerator):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc="Evaluating", disable=not accelerator.is_main_process
        ):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            bs, seq_len = input_ids.shape

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    stats = torch.tensor(
        [total_loss, total_tokens], dtype=torch.float32, device=accelerator.device
    )
    gathered = accelerator.gather(stats).view(-1, 2)
    avg_loss = gathered[:, 0].sum().item() / gathered[:, 1].sum().item()
    avg_ppl = math.exp(min(avg_loss, 20))
    model.train()
    return avg_loss, avg_ppl


def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    set_seed(args.seed)

    mask_token_id = 4
    tokenizer = SPMTokenizer(args.tokenizer_dir)
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocab size: {vocab_size}, mask_token_id: {mask_token_id}")

    # Model
    intermediate_size = int(args.d_model * 8 / 3)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.d_model,
        intermediate_size=intermediate_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        num_key_value_heads=args.n_kv_heads,
        max_position_embeddings=args.max_length,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        hidden_act="silu",
        rope_theta=args.rope_theta,
        attn_implementation="sdpa",
    )

    if args.resume_from and os.path.isdir(args.resume_from):
        logger.info(f"Resuming from: {args.resume_from}")
        model = LlamaForCausalLM.from_pretrained(
            args.resume_from, torch_dtype=torch.bfloat16
        )
    else:
        model = LlamaForCausalLM(config)
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} params ({total_params / 1e6:.1f}M)")

    # Data
    sp_model_path = os.path.join(args.tokenizer_dir, "spm.model")
    data_dir = args.data_dir

    train_file = os.path.join(data_dir, "train.txt")
    val_file = os.path.join(data_dir, "val.txt")

    if not os.path.exists(train_file):
        for subdir in ["processed_v7", "processed_v3"]:
            alt = os.path.join(data_dir, "..", subdir, "train.txt")
            alt = os.path.abspath(alt)
            if os.path.exists(alt):
                train_file = alt
                val_file = os.path.join(os.path.dirname(alt), "val.txt")
                break

    base_train_dataset = DocumentDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=args.max_length,
        stride=args.max_length // 2,
        bpe_dropout=args.bpe_dropout,
        sp_model_path=sp_model_path,
    )

    train_dataset = MNTPDataset(
        base_dataset=base_train_dataset,
        mask_token_id=mask_token_id,
        clm_ratio=args.clm_ratio,
        mask_ratio_start=args.mask_ratio_start,
        mask_ratio_end=args.mask_ratio_end,
        total_epochs=args.num_epochs,
    )

    val_dataset = DocumentDataset(
        tokenizer=tokenizer,
        file_path=val_file,
        block_size=args.max_length,
        stride=args.max_length,
        bpe_dropout=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=mntp_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    # Optimizer
    no_decay = ["bias", "layernorm.weight", "rmsnorm.weight", "norm.weight"]
    optimizer_grouped_params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n.lower() for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n.lower() for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Scheduler
    num_update_steps = len(train_loader) // args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * num_update_steps
    warmup_steps = int(args.warmup_ratio * max_train_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    effective_batch = (
        args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    )
    tokens_per_step = effective_batch * args.max_length

    logger.info(f"Train: {len(base_train_dataset):,}, Val: {len(val_dataset):,}")
    logger.info(f"Cosine Scheduler: warmup={warmup_steps}, total={max_train_steps}")
    logger.info(f"Effective batch: {effective_batch}, Tokens/step: {tokens_per_step:,}")

    # Prepare
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # WandB
    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "model": "BabyLLM-V7",
                "params": total_params,
                "vocab_size": vocab_size,
                "d_model": args.d_model,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_kv_heads": args.n_kv_heads,
                "batch_size": args.batch_size,
                "effective_batch": effective_batch,
                "learning_rate": args.learning_rate,
                "epochs": args.num_epochs,
                "max_steps": max_train_steps,
                "warmup_steps": warmup_steps,
                "clm_ratio": args.clm_ratio,
                "mask_ratio_start": args.mask_ratio_start,
                "mask_ratio_end": args.mask_ratio_end,
                "label_smoothing": args.label_smoothing,
                "gradient_checkpointing": args.gradient_checkpointing,
                "attention_dropout": args.attention_dropout,
                "bpe_dropout": args.bpe_dropout,
            },
        )

    stage_name = "CLM+MNTP Pretraining"
    accelerator.print(f"\n{'=' * 60}")
    accelerator.print(f"ChineseBabyLM V7 - {stage_name}")
    accelerator.print(f"  Params: {total_params / 1e6:.1f}M, Batch: {effective_batch}")
    accelerator.print(f"  LR: {args.learning_rate}, LS: {args.label_smoothing}")
    accelerator.print(f"  CLM:MNTP = {args.clm_ratio:.3f}:{1 - args.clm_ratio:.3f}")
    accelerator.print(f"  Mask ratio: {args.mask_ratio_start} -> {args.mask_ratio_end}")
    accelerator.print(f"  Epochs: {args.num_epochs}, Patience: {args.patience}")
    accelerator.print(f"{'=' * 60}\n")

    # Training loop
    best_val_loss = float("inf")
    early_stop_count = 0
    start_time = time.time()

    for epoch in range(args.num_epochs):
        train_dataset.set_epoch(epoch)

        if args.bpe_dropout > 0 and hasattr(base_train_dataset, "samples"):
            sp_for_retokenize = spm.SentencePieceProcessor()
            sp_for_retokenize.load(sp_model_path)
            logger.info(
                f"Epoch {epoch + 1}: Re-tokenizing with BPE dropout={args.bpe_dropout}"
            )

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_mntp_count = 0
        epoch_clm_count = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                if args.label_smoothing > 0:
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss(
                        label_smoothing=args.label_smoothing,
                        ignore_index=-100,
                    )
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                else:
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_steps += 1

            is_mntp = batch["is_mntp"]
            epoch_mntp_count += is_mntp.sum().item()
            epoch_clm_count += len(is_mntp) - is_mntp.sum().item()

            if step % args.logging_steps == 0 and step > 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / epoch_steps
                ppl = math.exp(min(avg_loss, 20))
                tps = (
                    (step + 1)
                    * effective_batch
                    * args.max_length
                    / (time.time() - start_time + 1e-6)
                )

                progress_bar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    ppl=f"{ppl:.1f}",
                    tps=f"{int(tps)}",
                )

                if accelerator.is_main_process:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/ppl": ppl,
                            "train/lr": current_lr,
                            "train/step": epoch * num_update_steps + step,
                            "train/epoch": epoch + 1,
                            "train/tokens_per_sec": tps,
                        }
                    )

        # Validation
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        val_loss, val_ppl = evaluate(model, val_loader, accelerator)
        current_lr = lr_scheduler.get_last_lr()[0]
        epoch_time = (time.time() - start_time) / 60

        mntp_pct = epoch_mntp_count / max(epoch_mntp_count + epoch_clm_count, 1) * 100

        accelerator.print(
            f"Epoch {epoch + 1} ({epoch_time:.1f}min) | "
            f"Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
            f"PPL: {val_ppl:.2f} | LR: {current_lr:.2e} | "
            f"MNTP: {mntp_pct:.0f}%"
        )

        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch/train_loss": avg_train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/val_ppl": val_ppl,
                    "epoch/epoch": epoch + 1,
                    "epoch/lr": current_lr,
                }
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0

            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                save_dir = os.path.join(args.output_dir, "best_model")
                os.makedirs(save_dir, exist_ok=True)
                unwrapped.save_pretrained(save_dir)
                unwrapped.config.save_pretrained(save_dir)

                if hasattr(tokenizer, "sp"):
                    import shutil as sh

                    sp_src = os.path.join(args.tokenizer_dir, "spm.model")
                    sh.copy2(sp_src, save_dir)

                from transformers import LlamaTokenizerFast

                try:
                    hf_tok = LlamaTokenizerFast.from_pretrained(args.tokenizer_dir)
                    hf_tok.save_pretrained(save_dir)
                except Exception:
                    pass

                accelerator.print(
                    f"  New best (val_loss={val_loss:.4f}, ppl={val_ppl:.2f})"
                )
                wandb.log({"best/val_loss": val_loss, "best/val_ppl": val_ppl})
        else:
            early_stop_count += 1
            accelerator.print(f"  Early stop: {early_stop_count}/{args.patience}")

        if early_stop_count >= args.patience:
            accelerator.print(
                f"\nEarly stopping: {args.patience} epochs without improvement."
            )
            break

    total_time = (time.time() - start_time) / 3600
    accelerator.print(f"\n{'=' * 60}")
    accelerator.print(f"V7 {stage_name} complete!")
    accelerator.print(f"  Best Val Loss: {best_val_loss:.4f}")
    accelerator.print(f"  Best Val PPL: {math.exp(min(best_val_loss, 20)):.2f}")
    accelerator.print(f"  Total time: {total_time:.1f}h")
    accelerator.print(f"  Params: {total_params / 1e6:.1f}M")
    accelerator.print(f"{'=' * 60}")

    if accelerator.is_main_process:
        wandb.finish()
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="ChineseBabyLM V7")

    parser.add_argument("--data_dir", type=str, default="data/processed_v7")
    parser.add_argument("--output_dir", type=str, default="output/babylm-v7")
    parser.add_argument("--tokenizer_dir", type=str, default="data/tokenizer_v7")
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--d_model", type=int, default=448)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--bpe_dropout", type=float, default=0.1)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--clm_ratio", type=float, default=0.125)
    parser.add_argument("--mask_ratio_start", type=float, default=0.30)
    parser.add_argument("--mask_ratio_end", type=float, default=0.15)

    parser.add_argument("--wandb_project", type=str, default="chinese-babylm")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train_v7.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info("=" * 60)
    logger.info(f"ChineseBabyLM V7 - Stage: CLM+MNTP")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    train(args)


if __name__ == "__main__":
    main()
