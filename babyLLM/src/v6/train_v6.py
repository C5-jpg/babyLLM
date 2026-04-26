"""
ChineseBabyLM V6 - SOTA Sprint Training Script

Core innovations over V5:
1. Three-stage training: CLM -> CLM+MLM (1:7 MNTP) -> Reverse KL KD
2. Label Smoothing 0.1 (new)
3. WSD (Warmup-Stable-Decay) scheduler option
4. ~80M params (640d, 12L, 10Q/5KV GQA)
5. Enhanced data cleaning output support

References:
- GPT-BERT (BabyLM 2024 winner): CLM+MLM 1:7 with MNTP
- MiniLLM (ICLR 2024): Reverse KL distillation
- WSD scheduler (ICLR 2025): River Valley paper
"""

import os
import sys
import math
import time
import logging
import argparse
import shutil
import glob
import random
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
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "v3"))
if V3_DIR not in sys.path:
    sys.path.append(V3_DIR)

import sentencepiece as spm
from spm_tokenizer import SPMTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def check_disk_space(path, min_free_gb=5.0):
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        logger.warning(f"Disk space low: {free_gb:.1f}GB free at {path}")
        return False
    logger.info(f"Disk space OK: {free_gb:.1f}GB free")
    return True


def safe_save_model(model, save_dir, accelerator, config=None, tokenizer=None):
    os.makedirs(save_dir, exist_ok=True)
    if config is not None:
        config.save_pretrained(save_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)
    try:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(save_dir)
    except Exception as e:
        logger.warning(f"save_pretrained failed: {e}, trying torch.save fallback...")
    weight_files = (
        glob.glob(os.path.join(save_dir, "model.safetensors"))
        + glob.glob(os.path.join(save_dir, "model-*.safetensors"))
        + glob.glob(os.path.join(save_dir, "pytorch_model.bin"))
    )
    if not weight_files:
        logger.warning("No weight files found, using torch.save fallback...")
        try:
            state_dict = accelerator.get_state_dict(model)
            fallback_path = os.path.join(save_dir, "pytorch_model.bin")
            torch.save(state_dict, fallback_path)
            weight_files = [fallback_path]
        except Exception as e:
            raise RuntimeError(f"Model save completely failed: {e}")
    total_size = sum(os.path.getsize(f) for f in weight_files)
    logger.info(f"Model saved: {len(weight_files)} files, {total_size / 1024**2:.1f}MB")
    return True


class DocumentAwareDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        file_path,
        block_size=1024,
        bpe_dropout=0.0,
        sp_model_path=None,
        encode_batch_size=4096,
    ):
        assert os.path.isfile(file_path), f"File not found: {file_path}"
        self.block_size = block_size
        self.bpe_dropout = bpe_dropout
        self.sp_model_path = sp_model_path
        self.sp = None
        self.encode_batch_size = encode_batch_size
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id or 0

        if bpe_dropout > 0 and sp_model_path and os.path.exists(sp_model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(sp_model_path)
            logger.info(f"BPE Dropout enabled: alpha={bpe_dropout}")

        logger.info(f"Reading data: {file_path}")
        self.lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.lines.append(line)
        logger.info(f"Documents: {len(self.lines):,}")

        self.examples = []
        self._tokenize(use_dropout=False)
        self._fixed_length = len(self.examples)
        logger.info(f"Training samples: {len(self.examples):,}")

    def _tokenize(self, use_dropout=False):
        all_token_ids = []
        if use_dropout and self.sp is not None:
            for line in self.lines:
                ids = self.sp.encode(
                    line, enable_sampling=True, alpha=self.bpe_dropout, nbest_size=-1
                )
                all_token_ids.extend(ids)
                all_token_ids.append(self.eos_id)
        else:
            if hasattr(self.tokenizer, "__call__"):
                for i in range(0, len(self.lines), self.encode_batch_size):
                    batch_lines = self.lines[i : i + self.encode_batch_size]
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

        stride = self.block_size // 2
        new_examples = []
        i = 0
        while i + self.block_size + 1 <= len(all_token_ids):
            new_examples.append(all_token_ids[i : i + self.block_size + 1])
            i += stride
        if len(new_examples) == 0 or (
            i < len(all_token_ids) - 1 and len(all_token_ids) - i > 1
        ):
            remaining = all_token_ids[i:]
            if len(remaining) > 1:
                padded = remaining + [self.pad_id] * (
                    self.block_size + 1 - len(remaining)
                )
                new_examples.append(padded[: self.block_size + 1])
        self.examples = new_examples

    def retokenize(self):
        if self.sp is None:
            return
        self._tokenize(use_dropout=True)
        if len(self.examples) < self._fixed_length:
            pad = self.examples[-1] if self.examples else [0] * (self.block_size + 1)
            self.examples.extend([pad] * (self._fixed_length - len(self.examples)))
        elif len(self.examples) > self._fixed_length:
            self.examples = self.examples[: self._fixed_length]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


class CLMMLMDataset(Dataset):
    """CLM + MLM mixed dataset (GPT-BERT MNTP approach)"""

    def __init__(
        self,
        base_dataset,
        clm_ratio=0.125,
        mask_ratio_start=0.30,
        mask_ratio_end=0.15,
        total_epochs=4,
        current_epoch=0,
        mask_token_id=4,
        vocab_size=32000,
    ):
        self.base = base_dataset
        self.clm_ratio = clm_ratio
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        progress = current_epoch / max(1, total_epochs)
        self.mask_ratio = (
            mask_ratio_start - (mask_ratio_start - mask_ratio_end) * progress
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        input_ids = item["input_ids"]
        labels = item["labels"]

        if random.random() < self.clm_ratio:
            return {"input_ids": input_ids, "labels": labels}

        masked_input = input_ids.clone()
        seq_len = input_ids.shape[0]
        mask = torch.rand(seq_len) < self.mask_ratio
        rand = torch.rand(seq_len)

        replace_mask = torch.tensor(
            [random.randint(5, self.vocab_size - 1) for _ in range(seq_len)]
        )
        masked_input[mask & (rand < 0.8)] = self.mask_token_id
        masked_input[mask & (rand >= 0.8) & (rand < 0.9)] = replace_mask[
            mask & (rand >= 0.8) & (rand < 0.9)
        ]

        mlm_labels = torch.zeros_like(labels).fill_(-100)
        mlm_labels[:-1] = input_ids[1:]
        mlm_labels[~mask] = -100

        return {"input_ids": masked_input, "labels": mlm_labels}


class KDDataset(Dataset):
    def __init__(self, base_dataset, teacher_logits_dir, top_k=10):
        self.base = base_dataset
        self.top_k = top_k
        self.teacher_logits = None
        self.teacher_indices = None
        logits_path = os.path.join(teacher_logits_dir, "teacher_logits.npy")
        indices_path = os.path.join(teacher_logits_dir, "teacher_indices.npy")
        if os.path.exists(logits_path) and os.path.exists(indices_path):
            self.teacher_logits = np.load(logits_path, mmap_mode="r")
            self.teacher_indices = np.load(indices_path, mmap_mode="r")
            logger.info(f"Teacher logits loaded: {self.teacher_logits.shape}")
        else:
            logger.warning(f"Teacher logits not found: {logits_path}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        result = {"input_ids": item["input_ids"], "labels": item["labels"]}
        if self.teacher_logits is not None and idx < len(self.teacher_logits):
            result["teacher_logits"] = torch.tensor(
                self.teacher_logits[idx], dtype=torch.float16
            )
            result["teacher_indices"] = torch.tensor(
                self.teacher_indices[idx], dtype=torch.long
            )
        else:
            seq_len = item["input_ids"].shape[0]
            result["teacher_logits"] = torch.zeros(
                seq_len, self.top_k, dtype=torch.float16
            )
            result["teacher_indices"] = torch.zeros(
                seq_len, self.top_k, dtype=torch.long
            )
        return result


def evaluate(model, val_loader, accelerator):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(
            val_loader,
            desc="Evaluating",
            ncols=100,
            disable=not accelerator.is_main_process,
        ):
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            input_ids = batch["input_ids"]
            if input_ids.dim() == 1:
                bs, seq_len = 1, input_ids.shape[0]
            else:
                bs, seq_len = input_ids.shape
            total_loss += outputs.loss.item() * bs * seq_len
            total_tokens += bs * seq_len
    stats = torch.tensor(
        [total_loss, total_tokens], dtype=torch.float32, device=accelerator.device
    )
    gathered = accelerator.gather(stats).view(-1, 2)
    avg_loss = gathered[:, 0].sum().item() / gathered[:, 1].sum().item()
    model.train()
    return avg_loss


def set_dropout(model, dropout_p):
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_p
            count += 1
    return count


def compute_ce_loss_with_sm(model, input_ids, labels, label_smoothing=0.0):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    if label_smoothing > 0:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )
    else:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
    return loss, logits


def compute_kd_loss(
    student_logits,
    teacher_logits_topk,
    teacher_indices_topk,
    labels,
    temperature=2.0,
    lambda_ce=0.5,
    lambda_kd=0.5,
    top_k=10,
    label_smoothing=0.0,
    use_reverse_kl=False,
):
    batch_size, seq_len, vocab_size = student_logits.shape

    ce_loss = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )

    flat_student = student_logits.view(-1, vocab_size)
    flat_teacher_indices = teacher_indices_topk.view(-1, top_k)

    student_topk = torch.gather(flat_student, dim=1, index=flat_teacher_indices)

    if use_reverse_kl:
        teacher_probs = F.softmax(
            teacher_logits_topk.view(-1, top_k).float() / temperature, dim=-1
        )
        student_probs = F.softmax(student_topk / temperature, dim=-1)
        kd_loss = F.kl_div(
            teacher_probs.log(), student_probs, reduction="batchmean"
        ) * (temperature**2)
    else:
        teacher_probs = F.softmax(
            teacher_logits_topk.view(-1, top_k).float() / temperature, dim=-1
        )
        student_log_probs = F.log_softmax(student_topk / temperature, dim=-1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
            temperature**2
        )

    total_loss = lambda_ce * ce_loss + lambda_kd * kd_loss
    return total_loss, {
        "ce_loss": ce_loss.item(),
        "kd_loss": kd_loss.item(),
        "total_loss": total_loss.item(),
    }


class WSDScheduler:
    """Warmup-Stable-Decay scheduler (ICLR 2025 River Valley)"""

    def __init__(
        self,
        optimizer,
        num_warmup_steps,
        num_stable_steps,
        num_decay_steps,
        min_lr_ratio=0.1,
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_stable_steps = num_stable_steps
        self.num_decay_steps = num_decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self._get_lr(base_lr)

    def _get_lr(self, base_lr):
        s = self.current_step
        min_lr = base_lr * self.min_lr_ratio
        if s <= self.num_warmup_steps:
            return base_lr * s / max(1, self.num_warmup_steps)
        elif s <= self.num_warmup_steps + self.num_stable_steps:
            return base_lr
        else:
            progress = (s - self.num_warmup_steps - self.num_stable_steps) / max(
                1, self.num_decay_steps
            )
            progress = min(progress, 1.0)
            return min_lr + 0.5 * (base_lr - min_lr) * (
                1 + math.cos(math.pi * progress)
            )

    def get_last_lr(self):
        return [self._get_lr(base_lr) for base_lr in self.base_lrs]


def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    stage_names = {
        "clm": "Stage 1: CLM Pretraining",
        "clm_mlm": "Stage 2: CLM+MLM Hybrid",
        "kd": "Stage 3: Knowledge Distillation",
    }
    stage_name = stage_names.get(args.stage, args.stage)

    accelerator.print("=" * 60)
    accelerator.print(f"ChineseBabyLM V6 - {stage_name}")
    accelerator.print("=" * 60)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            accelerator.print(f"  GPU {i}: {mem:.1f} GB")

    set_seed(args.seed)

    tokenizer_dir = args.tokenizer_dir or os.path.join(args.data_dir, "tokenizer_v3")
    accelerator.print(f"Loading tokenizer: {tokenizer_dir}")

    tokenizer = None
    try:
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_dir)
        test_ids = tokenizer.encode("test", add_special_tokens=False)
        if tokenizer.vocab_size <= 1000 or len(test_ids) == 0:
            raise ValueError("Invalid tokenizer")
    except Exception:
        spm_model_path = os.path.join(tokenizer_dir, "spm.model")
        if not os.path.exists(spm_model_path):
            spm_model_path = os.path.join(tokenizer_dir, "spiece.model")
        tokenizer = SPMTokenizer(spm_model_path)

    vocab_size = tokenizer.vocab_size
    accelerator.print(f"Vocab size: {vocab_size}")

    sp_model_path = os.path.join(tokenizer_dir, "spiece.model")
    if not os.path.exists(sp_model_path):
        sp_model_path = os.path.join(tokenizer_dir, "spm.model")

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.d_model,
        intermediate_size=int(args.d_model * 8 / 3),
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        num_key_value_heads=args.n_kv_heads,
        max_position_embeddings=args.max_length,
        rms_norm_eps=1e-5,
        rope_theta=args.rope_theta,
        tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        hidden_act="silu",
        attention_dropout=args.attention_dropout,
        attn_implementation="sdpa",
    )

    if args.resume_from and os.path.exists(args.resume_from):
        accelerator.print(f"Resuming from: {args.resume_from}")
        model = LlamaForCausalLM.from_pretrained(args.resume_from)
    else:
        model = LlamaForCausalLM(config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Model params: {total_params:,} ({total_params / 1e6:.1f}M)")

    data_dir = args.data_dir
    train_file = os.path.join(data_dir, "processed_v6", "train.txt")
    val_file = os.path.join(data_dir, "processed_v6", "val.txt")
    if not os.path.exists(train_file):
        train_file = os.path.join(data_dir, "processed_v3", "train.txt")
    if not os.path.exists(val_file):
        val_file = os.path.join(data_dir, "processed_v3", "val.txt")

    accelerator.print(f"Train data: {train_file}")
    accelerator.print(f"Val data: {val_file}")

    base_train_dataset = DocumentAwareDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=args.max_length,
        bpe_dropout=args.bpe_dropout,
        sp_model_path=sp_model_path,
    )
    val_dataset = DocumentAwareDataset(
        tokenizer=tokenizer,
        file_path=val_file,
        block_size=args.max_length,
        bpe_dropout=0.0,
    )

    if args.stage == "clm_mlm":
        mask_token_id = tokenizer.convert_tokens_to_ids(
            getattr(tokenizer, "mask_token", "<unk>")
        )
        if mask_token_id is None or mask_token_id == tokenizer.unk_token_id:
            mask_token_id = vocab_size - 1
        train_dataset = CLMMLMDataset(
            base_train_dataset,
            clm_ratio=args.clm_ratio,
            mask_ratio_start=args.mask_ratio_start,
            mask_ratio_end=args.mask_ratio_end,
            total_epochs=args.num_epochs,
            current_epoch=0,
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
        )
        accelerator.print(
            f"CLM+MLM mode: ratio={args.clm_ratio}, mask={args.mask_ratio_start}->{args.mask_ratio_end}"
        )
    elif args.stage == "kd" and args.teacher_logits_dir:
        train_dataset = KDDataset(
            base_train_dataset, args.teacher_logits_dir, args.top_k
        )
        accelerator.print(
            f"KD mode: reverse_kl={args.use_reverse_kl}, T={args.temperature}"
        )
    else:
        train_dataset = base_train_dataset

    accelerator.print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
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

    no_decay = ["bias", "layernorm.weight", "rmsnorm.weight"]
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
        optimizer_grouped_params, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8
    )

    num_update_steps = len(train_loader) // args.gradient_accumulation_steps
    max_steps = args.num_epochs * num_update_steps
    warmup_steps = int(args.warmup_ratio * max_steps)

    if args.use_wsd:
        stable_steps = int(max_steps * 0.75)
        decay_steps = max_steps - warmup_steps - stable_steps
        lr_scheduler = WSDScheduler(
            optimizer, warmup_steps, stable_steps, decay_steps, min_lr_ratio=0.1
        )
        accelerator.print(
            f"WSD Scheduler: warmup={warmup_steps}, stable={stable_steps}, decay={decay_steps}"
        )
    else:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        accelerator.print(f"Cosine Scheduler: warmup={warmup_steps}, total={max_steps}")

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
    )
    if isinstance(lr_scheduler, WSDScheduler):
        lr_scheduler = accelerator.prepare(lr_scheduler)

    effective_batch = (
        args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    )

    if accelerator.is_main_process:
        run_name = args.wandb_run_name or f"llama-v6-{args.d_model}d-{args.stage}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            config={
                "model": "BabyLLM-V6",
                "stage": args.stage,
                "d_model": args.d_model,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_kv_heads": args.n_kv_heads,
                "total_params": total_params,
                "batch_size_per_gpu": args.batch_size,
                "effective_batch_size": effective_batch,
                "learning_rate": args.learning_rate,
                "label_smoothing": args.label_smoothing,
                "use_wsd": args.use_wsd,
                "use_reverse_kl": args.use_reverse_kl,
                "clm_ratio": args.clm_ratio,
            },
        )

    output_dir = args.output_dir
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        check_disk_space(output_dir, min_free_gb=2.0)

    accelerator.print(f"\n{'=' * 60}")
    accelerator.print(f"Training: {stage_name}")
    accelerator.print(f"  Params: {total_params / 1e6:.1f}M, Batch: {effective_batch}")
    accelerator.print(f"  LR: {args.learning_rate}, LS: {args.label_smoothing}")
    accelerator.print(f"  Epochs: {args.num_epochs}, Patience: {args.patience}")
    accelerator.print(f"{'=' * 60}\n")

    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()
    is_kd = args.stage == "kd"
    is_mlm = args.stage == "clm_mlm"

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start = time.time()
        optimizer.zero_grad()

        if args.bpe_dropout > 0 and epoch > 0:
            accelerator.print(f"Epoch {epoch + 1}: BPE Dropout retokenize...")
            base_train_dataset.retokenize()

        if args.dropout_anneal and args.attention_dropout > 0:
            progress = epoch / args.num_epochs
            if progress > 0.7:
                new_dropout = args.attention_dropout * max(
                    0.0, 1.0 - (progress - 0.7) / 0.3
                )
                set_dropout(accelerator.unwrap_model(model), new_dropout)

        if is_mlm and hasattr(train_dataset, "base"):
            pass

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            ncols=140,
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                if is_kd and "teacher_logits" in batch:
                    loss, logits = compute_ce_loss_with_sm(
                        model, input_ids, labels, args.label_smoothing
                    )
                    total_loss, loss_dict = compute_kd_loss(
                        student_logits=logits,
                        teacher_logits_topk=batch["teacher_logits"],
                        teacher_indices_topk=batch["teacher_indices"],
                        labels=labels,
                        temperature=args.temperature,
                        lambda_ce=args.lambda_ce,
                        lambda_kd=args.lambda_kd,
                        top_k=args.top_k,
                        label_smoothing=args.label_smoothing,
                        use_reverse_kl=args.use_reverse_kl,
                    )
                else:
                    total_loss, logits = compute_ce_loss_with_sm(
                        model, input_ids, labels, args.label_smoothing
                    )
                    loss_dict = {"ce_loss": total_loss.item()}

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            bs, seq_len = input_ids.shape
            epoch_loss += total_loss.item() * bs * seq_len
            epoch_tokens += bs * seq_len
            current_lr = (
                lr_scheduler.get_last_lr()[0]
                if hasattr(lr_scheduler, "get_last_lr")
                else args.learning_rate
            )
            raw_loss = total_loss.item()
            ppl = math.exp(min(raw_loss, 20))

            progress_bar.set_postfix(
                loss=f"{raw_loss:.4f}",
                ppl=f"{ppl:.1f}",
                lr=f"{current_lr:.2e}",
                step=global_step,
            )

            if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                log_dict = {
                    "train/loss": raw_loss,
                    "train/ppl": ppl,
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                }
                if is_kd:
                    log_dict["train/ce_loss"] = loss_dict.get("ce_loss", 0)
                    log_dict["train/kd_loss"] = loss_dict.get("kd_loss", 0)
                wandb.log(log_dict, step=global_step)

            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                safe_save_model(model, ckpt_dir, accelerator, config, tokenizer)
                if args.save_total_limit > 0:
                    all_ckpts = sorted(
                        glob.glob(os.path.join(output_dir, "checkpoint-*")),
                        key=lambda x: int(x.split("-")[-1]),
                    )
                    for old_ckpt in all_ckpts[: -args.save_total_limit]:
                        shutil.rmtree(old_ckpt, ignore_errors=True)

        avg_train_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
        epoch_time = time.time() - epoch_start

        val_loss = evaluate(model, val_loader, accelerator)
        val_ppl = math.exp(min(val_loss, 20))

        accelerator.print(
            f"\nEpoch {epoch + 1} ({epoch_time / 60:.1f}min) | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {val_loss:.4f} | PPL: {val_ppl:.2f} | "
            f"LR: {current_lr:.2e}"
        )

        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch/train_loss": avg_train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/val_ppl": val_ppl,
                    "epoch/epoch": epoch + 1,
                },
                step=global_step,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_dir = os.path.join(output_dir, "best_model")
                safe_save_model(model, best_dir, accelerator, config, tokenizer)
                accelerator.print(
                    f"  New best (val_loss={val_loss:.4f}, ppl={val_ppl:.2f})"
                )
                wandb.log(
                    {"best/val_loss": val_loss, "best/val_ppl": val_ppl},
                    step=global_step,
                )
            else:
                patience_counter += 1
                accelerator.print(f"  Early stop: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            accelerator.print(
                f"\nEarly stopping triggered after {patience_counter} epochs."
            )
            break

    total_time = time.time() - start_time
    accelerator.print(f"\nV6 {stage_name} complete!")
    accelerator.print(f"  Best Val Loss: {best_val_loss:.4f}")
    accelerator.print(f"  Best Val PPL: {math.exp(min(best_val_loss, 20)):.2f}")
    accelerator.print(f"  Total time: {total_time / 3600:.1f}h")

    if accelerator.is_main_process:
        wandb.finish()
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="ChineseBabyLM V6")

    parser.add_argument(
        "--stage", type=str, default="clm", choices=["clm", "clm_mlm", "kd"]
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output/babylm-llama-v6")
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--d_model", type=int, default=640)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=10)
    parser.add_argument("--n_kv_heads", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--student_model_path", type=str, default=None)
    parser.add_argument("--teacher_logits_dir", type=str, default=None)
    parser.add_argument("--lambda_ce", type=float, default=0.5)
    parser.add_argument("--lambda_kd", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--use_reverse_kl", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--bpe_dropout", type=float, default=0.1)
    parser.add_argument("--dropout_anneal", action="store_true", default=True)
    parser.add_argument("--no_dropout_anneal", action="store_true")
    parser.add_argument("--use_wsd", action="store_true", default=False)

    parser.add_argument("--clm_ratio", type=float, default=0.125)
    parser.add_argument("--mask_ratio_start", type=float, default=0.30)
    parser.add_argument("--mask_ratio_end", type=float, default=0.15)

    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wandb_project", type=str, default="chinese-babylm")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")

    args = parser.parse_args()
    if args.no_dropout_anneal:
        args.dropout_anneal = False

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"train_v6_{args.stage}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info("=" * 60)
    logger.info(f"ChineseBabyLM V6 - Stage: {args.stage}")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    train(args)


if __name__ == "__main__":
    main()
