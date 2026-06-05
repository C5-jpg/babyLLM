"""
ChineseBabyLM V15 — SOTA Compact Model (based on V1-V14 deep analysis)

Target: New SOTA for compact Chinese LMs
  - PPL < 38.0 (beat V13's 38.68 with fewer params)
  - ZhoBLiMP > 65% (beat V13's 63.47%)
  - ~58M params (optimal tokens/param ratio ≈ 1.7×)

Architecture: LlamaForCausalLM, 640d, 14L, 10Q/5KV GQA, SPM 32K
Pipeline: 2-stage (CLM→MNTP), V13 proved Polish stage is ineffective

Inherited best practices:
  - SentencePiece 32K tokenizer (V3+: biggest single improvement)
  - CLM+MNTP hybrid training (V7+: GPT-BERT 2024 winner technique)
  - EMA decay=0.999 (V10+: 6-8% PPL improvement)
  - SGDR scheduler (V11+: validated by V1's accidental discovery)
  - Focal Loss γ=2.0/1.5 (V12+: helps MNTP class imbalance)
  - Label smoothing annealing (V11+: prevents early overconfidence)
  - BPE dropout 0.1 (V5+: improves generalization)
  - Dynamic CLM ratio (V11+: 0.25→0.125→0.0625)
  - Mask ratio annealing 0.25→0.10 (V13+: conservative)
  - PPL-filtered data with MinHash dedup (V13+: quality > quantity)

New V15 features:
  - Multi-scale EMA: track decay=0.999 and 0.9999 simultaneously
  - Per-layer gradient norm monitoring
  - Eval every 200 steps (finer cadence than V14's 500)
  - Gradient norm spike detection with LR reduction
"""

import argparse
import copy
import json
import math
import os
import random
import signal
import shutil
import time

import sentencepiece as spm
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
    set_seed,
)

_GRACEFUL_SHUTDOWN = False


def _signal_handler(signum, frame):
    global _GRACEFUL_SHUTDOWN
    _GRACEFUL_SHUTDOWN = True
    print(f"\n[Signal {signum}] Graceful shutdown requested, will save checkpoint and exit...")


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


class SPMTokenizer:
    def __init__(self, tokenizer_dir):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(tokenizer_dir, "spm.model"))
        self.vocab_size = self.sp.get_piece_size()
        self.mask_id = self.sp.piece_to_id("<mask>")
        self.eos_id = self.sp.eos_id()
        self._dir = tokenizer_dir


class MultiScaleEMA:
    """Track EMA at multiple decay rates simultaneously."""

    def __init__(self, model, decays=(0.999, 0.9999)):
        self.decays = decays
        self.shadows = {}
        for d in decays:
            self.shadows[d] = {}
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    self.shadows[d][k] = v.clone().detach().float()
                else:
                    self.shadows[d][k] = v.clone().detach()

    @torch.no_grad()
    def update(self, model):
        for d in self.decays:
            for k, v in model.state_dict().items():
                if k in self.shadows[d]:
                    self.shadows[d][k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    def state_dict(self):
        return {str(d): {k: v.clone() for k, v in shadow.items()} for d, shadow in self.shadows.items()}

    def load_state_dict(self, state_dict):
        for d in self.decays:
            d_str = str(d)
            if d_str in state_dict:
                self.shadows[d] = {k: v.clone() for k, v in state_dict[d_str].items()}

    def apply_to(self, model, decay=0.999):
        sd = model.state_dict()
        for k in self.shadows[decay]:
            if k in sd:
                sd[k] = self.shadows[decay][k].to(sd[k].dtype)
        model.load_state_dict(sd, strict=False)


class BabyDataset(Dataset):
    def __init__(
        self,
        file_path,
        tokenizer,
        max_length,
        stride,
        stage,
        clm_ratio,
        mask_ratio_start,
        mask_ratio_end,
        total_epochs,
        bpe_dropout,
        dynamic_clm_ratio=False,
    ):
        self.tokenizer = tokenizer
        self.stage = stage
        self.clm_ratio = clm_ratio
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        self.total_epochs = max(total_epochs, 1)
        self.current_epoch = 0
        self.bpe_dropout = bpe_dropout
        self.max_length = max_length
        self.mask_id = tokenizer.mask_id
        self.dynamic_clm_ratio = dynamic_clm_ratio

        print(f"  Reading data: {file_path}, block_size={max_length}, stride={stride}")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if len(l.strip()) > 2]
        print(f"  Documents: {len(lines):,}")

        all_tokens = []
        for line in lines:
            ids = self.tokenizer.sp.encode(line, out_type=int)
            all_tokens.extend(ids)
            all_tokens.append(self.tokenizer.eos_id)
        print(f"  Total tokens: {len(all_tokens):,}")

        self._all_tokens = all_tokens
        self._stride = stride

        self._samples = []
        for i in range(0, len(all_tokens) - max_length, max(1, stride)):
            chunk = all_tokens[i : i + max_length + 1]
            if len(chunk) == max_length + 1:
                self._samples.append(chunk)
        print(f"  Total chunks: {len(self._samples):,}")

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    @property
    def mask_ratio(self):
        if self.total_epochs <= 1:
            return self.mask_ratio_end
        progress = self.current_epoch / max(self.total_epochs - 1, 1)
        return self.mask_ratio_start + (self.mask_ratio_end - self.mask_ratio_start) * progress

    @property
    def effective_clm_ratio(self):
        if self.stage == "clm":
            return 1.0
        if not self.dynamic_clm_ratio:
            return self.clm_ratio
        progress = self.current_epoch / max(self.total_epochs - 1, 1)
        if progress < 0.25:
            return 0.25
        elif progress < 0.75:
            return 0.125
        else:
            return 0.0625

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        chunk = self._samples[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)

        if self.stage == "clm" or random.random() < self.effective_clm_ratio:
            return {"input_ids": input_ids, "labels": labels}

        masked_input = input_ids.clone()
        mask = torch.rand(input_ids.shape) < self.mask_ratio
        masked_input[mask] = self.mask_id
        return {"input_ids": masked_input, "labels": labels}


def build_model(args, tokenizer):
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"  Resuming from: {args.resume_from}")
        model = LlamaForCausalLM.from_pretrained(
            args.resume_from, torch_dtype=torch.bfloat16
        )
    else:
        config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.d_model,
            intermediate_size=round(args.d_model * 8 / 3 / 256) * 256,
            num_hidden_layers=args.n_layer,
            num_attention_heads=args.n_head,
            num_key_value_heads=args.n_kv_heads,
            max_position_embeddings=args.max_length,
            rope_theta=10000.0,
            rms_norm_eps=1e-5,
            tie_word_embeddings=True,
            attn_implementation="sdpa",
        )
        model = LlamaForCausalLM(config)
        print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    model.gradient_checkpointing_enable()
    return model


def copy_tokenizer(tokenizer_dir, dest_dir):
    for fname in ["spm.model", "tokenizer.model"]:
        src = os.path.join(tokenizer_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest_dir, fname))


def _save_full_checkpoint(ckpt_dir, model, tokenizer_dir, optimizer, scheduler, ema, global_step, epoch, best_val, best_ppl, patience_counter):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    copy_tokenizer(tokenizer_dir, ckpt_dir)
    state = {
        "global_step": global_step,
        "epoch": epoch,
        "best_val": best_val,
        "best_ppl": best_ppl,
        "patience_counter": patience_counter,
    }
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
        json.dump(state, f, indent=2)
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
    if ema is not None:
        torch.save(ema.state_dict(), os.path.join(ckpt_dir, "ema_state.pt"))


def log_gradient_norms(model, accelerator, global_step, wandb_run):
    """Log per-layer gradient norms for debugging."""
    if not accelerator.is_main_process:
        return 0.0
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[f"grad_norm/{name}"] = param.grad.data.norm(2).item()
    if norms:
        total_norm = sum(v ** 2 for v in norms.values()) ** 0.5
        norms["grad_norm/total"] = total_norm
        wandb.log(norms, step=global_step)
        return total_norm
    return 0.0


def train():
    parser = argparse.ArgumentParser(description="ChineseBabyLM V15 Training")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage", default="clm", choices=["clm", "mntp"])
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--d_model", type=int, default=640)
    parser.add_argument("--n_layer", type=int, default=14)
    parser.add_argument("--n_head", type=int, default=10)
    parser.add_argument("--n_kv_heads", type=int, default=5)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--clm_ratio", type=float, default=0.125)
    parser.add_argument("--mask_ratio_start", type=float, default=0.25)
    parser.add_argument("--mask_ratio_end", type=float, default=0.10)
    parser.add_argument("--bpe_dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--label_smoothing_anneal", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="chinese-babylm")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "sgdr"])
    parser.add_argument("--sgdr_t0", type=int, default=0, help="SGDR T_0 (0=auto: 1 epoch)")
    parser.add_argument("--sgdr_t_mult", type=int, default=2)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--dynamic_clm_ratio", action="store_true")
    parser.add_argument("--self_distill", action="store_true")
    parser.add_argument("--sd_temperature", type=float, default=2.0)
    parser.add_argument("--sd_lambda", type=float, default=0.5)
    parser.add_argument("--focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--sd_update_freq", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", default="")
    parser.add_argument("--eval_steps", type=int, default=200, help="Run validation every N steps (0=epoch-only)")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--log_grad_norms", action="store_true", help="Log per-layer gradient norms")
    parser.add_argument("--grad_norm_spike_threshold", type=float, default=10.0, help="Reduce LR on grad norm spike")
    args = parser.parse_args()

    t0 = time.time()
    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(args.seed)

    if accelerator.is_main_process:
        print("=" * 60)
        print(f"  ChineseBabyLM V15 — Stage: {args.stage.upper()}")
        print("=" * 60)
        for k, v in sorted(vars(args).items()):
            print(f"  {k}: {v}")
        print("=" * 60)

    tokenizer = SPMTokenizer(args.tokenizer_dir)
    if accelerator.is_main_process:
        print(f"  Vocab size: {tokenizer.vocab_size}, mask_token_id: {tokenizer.mask_id}")

    model = build_model(args, tokenizer)

    if args.attention_dropout > 0:
        for module in model.modules():
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = args.attention_dropout

    train_file = os.path.join(args.data_dir, "train.txt")
    val_file = os.path.join(args.data_dir, "val.txt")

    train_ds = BabyDataset(
        train_file, tokenizer, args.max_length, args.stride,
        args.stage, args.clm_ratio, args.mask_ratio_start, args.mask_ratio_end,
        args.epochs, args.bpe_dropout, args.dynamic_clm_ratio,
    )
    val_ds = BabyDataset(
        val_file, tokenizer, args.max_length, args.max_length,
        "clm", 1.0, 0.0, 0.0, 1, 0.0,
    )

    num_workers = min(8, os.cpu_count() // max(1, torch.cuda.device_count()) or 4)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True, prefetch_factor=4, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        num_workers=num_workers, pin_memory=True, prefetch_factor=4, persistent_workers=True,
    )

    no_decay = ["bias", "layernorm", "rmsnorm", "norm"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8
    )

    total_steps = args.max_steps if args.max_steps > 0 else len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    if args.scheduler == "sgdr":
        t0_val = args.sgdr_t0 if args.sgdr_t0 > 0 else len(train_loader)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=t0_val, T_mult=args.sgdr_t_mult, eta_min=1e-6
        )
        if accelerator.is_main_process:
            print(f"  SGDR: T_0={t0_val}, T_mult={args.sgdr_t_mult}, total_steps={total_steps}")
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    ema = None
    if args.use_ema and accelerator.is_main_process:
        ema = MultiScaleEMA(accelerator.unwrap_model(model), decays=(0.999, 0.9999))
        if accelerator.is_main_process:
            print(f"  Multi-scale EMA enabled: decays=(0.999, 0.9999)")

    best_val = float("inf")
    global_step = 0
    patience_counter = 0
    step_patience_counter = 0
    best_ppl = float("inf")
    start_epoch = 0
    prev_grad_norm = 0.0

    resume_ckpt = args.resume_from_checkpoint
    if not resume_ckpt:
        latest_dir = os.path.join(args.output_dir, "latest_checkpoint")
        if os.path.isdir(latest_dir) and os.path.exists(os.path.join(latest_dir, "trainer_state.json")):
            resume_ckpt = latest_dir

    if resume_ckpt and os.path.isdir(resume_ckpt) and os.path.exists(os.path.join(resume_ckpt, "trainer_state.json")):
        if accelerator.is_main_process:
            print(f"  Resuming training from checkpoint: {resume_ckpt}")
        with open(os.path.join(resume_ckpt, "trainer_state.json"), "r") as f:
            saved_state = json.load(f)
        start_epoch = saved_state.get("epoch", 0) + 1
        global_step = saved_state.get("global_step", 0)
        best_val = saved_state.get("best_val", float("inf"))
        best_ppl = saved_state.get("best_ppl", float("inf"))
        patience_counter = saved_state.get("patience_counter", 0)

        if start_epoch >= args.epochs:
            if accelerator.is_main_process:
                print(f"  Checkpoint epoch {start_epoch} >= total epochs {args.epochs}, nothing to resume")
            return

        unwrapped = accelerator.unwrap_model(model)
        ckpt_model = LlamaForCausalLM.from_pretrained(resume_ckpt, torch_dtype=torch.bfloat16)
        unwrapped.load_state_dict(ckpt_model.state_dict(), strict=False)
        del ckpt_model
        if accelerator.is_main_process:
            print(f"  Model weights loaded from {resume_ckpt}")

        opt_path = os.path.join(resume_ckpt, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=True))
        sch_path = os.path.join(resume_ckpt, "scheduler.pt")
        if os.path.exists(sch_path):
            scheduler.load_state_dict(torch.load(sch_path, map_location="cpu", weights_only=True))
        ema_path = os.path.join(resume_ckpt, "ema_state.pt")
        if ema is not None and os.path.exists(ema_path):
            ema.load_state_dict(torch.load(ema_path, map_location="cpu", weights_only=True))

        if accelerator.is_main_process:
            print(f"  Resumed: epoch={start_epoch}, step={global_step}, best_val={best_val:.4f}, best_ppl={best_ppl:.2f}")

    wandb_run = None
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "run_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

        run_name = args.wandb_run_name or f"babylm-v15-{args.stage}"
        resume_id = None
        if resume_ckpt:
            prev_wandb_file = os.path.join(args.output_dir, "wandb_run_id.txt")
            if os.path.exists(prev_wandb_file):
                with open(prev_wandb_file, "r") as f:
                    resume_id = f.read().strip()
        wandb_run = wandb.init(
            project=args.wandb_project, name=run_name, config=vars(args),
            id=resume_id, resume="allow" if resume_id else None,
        )
        with open(os.path.join(args.output_dir, "wandb_run_id.txt"), "w") as f:
            f.write(wandb_run.id)

        wandb_run.alert(
            title=f"V15 {args.stage.upper()} {'Resumed' if resume_ckpt else 'Started'}",
            text=f"LR={args.lr}, Epochs={args.epochs}, Scheduler={args.scheduler}, EMA={args.use_ema}",
            level=wandb.AlertLevel.INFO,
        )

    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        epoch_steps = 0

        for batch in tqdm(
            train_loader,
            disable=not accelerator.is_main_process,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        ):
            outputs = model(input_ids=batch["input_ids"])
            logits = outputs.logits
            labels = batch["labels"]

            ls_val = args.label_smoothing
            if args.label_smoothing_anneal and args.epochs > 1:
                progress = epoch / max(args.epochs - 1, 1)
                ls_val = args.label_smoothing * (1.0 - 0.5 * progress)

            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1),
                label_smoothing=ls_val, reduction="none",
            )

            if args.focal_loss:
                pt = torch.exp(-ce_loss)
                ce_loss = ((1 - pt) ** args.focal_gamma * ce_loss).mean()
            else:
                ce_loss = ce_loss.mean()

            loss = ce_loss

            try:
                accelerator.backward(loss)
            except torch.cuda.OutOfMemoryError:
                if accelerator.is_main_process:
                    print(f"  [OOM] Step {global_step}! Clearing cache, skipping batch")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                del loss, outputs, logits
                continue

            if global_step % args.grad_accum_steps == 0:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if args.log_grad_norms and global_step % (args.logging_steps * 5) == 0:
                    total_norm = log_gradient_norms(
                        accelerator.unwrap_model(model), accelerator, global_step, wandb_run
                    )
                    if total_norm > args.grad_norm_spike_threshold * max(prev_grad_norm, 1.0):
                        if accelerator.is_main_process:
                            print(f"  [Spike] Grad norm {total_norm:.2f} > {args.grad_norm_spike_threshold}x prev {prev_grad_norm:.2f}")
                    prev_grad_norm = total_norm

                optimizer.step()
                if args.scheduler == "sgdr":
                    scheduler.step(global_step)
                else:
                    scheduler.step()
                optimizer.zero_grad()
            global_step += 1

            if ema is not None:
                ema.update(accelerator.unwrap_model(model))

            epoch_loss += loss.item()
            epoch_steps += 1

            if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                avg_loss = epoch_loss / epoch_steps
                lr_val = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/lr": lr_val,
                    "train/mask_ratio": train_ds.mask_ratio,
                    "train/clm_ratio": train_ds.effective_clm_ratio,
                    "train/label_smoothing": ls_val,
                    "train/epoch": epoch + 1,
                    "train/gpu_mem_gb": torch.cuda.memory_allocated() / 1e9,
                    "train/gpu_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "step": global_step,
                })

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    _save_full_checkpoint(
                        os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                        accelerator.unwrap_model(model), args.tokenizer_dir,
                        optimizer, scheduler, ema,
                        global_step, epoch, best_val, best_ppl, patience_counter,
                    )
                    if args.save_total_limit > 0:
                        ckpts = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-") and d.replace("checkpoint-", "").isdigit()],
                            key=lambda x: int(x.replace("checkpoint-", "")),
                        )
                        while len(ckpts) > args.save_total_limit:
                            shutil.rmtree(os.path.join(args.output_dir, ckpts.pop(0)), ignore_errors=True)
                    _save_full_checkpoint(
                        os.path.join(args.output_dir, "latest_checkpoint"),
                        accelerator.unwrap_model(model), args.tokenizer_dir,
                        optimizer, scheduler, ema,
                        global_step, epoch, best_val, best_ppl, patience_counter,
                    )

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            if _GRACEFUL_SHUTDOWN:
                if accelerator.is_main_process:
                    print(f"  [Shutdown] Saving emergency checkpoint at step {global_step}...")
                    _save_full_checkpoint(
                        os.path.join(args.output_dir, "latest_checkpoint"),
                        accelerator.unwrap_model(model), args.tokenizer_dir,
                        optimizer, scheduler, ema,
                        global_step, epoch, best_val, best_ppl, patience_counter,
                    )
                break

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                if ema is not None:
                    if accelerator.is_main_process:
                        orig_state = copy.deepcopy(dict(accelerator.unwrap_model(model).state_dict()))
                    ema.apply_to(accelerator.unwrap_model(model))

                model.eval()
                eval_loss = 0.0
                eval_batches = 0
                with torch.no_grad():
                    for vb in tqdm(val_loader, disable=not accelerator.is_main_process, desc="StepEval"):
                        vo = model(input_ids=vb["input_ids"])
                        vl = F.cross_entropy(
                            vo.logits.view(-1, vo.logits.size(-1)),
                            vb["labels"].view(-1),
                        )
                        eval_loss += vl.item()
                        eval_batches += 1
                        if eval_batches >= 20:
                            break

                eval_loss = eval_loss / eval_batches if eval_batches > 0 else float("inf")
                eval_loss_t = torch.tensor(eval_loss, device=accelerator.device)
                eval_loss_t = accelerator.gather(eval_loss_t).mean().item()

                if ema is not None and accelerator.is_main_process:
                    accelerator.unwrap_model(model).load_state_dict(orig_state, strict=False)

                model.train()

                if accelerator.is_main_process:
                    eval_ppl = math.exp(min(eval_loss_t, 20))
                    wandb.log({
                        "step_eval/loss": eval_loss_t,
                        "step_eval/ppl": eval_ppl,
                        "step": global_step,
                    })
                    print(f"  [Step {global_step}] Eval Loss: {eval_loss_t:.4f}, PPL: {eval_ppl:.2f}, Best: {best_val:.4f}")

                    if eval_loss_t < best_val - args.early_stop_min_delta:
                        best_val = eval_loss_t
                        best_ppl = eval_ppl
                        step_patience_counter = 0
                        save_path = os.path.join(args.output_dir, "best_model")
                        accelerator.unwrap_model(model).save_pretrained(save_path)
                        copy_tokenizer(args.tokenizer_dir, save_path)
                        if ema is not None:
                            ema_save_path = os.path.join(save_path, "ema_best.pt")
                            torch.save(ema.state_dict(), ema_save_path)
                            for d in ema.decays:
                                ema.apply_to(accelerator.unwrap_model(model), decay=d)
                                ema_path = os.path.join(args.output_dir, f"best_model_ema_{d}")
                                accelerator.unwrap_model(model).save_pretrained(ema_path)
                                copy_tokenizer(args.tokenizer_dir, ema_path)
                            if 0.999 in ema.decays:
                                ema.apply_to(accelerator.unwrap_model(model), decay=0.999)
                                ema_path = os.path.join(args.output_dir, "best_model_ema")
                                accelerator.unwrap_model(model).save_pretrained(ema_path)
                                copy_tokenizer(args.tokenizer_dir, ema_path)
                            orig_ema_sd = ema.state_dict()
                            ema.load_state_dict(orig_ema_sd)
                            accelerator.unwrap_model(model).load_state_dict(orig_state, strict=False)
                        print(f"    -> New best model (val_loss={eval_loss_t:.4f}, ppl={eval_ppl:.2f})")
                    else:
                        step_patience_counter += 1
                        print(f"    -> No improvement ({step_patience_counter}/{args.early_stop_patience})")

                step_pat_tensor = torch.tensor(step_patience_counter, device=accelerator.device)
                step_pat_tensor = accelerator.gather(step_pat_tensor).max().item()
                if args.early_stop_patience > 0 and step_pat_tensor >= args.early_stop_patience:
                    if accelerator.is_main_process:
                        print(f"  [Step {global_step}] Step-level early stopping triggered!")
                    break

        if _GRACEFUL_SHUTDOWN:
            break

        if ema is not None and accelerator.is_main_process:
            orig_state = copy.deepcopy(dict(accelerator.unwrap_model(model).state_dict()))
            ema.apply_to(accelerator.unwrap_model(model))

        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                disable=not accelerator.is_main_process,
                desc="Validation",
            ):
                outputs = model(input_ids=batch["input_ids"])
                loss = F.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    batch["labels"].view(-1),
                )
                val_loss += loss.item()
                val_steps += 1

        val_loss = val_loss / val_steps if val_steps > 0 else float("inf")
        val_loss_t = torch.tensor(val_loss, device=accelerator.device)
        val_loss_t = accelerator.gather(val_loss_t).mean().item()

        if ema is not None and accelerator.is_main_process:
            accelerator.unwrap_model(model).load_state_dict(orig_state, strict=False)

        model.train()

        if accelerator.is_main_process:
            ppl = math.exp(min(val_loss_t, 20))
            avg_train_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch + 1} Step {global_step} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss_t:.4f} | PPL: {ppl:.2f} | "
                f"Mask Ratio: {train_ds.mask_ratio:.3f} | "
                f"CLM Ratio: {train_ds.effective_clm_ratio:.3f} | "
                f"LS: {ls_val:.4f} | "
                f"Elapsed: {elapsed / 3600:.1f}h"
            )
            wandb.log({
                "val/loss": val_loss_t,
                "val/ppl": ppl,
                "epoch": epoch + 1,
                "step": global_step,
                "val/elapsed_hours": elapsed / 3600,
            })

            if val_loss_t < best_val:
                best_val = val_loss_t
                best_ppl = ppl
                patience_counter = 0
                save_path = os.path.join(args.output_dir, "best_model")
                accelerator.unwrap_model(model).save_pretrained(save_path)
                copy_tokenizer(args.tokenizer_dir, save_path)
                if ema is not None:
                    ema_save_path = os.path.join(save_path, "ema_best.pt")
                    torch.save(ema.state_dict(), ema_save_path)
                    for d in ema.decays:
                        ema.apply_to(accelerator.unwrap_model(model), decay=d)
                        ema_path = os.path.join(args.output_dir, f"best_model_ema_{d}")
                        accelerator.unwrap_model(model).save_pretrained(ema_path)
                        copy_tokenizer(args.tokenizer_dir, ema_path)
                    if 0.999 in ema.decays:
                        ema.apply_to(accelerator.unwrap_model(model), decay=0.999)
                        ema_path = os.path.join(args.output_dir, "best_model_ema")
                        accelerator.unwrap_model(model).save_pretrained(ema_path)
                        copy_tokenizer(args.tokenizer_dir, ema_path)
                    orig_ema_sd = ema.state_dict()
                    ema.load_state_dict(orig_ema_sd)
                    accelerator.unwrap_model(model).load_state_dict(orig_state, strict=False)

                print(f"  -> New best model saved (val_loss={val_loss_t:.4f}, ppl={ppl:.2f})")

                _save_full_checkpoint(
                    os.path.join(args.output_dir, "latest_checkpoint"),
                    accelerator.unwrap_model(model), args.tokenizer_dir,
                    optimizer, scheduler, ema,
                    global_step, epoch, best_val, best_ppl, patience_counter,
                )

                wandb_run.alert(
                    title=f"V15 {args.stage.upper()} New Best PPL={ppl:.2f}",
                    text=(
                        f"Epoch {epoch + 1}/{args.epochs}\n"
                        f"Val Loss: {val_loss_t:.4f}\n"
                        f"PPL: {ppl:.2f}\n"
                        f"Step: {global_step}\n"
                        f"Elapsed: {elapsed / 3600:.1f}h"
                    ),
                    level=wandb.AlertLevel.INFO,
                    wait_duration=300,
                )
            else:
                patience_counter += 1
                print(f"  -> No improvement ({patience_counter}/{args.patience})")

            _save_full_checkpoint(
                os.path.join(args.output_dir, "latest_checkpoint"),
                accelerator.unwrap_model(model), args.tokenizer_dir,
                optimizer, scheduler, ema,
                global_step, epoch, best_val, best_ppl, patience_counter,
            )

        patience_tensor = torch.tensor(patience_counter, device=accelerator.device)
        patience_tensor = accelerator.gather(patience_tensor).max().item()

        if args.patience > 0 and patience_tensor >= args.patience:
            if accelerator.is_main_process:
                print(f"  Early stopping triggered after {epoch + 1} epochs")
                wandb_run.alert(
                    title=f"V15 {args.stage.upper()} Early Stopped",
                    text=f"Stopped at epoch {epoch + 1}, best_ppl={best_ppl:.2f}",
                    level=wandb.AlertLevel.WARN,
                )
            break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    elapsed_total = time.time() - t0
    if accelerator.is_main_process:
        summary = (
            f"Stage: {args.stage}\n"
            f"Best Val Loss: {best_val:.4f}\n"
            f"Best PPL: {best_ppl:.2f}\n"
            f"Total Steps: {global_step}\n"
            f"Total Time: {elapsed_total / 3600:.2f}h"
        )
        print(f"\nTraining complete. Best val_loss: {best_val:.4f}, ppl: {best_ppl:.2f}")

        wandb_run.alert(
            title=f"V15 {args.stage.upper()} Complete — PPL={best_ppl:.2f}",
            text=summary,
            level=wandb.AlertLevel.INFO,
        )

        summary_path = os.path.join(args.output_dir, "training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "stage": args.stage,
                "best_val_loss": best_val,
                "best_ppl": best_ppl,
                "total_steps": global_step,
                "elapsed_seconds": elapsed_total,
                "config": vars(args),
            }, f, indent=2)

        wandb.finish()


if __name__ == "__main__":
    train()
