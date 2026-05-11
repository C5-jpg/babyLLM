"""
ChineseBabyLM V11 — SOTA Sprint Training Script

V10 baseline: PPL=42.89, 38.7M params, 8K Unigram SPM
V11 improvements:
  - SGDR scheduler (CosineAnnealingWarmRestarts) for Stage 1
  - EMA (Exponential Moving Average, decay=0.999)
  - Self-Distillation (EMA teacher → student KL loss)
  - Dynamic CLM:MTP ratio (0.25 → 0.125 → 0.0625)
  - Label smoothing anneal (linearly decay over epochs)
  - Full wandb monitoring per stage

Architecture: LlamaForCausalLM, 512d, 12L, 8Q/4KV GQA, 8K Unigram, ~38.7M params
"""

import argparse
import copy
import json
import math
import os
import random
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


class SPMTokenizer:
    def __init__(self, tokenizer_dir):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(tokenizer_dir, "spm.model"))
        self.vocab_size = self.sp.get_piece_size()
        self.mask_id = self.sp.piece_to_id("<mask>")
        self.eos_id = self.sp.eos_id()
        self._dir = tokenizer_dir


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k] = v.clone().detach().float()
            else:
                self.shadow[k] = v.clone().detach()

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}

    def apply_to(self, model):
        sd = model.state_dict()
        for k in self.shadow:
            if k in sd:
                sd[k] = self.shadow[k].to(sd[k].dtype)
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
            intermediate_size=int(args.d_model * 8 / 3),
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


def train():
    parser = argparse.ArgumentParser(description="ChineseBabyLM V11 Training")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage", default="clm", choices=["clm", "mntp"])
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--clm_ratio", type=float, default=0.125)
    parser.add_argument("--mask_ratio_start", type=float, default=0.25)
    parser.add_argument("--mask_ratio_end", type=float, default=0.12)
    parser.add_argument("--bpe_dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--label_smoothing_anneal", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=2000)
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
    args = parser.parse_args()

    t0 = time.time()
    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(args.seed)

    if accelerator.is_main_process:
        print("=" * 60)
        print(f"  ChineseBabyLM V11 — Stage: {args.stage.upper()}")
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
        ema = EMA(accelerator.unwrap_model(model), decay=args.ema_decay)
        if accelerator.is_main_process:
            print(f"  EMA enabled: decay={args.ema_decay}")

    teacher_model = None
    if args.self_distill and ema is not None:
        teacher_model = copy.deepcopy(accelerator.unwrap_model(model))
        sd_state = {k: v.to(v.dtype) for k, v in ema.shadow.items()}
        teacher_model.load_state_dict(sd_state, strict=False)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        if accelerator.is_main_process:
            print(f"  Self-Distillation: T={args.sd_temperature}, lambda={args.sd_lambda}")

    wandb_run = None
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "run_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

        run_name = args.wandb_run_name or f"babylm-v11-{args.stage}"
        wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

        wandb_run.alert(
            title=f"V11 {args.stage.upper()} Started",
            text=f"LR={args.lr}, Epochs={args.epochs}, Scheduler={args.scheduler}, EMA={args.use_ema}, SD={args.self_distill}",
            level=wandb.AlertLevel.INFO,
        )

    best_val = float("inf")
    global_step = 0
    patience_counter = 0
    best_ppl = float("inf")

    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        epoch_steps = 0

        if teacher_model is not None and ema is not None:
            sd_state = {k: v.to(v.dtype) for k, v in ema.shadow.items()}
            teacher_model.load_state_dict(sd_state, strict=False)
            teacher_model.eval()
            if accelerator.is_main_process:
                print(f"  Teacher model updated with EMA weights for epoch {epoch + 1}")

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
                label_smoothing=ls_val,
            )

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(input_ids=batch["input_ids"]).logits

                T = args.sd_temperature
                kd_loss = F.kl_div(
                    F.log_softmax(logits.view(-1, logits.size(-1)) / T, dim=-1),
                    F.softmax(teacher_logits.detach().view(-1, teacher_logits.size(-1)) / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                loss = (1.0 - args.sd_lambda) * ce_loss + args.sd_lambda * kd_loss
            else:
                loss = ce_loss

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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
                    "step": global_step,
                })

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.unwrap_model(model).save_pretrained(ckpt_path)
                    copy_tokenizer(args.tokenizer_dir, ckpt_path)
                    if ema is not None:
                        ema_path = os.path.join(ckpt_path, "ema_state.pt")
                        torch.save(ema.state_dict(), ema_path)

            if args.max_steps > 0 and global_step >= args.max_steps:
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
                    ema.apply_to(accelerator.unwrap_model(model))
                    ema_path = os.path.join(args.output_dir, "best_model_ema")
                    accelerator.unwrap_model(model).save_pretrained(ema_path)
                    copy_tokenizer(args.tokenizer_dir, ema_path)
                    orig_ema_sd = torch.load(ema_save_path)
                    ema.load_state_dict(orig_ema_sd)
                    accelerator.unwrap_model(model).load_state_dict(orig_state, strict=False)

                print(f"  -> New best model saved (val_loss={val_loss_t:.4f}, ppl={ppl:.2f})")

                wandb_run.alert(
                    title=f"V11 {args.stage.upper()} New Best PPL={ppl:.2f}",
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

        patience_tensor = torch.tensor(patience_counter, device=accelerator.device)
        patience_tensor = accelerator.gather(patience_tensor).max().item()

        if args.patience > 0 and patience_tensor >= args.patience:
            if accelerator.is_main_process:
                print(f"  Early stopping triggered after {epoch + 1} epochs")
                wandb_run.alert(
                    title=f"V11 {args.stage.upper()} Early Stopped",
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
            title=f"V11 {args.stage.upper()} Complete — PPL={best_ppl:.2f}",
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
