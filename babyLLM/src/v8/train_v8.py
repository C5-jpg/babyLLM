import os, sys, math, time, argparse, random, json, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup, set_seed
from accelerate import Accelerator
import wandb
import sentencepiece as spm
from tqdm import tqdm

class SPMTokenizer:
    def __init__(self, tokenizer_dir):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(tokenizer_dir, "spm.model"))
        self.vocab_size = self.sp.get_piece_size()
        self.mask_id = self.sp.piece_to_id("<mask>")
        self.eos_id = self.sp.eos_id()

class BabyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, stage, clm_ratio, mask_ratio):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stage = stage
        self.clm_ratio = clm_ratio
        self.mask_ratio = mask_ratio
        self.samples = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if len(l.strip()) > 2]
        
        all_tokens = []
        for line in lines:
            all_tokens.extend(self.tokenizer.sp.encode(line, out_type=int))
            all_tokens.append(self.tokenizer.eos_id)
        
        for i in range(0, len(all_tokens) - max_length, max_length):
            chunk = all_tokens[i:i+max_length+1]
            if len(chunk) == max_length + 1:
                self.samples.append(chunk)
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        chunk = self.samples[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        
        if self.stage == "clm" or random.random() < self.clm_ratio:
            return {"input_ids": input_ids, "labels": labels}
            
        masked_input = input_ids.clone()
        mask = torch.rand(input_ids.shape) < self.mask_ratio
        masked_input[mask] = self.tokenizer.mask_id
        return {"input_ids": masked_input, "labels": labels}

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stage", type=str, default="clm")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--clm_ratio", type=float, default=0.125)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(42)
    
    tokenizer = SPMTokenizer(args.tokenizer_dir)
    
    if args.resume_from and os.path.exists(args.resume_from):
        model = LlamaForCausalLM.from_pretrained(args.resume_from, torch_dtype=torch.bfloat16)
    else:
        config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.d_model,
            intermediate_size=int(args.d_model * 8 / 3),
            num_hidden_layers=args.n_layer,
            num_attention_heads=args.n_head,
            num_key_value_heads=args.n_kv_heads,
            max_position_embeddings=1024,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            attn_implementation="sdpa"
        )
        model = LlamaForCausalLM(config)
        
    model.gradient_checkpointing_enable()
    
    train_ds = BabyDataset(os.path.join(args.data_dir, "train.txt"), tokenizer, 1024, args.stage, args.clm_ratio, args.mask_ratio)
    val_ds = BabyDataset(os.path.join(args.data_dir, "val.txt"), tokenizer, 1024, "clm", 1.0, 0.0)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        wandb.init(project="chinese-babylm", name=f"babylm-v8-{args.stage}", config=args)
        
    best_val = float("inf")
    
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, disable=not accelerator.is_main_process)):
            outputs = model(input_ids=batch["input_ids"])
            logits = outputs.logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if step % 50 == 0 and accelerator.is_main_process and step > 0:
                wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]})
                
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, disable=not accelerator.is_main_process):
                outputs = model(input_ids=batch["input_ids"])
                loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), batch["labels"].view(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_loss_t = torch.tensor(val_loss, device=accelerator.device)
        val_loss_t = accelerator.gather(val_loss_t).mean().item()
        
        if accelerator.is_main_process:
            ppl = math.exp(min(val_loss_t, 20))
            print(f"Epoch {epoch+1} Val Loss: {val_loss_t:.4f} PPL: {ppl:.2f}")
            wandb.log({"val/loss": val_loss_t, "val/ppl": ppl, "epoch": epoch+1})
            
            if val_loss_t < best_val:
                best_val = val_loss_t
                unwrapped = accelerator.unwrap_model(model)
                save_path = os.path.join(args.output_dir, "best_model")
                unwrapped.save_pretrained(save_path)
                shutil.copy2(os.path.join(args.tokenizer_dir, "spm.model"), os.path.join(save_path, "spm.model"))
                
    if accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    train()
