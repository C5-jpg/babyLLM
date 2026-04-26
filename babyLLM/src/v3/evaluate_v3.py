"""
ChineseBabyLM V3 - LLaMA 模型评测脚本
"""
import os
import sys
import math
import torch
import argparse
from tqdm import tqdm
from transformers import LlamaForCausalLM
import wandb

# 修复 Bug1: 使用 SPMTokenizer（与训练一致的 SentencePiece tokenizer）
# 原错误: LlamaTokenizerFast 词表与训练时不同，导致所有 PPL 数值完全失真
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spm_tokenizer import SPMTokenizer

def load_model_and_tokenizer(model_path, device="cuda"):
    print(f"加载模型: {model_path}")
    # 优先从模型目录加载 spm.model
    spm_path = os.path.join(model_path, "spm.model")
    if os.path.exists(spm_path):
        print(f"  [SPMTokenizer] {spm_path}")
        tokenizer = SPMTokenizer(spm_path)
    else:
        # 回退到 data/tokenizer_v3
        fallback = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "tokenizer_v3"
        ))
        print(f"  [SPMTokenizer fallback] {fallback}")
        tokenizer = SPMTokenizer.from_pretrained(fallback)
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto" if torch.cuda.device_count() > 1 else None)
    if torch.cuda.device_count() <= 1:
        model = model.to(device)
    model.eval()
    return model, tokenizer, device

def compute_ppl_on_data(model, tokenizer, device, file_path, block_size=1024):
    print(f"\n计算 PPL (block_size={block_size})...")
    eos_id = tokenizer.eos_token_id
    all_token_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            line = line.strip()
            if line:
                all_token_ids.extend(tokenizer.encode(line, add_special_tokens=False) + [eos_id])
                
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in tqdm(range(0, len(all_token_ids) - block_size, block_size), desc="Computing PPL"):
            chunk = all_token_ids[i:i + block_size]
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item() * block_size
            total_tokens += block_size
            
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(min(avg_loss, 20))
    print(f"  评估 tokens: {total_tokens:,} | 平均 Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")
    return avg_loss, ppl

def test_generation(model, tokenizer, device, prompts, max_new_tokens=150):
    print("\n" + "=" * 60 + "\n📝 文本生成测试\n" + "=" * 60)
    model.config.pad_token_id = model.config.eos_token_id
    results = []
    
    for prompt in prompts:
        print(f"\n--- Prompt: {prompt!r} ---")
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        
        with torch.no_grad():
            greedy_output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
            print(f"  [Greedy]: {greedy_text}")
            
            sample_output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8, top_p=0.95, repetition_penalty=1.2)
            sample_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
            print(f"  [Sample]: {sample_text}")
            
        results.append({"prompt": prompt, "greedy": greedy_text, "sample": sample_text})
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/babylm-llama-v3/best_model")
    parser.add_argument("--val_file", type=str, default="data/processed_v3/val.txt")
    parser.add_argument("--wandb_project", type=str, default="chinese-babylm")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ 模型路径不存在: {args.model_path}")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, device)
    
    ppl_loss, ppl = compute_ppl_on_data(model, tokenizer, device, args.val_file)
    
    prompts = ["今天", "我喜欢", "小明和小红", "在一个很远的", "春天来了，", "从前有一座山", "老师说", "中国的首都是"]
    results = test_generation(model, tokenizer, device, prompts)
    
    try:
        wandb.init(project=args.wandb_project, name="llama-v3-eval", mode="online")
        wandb.log({"eval/ppl": ppl, "eval/loss": ppl_loss})
        table = wandb.Table(columns=["prompt", "greedy", "sample"], data=[[r["prompt"], r["greedy"], r["sample"]] for r in results])
        wandb.log({"eval/generation": table})
        wandb.finish()
    except Exception as e:
        print(f"⚠️ WandB 日志失败: {e}")

if __name__ == "__main__":
    main()
