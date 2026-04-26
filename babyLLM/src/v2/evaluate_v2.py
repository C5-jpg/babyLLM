"""
ChineseBabyLM V2 - LLaMA 模型评测脚本
支持 LlamaForCausalLM 模型的完整评测:
1. PPL 计算（独立验证集）
2. 文本生成质量测试
3. Tokenizer 分析
4. 结果记录到 WandB
"""
import os
import math
import json
import argparse
import time
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast


def load_model_and_tokenizer(model_path, device="cuda"):
    """加载 LLaMA 模型和 tokenizer"""
    print(f"加载模型: {model_path}")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.device_count() > 1 else None,
    )
    
    if torch.cuda.device_count() <= 1:
        model = model.to(device)
    
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"设备: {device}, 参数量: {param_count:.1f}M")
    return model, tokenizer, device


def compute_ppl_on_data(model, tokenizer, device, file_path, max_lines=None, block_size=1024):
    """在数据上计算 PPL（文档感知版本）"""
    print(f"\n计算 PPL (block_size={block_size})...")
    if max_lines:
        print(f"  使用前 {max_lines} 行")
    
    # Tokenize 所有数据
    eos_id = tokenizer.eos_token_id
    all_token_ids = []
    line_count = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing for eval"):
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line)
            all_token_ids.extend(ids)
            all_token_ids.append(eos_id)
            line_count += 1
            if max_lines and line_count >= max_lines:
                break
    
    print(f"  总 tokens: {len(all_token_ids):,}")
    
    # 分块计算 loss
    total_loss = 0.0
    total_tokens = 0
    num_chunks = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_token_ids) - block_size, block_size),
                      desc="Computing PPL"):
            chunk = all_token_ids[i:i + block_size]
            if len(chunk) < block_size:
                break
            
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            outputs = model(input_ids, labels=input_ids)
            
            # loss 是 per-token 平均
            total_loss += outputs.loss.item() * block_size
            total_tokens += block_size
            num_chunks += 1
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(min(avg_loss, 20))
    
    print(f"\n📊 PPL 结果:")
    print(f"  采样 chunks: {num_chunks}")
    print(f"  评估 tokens: {total_tokens:,}")
    print(f"  平均 Loss: {avg_loss:.4f}")
    print(f"  PPL: {ppl:.2f}")
    
    return avg_loss, ppl


def test_generation(model, tokenizer, device, prompts, max_new_tokens=150):
    """测试文本生成质量"""
    print("\n" + "=" * 60)
    print("📝 文本生成测试")
    print("=" * 60)
    
    model.config.pad_token_id = model.config.eos_token_id
    
    results = []
    for prompt in prompts:
        print(f"\n--- Prompt: {prompt!r} ---")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Greedy decoding
        with torch.no_grad():
            greedy_output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
            )
        
        greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        print(f"  [Greedy]: {greedy_text}")
        
        # Sampling
        with torch.no_grad():
            sample_output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                num_return_sequences=1,
            )
        
        sample_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
        print(f"  [Sample]: {sample_text}")
        
        results.append({
            "prompt": prompt,
            "greedy": greedy_text,
            "sample": sample_text,
        })
    
    return results


def compute_tokenizer_stats(tokenizer):
    """Tokenizer 统计信息"""
    print("\n" + "=" * 60)
    print("🔤 Tokenizer 分析")
    print("=" * 60)
    
    test_texts = [
        "这是一个测试句子，用来验证tokenizer是否正常工作。",
        "今天天气真好，我想出去玩。",
        "小猫在阳光下睡觉。",
        "The quick brown fox jumps over the lazy dog.",
        "我喜欢吃苹果和香蕉。",
        "中国的首都是北京。",
        "人工智能正在改变我们的生活方式。",
    ]
    
    total_chars = 0
    total_tokens = 0
    has_unk_count = 0
    
    for text in test_texts:
        ids = tokenizer.encode(text)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        total_chars += len(text)
        total_tokens += len(ids)
        has_unk = "<unk>" in tokens
        if has_unk:
            has_unk_count += 1
        flag = " ⚠️ UNK!" if has_unk else " ✅"
        print(f"\n  文本: {text}")
        print(f"  Tokens ({len(ids)}){flag}: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
    
    avg_ratio = total_tokens / total_chars if total_chars > 0 else 0
    print(f"\n  平均 Token/字符比: {avg_ratio:.3f}")
    print(f"  词表大小: {tokenizer.vocab_size}")
    print(f"  含 UNK 的文本数: {has_unk_count}/{len(test_texts)}")
    
    return avg_ratio


def analyze_model_config(model_path):
    """分析模型配置"""
    print("\n" + "=" * 60)
    print("⚙️ 模型配置分析")
    print("=" * 60)
    
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        
        print(f"  架构: {config.get('model_type', 'LLaMA')}")
        print(f"  隐藏维度: {config.get('hidden_size', 'N/A')}")
        print(f"  层数: {config.get('num_hidden_layers', 'N/A')}")
        print(f"  注意力头: {config.get('num_attention_heads', 'N/A')}")
        print(f"  KV 头: {config.get('num_key_value_heads', 'N/A')}")
        print(f"  FFN 维度: {config.get('intermediate_size', 'N/A')}")
        print(f"  最大位置: {config.get('max_position_embeddings', 'N/A')}")
        print(f"  词表大小: {config.get('vocab_size', 'N/A')}")
        print(f"  激活函数: {config.get('hidden_act', 'N/A')}")
        
        return config
    else:
        print("  ⚠️ config.json 不存在")
        return {}


def log_to_wandb(model_path, val_file, wandb_project, wandb_run_name):
    """将评测结果记录到 WandB"""
    try:
        import wandb
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, device = load_model_and_tokenizer(model_path, device)
        
        # 初始化 WandB
        wandb.init(
            project=wandb_project,
            name=f"{wandb_run_name}-eval",
            mode="online",
        )
        
        # PPL 评测
        if os.path.exists(val_file):
            ppl_loss, ppl = compute_ppl_on_data(model, tokenizer, device, val_file)
            wandb.log({"eval/ppl": ppl, "eval/loss": ppl_loss})
        
        # 生成测试
        prompts = [
            "今天", "我喜欢", "中国的首都是",
            "春天来了", "从前有一座山",
            "人工智能", "小明和小红",
        ]
        results = test_generation(model, tokenizer, device, prompts)
        
        # 创建生成表格
        table = wandb.Table(
            columns=["prompt", "greedy", "sample"],
            data=[[r["prompt"], r["greedy"], r["sample"]] for r in results],
        )
        wandb.log({"eval/generation": table})
        
        wandb.finish()
        print("\n✅ 评测结果已记录到 WandB")
        
    except ImportError:
        print("⚠️ wandb 未安装，跳过 WandB 日志")


def main():
    parser = argparse.ArgumentParser(description="ChineseBabyLM V2 评测")
    parser.add_argument("--model_path", type=str, default="output/babylm-llama-v2/best_model")
    parser.add_argument("--val_file", type=str, default="data/processed_v2/val.txt")
    parser.add_argument("--max_lines", type=int, default=None, help="最多评估行数")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--wandb_project", type=str, default="chinese-babylm")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ChineseBabyLM V2 - LLaMA 模型评测")
    print("=" * 60)
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"❌ 模型路径不存在: {args.model_path}")
        print("可用的模型:")
        output_dir = os.path.dirname(args.model_path)
        if os.path.exists(output_dir):
            for d in os.listdir(output_dir):
                print(f"  {d}")
        return
    
    # 1. 模型配置
    config = analyze_model_config(args.model_path)
    
    # 2. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, device)
    
    # 3. Tokenizer 分析
    token_ratio = compute_tokenizer_stats(tokenizer)
    
    # 4. PPL 计算
    val_file = args.val_file
    if not os.path.exists(val_file):
        # 回退
        val_file = "data/processed/train.txt"
        print(f"\n⚠️ 验证集不存在，使用: {val_file}")
    
    if os.path.exists(val_file):
        ppl_loss, ppl = compute_ppl_on_data(
            model, tokenizer, device, val_file,
            max_lines=args.max_lines, block_size=args.block_size,
        )
    
    # 5. 文本生成测试
    prompts = [
        "今天",
        "我喜欢",
        "小明和小红",
        "在一个很远的",
        "春天来了，",
        "从前有一座山",
        "老师说",
        "中国的首都是",
        "人工智能正在",
        "小猫在阳光下",
    ]
    results = test_generation(model, tokenizer, device, prompts, max_new_tokens=100)
    
    print("\n" + "=" * 60)
    print("✅ 评测完成!")
    print("=" * 60)
    
    # 6. WandB 日志
    if not args.no_wandb:
        try:
            log_to_wandb(args.model_path, val_file, args.wandb_project, args.wandb_run_name)
        except Exception as e:
            print(f"⚠️ WandB 日志失败: {e}")


if __name__ == "__main__":
    main()