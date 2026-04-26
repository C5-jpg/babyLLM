"""
ChineseBabyLM 模型评测脚本
快速评测 best_model 的生成质量和基本指标
"""
import os
import torch
import math
import json
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

MODEL_PATH = "output/babylm-gpt2/best_model"
DATA_PATH = "data/processed/train.txt"

def load_model_and_tokenizer(model_path):
    """加载模型和tokenizer"""
    print(f"加载模型: {model_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"设备: {device}, 参数量: {param_count:.1f}M")
    return model, tokenizer, device


def compute_ppl_on_data(model, tokenizer, device, file_path, max_lines=1000, block_size=512):
    """在数据上计算 PPL"""
    print(f"\n计算 PPL (使用 {max_lines} 行数据)...")
    
    token_buffer = []
    total_loss = 0.0
    total_tokens = 0
    num_chunks = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line)
            token_buffer.extend(ids)
    
    # 分块计算 loss
    with torch.no_grad():
        while len(token_buffer) >= block_size + 1:
            chunk = token_buffer[:block_size]
            token_buffer = token_buffer[block_size:]
            
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item() * block_size
            total_tokens += block_size
            num_chunks += 1
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(min(avg_loss, 20))
    print(f"  采样 chunks: {num_chunks}, tokens: {total_tokens:,}")
    print(f"  平均 Loss: {avg_loss:.4f}")
    print(f"  PPL: {ppl:.2f}")
    return avg_loss, ppl


def test_generation(model, tokenizer, device, prompts, max_new_tokens=100):
    """测试文本生成质量"""
    print("\n" + "=" * 60)
    print("文本生成测试")
    print("=" * 60)
    
    model.config.pad_token_id = model.config.eos_token_id
    
    for prompt in prompts:
        print(f"\n--- Prompt: {prompt!r} ---")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                num_return_sequences=1,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"生成: {generated}")
    
    return outputs


def compute_tokenizer_stats(tokenizer):
    """Tokenizer 统计信息"""
    print("\n" + "=" * 60)
    print("Tokenizer 分析")
    print("=" * 60)
    
    test_texts = [
        "这是一个测试句子。",
        "今天天气真好，我想出去玩。",
        "小猫在阳光下睡觉。",
        "The quick brown fox jumps over the lazy dog.",
        "我喜欢吃苹果和香蕉。",
    ]
    
    total_chars = 0
    total_tokens = 0
    
    for text in test_texts:
        ids = tokenizer.encode(text)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        total_chars += len(text)
        total_tokens += len(ids)
        print(f"\n  文本: {text}")
        print(f"  Tokens ({len(ids)}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
    
    avg_ratio = total_tokens / total_chars if total_chars > 0 else 0
    print(f"\n平均 Token/字符比: {avg_ratio:.3f}")
    print(f"词表大小: {tokenizer.vocab_size}")
    
    return avg_ratio


def analyze_model_config():
    """分析模型配置"""
    print("\n" + "=" * 60)
    print("模型配置分析")
    print("=" * 60)
    
    config_path = os.path.join(MODEL_PATH, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"  架构: GPT-2")
    print(f"  隐藏维度: {config['n_embd']}")
    print(f"  层数: {config['n_layer']}")
    print(f"  注意力头: {config['n_head']}")
    print(f"  最大位置: {config['n_positions']}")
    print(f"  词表大小: {config['vocab_size']}")
    print(f"  激活函数: {config.get('activation_function', 'N/A')}")
    
    # 计算参数量
    n_embd = config['n_embd']
    n_layer = config['n_layer']
    n_head = config['n_head']
    vocab_size = config['vocab_size']
    n_positions = config['n_positions']
    
    # GPT-2 参数量估算
    # Token embedding: vocab_size * n_embd
    # Position embedding: n_positions * n_embd
    # Each transformer layer: ~ 4 * n_embd^2 (attention + FFN) + 4 * n_embd (biases)
    embed_params = (vocab_size + n_positions) * n_embd
    layer_params = n_layer * (4 * n_embd * n_embd + 4 * n_embd + 2 * n_embd * n_embd // n_head * n_head)
    # 更精确: attention: 4*n_embd^2 (Q,K,V,O), FFN: 2*4*n_embd^2, LN: 4*n_embd*2
    layer_params = n_layer * (4 * n_embd * n_embd + 2 * 4 * n_embd * n_embd + 4 * n_embd)
    total = embed_params + layer_params + n_embd  # final LN
    
    print(f"\n  估算参数量: {total/1e6:.1f}M")
    
    return config


def main():
    print("=" * 60)
    print("ChineseBabyLM 模型评测")
    print("=" * 60)
    
    # 1. 模型配置分析
    config = analyze_model_config()
    
    # 2. 加载模型
    model, tokenizer, device = load_model_and_tokenizer(MODEL_PATH)
    
    # 3. Tokenizer 分析
    token_ratio = compute_tokenizer_stats(tokenizer)
    
    # 4. 计算 PPL
    if os.path.exists(DATA_PATH):
        ppl_loss, ppl = compute_ppl_on_data(model, tokenizer, device, DATA_PATH, max_lines=2000)
    
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
    ]
    test_generation(model, tokenizer, device, prompts, max_new_tokens=80)
    
    print("\n" + "=" * 60)
    print("评测完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()