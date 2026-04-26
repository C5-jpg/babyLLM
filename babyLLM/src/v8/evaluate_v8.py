import argparse, os, math, json, torch
from transformers import LlamaForCausalLM
import sentencepiece as spm
from tqdm import tqdm
import torch.nn.functional as F

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_file", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(args.model_path, "spm.model"))
    
    all_tokens = []
    with open(args.val_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_tokens.extend(sp.encode(line, out_type=int))
                all_tokens.append(sp.eos_id())
    
    max_length = 1024
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_tokens) - max_length, max_length)):
            chunk = all_tokens[i:i+max_length+1]
            if len(chunk) < max_length + 1: continue
            input_ids = torch.tensor([chunk[:-1]], device=device)
            labels = torch.tensor([chunk[1:]], device=device)
            outputs = model(input_ids=input_ids)
            loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
            total_loss += loss.item() * max_length
            total_tokens += max_length

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(min(avg_loss, 20))
    print(json.dumps({"loss": avg_loss, "ppl": ppl}))

if __name__ == "__main__":
    eval()
