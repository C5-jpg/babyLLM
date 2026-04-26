"""
ChineseBabyLM V2 - ByteLevel BPE Tokenizer 训练脚本
修复: 使用 ByteLevel 预分词替代 WhitespaceSplit，完美支持中文
"""
import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

SAVE_DIR = "data/tokenizer_v2"
TRAIN_FILE = "data/processed/train.txt"
VOCAB_SIZE = 32000

os.makedirs(SAVE_DIR, exist_ok=True)

assert os.path.exists(TRAIN_FILE), f"训练文件不存在: {TRAIN_FILE}"

print("=" * 60)
print(f"训练 ByteLevel BPE Tokenizer (词表: {VOCAB_SIZE})")
print(f"训练数据: {TRAIN_FILE}")
print("=" * 60)

# 初始化 BPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# ✅ 关键修复: 使用 ByteLevel 预分词，天然支持中文 UTF-8 编码
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# 特殊 token
special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=special_tokens,
    show_progress=True,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

print("\n开始训练 tokenizer...")
tokenizer.train([TRAIN_FILE], trainer=trainer)

# 保存 tokenizer.json
tokenizer.save(os.path.join(SAVE_DIR, "tokenizer.json"))
print(f"\nTokenizer 训练完成! 词表大小: {tokenizer.get_vocab_size()}")

# 创建 HuggingFace 格式 tokenizer
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(SAVE_DIR, "tokenizer.json"),
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    mask_token="<mask>",
)
hf_tokenizer.save_pretrained(SAVE_DIR)
print(f"HuggingFace tokenizer 保存到: {SAVE_DIR}")

# 测试
test_texts = [
    "这是一个测试句子，用来验证tokenizer是否正常工作。",
    "今天天气真好，我想出去玩。",
    "小猫在阳光下睡觉。",
    "我喜欢吃苹果和香蕉。",
    "中国的首都是北京。",
]

print("\n" + "=" * 60)
print("Tokenizer 测试")
print("=" * 60)
total_chars = 0
total_tokens = 0
for text in test_texts:
    encoding = hf_tokenizer.encode(text)
    tokens = hf_tokenizer.convert_ids_to_tokens(encoding)
    total_chars += len(text)
    total_tokens += len(encoding)
    has_unk = "<unk>" in tokens
    flag = " ⚠️ HAS UNK!" if has_unk else " ✅"
    print(f"\n  文本: {text}")
    print(f"  Tokens ({len(encoding)}){flag}: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")

print(f"\n平均 Token/字符比: {total_tokens / total_chars:.3f}")
print(f"词表大小: {hf_tokenizer.vocab_size}")
print("\n✅ Tokenizer 训练完成!")