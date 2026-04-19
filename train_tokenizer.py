"""训练 BPE Tokenizer"""
import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

os.makedirs("data/tokenizer", exist_ok=True)

train_file = "data/processed/train.txt"
assert os.path.exists(train_file), f"训练文件不存在: {train_file}"

print(f"从 {train_file} 训练 BPE Tokenizer (32K 词表)...")
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Punctuation(behavior="isolated"),
])
tokenizer.decoder = decoders.BPEDecoder()

special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=special_tokens,
    show_progress=True,
    min_frequency=2,
)

tokenizer.train([train_file], trainer=trainer)
tokenizer.save("data/tokenizer/tokenizer.json")
print(f"Tokenizer 训练完成! 词表大小: {tokenizer.get_vocab_size()}")

# 创建 tokenizer_config.json
config = {
    "model_type": "gpt2",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "tokenizer_class": "PreTrainedTokenizerFast",
}
with open("data/tokenizer/tokenizer_config.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

# 测试
test_text = "这是一个测试句子，用来验证tokenizer是否正常工作。"
encoding = tokenizer.encode(test_text)
print(f"测试: {test_text}")
print(f"Tokens: {encoding.tokens}")
print(f"IDs: {encoding.ids}")
print("Tokenizer 保存到 data/tokenizer/")