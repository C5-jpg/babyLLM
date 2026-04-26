#!/usr/bin/env python3
"""Fix tokenizer_v4: rebuild tokenizer.json from spiece.model"""
import sentencepiece as spm
import json
import os
import shutil

TOKENIZER_DIR = '/home/kehe/babyllm/babyLLM/data/tokenizer_v4'
SPM_MODEL = os.path.join(TOKENIZER_DIR, 'spiece.model')

# Load spm model
sp = spm.SentencePieceProcessor()
sp.load(SPM_MODEL)
vocab_size = sp.get_piece_size()
print(f'spm vocab size: {vocab_size}')

# Build vocab and merges for BPE tokenizer.json
vocab = {}
for i in range(vocab_size):
    piece = sp.id_to_piece(i)
    vocab[piece] = i

print(f'vocab dict size: {len(vocab)}')

# Build added_tokens list
added_tokens = [
    {"id": 0, "content": "<unk>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
    {"id": 1, "content": "<s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
    {"id": 2, "content": "</s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
    {"id": 3, "content": "<pad>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
]

# Build tokenizer.json in Unigram format (since we have a SentencePiece model)
# Use the scores from the model
scores = []
for i in range(vocab_size):
    piece = sp.id_to_piece(i)
    score = sp.get_score(i)
    scores.append([piece, score])

tokenizer_json = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": added_tokens,
    "normalizer": {
        "type": "Sequence",
        "normalizers": [
            {"type": "Prepend", "prepend": "\u2581"},
            {"type": "Replace", "pattern": {"String": " "}, "content": "\u2581"}
        ]
    },
    "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581", "str_rep": "\u2581", "add_prefix_space": True},
    "post_processor": None,
    "decoder": {"type": "Metaspace", "replacement": "\u2581", "str_rep": "\u2581", "add_prefix_space": True},
    "model": {
        "type": "Unigram",
        "unk_id": 0,
        "vocab": scores,
        "byte_fallback": True,
    }
}

out_path = os.path.join(TOKENIZER_DIR, 'tokenizer.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)
print(f'Written tokenizer.json with {len(scores)} vocab entries')

# Update tokenizer_config.json
config = {
    "add_prefix_space": None,
    "backend": "tokenizers",
    "bos_token": "<s>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "</s>",
    "from_slow": True,
    "model_max_length": 1024,
    "pad_token": "<pad>",
    "tokenizer_class": "LlamaTokenizer",
    "unk_token": "<unk>",
    "use_default_system_prompt": False
}
config_path = os.path.join(TOKENIZER_DIR, 'tokenizer_config.json')
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
print('Updated tokenizer_config.json')

# Verify
from transformers import LlamaTokenizerFast
tokenizer = LlamaTokenizerFast.from_pretrained(TOKENIZER_DIR)
print(f'Verification - vocab_size: {tokenizer.vocab_size}')
print(f'Verification - len(tokenizer): {len(tokenizer)}')
print(f'Verification - pad_token_id: {tokenizer.pad_token_id}')
test = tokenizer.encode('今天天气真好', add_special_tokens=False)
print(f'Test encode: {test[:10]}')
print('Done!')
