"""SentencePiece Tokenizer 包装器 - 兼容 HuggingFace 接口"""
import sentencepiece as spm
from typing import List, Optional, Union

class SPMTokenizer:
    """直接封装 SentencePiece，提供与 HuggingFace tokenizer 兼容的接口"""
    
    def __init__(self, model_path: str):
        self._model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        
        # Special token IDs (from training config)
        self.unk_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 3
        
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        
        self._vocab_size = self.sp.GetPieceSize()
        
        # HuggingFace 兼容属性
        self.model_max_length = 2048
        self.is_fast = False
        self.name_or_path = model_path
    
    @property
    def vocab_size(self):
        return self._vocab_size
    
    def __len__(self):
        return self._vocab_size
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = self.sp.Encode(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            special = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
            ids = [i for i in ids if i not in special]
        return self.sp.Decode(ids)
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.sp.IdToPiece(i) for i in ids]
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        if isinstance(tokens, str):
            return self.sp.PieceToId(tokens)
        return [self.sp.PieceToId(t) for t in tokens]
    
    def save_pretrained(self, save_dir: str):
        import shutil, os
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy2(self._model_path, os.path.join(save_dir, "spm.model"))
        # 写入 config
        config = {
            "model_type": "llama",
            "tokenizer_class": "SPMTokenizer",
            "vocab_size": self._vocab_size,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
        }
        import json
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, path: str):
        import os
        model_path = os.path.join(path, "spm.model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"spm.model not found in {path}")
        return cls(model_path)