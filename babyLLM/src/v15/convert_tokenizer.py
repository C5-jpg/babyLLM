"""
Convert SentencePiece tokenizer to HuggingFace-compatible format.

Produces tokenizer.json, tokenizer_config.json, and special_tokens_map.json
so that AutoTokenizer.from_pretrained() works with the official eval pipeline.
"""

import argparse
import json
import os
import shutil

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Sequence
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import Decoder


class SPMDummyDecoder(Decoder):
    def decode(self, tokens):
        return "".join(tokens)

    @staticmethod
    def from_str(data):
        return SPMDummyDecoder()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm_model", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)
    vocab_size = sp.get_piece_size()

    unk_id = sp.unk_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    pad_id = sp.pad_id() if sp.pad_id() >= 0 else eos_id
    mask_token = "<mask>"
    mask_id = sp.piece_to_id(mask_token)
    if mask_id < 0:
        mask_id = vocab_size - 1

    unk_token = sp.id_to_piece(unk_id)
    bos_token = sp.id_to_piece(bos_id)
    eos_token = sp.id_to_piece(eos_id)
    pad_token = sp.id_to_piece(pad_id) if pad_id >= 0 and pad_id < vocab_size else "<pad>"

    vocab = []
    scores = []
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        vocab.append((piece, score))
        scores.append(score)

    tokenizer = Tokenizer(Unigram(vocab, unk_id))

    pre_tokenizer = Sequence([])
    tokenizer.pre_tokenizer = pre_tokenizer

    tokenizer.post_processor = TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        pair=f"{bos_token} $A {eos_token} {bos_token} $B {eos_token}",
        special_tokens=[
            (bos_token, bos_id),
            (eos_token, eos_id),
        ],
    )

    tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))

    tokenizer_config = {
        "model_type": "llama",
        "tokenizer_class": "LlamaTokenizerFast",
        "bos_token": bos_token,
        "eos_token": eos_token,
        "unk_token": unk_token,
        "pad_token": pad_token,
        "mask_token": mask_token,
        "model_max_length": 1024,
        "add_bos_token": False,
        "add_eos_token": False,
        "legacy": False,
    }
    with open(os.path.join(args.output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    special_tokens_map = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "unk_token": unk_token,
        "pad_token": pad_token,
        "mask_token": mask_token,
    }
    with open(os.path.join(args.output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)

    shutil.copy2(args.spm_model, os.path.join(args.output_dir, "tokenizer.model"))

    print(f"Converted {vocab_size} tokens to {args.output_dir}")
    print(f"  unk={unk_token}({unk_id}), bos={bos_token}({bos_id}), eos={eos_token}({eos_id}), pad={pad_token}({pad_id}), mask={mask_token}({mask_id})")


if __name__ == "__main__":
    main()
