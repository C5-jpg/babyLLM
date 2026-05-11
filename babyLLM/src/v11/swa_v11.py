"""
ChineseBabyLM V11 — SWA (Stochastic Weight Averaging)

Averages weights from multiple checkpoints to produce a more robust model.
"""

import argparse
import os
import shutil

import torch
from transformers import LlamaForCausalLM


def main():
    parser = argparse.ArgumentParser(description="SWA Weight Averaging")
    parser.add_argument("--checkpoint_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Weights for each checkpoint (uniform if not specified)")
    args = parser.parse_args()

    n = len(args.checkpoint_dirs)
    if args.weights is None:
        weights = [1.0 / n] * n
    else:
        assert len(args.weights) == n, f"Got {len(args.weights)} weights for {n} checkpoints"
        total = sum(args.weights)
        weights = [w / total for w in args.weights]

    print(f"SWA averaging {n} checkpoints:")
    for d, w in zip(args.checkpoint_dirs, weights):
        print(f"  {d} (weight={w:.3f})")

    avg_state = None
    for ckpt_dir, w in zip(args.checkpoint_dirs, weights):
        print(f"  Loading {ckpt_dir} ...")
        model = LlamaForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16)
        state = {k: v.float() for k, v in model.state_dict().items()}
        if avg_state is None:
            avg_state = {k: v * w for k, v in state.items()}
        else:
            for k, v in state.items():
                avg_state[k] = avg_state[k] + v * w
        del model, state

    print(f"  Loading base model structure from {args.checkpoint_dirs[0]} ...")
    base = LlamaForCausalLM.from_pretrained(args.checkpoint_dirs[0], torch_dtype=torch.bfloat16)
    base_sd = base.state_dict()
    for k in avg_state:
        if k in base_sd:
            base_sd[k] = avg_state[k].to(torch.bfloat16)
    base.load_state_dict(base_sd, strict=False)

    os.makedirs(args.output_dir, exist_ok=True)
    base.save_pretrained(args.output_dir)

    for fname in ["spm.model", "tokenizer.model"]:
        src = os.path.join(args.tokenizer_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, fname))

    total_params = sum(p.numel() for p in base.parameters())
    print(f"  SWA model saved to {args.output_dir}")
    print(f"  Params: {total_params:,}")

    del base, avg_state


if __name__ == "__main__":
    main()
