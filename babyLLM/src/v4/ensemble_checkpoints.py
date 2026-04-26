"""
ChineseBabyLM V4 - 模型集成/权重平均工具

功能:
实现模型 Soup (权重平均，Model Weight Averaging):
  - 对多个训练 checkpoint 进行等权/加权平均
  - 平均后的模型通常比单个最佳 checkpoint PPL 更低、BLiMP 更高 (+2-5%)
  - 支持简单平均（Uniform Soup）和加权平均（Greedy Soup）

参考论文: "Model soups: averaging weights of multiple fine-tuned models improves accuracy" (2022)

用法:
  # 等权平均（Uniform Soup）
  python ensemble_checkpoints.py \
      --checkpoint_dirs output/babylm-llama-v4/checkpoint-5000 \
                        output/babylm-llama-v4/checkpoint-10000 \
                        output/babylm-llama-v4/best_model \
      --output_dir output/babylm-llama-v4/ensemble_soup

  # 根据 Val Loss 加权平均
  python ensemble_checkpoints.py \
      --checkpoint_dirs output/babylm-llama-v4/checkpoint-5000 \
                        output/babylm-llama-v4/checkpoint-10000 \
                        output/babylm-llama-v4/best_model \
      --val_losses 4.5 4.2 4.1 \
      --output_dir output/babylm-llama-v4/ensemble_soup
"""
import os
import sys
import copy
import json
import argparse
import torch
from collections import OrderedDict
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from tqdm import tqdm


def load_model_weights(checkpoint_dir: str, device: str = "cpu") -> OrderedDict:
    """
    加载模型权重到 CPU 内存（不实例化完整模型，节省 GPU 内存）。
    返回 state_dict。
    """
    print(f"  加载权重: {checkpoint_dir}")
    # 使用 map_location="cpu" 避免 GPU OOM
    model = LlamaForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float32,  # fp32 做平均更精确
        low_cpu_mem_usage=True,
    )
    sd = copy.deepcopy(model.state_dict())
    del model
    torch.cuda.empty_cache()
    return sd


def uniform_soup(checkpoint_dirs: list, output_dir: str, device: str = "cpu"):
    """
    等权平均所有 checkpoint 的权重（Uniform Soup）。
    对应公式: θ_soup = (1/N) * Σ θ_i
    """
    print(f"\n🍲 Uniform Soup - 对 {len(checkpoint_dirs)} 个 checkpoint 进行等权平均")

    # 加载第一个 checkpoint 的结构
    print(f"\n[1/{len(checkpoint_dirs)}] 初始化 soup...")
    soup_sd = load_model_weights(checkpoint_dirs[0], device)

    # 累加其余 checkpoint
    for i, ckpt_dir in enumerate(checkpoint_dirs[1:], start=2):
        print(f"\n[{i}/{len(checkpoint_dirs)}] 累加权重...")
        sd = load_model_weights(ckpt_dir, device)
        for key in soup_sd:
            soup_sd[key] = soup_sd[key] + sd[key]
        del sd

    # 求平均
    print("\n计算平均值...")
    n = len(checkpoint_dirs)
    for key in tqdm(soup_sd, desc="平均权重"):
        soup_sd[key] = soup_sd[key] / n

    return soup_sd


def weighted_soup(checkpoint_dirs: list, weights: list, output_dir: str, device: str = "cpu"):
    """
    加权平均 checkpoint 的权重（Weighted Soup）。
    对应公式: θ_soup = Σ (w_i * θ_i)
    weights 会自动归一化为和为 1。
    """
    # 归一化权重
    total = sum(weights)
    norm_weights = [w / total for w in weights]
    print(f"\n🍲 Weighted Soup - 对 {len(checkpoint_dirs)} 个 checkpoint 进行加权平均")
    for ckpt, w in zip(checkpoint_dirs, norm_weights):
        print(f"  权重 {w:.4f}: {os.path.basename(ckpt)}")

    # 加载第一个 checkpoint 并乘以权重
    print(f"\n[1/{len(checkpoint_dirs)}] 初始化 soup...")
    first_sd = load_model_weights(checkpoint_dirs[0], device)
    soup_sd = OrderedDict()
    for key in first_sd:
        soup_sd[key] = first_sd[key] * norm_weights[0]
    del first_sd

    # 累加其余 checkpoint（乘以对应权重）
    for i, (ckpt_dir, w) in enumerate(zip(checkpoint_dirs[1:], norm_weights[1:]), start=2):
        print(f"\n[{i}/{len(checkpoint_dirs)}] 累加权重（weight={w:.4f}）...")
        sd = load_model_weights(ckpt_dir, device)
        for key in soup_sd:
            soup_sd[key] = soup_sd[key] + sd[key] * w
        del sd

    return soup_sd


def save_soup_model(soup_sd: OrderedDict, reference_checkpoint: str, output_dir: str):
    """
    将 soup 权重保存为标准 HuggingFace 格式。
    使用第一个 checkpoint 的 config 和 tokenizer。
    """
    print(f"\n保存 Soup 模型到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 加载参考 checkpoint 的结构（用于保存格式）
    model = LlamaForCausalLM.from_pretrained(
        reference_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # 将 fp32 平均权重转为 bf16 并加载
    bf16_sd = OrderedDict()
    for key, val in soup_sd.items():
        bf16_sd[key] = val.to(torch.bfloat16)

    model.load_state_dict(bf16_sd)
    model.save_pretrained(output_dir)
    print(f"✅ Soup 模型已保存")

    # 复制 tokenizer
    try:
        tokenizer = LlamaTokenizerFast.from_pretrained(reference_checkpoint)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Tokenizer 已保存")
    except Exception as e:
        print(f"⚠️  Tokenizer 保存失败: {e}")

    del model
    torch.cuda.empty_cache()

    # 保存 soup metadata
    metadata = {
        "soup_type": "uniform" if len(set([1.0] * 100)) == 1 else "weighted",
        "num_checkpoints": len(soup_sd),
        "reference_checkpoint": reference_checkpoint,
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "soup_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def auto_find_checkpoints(output_dir: str):
    """
    自动扫描输出目录，找到所有 checkpoint 和 best_model。
    返回按 step 排序的 checkpoint 列表。
    """
    checkpoints = []

    if not os.path.exists(output_dir):
        return checkpoints

    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            config_file = os.path.join(path, "config.json")
            if os.path.exists(config_file):
                if name.startswith("checkpoint-"):
                    try:
                        step = int(name.split("-")[-1])
                        checkpoints.append((step, path))
                    except ValueError:
                        pass
                elif name == "best_model":
                    checkpoints.append((999999, path))  # best_model 放最后

    checkpoints.sort(key=lambda x: x[0])
    return [path for _, path in checkpoints]


def main():
    parser = argparse.ArgumentParser(
        description="ChineseBabyLM V4 - 模型 Soup（权重平均集成）"
    )
    parser.add_argument(
        "--checkpoint_dirs", type=str, nargs="+", default=None,
        help="要集成的 checkpoint 目录列表（不指定则自动扫描 --model_output_dir）"
    )
    parser.add_argument(
        "--model_output_dir", type=str, default="output/babylm-llama-v4",
        help="模型输出目录（用于自动扫描 checkpoint）"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Soup 模型保存目录（默认: model_output_dir/ensemble_soup）"
    )
    parser.add_argument(
        "--val_losses", type=float, nargs="+", default=None,
        help="各 checkpoint 的验证集 Loss（用于加权平均，不指定则等权平均）"
    )
    parser.add_argument(
        "--top_k", type=int, default=None,
        help="只取 val_loss 最低的前 k 个 checkpoint（需要同时提供 --val_losses）"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="运行设备（建议 cpu，避免 GPU OOM）"
    )
    args = parser.parse_args()

    # 确定输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_output_dir, "ensemble_soup")

    # 确定 checkpoint 列表
    if args.checkpoint_dirs is None:
        print(f"自动扫描 checkpoint: {args.model_output_dir}")
        checkpoint_dirs = auto_find_checkpoints(args.model_output_dir)
        if not checkpoint_dirs:
            print(f"❌ 未找到任何 checkpoint，请检查目录: {args.model_output_dir}")
            sys.exit(1)
        print(f"找到 {len(checkpoint_dirs)} 个 checkpoint:")
        for d in checkpoint_dirs:
            print(f"  {d}")
    else:
        checkpoint_dirs = args.checkpoint_dirs
        # 验证路径存在
        for d in checkpoint_dirs:
            if not os.path.exists(d):
                print(f"❌ checkpoint 目录不存在: {d}")
                sys.exit(1)

    # top_k 过滤
    if args.top_k and args.val_losses:
        if len(args.val_losses) != len(checkpoint_dirs):
            print(f"❌ val_losses 数量 ({len(args.val_losses)}) 与 checkpoint 数量 ({len(checkpoint_dirs)}) 不匹配")
            sys.exit(1)
        # 按 val_loss 排序，取前 k 个
        paired = sorted(zip(args.val_losses, checkpoint_dirs), key=lambda x: x[0])
        paired = paired[:args.top_k]
        args.val_losses = [p[0] for p in paired]
        checkpoint_dirs = [p[1] for p in paired]
        print(f"\n选取 Val Loss 最低的前 {args.top_k} 个 checkpoint:")
        for loss, d in zip(args.val_losses, checkpoint_dirs):
            print(f"  Loss={loss:.4f}: {d}")

    if len(checkpoint_dirs) < 2:
        print("⚠️  需要至少 2 个 checkpoint 才能进行集成，当前只有 1 个")
        print("   直接复制该 checkpoint 作为输出...")
        import shutil
        shutil.copytree(checkpoint_dirs[0], args.output_dir, dirs_exist_ok=True)
        print(f"✅ 已复制到: {args.output_dir}")
        return

    print("=" * 60)
    print("ChineseBabyLM V4 - 模型 Soup 集成")
    print("=" * 60)
    print(f"  集成数量: {len(checkpoint_dirs)} 个 checkpoint")
    print(f"  方式: {'加权平均（按 Val Loss 倒数）' if args.val_losses else '等权平均'}")
    print(f"  输出: {args.output_dir}")
    print("=" * 60)

    # 执行集成
    if args.val_losses:
        if len(args.val_losses) != len(checkpoint_dirs):
            print(f"❌ val_losses 数量 ({len(args.val_losses)}) 与 checkpoint 数量 ({len(checkpoint_dirs)}) 不匹配")
            sys.exit(1)
        # 使用 1/loss 作为权重（loss 越低，权重越高）
        weights = [1.0 / loss for loss in args.val_losses]
        soup_sd = weighted_soup(checkpoint_dirs, weights, args.output_dir, args.device)
    else:
        soup_sd = uniform_soup(checkpoint_dirs, args.output_dir, args.device)

    # 保存模型
    save_soup_model(soup_sd, checkpoint_dirs[0], args.output_dir)

    print("\n" + "=" * 60)
    print("🎉 模型 Soup 集成完成!")
    print(f"  输出: {args.output_dir}")
    print("  建议: 使用 evaluate_v4.py 和 evaluate_blimp_v4.py 验证 Soup 模型效果")
    print("  参考命令:")
    print(f"    python evaluate_blimp_v4.py --model_dir {args.output_dir} --use_demo_data")
    print("=" * 60)


if __name__ == "__main__":
    main()
