"""
ChineseBabyLM V3 - BLiMP 竞赛评测脚本

功能:
实现语法可接受性判断（Linguistic Acceptability，BLiMP-style）:
  - 使用 V3 的 SPMTokenizer（与训练一致）
  - 给定正确句/错误句对，计算模型对每句的 log-probability
  - 正确句的 log-prob 高于错误句则得分
  - 输出准确率（目标 > 70%，SOTA: 80-85%）

支持:
  1. 标准 JSON 格式数据集（含 sentence_good / sentence_bad 字段）
  2. --use_demo_data：内置中文语法对比示例（无需外部数据即可运行）
  3. 批量评测 + 结果 JSON 导出

用法:
  python evaluate_blimp_v3.py --model_path output/babylm-llama-v3/best_model --use_demo_data
  python evaluate_blimp_v3.py --model_path output/babylm-llama-v3/best_model --data_file blimp_data.json
"""
import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, set_seed

# 使用与训练一致的 SPMTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spm_tokenizer import SPMTokenizer

set_seed(42)

# ============================================================
# 内置示例数据（中文语法可接受性对比）
# ============================================================
DEMO_BLIMP_PAIRS = [
    # 主谓一致
    {"sentence_good": "小猫在花园里玩耍。",        "sentence_bad": "小猫在花园里玩耍玩。",      "category": "verb_repetition"},
    {"sentence_good": "他们都去学校了。",           "sentence_bad": "他们都去了学校了。",          "category": "aspect_marker"},
    {"sentence_good": "妈妈给我买了一本书。",      "sentence_bad": "妈妈给我买了一本本书。",      "category": "classifier_duplication"},
    {"sentence_good": "这个苹果很甜。",            "sentence_bad": "这个苹果是很甜。",            "category": "copula_usage"},
    {"sentence_good": "我昨天看了一部电影。",      "sentence_bad": "我昨天看了一部电影电影。",    "category": "noun_repetition"},
    # 量词使用
    {"sentence_good": "桌子上有三本书。",          "sentence_bad": "桌子上有三条书。",            "category": "classifier"},
    {"sentence_good": "他养了两只猫。",            "sentence_bad": "他养了两本猫。",              "category": "classifier"},
    {"sentence_good": "她穿着一件红色的裙子。",    "sentence_bad": "她穿着一张红色的裙子。",      "category": "classifier"},
    {"sentence_good": "我喝了一杯茶。",            "sentence_bad": "我喝了一只茶。",              "category": "classifier"},
    {"sentence_good": "公园里有很多棵树。",        "sentence_bad": "公园里有很多本树。",          "category": "classifier"},
    # 动词时态/体标记
    {"sentence_good": "他已经吃完了饭。",          "sentence_bad": "他已经吃完饭了吃。",          "category": "aspect"},
    {"sentence_good": "我正在写作业。",            "sentence_bad": "我正在写写作业。",            "category": "aspect"},
    {"sentence_good": "她快要毕业了。",            "sentence_bad": "她快要毕业毕业了。",          "category": "aspect"},
    # 否定结构
    {"sentence_good": "他没有来上课。",            "sentence_bad": "他不有来上课。",              "category": "negation"},
    {"sentence_good": "我不喜欢吃辣的食物。",      "sentence_bad": "我没喜欢吃辣的食物。",        "category": "negation"},
    {"sentence_good": "这件事她不知道。",          "sentence_bad": "这件事她没知道。",            "category": "negation"},
    # 比较结构
    {"sentence_good": "这本书比那本厚。",          "sentence_bad": "这本书比那本更加厚比。",      "category": "comparison"},
    {"sentence_good": "他跑得比我快。",            "sentence_bad": "他跑得我比快。",              "category": "comparison"},
    # 把字句和被字句
    {"sentence_good": "他把书放在桌子上。",        "sentence_bad": "他把放书在桌子上。",          "category": "ba_construction"},
    {"sentence_good": "书被他放在桌子上。",        "sentence_bad": "书被放他在桌子上。",          "category": "bei_construction"},
    # 定语位置
    {"sentence_good": "漂亮的花在窗台上。",        "sentence_bad": "花漂亮的在窗台上。",          "category": "attributive_position"},
    {"sentence_good": "我有一本有趣的书。",        "sentence_bad": "我有一本书有趣的。",          "category": "attributive_position"},
    # 常见短语流畅性
    {"sentence_good": "春天来了，花儿开放了。",    "sentence_bad": "春天来了，花儿开放开放了。",  "category": "fluency"},
    {"sentence_good": "小朋友们在操场上玩耍。",    "sentence_bad": "小朋友们在操场上玩耍耍。",    "category": "fluency"},
    {"sentence_good": "今天天气非常好。",          "sentence_bad": "今天天气非常非常非常好。",    "category": "fluency"},
    {"sentence_good": "中国是一个美丽的国家。",    "sentence_bad": "中国是一个一个美丽的国家。",  "category": "fluency"},
    {"sentence_good": "他每天早上七点起床。",      "sentence_bad": "他每天早上七七点起床。",      "category": "fluency"},
    {"sentence_good": "图书馆里有很多书。",        "sentence_bad": "图书馆里有很多多书。",        "category": "fluency"},
    {"sentence_good": "她喜欢看电影和听音乐。",    "sentence_bad": "她喜欢看电影看和听音乐。",    "category": "fluency"},
    {"sentence_good": "孩子们在公园里开心地玩。",  "sentence_bad": "孩子们在公园里开心心地玩。",  "category": "fluency"},
]


def load_tokenizer(model_path: str) -> SPMTokenizer:
    """加载与 V3 训练一致的 SPMTokenizer"""
    # 优先从模型目录加载
    spm_path = os.path.join(model_path, "spm.model")
    if os.path.exists(spm_path):
        print(f"  [SPMTokenizer] {spm_path}")
        return SPMTokenizer(spm_path)

    # 回退到 data/tokenizer_v3
    fallback = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "tokenizer_v3"
    ))
    print(f"  [SPMTokenizer fallback] {fallback}")
    return SPMTokenizer.from_pretrained(fallback)


def sentence_log_prob(model, tokenizer: SPMTokenizer, sentence: str, device: str) -> float:
    """
    计算句子的平均 token log-probability（越高越好）。
    使用 teacher-forcing: 模型看到完整序列，预测每个 token，取平均 log-prob。
    """
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    if not ids:
        return -float('inf')

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss.item()

    return -neg_log_likelihood  # 返回平均 log-prob（越大越好）


def evaluate_blimp(
    model,
    tokenizer,
    pairs: list,
    device: str = "cuda",
) -> tuple:
    """
    对 BLiMP 句对列表进行评测。
    返回 (summary_dict, results_list)
    """
    model.eval()
    results = []
    correct = 0
    total = 0
    category_stats = {}

    print(f"\n评测 {len(pairs)} 个句对 (V3 SPMTokenizer)...")
    for pair in tqdm(pairs, desc="BLiMP评测"):
        sent_good = pair.get("sentence_good", pair.get("good", ""))
        sent_bad  = pair.get("sentence_bad",  pair.get("bad",  ""))
        category  = pair.get("category", "unknown")

        if not sent_good or not sent_bad:
            continue

        lp_good = sentence_log_prob(model, tokenizer, sent_good, device)
        lp_bad  = sentence_log_prob(model, tokenizer, sent_bad,  device)

        is_correct = lp_good > lp_bad
        correct += int(is_correct)
        total += 1

        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0}
        category_stats[category]["total"] += 1
        if is_correct:
            category_stats[category]["correct"] += 1

        results.append({
            "sentence_good": sent_good,
            "sentence_bad": sent_bad,
            "category": category,
            "lp_good": round(lp_good, 4),
            "lp_bad": round(lp_bad, 4),
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    category_accuracy = {
        cat: round(v["correct"] / v["total"], 4)
        for cat, v in category_stats.items() if v["total"] > 0
    }

    summary = {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "category_accuracy": category_accuracy,
        "category_stats": category_stats,
    }

    print("\n" + "=" * 60)
    print("BLiMP 评测结果 (V3)")
    print("=" * 60)
    print(f"总准确率: {accuracy * 100:.2f}%  ({correct}/{total})")

    if accuracy >= 0.75:
        print("🏆 优秀！达到竞赛冠军水平 (≥75%)")
    elif accuracy >= 0.70:
        print("✅ 良好！达到竞赛参赛目标 (≥70%)")
    elif accuracy >= 0.60:
        print("📈 中等，有提升空间 (60-70%)")
    elif accuracy >= 0.52:
        print("🔄 基础水平，高于随机 (52-60%)")
    else:
        print("⚠️  接近随机水平 (<52%)，需要更多训练")

    print("\n按类别准确率:")
    for cat, acc in sorted(category_accuracy.items(), key=lambda x: -x[1]):
        n = category_stats[cat]["total"]
        c = category_stats[cat]["correct"]
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {cat:30s} [{bar}] {acc*100:.1f}% ({c}/{n})")

    print("\n部分错误案例:")
    errors = [r for r in results if not r["correct"]][:5]
    for r in errors:
        print(f"  ✗ 正确句: {r['sentence_good']!r:40s}  lp={r['lp_good']:.4f}")
        print(f"    错误句: {r['sentence_bad']!r:40s}  lp={r['lp_bad']:.4f}")

    print("=" * 60)
    return summary, results


def main():
    parser = argparse.ArgumentParser(description="ChineseBabyLM V3 - BLiMP 语法可接受性评测")
    parser.add_argument("--model_path", type=str, required=True,
                        help="V3 模型目录（含 config.json 和 spm.model）")
    parser.add_argument("--data_file", type=str, default=None,
                        help="BLiMP 数据集 JSON 文件（含 sentence_good/sentence_bad 字段）")
    parser.add_argument("--use_demo_data", action="store_true",
                        help="使用内置中文示例数据")
    parser.add_argument("--output_file", type=str, default=None,
                        help="评测结果输出 JSON 文件")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_pairs", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ 模型路径不存在: {args.model_path}")
        sys.exit(1)

    # 加载 tokenizer
    print(f"\n加载 SPMTokenizer...")
    tokenizer = load_tokenizer(args.model_path)
    print(f"  词表大小: {tokenizer.vocab_size}")

    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"设备: {args.device}")

    # 加载数据
    if args.use_demo_data:
        print("\n使用内置示例数据...")
        pairs = DEMO_BLIMP_PAIRS
    elif args.data_file:
        print(f"\n加载数据集: {args.data_file}")
        with open(args.data_file, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        print(f"  加载 {len(pairs)} 个句对")
    else:
        print("❌ 请指定 --data_file 或 --use_demo_data")
        sys.exit(1)

    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    summary, results = evaluate_blimp(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        device=args.device,
    )

    if args.output_file:
        output_data = {
            "model_path": args.model_path,
            "summary": summary,
            "results": results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 评测结果已保存到: {args.output_file}")

    print(f"\n最终 BLiMP 准确率: {summary['accuracy'] * 100:.2f}%")
    return summary["accuracy"]


if __name__ == "__main__":
    main()
