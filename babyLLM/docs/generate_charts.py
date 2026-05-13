#!/usr/bin/env python3
"""Generate all visualization charts for the BabyLM README."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# ── Global style ──────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm

# Try to find and register CJK font
_cjk_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_cjk_font_path):
    fm.fontManager.addfont(_cjk_font_path)
    _cjk_prop = fm.FontProperties(fname=_cjk_font_path)
    _cjk_family = _cjk_prop.get_name()
else:
    _cjk_family = "DejaVu Sans"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": [_cjk_family, "Noto Sans CJK SC", "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
})

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
# Versions with complete data (skip V1/V3/V4/V6 — no comparable PPL)
VERSIONS = ["V2", "V3", "V5", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15"]
PPL =     [597,  542,  525,  50.84, 50.84, 50.84, 42.89, 40.72, 38.84, 38.68, 41.82, 45.14]
PARAMS =  [125,  125,  51,   30.1,  35,    35,    38.7,  38.7,  54.2,  94.2,  59.2,  68.2]
STATUS =  ["ok", "fail","ok", "ok",  "ok",  "ok",  "ok",  "ok",  "ok",  "SOTA","ok",  "ok"]

COLOR_MAP = {"fail": "#e74c3c", "ok": "#3498db", "SOTA": "#f1c40f"}
COLORS = [COLOR_MAP[s] for s in STATUS]

# Versions with stage data
STAGE_VERSIONS = ["V10", "V11", "V12", "V13", "V14", "V15"]
STAGE_PPL = {
    "V10": [55.60, 50.68, 42.89],
    "V11": [46.24, 40.85, 40.77, 40.75, 40.72],
    "V12": [41.83, 38.96, 39.01, 39.04, 39.04],
    "V13": [42.19, 40.73, 40.32],
    "V14": [45.68, 44.12, 41.55, 41.82, 41.84],
    "V15": [45.38, 45.14],
}
STAGE_LABELS = {
    "V10": ["CLM", "MNTP", "Polish"],
    "V11": ["CLM\nSGDR", "MNTP", "Polish", "Anneal", "Self\nDistill"],
    "V12": ["CLM\nSGDR", "MNTP", "Polish", "Self\nDistill", "Anneal"],
    "V13": ["CLM\nSGDR", "MNTP", "Polish"],
    "V14": ["CLM\nSGDR", "MNTP", "Polish", "Self\nDistill", "Anneal"],
    "V15": ["CLM\nSGDR", "MNTP"],
}

# Official eval
EVAL_VERSIONS = ["V13", "V14", "V15"]
EVAL_TASKS = ["ZhoBLiMP", "Hanzi\nStructure", "Hanzi\nPinyin", "AFQMC", "OCNLI", "TNEWS", "WSC2020"]
EVAL_DATA = {
    "V13": [63.5, 64.7, 49.5, 69.0, 64.0, 53.9, 63.5],
    "V14": [64.3, 62.4, 41.9, 69.0, 66.0, 54.1, 63.5],
    "V15": [62.4, 63.9, 47.4, 69.0, 65.9, 54.4, 63.8],
}

# Technique impact (estimated PPL improvement %)
TECHNIQUES = [
    "SPM Tokenizer\n(BPE to SPM)",
    "MNTP Hybrid\nTraining",
    "EMA Weight\nAveraging",
    "SGDR Learning\nRate Schedule",
    "Focal Loss",
    "PPL Data\nFiltering",
    "Multi-stage\nPipeline",
    "Label\nSmoothing",
]
TECH_IMPACT = [74, 16, 7, 3, 5, 4, 8, 2]  # percentage contribution


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1: PPL Evolution Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
def chart_ppl_evolution():
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(VERSIONS, PPL, color=COLORS, edgecolor="white", linewidth=0.8, width=0.7)
    ax.set_yscale("log")
    ax.set_ylabel("Perplexity (PPL) — log scale", fontsize=12)
    ax.set_xlabel("Version", fontsize=12)
    ax.set_title("BabyLM V2–V15  PPL Evolution", fontsize=14, fontweight="bold")
    for bar, ppl, status in zip(bars, PPL, STATUS):
        label = f"{ppl:.1f}"
        if status == "SOTA":
            label += "\nSOTA"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    # Legend
    legend_elements = [
        Line2D([0], [0], color="#3498db", lw=4, label="Complete"),
        Line2D([0], [0], color="#f1c40f", lw=4, label="SOTA (V13)"),
        Line2D([0], [0], color="#e74c3c", lw=4, label="Failed"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    ax.set_ylim(top=max(PPL) * 2)
    save(fig, "ppl_evolution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2: Params vs PPL Scatter
# ═══════════════════════════════════════════════════════════════════════════════
def chart_params_vs_ppl():
    fig, ax = plt.subplots(figsize=(10, 7))
    # Exclude V2/V3 (outlier PPL) for cleaner chart
    keep = [i for i, v in enumerate(VERSIONS) if v not in ("V2", "V3")]
    px = [PARAMS[i] for i in keep]
    py = [PPL[i] for i in keep]
    vl = [VERSIONS[i] for i in keep]
    vc = [COLORS[i] for i in keep]

    ax.scatter(px, py, s=200, c=vc, edgecolors="white", linewidths=1.5, zorder=5)
    for x, y, label in zip(px, py, vl):
        offset = (8, 8) if label != "V15" else (-15, -15)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=offset, fontsize=11, fontweight="bold")

    # Pareto frontier (lower-left is better)
    pts = sorted(zip(px, py, vl), key=lambda t: t[0])
    frontier_x, frontier_y = [], []
    best_y = float("inf")
    for x, y, _ in pts:
        if y < best_y:
            frontier_x.append(x)
            frontier_y.append(y)
            best_y = y
    ax.plot(frontier_x, frontier_y, "--", color="#e74c3c", alpha=0.6, lw=2,
            label="Pareto frontier")
    ax.fill_between(frontier_x, frontier_y, [max(py) * 1.5] * len(frontier_y),
                    alpha=0.05, color="#e74c3c")

    ax.set_xlabel("Parameters (M)", fontsize=12)
    ax.set_ylabel("Best PPL (lower is better)", fontsize=12)
    ax.set_title("Parameters vs PPL — Efficiency Frontier", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    save(fig, "params_vs_ppl.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3: Version Timeline
# ═══════════════════════════════════════════════════════════════════════════════
def chart_version_timeline():
    versions_info = [
        ("V1",  "GPT-2\nBaseline",      343,    "Baseline"),
        ("V2",  "LLaMA\nArch",           597,    "Arch Migrate"),
        ("V3",  "SPM\nTokenizer",        542,    "NCCL Failed"),
        ("V4",  "Deep\nModel",           None,   "Too Large"),
        ("V5",  "Small +\nKD",           525,    "Knowledge Distill"),
        ("V6",  "3-Stage\nPipeline",     None,   "Data Lost"),
        ("V7",  "MNTP\nHybrid",          50.8,   "8K Vocab"),
        ("V8",  "Simplified\n3-Stage",   50.8,   "PPL=50.84"),
        ("V9",  "Probe\nExperiments",    50.8,   "Hparam Search"),
        ("V10", "Production\nPipeline",  42.9,   "PPL=42.89"),
        ("V11", "EMA +\nSGDR",          40.7,   "PPL=40.72"),
        ("V12", "Focal\nLoss",           38.8,   "Best Efficiency"),
        ("V13", "PPL\nFiltering",        38.7,   "SOTA"),
        ("V14", "Efficiency\nBuild",     41.8,   "52M Params"),
        ("V15", "Optimized\nArch",       45.1,   "68M Params"),
    ]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(-0.5, len(versions_info) - 0.5)
    ax.set_ylim(-2, 5)
    ax.axis("off")
    ax.set_title("BabyLM V1–V15  Version Evolution Timeline",
                 fontsize=16, fontweight="bold", pad=20)

    # Draw timeline spine
    ax.plot([-0.3, len(versions_info) - 0.7], [1.5, 1.5], "-",
            color="#bdc3c7", lw=3, zorder=1)

    for i, (ver, desc, ppl, note) in enumerate(versions_info):
        # Alternate above/below
        y_box = 2.8 if i % 2 == 0 else 0.2
        y_con = 1.5

        # Color by status
        if ppl is None:
            color = "#e74c3c"  # failed
        elif ppl < 40:
            color = "#f1c40f"  # SOTA
        elif ppl < 45:
            color = "#2ecc71"  # good
        else:
            color = "#3498db"  # ok

        # Node on spine
        ax.plot(i, y_con, "o", color=color, ms=12, zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        # Connector line
        y_line = y_box + (0.4 if i % 2 == 0 else -0.4)
        ax.plot([i, i], [y_con, y_line], "-", color="#bdc3c7", lw=1.5)

        # Box
        box = FancyBboxPatch((i - 0.42, y_box - 0.35), 0.84, 0.9,
                             boxstyle="round,pad=0.05", facecolor=color,
                             alpha=0.15, edgecolor=color, linewidth=1.5,
                             transform=ax.transData)
        ax.add_patch(box)
        # Text
        y_text = y_box + (0.1 if i % 2 == 0 else -0.1)
        ax.text(i, y_text + 0.2, ver, ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)
        ax.text(i, y_text - 0.05, desc, ha="center", va="center",
                fontsize=8, color="#2c3e50")
        ppl_str = f"PPL={ppl:.1f}" if ppl else note
        ax.text(i, y_text - 0.25, ppl_str, ha="center", va="center",
                fontsize=7, color="#7f8c8d", style="italic")

    save(fig, "version_timeline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4: Training Pipeline Architecture
# ═══════════════════════════════════════════════════════════════════════════════
def chart_training_pipeline():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("BabyLM Multi-Stage Training Pipeline",
                 fontsize=16, fontweight="bold", pad=20)

    def draw_box(x, y, w, h, label, color, sublabel=""):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, alpha=0.2, edgecolor=color,
                             linewidth=2)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2 + (0.15 if sublabel else 0), label,
                ha="center", va="center", fontsize=11, fontweight="bold", color="#2c3e50")
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.25, sublabel,
                    ha="center", va="center", fontsize=8, color="#7f8c8d")

    def arrow(x1, y1, x2, y2, text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5))
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.1, my, text, fontsize=8, color="#7f8c8d",
                    rotation=0, va="bottom")

    # Row 1: Data pipeline
    draw_box(0.5, 8.5, 2.5, 1, "Raw Data", "#e74c3c", "~100M chars")
    arrow(3.0, 9.0, 3.7, 9.0, "")
    draw_box(3.7, 8.5, 2.5, 1, "Preprocessing", "#e67e22", "Clean + Dedup")
    arrow(6.2, 9.0, 6.9, 9.0, "")
    draw_box(6.9, 8.5, 2.5, 1, "PPL Filtering", "#f1c40f", "max_ppl=250")
    arrow(9.4, 9.0, 10.1, 9.0, "")
    draw_box(10.1, 8.5, 3.2, 1, "Tokenization", "#2ecc71", "SPM Unigram 8K/32K")

    # Row 2: Stage 1
    draw_box(0.5, 5.5, 3, 2, "Stage 1: CLM", "#3498db",
             "Causal LM + SGDR\nEMA + Focal Loss\n8-10 epochs")
    arrow(3.5, 6.5, 4.3, 6.5, "best_model_ema")

    # Stage 2
    draw_box(4.3, 5.5, 3, 2, "Stage 2: MNTP", "#9b59b6",
             "Masked Next Token Pred\nDynamic CLM ratio\nEMA + Focal Loss")
    arrow(7.3, 6.5, 8.1, 6.5, "best_model_ema")

    # Stage 3 (optional)
    draw_box(8.1, 5.5, 3, 2, "Stage 3: Polish", "#1abc9c",
             "Pure CLM fine-tune\nNo regularization\nOptional")
    arrow(11.1, 6.5, 11.6, 6.5, "")

    # Best model
    draw_box(11.6, 5.8, 2, 1.4, "Best Model", "#f1c40f", "EMA checkpoint")

    # Row 3: Key techniques sidebar
    techniques = [
        "RoPE + GQA + SwiGLU + RMSNorm",
        "EMA (decay=0.999)",
        "SGDR scheduler (T0=2k, Tmult=2)",
        "Focal Loss (gamma=1.5~2.0)",
        "Label Smoothing (0.1 → anneal)",
        "Gradient Checkpointing + bf16",
        "Multi-scale EMA (0.999 + 0.9999)",
    ]
    ax.text(0.5, 4.2, "Key Techniques:", fontsize=11, fontweight="bold", color="#2c3e50")
    for j, tech in enumerate(techniques):
        ax.text(0.7, 3.7 - j * 0.45, f"  {tech}", fontsize=9, color="#34495e")

    # Row 3 right: Evaluation
    draw_box(8, 2, 5.5, 3, "Evaluation", "#e74c3c",
             "PPL / ZhoBLiMP / Hanzi\nAFQMC / OCNLI / TNEWS\nWSC2020 / Generation Quality")

    save(fig, "training_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 5: Official Eval Radar Chart
# ═══════════════════════════════════════════════════════════════════════════════
def chart_official_eval_radar():
    N = len(EVAL_TASKS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#f1c40f", "#3498db", "#e74c3c"]

    for ver, color in zip(EVAL_VERSIONS, colors):
        values = EVAL_DATA[ver] + EVAL_DATA[ver][:1]
        ax.plot(angles, values, "o-", linewidth=2, label=ver, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(EVAL_TASKS, fontsize=10)
    ax.set_ylim(35, 75)
    ax.set_title("Official Evaluation — CLUE Benchmark Tasks",
                 fontsize=14, fontweight="bold", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    save(fig, "official_eval_radar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 6: Stage-by-Stage PPL Improvement
# ═══════════════════════════════════════════════════════════════════════════════
def chart_ppl_by_stage():
    fig, axes = plt.subplots(1, len(STAGE_VERSIONS), figsize=(16, 5), sharey=True)
    fig.suptitle("Stage-by-Stage PPL Improvement (V10–V15)",
                 fontsize=14, fontweight="bold", y=1.02)

    stage_colors = ["#3498db", "#9b59b6", "#1abc9c", "#e67e22", "#e74c3c"]

    for ax, ver in zip(axes, STAGE_VERSIONS):
        ppls = STAGE_PPL[ver]
        labels = STAGE_LABELS[ver]
        colors = stage_colors[:len(ppls)]
        bars = ax.bar(range(len(ppls)), ppls, color=colors, edgecolor="white", width=0.7)
        for bar, ppl in zip(bars, ppls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{ppl:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(ver, fontsize=12, fontweight="bold")
        ax.set_ylim(35, 60)

    axes[0].set_ylabel("PPL", fontsize=11)
    fig.tight_layout()
    save(fig, "ppl_by_stage.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 7: Efficiency Frontier
# ═══════════════════════════════════════════════════════════════════════════════
def chart_efficiency_frontier():
    # PPL per 10M params
    fig, ax = plt.subplots(figsize=(10, 7))
    keep = [i for i, v in enumerate(VERSIONS) if v not in ("V2", "V3")]
    px = [PARAMS[i] for i in keep]
    ppl_10m = [PPL[i] / PARAMS[i] * 10 for i in keep]
    vl = [VERSIONS[i] for i in keep]
    vc = [COLORS[i] for i in keep]

    ax.scatter(px, ppl_10m, s=200, c=vc, edgecolors="white", linewidths=1.5, zorder=5)
    for x, y, label in zip(px, ppl_10m, vl):
        offset = (8, 8) if label not in ("V8", "V9") else (-15, -15)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=offset, fontsize=11, fontweight="bold")

    # Pareto frontier (lower is better)
    pts = sorted(zip(px, ppl_10m, vl), key=lambda t: t[0])
    fx, fy = [], []
    best = float("inf")
    for x, y, _ in pts:
        if y < best:
            fx.append(x)
            fy.append(y)
            best = y
    ax.plot(fx, fy, "--", color="#e74c3c", alpha=0.6, lw=2, label="Efficiency frontier")
    ax.fill_between(fx, fy, [max(ppl_10m) * 1.5] * len(fy), alpha=0.05, color="#e74c3c")

    ax.set_xlabel("Parameters (M)", fontsize=12)
    ax.set_ylabel("PPL per 10M Params (lower is better)", fontsize=12)
    ax.set_title("Parameter Efficiency Frontier", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    save(fig, "efficiency_frontier.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 8: Technique Impact Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def chart_technique_impact():
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(TECHNIQUES))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(TECHNIQUES)))[::-1]

    bars = ax.barh(y_pos, TECH_IMPACT, color=colors, edgecolor="white", height=0.6)
    for bar, impact in zip(bars, TECH_IMPACT):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{impact}%", ha="left", va="center", fontsize=11, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(TECHNIQUES, fontsize=10)
    ax.set_xlabel("Estimated PPL Improvement Contribution (%)", fontsize=12)
    ax.set_title("Technique Impact Analysis — PPL Improvement Breakdown",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(TECH_IMPACT) * 1.2)
    save(fig, "technique_impact.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating BabyLM README charts...")
    chart_ppl_evolution()
    chart_params_vs_ppl()
    chart_version_timeline()
    chart_training_pipeline()
    chart_official_eval_radar()
    chart_ppl_by_stage()
    chart_efficiency_frontier()
    chart_technique_impact()
    print("All 8 charts generated in docs/assets/")
