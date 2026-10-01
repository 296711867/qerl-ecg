# -*- coding: utf-8 -*-
"""从已保存的 JSON 表重绘 Fig 4 / Fig 5（标签避让版），不重跑评估。

数据源: outputs/tables/cv_summary{SFX}.json (table3) + ablation{SFX}.json (lambda_sweep)
输出:   outputs/figures/pareto_cv{SFX}.png / lambda_pareto{SFX}.png (300dpi)

用法:   python scripts/replot_paretos.py [suffix]     # 默认 suffix=os

标签锚点与 eval_cv.py / eval_ablation.py 内联版本保持一致；中部聚簇区
（Threshold / Fixed 5s / RL-DDDQN）互不压点、互不压线。重评估后重跑本脚本即可。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "os"
SFX = f"_{SUFFIX}" if SUFFIX else ""

BLUE, RED, GRAY = "#4C72B0", "#C44E52", "#8C8C8C"


def bbox():
    return dict(fc="white", ec="none", alpha=0.75, pad=1)


def plot_pareto():
    cv = json.loads((ROOT / f"outputs/tables/cv_summary{SFX}.json").read_text(encoding="utf-8"))
    pts = {}
    for pol, ms in cv["table3"].items():
        ys = [m["macro_f1"] for m in ms]
        xs = [m["mean_decision_time"] for m in ms]
        pts[pol] = (float(np.mean(xs)), float(np.mean(ys)), float(np.std(ys, ddof=1)))

    layout = {  # (dx, dy, ha)：与 eval_cv.py 内联版一致
        "Fixed 1s":  (13, 0, "left"),
        "Fixed 2s":  (0, -13, "center"),
        "Fixed 3s":  (0, -13, "center"),
        "Fixed 5s":  (12, 0, "left"),
        "Fixed 10s": (3, -10, "right"),
        "Threshold": (0, -15, "center"),
        "RL-DDDQN (ours)": (12, 0, "left"),
    }

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for pol, (mx, my, sd) in pts.items():
        is_rl, is_th = pol.startswith("RL"), pol.startswith("Thresh")
        color = RED if is_rl else (GRAY if is_th else BLUE)
        ax.scatter(mx, my, marker="*" if is_rl else ("s" if is_th else "o"),
                   s=200 if is_rl else 60, color=color, zorder=3)
        dx, dy, ha = layout.get(pol, (8, -8, "left"))
        ax.annotate(f"{pol}\n(mean\u00b1{sd:.3f})", (mx, my), fontsize=7, color=color,
                    xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center" if dy == 0 else ("top" if dy < 0 else "bottom"),
                    zorder=5, bbox=bbox())
    ax.set(xlabel="Mean decision time (s)", ylabel="Macro-F1",
           title="Policy comparison (5-fold CV)")
    ax.margins(x=0.06, y=0.09)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / f"outputs/figures/pareto_cv{SFX}.png"
    fig.savefig(out, dpi=300)
    print("saved", out)


def plot_lambda():
    ab = json.loads((ROOT / f"outputs/tables/ablation{SFX}.json").read_text(encoding="utf-8"))
    rows = sorted(ab["lambda_sweep"], key=lambda r: r["lam"])

    layout = {  # (dx, dy, ha)：与 eval_ablation.py 内联版一致
        0.00: (0, -14, "center"),
        0.01: (9, -18, "left"),
        0.03: (-9, 10, "right"),
        0.05: (0, -14, "center"),
        0.10: (9, 8, "left"),
    }

    fig, ax = plt.subplots(figsize=(6, 4.2))
    sweep = ab["lambda_sweep"]                                          # 评估顺序连线
    dts = [r["mean_decision_time"] for r in sweep]
    f1s = [r["macro_f1"] for r in sweep]
    ax.plot(dts, f1s, "o-", color=BLUE, alpha=0.55)
    for r in rows:
        lam = r["lam"]
        ax.scatter(r["mean_decision_time"], r["macro_f1"], s=28, color=BLUE,
                   edgecolors="white", linewidths=1.2, zorder=3)
        dx, dy, ha = layout.get(round(lam, 2), (8, -8, "left"))
        ax.annotate(f"\u03bb={lam:.2f}", (r["mean_decision_time"], r["macro_f1"]),
                    fontsize=8, xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="top" if dy < 0 else "bottom", zorder=5, bbox=bbox())
    ax.set(xlabel="Mean decision time (s)", ylabel="Macro-F1",
           title="Reward wait-cost \u03bb: accuracy\u2013latency frontier (test)")
    ax.margins(x=0.09, y=0.12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / f"outputs/figures/lambda_pareto{SFX}.png"
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    plot_pareto()
    plot_lambda()
