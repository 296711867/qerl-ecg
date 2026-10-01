# -*- coding: utf-8 -*-
"""聚合阶段1结果：窗口长度-F1 曲线图 + 主结果表。

读取 outputs/tables/{task}_w{W}_f{F}_metrics.json，输出:
- outputs/figures/window_curve.png   （go/no-go 门槛图）
- outputs/tables/stage1_summary.md
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["N", "AFIB/AFL", "AVB", "SND", "SR-mPAC-BT", "SR-mPVC-BT", "SVTA", "VT", "MAT"]


def main():
    tables = ROOT / "outputs/tables"
    figs = ROOT / "outputs/figures"
    figs.mkdir(exist_ok=True, parents=True)

    rows = []
    for f in sorted(tables.glob("*_w*_f*_metrics.json")):
        if "smoke" in f.stem or "rl_" in f.stem:
            continue
        parts = f.stem.replace("_metrics", "").split("_")
        task, w = parts[0], int(parts[1][1:])
        best = json.loads(f.read_text())["best_val"]
        rows.append(dict(task=task, w=w, **{k: best[k] for k in
                       ("macro_f1", "balanced_acc", "accuracy") if k in best},
                         per_class=best.get("per_class_recall", {})))

    if not rows:
        print("no results yet")
        return

    nine = sorted([r for r in rows if r["task"] == "9class"], key=lambda r: r["w"])
    binr = sorted([r for r in rows if r["task"] == "binary"], key=lambda r: r["w"])

    # 曲线图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if nine:
        axes[0].plot([r["w"] for r in nine], [r["macro_f1"] for r in nine], "o-",
                     label="Macro-F1", color="tab:red")
        axes[0].plot([r["w"] for r in nine], [r["balanced_acc"] for r in nine], "s--",
                     label="Balanced Acc", color="tab:blue")
        axes[0].set(xlabel="Observation window (s)", ylabel="Score",
                    title="9-class: window length vs performance (val, fold 0)")
        axes[0].legend(); axes[0].grid(alpha=0.3)
    if binr:
        axes[1].plot([r["w"] for r in binr], [r["macro_f1"] for r in binr], "o-",
                     label="Macro-F1", color="tab:green")
        axes[1].set(xlabel="Observation window (s)", ylabel="Macro-F1",
                    title="Binary (Normal vs Abnormal)")
        axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "window_curve.png", dpi=150)
    print(f"saved window_curve.png")

    # 摘要表
    lines = ["# Stage 1 结果汇总（val, fold 0）\n",
             "| task | window(s) | macro-F1 | balanced acc | accuracy |",
             "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["task"], r["w"])):
        lines.append(f"| {r['task']} | {r['w']} | {r['macro_f1']:.4f} | "
                     f"{r['balanced_acc']:.4f} | {r['accuracy']:.4f} |")
    # 少数类 recall
    if nine:
        lines.append("\n## 9类逐类 recall（最佳窗口）\n")
        best9 = max(nine, key=lambda r: r["macro_f1"])
        lines.append(f"最佳: w={best9['w']}s\n")
        lines.append("| class | recall |")
        lines.append("|---|---|")
        for c, v in best9["per_class"].items():
            lines.append(f"| {c} | {v:.3f} |")
    (tables / "stage1_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
