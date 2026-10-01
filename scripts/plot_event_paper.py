"""Rebuild the two manuscript figures from the causal-policy result table."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/v2/figures"
OUT.mkdir(parents=True, exist_ok=True)
RESULT = ROOT / "outputs/tables/event_cv_v2b_ce_sqrt_causal.json"


def save(fig, stem):
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def protocol():
    fig, ax = plt.subplots(figsize=(9.2, 3.25))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.1, 1.55, 1.8, 1.1, "Public ECG +\nexpert annotations", "#DCECF5"),
        (2.2, 1.55, 1.8, 1.1, "One episode per\nlabelled rhythm run", "#DCECF5"),
        (4.3, 1.55, 1.8, 1.1, "Causal 1–10 s\nECG prefixes", "#E8F2E2"),
        (6.4, 1.55, 1.35, 1.1, "Shared\nclassifier", "#E8F2E2"),
        (8.05, 1.55, 1.85, 1.1, "Stopping rules +\npatient-level tests", "#FDEBD8"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                                    linewidth=1.1, edgecolor="#3C5367", facecolor=color))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    for x1, x2 in ((1.97, 2.16), (4.07, 4.25), (6.17, 6.35), (7.82, 7.99)):
        ax.add_patch(FancyArrowPatch((x1, 2.1), (x2, 2.1), arrowstyle="-|>",
                                     mutation_scale=14, linewidth=1.2, color="#3C5367"))
    ax.text(5.0, 3.3, "482 patients  •  4,793 eligible episodes  •  nine rhythm classes",
            ha="center", fontsize=11, color="#17324A")
    ax.text(5.0, 0.95, "At each second: observed ECG → class posterior → wait / classify / abstain",
            ha="center", fontsize=10.5)
    ax.text(5.0, 0.46,
            "Oracle annotation supplies episode start and end; the new DQN does not receive future duration.",
            ha="center", fontsize=9.2, color="#6A3E24")
    save(fig, "fig1_protocol")


def results():
    data = json.loads(RESULT.read_text(encoding="utf-8"))["folds"]
    names = ["Fixed 1s", "Fixed 3s", "Fixed 5s", "Fixed 10s",
             "Threshold", "Time threshold", "RL"]
    labels = ["Fixed 1 s", "Fixed 3 s", "Fixed 5 s", "Fixed 10 s",
              "Global threshold", "Time threshold", "DQN"]
    colors = ["#888888"] * 4 + ["#9467BD", "#0072B2", "#D55E00"]
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for name, label, color, marker in zip(names, labels, colors, markers):
        f1 = np.array([r["metrics"][name]["macro_f1_all"] for r in data])
        t = np.array([r["metrics"][name]["mean_decision_time"] for r in data])
        ax.errorbar(t.mean(), f1.mean(), xerr=t.std(ddof=1), yerr=f1.std(ddof=1),
                    fmt=marker, markersize=8, capsize=3,
                    color=color, markeredgecolor="white", linewidth=1.4, label=label)
    ax.set(xlabel="Observed seconds after first annotated beat (fold mean ± SD)",
           ylabel="Event-level macro-F1 (fold mean ± SD)", xlim=(0.4, 9.1), ylim=(0.30, 0.60))
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.tick_params(labelsize=9)
    save(fig, "fig2_accuracy_time")


if __name__ == "__main__":
    protocol()
    results()
    print(OUT)
