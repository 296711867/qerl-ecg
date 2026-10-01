# -*- coding: utf-8 -*-
"""论文 Figure 1: QERL-ECG 框架总览图（matplotlib 绘制，300dpi）。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/figures/framework.png"

BLUE, ORANGE, GREEN, RED, GRAY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C"


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec=BLUE, fs=9, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=GRAY, style="-|>", lw=1.6, ls="-",
          rad=0.0, label=None, lfs=7.5, lox=0, loy=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, color=color, lw=lw,
                                 linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + lox, (y1 + y2) / 2 + loy, label, fontsize=lfs,
                ha="center", va="center", color=color,
                bbox=dict(fc="white", ec="none", pad=0.8))


def synth_ecg(fs=100, n=10, seed=3):
    """合成 10 秒 ECG 示意波形。"""
    rng = np.random.RandomState(seed)
    t = np.arange(n * fs) / fs
    ecg = 0.08 * np.sin(2 * np.pi * t * 1.2)          # 基线
    for rr in rng.uniform(0.75, 0.95, size=14):       # R 峰序列
        pass
    beats = np.arange(0.4, 10, 0.83)
    for b in beats:
        for center, amp, sig in [(b, 1.0, 0.028), (b - 0.05, -0.18, 0.03),
                                 (b + 0.20, 0.22, 0.045)]:
            ecg += amp * np.exp(-0.5 * ((t - center) / sig) ** 2)
    ecg += 0.02 * rng.randn(len(t))
    return t, ecg


def main():
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.set_xlim(0, 125); ax.set_ylim(0, 64); ax.axis("off")

    # ===== 1. ECG 输入（含逐秒观察刻度）=====
    ax_in = fig.add_axes([0.025, 0.60, 0.185, 0.30])
    t, ecg = synth_ecg()
    ax_in.plot(t, ecg, lw=0.8, color="k")
    for s in range(1, 11):
        ax_in.axvline(s, color=GRAY, lw=0.4, ls=":", alpha=0.6)
    ax_in.axvspan(0, 4, color=BLUE, alpha=0.10)
    ax_in.set_xlim(0, 10); ax_in.set_xticks(range(0, 11, 2))
    ax_in.tick_params(labelsize=7)
    ax_in.set_title("Intraoperative ECG stream (II, 100 Hz)", fontsize=8.5)
    ax_in.set_ylabel("mV", fontsize=7)
    ax.annotate("", xy=(0.42, -0.32), xytext=(0.03, -0.32),
                xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.2))
    ax_in.text(0.22, -0.45, "observation time", fontsize=7.5,
               transform=ax_in.transAxes, ha="center")

    # ===== 2. 编码器与双头 =====
    box(ax, 27, 40, 22, 12, "Multi-scale TCN\n(k=5/9/15, dilated)\n+ GRU encoder", fs=9)
    box(ax, 57, 48, 20, 9, "Rhythm head\n$P(y\\,|\\,x_{1:t})$, 9 classes", ec=ORANGE, fs=8.5)
    box(ax, 57, 36, 20, 9, "Quality head\n$\\hat{q}_t\\in[0,1]$", ec=GREEN, fs=8.5)
    arrow(ax, 49, 46, 57, 52.5); arrow(ax, 49, 46, 57, 40.5)

    # ===== 3. 状态 =====
    box(ax, 83, 40, 17, 12, "State $s_t$\n$[P, \\max P, $ margin,\n$H(P),\\hat q_t, t/T]$",
        fc="#F5F7FA", fs=8)
    arrow(ax, 77, 52.5, 83, 48); arrow(ax, 77, 40.5, 83, 45)

    # ===== 4. RL 决策器 =====
    box(ax, 83, 14, 17, 12, "Double-Dueling\nDQN\n$Q(s_t, a)$", fc="#FDF6EC",
        ec=RED, fs=9.5, lw=1.8)
    arrow(ax, 91.5, 40, 91.5, 26, color=RED)

    # ===== 5. 动作 =====
    box(ax, 108, 52, 15, 7, "WAIT 1s / 2s\n(keep observing)", ec=BLUE, fs=8)
    box(ax, 108, 38, 15, 7, "STOP\n$\\to$ label $\\hat y$, time $t$", ec=ORANGE, fs=8)
    box(ax, 108, 24, 15, 7, "ABSTAIN\n(flag low quality)", ec=GREEN, fs=8)
    arrow(ax, 100, 22, 108, 55, color=RED, rad=-0.25)
    arrow(ax, 100, 20, 108, 41.5, color=RED, rad=-0.12)
    arrow(ax, 100, 18, 108, 27.5, color=RED, rad=0.05)

    # WAIT 回环：动作 → 观察窗延长（明确指回 ECG 波形面板）
    arrow(ax, 115.5, 52, 115.5, 62, color=BLUE, lw=1.2)
    ax.add_patch(FancyArrowPatch((115.5, 62), (24, 62), arrowstyle="-",
                                 color=BLUE, lw=1.2))
    arrow(ax, 24, 62, 24, 51.5, color=BLUE, lw=1.2)
    ax.text(70, 63.2, "next second of ECG becomes available", fontsize=8,
            color=BLUE, ha="center")

    # ===== 6. 奖励注释 =====
    box(ax, 27, 8, 50, 12,
        "Reward:  correct STOP $+1$   |   wrong STOP $-2$\n"
        "waiting $-0.03/s$   |   abstain on bad quality $+0.3$ / else $-0.4$",
        ec=GRAY, fs=8.5, fc="#FAFAFA")
    arrow(ax, 83, 14, 77, 14, color=GRAY, ls="--", lw=1.0)

    ax.text(2, 4, "QERL-ECG: quality-aware reinforcement learning for adaptive "
            "early intraoperative arrhythmia recognition",
            fontsize=9, style="italic", color=GRAY)

    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
