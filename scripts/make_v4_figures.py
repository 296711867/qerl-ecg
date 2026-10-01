# -*- coding: utf-8 -*-
"""Figures for manuscript v4. Run scripts/v4_collect.py first.

Usage: python scripts/make_v4_figures.py [fig1 fig2 ...]   (default: all)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.event_dataset import CACHE, CLASSES, FS  # noqa: E402

OUT = ROOT / "paper/v4/figures"
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
FULL, HALF = 183 * MM, 89 * MM

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 600, "svg.fonttype": "none", "pdf.fonttype": 42,
})

C = dict(fixed="#8C8C8C", thr="#9DB9D8", tt="#2E6DB4", dqn="#D2552B",
         matched="#7A4FA0", oracle="#3A9A5B", ink="#222222", soft="#F4F4F4")
SHORT = ["N", "AF/AFL", "AVB", "SND", "PAC", "PVC", "SVTA", "VT", "MAT"]
CLASS_COL = ["#6E6E6E", "#2E6DB4", "#7A4FA0", "#A0522D", "#3A9A5B",
             "#D2552B", "#C9A227", "#B22234", "#1B9AAA"]

D = dict(np.load(ROOT / "outputs/tables/v4_decisions.npz"))
V3 = json.loads((ROOT / "outputs/tables/v3_analysis.json").read_text())
EPIS = pd.read_parquet(ROOT / "data/processed/event_episodes_v2.parquet")


def save(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def tt_curve(fold):
    tau0, b = D["tt_params"][fold]
    w = np.arange(1, 11)
    return w, np.clip(tau0 + b * (w - 1) / 9, 0.01, 0.99)


def episode_ecg(i, seconds=10):
    row = EPIS.set_index("event_id").loc[int(D["event_id"][i])]
    ecg, t0 = CACHE.get(int(row.case_id))
    s = int(round((row.t_start - t0) * FS))
    x = ecg[s:s + seconds * FS].astype(float)
    return (x - np.median(x)) / (np.percentile(x, 99) - np.percentile(x, 1) + 1e-6)


# --------------------------------------------------------------------------- Fig 1
def fig1():
    """Two real test episodes: the growing ECG prefix, the posterior and the stops."""
    # (index, title, offsets of the TT / DQN annotation text in data units)
    examples = [(34, "Atrial fibrillation: both rules stop at 2 s",
                 {"tt": (1.3, -0.3), "dqn": (1.3, -0.14)}),
                (246, "Sinus rhythm with PVCs: the first ectopic beat appears after 2 s",
                 {"tt": (0.6, -0.42), "dqn": (-3.9, 0.15)})]
    fig = plt.figure(figsize=(FULL, 92 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.42, wspace=0.14,
                          left=0.06, right=0.99, top=0.9, bottom=0.11)
    for col, (i, title, offs) in enumerate(examples):
        y, fold = int(D["labels9"][i]), int(D["fold"][i])
        st, sd = int(D["stop_tt"][i]), int(D["stop_dqn"][i])
        P = D["probs"][i]
        x = episode_ecg(i)
        t = np.arange(len(x)) / FS

        ax = fig.add_subplot(gs[0, col])
        for k in range(10):
            ax.axvspan(k, k + 1, color=C["soft"] if k % 2 == 0 else "white", lw=0, zorder=0)
        ax.plot(t, x, color=C["ink"], lw=0.55, zorder=2)
        marks = [(st, "tt", "TT stop"), (sd, "dqn", "DQN stop")] if st != sd else             [(st, "dqn", "TT and DQN stop")]
        for s, key, lab in marks:
            ax.axvline(s, color=C[key], lw=1.1, zorder=3)
            ax.text(s, 1.02, lab, color=C[key], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6.3, fontweight="bold")
        ax.set_xlim(0, 10)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xticks(range(11))
        ax.tick_params(labelbottom=False)
        ax.set_title(f"{title}\n(true label {CLASSES[y]}, patient-held-out fold {fold + 1})",
                     loc="left", pad=11, fontsize=7)
        panel(ax, "ab"[col], x=-0.07, y=1.28)

        ax2 = fig.add_subplot(gs[1, col])
        w = np.arange(1, 11)
        top = np.argsort(P.max(0))[::-1][:3]
        for c in range(9):
            if c in top:
                ax2.plot(w, P[:, c], "-o", ms=2.6, lw=1.1, color=CLASS_COL[c],
                         label=SHORT[c], zorder=3)
            else:
                ax2.plot(w, P[:, c], "-", lw=0.4, color="#CFCFCF", zorder=1)
        tw, thr = tt_curve(fold)
        ax2.step(tw, thr, where="mid", color=C["tt"], ls="--", lw=0.9,
                 label="TT threshold", zorder=2)
        for s, key, mk in ((st, "tt", "s"), (sd, "dqn", "D")):
            c = P[s - 1].argmax()
            ax2.plot(s, P[s - 1, c], marker=mk, ms=6.5, mfc="white", mec=C[key], mew=1.3,
                     zorder=5, ls="none")
            ok = "correct" if c == y else "wrong"
            ax2.annotate(f"{'TT' if key == 'tt' else 'DQN'}: {SHORT[c]} ({ok})",
                         xy=(s, P[s - 1, c]), xytext=(s + offs[key][0], P[s - 1, c] + offs[key][1]),
                         color=C[key], fontsize=6.3, fontweight="bold",
                         arrowprops=dict(arrowstyle="-", color=C[key], lw=0.6),
                         bbox=dict(fc="white", ec="none", pad=0.6), zorder=6)
        ax2.set_xlim(0.5, 10.5)
        ax2.set_ylim(-0.02, 1.12)
        ax2.set_xticks(w)
        ax2.set_xlabel("Observed prefix since annotated onset (s)")
        if col == 0:
            ax2.set_ylabel("Classifier posterior")
        leg = ax2.legend(loc="center right", handlelength=1.6, borderaxespad=0.3,
                         bbox_to_anchor=(1.0, 0.5), frameon=True, framealpha=0.95,
                         edgecolor="none", facecolor="white")
        leg.set_zorder(7)
    save(fig, "fig1_examples")


# --------------------------------------------------------------------------- Fig 2
def blk(ax, x, y, w, h, fc, title=None, body=None, ec="#555555", lw=0.7, r=1.2, fs=6.4,
        title_dy=1.8, body_y=None, body_va="top", color="#222222"):
    """Rounded box with an optional bold title at the top and a body at body_y."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if title:
        ax.text(x + w / 2, y + h - title_dy, title, ha="center", va="top", fontsize=fs,
                fontweight="bold", zorder=3, color=color)
    if body:
        if body_y is None:
            body_y, body_va = (y + h / 2, "center") if not title else (y + h - title_dy - 4.2, "top")
        ax.text(x + w / 2, body_y, body, ha="center", va=body_va, fontsize=fs - 0.5,
                zorder=3, color=color, linespacing=1.35)


def arrow(ax, p, q, color="#444444", lw=0.9, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, color=color,
                                 lw=lw, zorder=4, linestyle=ls, shrinkA=0, shrinkB=0))


def poly(ax, pts, color="#444444", lw=0.9, head=True):
    """Orthogonal connector through pts, with an arrow head on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=4, solid_capstyle="butt")
    if head:
        arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


def fig2():
    """Architecture: prefix classifier (top) and stopping controller (bottom, right to left).

    NOTE(2026-09-29): 投稿版 fig2_architecture.{png,svg,pdf} 已被手写 SVG 复刻版替换
    （源文件 C:/Awork/01-Web-project/01-项目架构图绘制/ecg-paper/fig-prefix-classifier.svg）。
    重跑本函数会用 matplotlib 旧版覆盖该替换；除非有意恢复，请跳过 fig2。
    """
    fig = plt.figure(figsize=(FULL, 100 * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 183)
    ax.set_ylim(0, 100)
    ax.axis("off")
    blue, green, orange, grey = "#E4EEF8", "#E6F2E8", "#FBE9E1", "#F2F2F2"

    # frames
    ax.add_patch(FancyBboxPatch((2, 54.5), 179, 44, boxstyle="round,pad=0,rounding_size=2",
                                fc="none", ec="#9DB9D8", lw=0.8, ls="--"))
    ax.text(4.5, 95.8, "A   Prefix classifier (TCN-GRU, 346 k parameters) — re-run on the "
            "whole prefix every time it grows", fontsize=7.2, fontweight="bold",
            color=C["tt"], va="center")
    ax.add_patch(FancyBboxPatch((2, 1.5), 179, 44, boxstyle="round,pad=0,rounding_size=2",
                                fc="none", ec="#E3A58B", lw=0.8, ls="--"))
    ax.text(9, 42.8, "B   Stopping controller — is the prefix long enough?", fontsize=7.2,
            fontweight="bold", color=C["dqn"], va="center")

    # ---------------- A (left to right), boxes span y 59-91
    y0, h, ym = 59, 32, 75
    blk(ax, 9, y0, 27, h, grey, "ECG prefix $x_{1:w}$",
        "lead II, 100 Hz\n$w$ = 1 … 10 s since onset\nz-scored, left-padded\nto 1,000 samples")
    x = episode_ecg(246, 3)
    ax.add_patch(Rectangle((11, 61.5), 5, 8, fc="white", ec="#BBBBBB", lw=0.4, zorder=2.5))
    ax.text(13.5, 65.5, "pad", fontsize=5, ha="center", va="center", color="#999999", zorder=3)
    ax.plot(np.linspace(16, 34, len(x)), 65.2 + 3.2 * x, color=C["ink"], lw=0.45, zorder=3)

    blk(ax, 41, y0, 27, h, blue, "Multi-scale conv", "concat → BN → ReLU\n96 × 1,000",
        body_y=66.5)
    for j, k in enumerate((5, 9, 15)):
        blk(ax, 43 + j * 7.8, 73, 7, 9, "white", body=f"k = {k}\n32 ch", fs=5.9, r=0.6)

    blk(ax, 73, y0, 42, h, blue, "Dilated residual TCN",
        "each block: 2 × (conv3 → BN → ReLU → dropout)\n+ identity skip, 96 channels\nreceptive field 0.75 s",
        body_y=69.5)
    for j, dl in enumerate((1, 2, 4, 8)):
        bx = 75.5 + j * 9.9
        blk(ax, bx, 73, 7.8, 9, "white", body=f"d = {dl}", fs=5.9, r=0.6)
        if j < 3:
            arrow(ax, (bx + 7.8, 77.5), (bx + 9.9, 77.5), lw=0.6)

    blk(ax, 120, 78, 16, 13, blue, "1×1 conv", "96 → 128")
    blk(ax, 120, y0, 16, 16, blue, "GRU", "128 units\nlast state $h_w$")
    arrow(ax, (128, 78), (128, 75))

    blk(ax, 141, 77, 37, 14, green, "Rhythm head", "linear 128 → 9, softmax\nposterior $p_w$")
    blk(ax, 141, y0, 37, 14, green, "Quality head", "128 → 64 → 1, sigmoid\nquality score $q_w$")
    arrow(ax, (36, 77.5), (41, 77.5))
    arrow(ax, (68, 77.5), (73, 77.5))
    arrow(ax, (115, 84.5), (120, 84.5))
    poly(ax, [(136, 67), (138.5, 67), (138.5, 84), (141, 84)])
    arrow(ax, (138.5, 66), (141, 66))

    # ---------------- B (right to left), boxes span y 5-37
    blk(ax, 141, 5, 37, 32, orange, "State $s_w$ (15 values)",
        "posterior $p_w$ (9)\nmax $p_w$ and top-2 margin\nentropy $H(p_w)$\nquality $q_w$\n"
        "elapsed time $w/10$, bias\n\nno access to when the\nannotated segment ends")
    poly(ax, [(159.5, 59), (159.5, 37)])

    blk(ax, 70, 23.5, 56, 13.5, "white", "Rule: time-varying threshold (TT)",
        "stop if max $p_w$ ≥ clip($\\tau_0 + b\\,(w-1)/9$), otherwise wait 1 s\n"
        "2 parameters selected by validation reward", ec=C["tt"], lw=1.1)
    blk(ax, 70, 5, 56, 15.5, "white", "Learned: Double-Dueling DQN",
        "MLP 15 → 256 → 256 → value $V(s)$ + advantage $A(s,a)$\n"
        "greedy $a$ = argmax $Q(s,a)$; 71 k parameters\n"
        "reward +1 correct, −2 wrong, −0.03 s$^{-1}$; abstain +0.3 / −1",
        ec=C["dqn"], lw=1.1)
    ax.text(98, 21.9, "or", ha="center", va="center", fontsize=6.3, style="italic")
    arrow(ax, (141, 30.25), (126, 30.25))
    arrow(ax, (141, 12.75), (126, 12.75))

    ax.text(34, 38.5, "Action", ha="center", fontsize=6.5, fontweight="bold", va="center")
    acts = [("wait 1 s", "white"), ("wait 2 s (DQN only)", "white"),
            ("stop: output argmax $p_w$", green), ("abstain (DQN only)", "white")]
    ay = [30, 22.2, 14.4, 6.6]
    for (lab, fc), yy in zip(acts, ay):
        blk(ax, 17, yy, 34, 6.2, fc, body=lab, fs=6.2, r=0.8)
    arrow(ax, (70, 30.25), (51, 25.3))
    arrow(ax, (70, 12.75), (51, 17.5))
    ax.text(34, 3.3, "forced stop at the annotated segment end", fontsize=5.6,
            ha="center", color="#555555", style="italic")

    # wait loop: back to the classifier with a longer prefix
    loop = C["tt"]
    poly(ax, [(5, 25.3), (5, 50), (22.5, 50), (22.5, 59)], color=loop, lw=1.1)
    ax.plot([5, 17], [33.1, 33.1], color=loop, lw=1.1, zorder=4)
    ax.plot([5, 17], [25.3, 25.3], color=loop, lw=1.1, zorder=4)
    ax.text(25, 50, "wait:  $w \\leftarrow w+1$ or $w+2$  →  classify the longer prefix",
            fontsize=6.2, color=loop, va="center", ha="left",
            bbox=dict(fc="white", ec="none", pad=0.4))
    save(fig, "fig2_architecture")


# --------------------------------------------------------------------------- Fig 3
def fig3():
    """Data composition and how recognizability grows with prefix length."""
    y, wm, P = D["labels9"], D["wmax"], D["probs"]
    pred = P.argmax(2)
    fig = plt.figure(figsize=(FULL, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.2, 1.0], wspace=0.55,
                          left=0.07, right=0.985, top=0.88, bottom=0.2)

    ax = fig.add_subplot(gs[0])
    lab = EPIS.label.map({c: k for k, c in enumerate(CLASSES)}).values
    bins = [(1, 2, "1–2 s"), (3, 4, "3–4 s"), (5, 9, "5–9 s"), (10, 10, "10 s")]
    shades = ["#F2C4B0", "#E08A6A", "#9DB9D8", "#2E6DB4"]
    order = np.argsort([-(lab == c).sum() for c in range(9)])[::-1]
    left = np.zeros(9)
    tot = np.array([(lab == c).sum() for c in order])
    for (lo, hi, name), sh in zip(bins, shades):
        cnt = np.array([((lab == c) & (EPIS.w_max.values >= lo) & (EPIS.w_max.values <= hi)).sum()
                        for c in order])
        ax.barh(range(9), cnt / tot, left=left, color=sh, height=0.72, label=name, lw=0)
        left += cnt / tot
    pts = EPIS.groupby("label").case_id.nunique()
    for r, c in enumerate(order):
        ax.text(1.03, r, f"{tot[r]:,}  ({pts[CLASSES[c]]})", va="center", fontsize=5.9)
    ax.text(1.03, 8.95, "n (patients)", fontsize=5.6, va="center", color="#555555")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 9.4)
    ax.set_yticks(range(9))
    ax.set_yticklabels([SHORT[c] for c in order])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50%", "100%"])
    ax.set_xlabel("Share of episodes by available duration")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=4, fontsize=5.8,
              handlelength=0.9, columnspacing=0.8, handletextpad=0.4)
    ax.set_title("4,793 episodes, 482 patients", loc="left")
    panel(ax, "a", x=-0.3)

    ax = fig.add_subplot(gs[1])
    w = np.arange(1, 11)
    for c in [0, 1, 7, 5, 6, 4, 3, 2, 8]:
        rec, ws = [], []
        for k in w:
            m = (y == c) & (wm >= k)
            if m.sum() >= 10:
                rec.append((pred[m, k - 1] == c).mean())
                ws.append(k)
        ax.plot(ws, rec, "-o", ms=2, lw=1.0, color=CLASS_COL[c])
        nudge = {0: 0.03, 1: -0.03, 4: 0.025, 2: -0.025}.get(c, 0)
        ax.text(ws[-1] + 0.25, rec[-1] + nudge, SHORT[c], color=CLASS_COL[c], fontsize=5.8,
                va="center")
    ax.set_xlim(0.7, 11.8)
    ax.set_ylim(-0.03, 1)
    ax.set_xticks(w)
    ax.set_xlabel("Prefix length (s)")
    ax.set_ylabel("Recall (episodes lasting ≥ t)")
    ax.set_title("Per-rhythm recall vs prefix length", loc="left")
    panel(ax, "b", x=-0.2)

    ax = fig.add_subplot(gs[2])
    i = np.arange(len(y))
    corr = pred == y[:, None]
    valid = w[None, :] <= wm[:, None]
    first = np.where((corr & valid).any(1), (corr & valid).argmax(1) + 1, 99)
    cum_any = [(first <= k).mean() for k in w]
    cum_tt = [((D["stop_tt"] <= k) & (pred[i, D["stop_tt"] - 1] == y)).mean() for k in w]
    cum_dqn = [((D["stop_dqn"] <= k) & (pred[i, D["stop_dqn"] - 1] == y)).mean() for k in w]
    fix = [(pred[i, np.minimum(wm, k) - 1] == y).mean() for k in w]
    ax.plot(w, cum_any, "-", color=C["oracle"], lw=1.2, label="correct at some prefix ≤ t\n(hindsight bound)")
    ax.plot(w, fix, "-o", ms=2, color=C["fixed"], lw=1.0, label="fixed length t")
    ax.plot(w, cum_tt, "-", color=C["tt"], lw=1.2, label="TT: stopped correctly by t")
    ax.plot(w, cum_dqn, "-", color=C["dqn"], lw=1.2, label="DQN: stopped correctly by t")
    ax.set_xticks(w)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("Fraction of all episodes")
    ax.set_title("Correct decisions accumulate early", loc="left")
    ax.legend(loc="lower right", fontsize=5.6, handlelength=1.4)
    panel(ax, "c", x=-0.24)
    save(fig, "fig3_data_recognizability")


# --------------------------------------------------------------------------- Fig 4
RUNS = [("v2b_ce_sqrt", "Seed A (principal)"), ("v3_cs1", "Seed B"), ("v3_cs2", "Seed C"),
        ("v3_p1", "Partition 1"), ("v3_p2", "Partition 2")]


def fig4():
    """Operating points (top) and paired differences across five repeated runs (bottom)."""
    fig = plt.figure(figsize=(FULL, 150 * MM))
    top = fig.add_gridspec(1, 3, width_ratios=[1, 1.35, 0.62], wspace=0.28,
                           left=0.065, right=0.99, top=0.955, bottom=0.62)
    bot = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.06,
                           left=0.2, right=0.985, top=0.53, bottom=0.065)
    fr = V3["frontier"]
    clouds = (("fixed", C["fixed"], "fixed length, 1–10 s"),
              ("threshold", C["thr"], "global threshold, all settings"),
              ("time_threshold", "#B7C9E2", "TT, all settings"))
    sel = [("selected Fixed 10s", "Fixed 10 s", C["fixed"], "o"),
           ("selected Threshold", "Global threshold (selected)", C["thr"], "v"),
           ("selected Time threshold", "TT (selected)", C["tt"], "s"),
           ("selected Matched threshold", "TT, time-matched to DQN", C["matched"], "^"),
           ("selected RL", "DQN", C["dqn"], "D")]
    env = np.array(fr["frontier"])

    def draw(ax, ms):
        for key, col, lab in clouds:
            a = np.array(fr[key])
            ax.scatter(a[:, 0], a[:, 1], s=9 if key == "fixed" else 3, color=col, lw=0,
                       label=lab, zorder=3 if key == "fixed" else 2)
        ax.plot(env[:, 0], env[:, 1], color="#555555", lw=0.7, ls=":", zorder=3,
                label="best rule setting at each time")
        for k, lab, col, mk in sel:
            t, f = fr[k]
            ax.plot(t, f, marker=mk, ms=ms, color=col, mec="white", mew=0.7, ls="none",
                    zorder=6, label=lab)

    ax = fig.add_subplot(top[0])
    draw(ax, 4.5)
    ax.add_patch(Rectangle((2.6, 0.515), 6.0, 0.047, fc="none", ec="#777777", lw=0.6, ls="--",
                           zorder=7))
    ax.set_xlim(0.7, 8.8)
    ax.set_ylim(0.35, 0.58)
    ax.set_xlabel("Mean observation time (s)")
    ax.set_ylabel("Macro-F1 (pooled, principal run)")
    ax.set_title("All operating points", loc="left")
    panel(ax, "a", x=-0.27)

    ax = fig.add_subplot(top[1])
    draw(ax, 7)
    ax.set_xlim(2.6, 8.6)
    ax.set_ylim(0.515, 0.562)
    ax.set_xlabel("Mean observation time (s)")
    ax.set_title("Zoom on the dashed box", loc="left")
    t_r, f_r = fr["selected RL"]
    ax.annotate("DQN lies above every rule\nsetting that is at least as fast",
                xy=(t_r, f_r), xytext=(5.0, 0.5585), fontsize=6, color=C["dqn"],
                arrowprops=dict(arrowstyle="-", color=C["dqn"], lw=0.6), va="center")
    t_f, f_f = fr["selected Fixed 10s"]
    t_t, f_t = fr["selected Time threshold"]
    ax.annotate("", xy=(t_t + 0.1, f_t - 0.004), xytext=(t_f - 0.1, f_f - 0.004),
                arrowprops=dict(arrowstyle="-|>", color=C["tt"], lw=0.8, mutation_scale=7))
    ax.text((t_t + t_f) / 2, f_f - 0.0055, f"−{t_f - t_t:.1f} s at equal F1", color=C["tt"],
            fontsize=6, ha="center", va="top")
    panel(ax, "b", x=-0.12)
    h, lab = ax.get_legend_handles_labels()
    lax = fig.add_subplot(top[2])
    lax.axis("off")
    lax.legend(h, lab, loc="center left", fontsize=6, handlelength=1.4, borderaxespad=0,
               labelspacing=0.7)

    comps = [("rl_vs_tt", "DQN − TT", C["dqn"]),
             ("rl_vs_matched", "DQN − TT time-matched to DQN", C["matched"]),
             ("tt_vs_fixed", "TT − Fixed 10 s", C["tt"])]
    rows = [(V3["per_run"][k], lab, False) for k, lab in RUNS]
    rows += [(V3["combined"]["classifier seeds"], "Average, seeds A–C", True),
             (V3["combined"]["patient partitions"], "Average, seed A + partitions 1–2", True)]
    axf = fig.add_subplot(bot[0])
    axt = fig.add_subplot(bot[1], sharey=axf)
    ticks, tl, ypos = [], [], 0.0
    headers = []
    for key, glab, col in comps:
        headers.append((ypos, glab, col))
        ypos += 1.1
        for ri, (res, rlab, avg) in enumerate(rows):
            if avg and not rows[ri - 1][2]:
                ypos += 0.35
            yy = -ypos
            r = res[key]
            for a_, val, ci in ((axf, r["f1"], r["f1_ci"]), (axt, r["time"], r["time_ci"])):
                a_.plot(ci, [yy, yy], color=col, lw=1.7 if avg else 0.9, solid_capstyle="butt")
                a_.plot(val, yy, marker="D" if avg else "o", ms=4.2 if avg else 3.2, color=col,
                        mfc=col if avg else "white", mew=0.9, zorder=5)
            ticks.append(yy)
            tl.append(rlab)
            ypos += 1
        ypos += 0.6
    for yh, glab, col in headers:
        axf.text(-0.0485, -yh, glab, color=col, fontweight="bold", fontsize=6.6, va="center")
    axf.set_yticks(ticks)
    axf.set_yticklabels(tl, fontsize=6)
    axf.set_ylim(-ypos + 0.3, 0.8)
    for a_ in (axf, axt):
        a_.axvline(0, color="#999999", lw=0.6, zorder=0)
    axf.set_xlim(-0.05, 0.04)
    axf.set_xlabel("Δ macro-F1 (95% patient-bootstrap CI)")
    axf.set_title("Paired difference in macro-F1", loc="left")
    panel(axf, "c", x=-0.42)
    axt.tick_params(labelleft=False)
    axt.set_xlim(-5.2, 1.3)
    axt.set_xlabel("Δ mean observation time (s)")
    axt.set_title("Paired difference in time", loc="left")
    panel(axt, "d", x=-0.05)
    save(fig, "fig4_main_results")


# --------------------------------------------------------------------------- Fig 5
def fig5():
    """What the learned trigger does differently from the rule."""
    y, wm, P = D["labels9"], D["wmax"], D["probs"]
    act = D["act"]
    fig = plt.figure(figsize=(FULL, 64 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 0.95], wspace=0.42,
                          left=0.06, right=0.985, top=0.88, bottom=0.27)

    ax = fig.add_subplot(gs[0])
    w = np.arange(1, 11)
    conf = P.max(2)
    edges = np.linspace(0.2, 1.0, 17)
    grid = np.full((len(edges) - 1, 10), np.nan)
    for k in range(9):  # w = 10 is a forced stop
        m = (wm >= k + 1)
        for b in range(len(edges) - 1):
            mm = m & (conf[:, k] >= edges[b]) & (conf[:, k] < edges[b + 1] + (1e-9 if b == len(edges) - 2 else 0))
            if mm.sum() >= 5:
                grid[b, k] = (act[mm, k] == 2).mean()
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                   extent=[0.5, 10.5, edges[0], edges[-1]], interpolation="nearest")
    for f in range(5):
        tw, thr = tt_curve(f)
        ax.step(tw[:9], thr[:9], where="mid", color="black", lw=0.6, alpha=0.7,
                label="TT thresholds (5 folds)" if f == 0 else None)
    ax.add_patch(Rectangle((9.5, edges[0]), 1, edges[-1] - edges[0], fc="#EEEEEE",
                           ec="none", hatch="////", zorder=1.5))
    ax.text(10, 0.6, "forced stop", rotation=90, ha="center", va="center", fontsize=5.8,
            color="#333333", zorder=3,
            bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_ylim(edges[0], edges[-1])
    ax.set_xticks(w)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Top posterior max $p_w$")
    ax.set_title("DQN greedy action on test states", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    ax.legend_ = None
    cb.set_label("P(action = stop)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=5.8,
              handlelength=1.2)
    panel(ax, "a", x=-0.2)

    ax = fig.add_subplot(gs[1])
    i = np.arange(len(y))
    ctt = P[i, D["stop_tt"] - 1].argmax(1) == y
    cdq = P[i, D["stop_dqn"] - 1].argmax(1) == y
    order = [7, 1, 0, 5, 6, 4, 3, 2, 8]
    for r, c in enumerate(order):
        m = y == c
        a, b = D["stop_tt"][m].mean(), D["stop_dqn"][m].mean()
        ax.plot([a, b], [r, r], color="#BBBBBB", lw=1.2, zorder=1)
        ax.plot(a, r, "s", ms=4, color=C["tt"], zorder=2, label="TT" if r == 0 else None)
        ax.plot(b, r, "D", ms=4, color=C["dqn"], zorder=2, label="DQN" if r == 0 else None)
        ax.text(8.9, r, f"{ctt[m].mean():.2f} → {cdq[m].mean():.2f}", va="center",
                fontsize=5.8)
    ax.text(8.9, -0.75, "accuracy\nTT → DQN", fontsize=5.8, va="bottom", color="#555555")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[c] for c in order])
    ax.set_ylim(-1.6, len(order) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(1, 11)
    ax.set_xticks([2, 4, 6, 8])
    ax.set_xlabel("Mean stop time (s)")
    ax.set_title("Where the extra waiting goes", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, fontsize=6,
              handlelength=1)
    panel(ax, "b", x=-0.22)

    ax = fig.add_subplot(gs[2])
    st, sd = D["stop_tt"], D["stop_dqn"]
    cats = [("same time", st == sd), ("DQN later", sd > st), ("DQN earlier", sd < st)]
    xs = np.arange(len(cats))
    vals = []
    for name, m in cats:
        vals.append([(ctt & cdq & m).sum(), (~ctt & cdq & m).sum(), (ctt & ~cdq & m).sum(),
                     (~ctt & ~cdq & m).sum()])
    vals = np.array(vals)
    parts = [("both correct", "#D9D9D9"), ("only DQN correct", C["dqn"]),
             ("only TT correct", C["tt"]), ("both wrong", "#6E6E6E")]
    bottom = np.zeros(len(cats))
    for j, (lab, col) in enumerate(parts):
        ax.bar(xs, vals[:, j], bottom=bottom, color=col, width=0.62, label=lab, lw=0)
        bottom += vals[:, j]
    for k in range(len(cats)):
        tag = ("same output" if vals[k, 1] + vals[k, 2] == 0
               else f"DQN +{vals[k, 1]}\n/ −{vals[k, 2]}")
        ax.text(xs[k], bottom[k] + 40, f"n={int(bottom[k])}\n{tag}",
                ha="center", fontsize=5.8, va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in cats])
    ax.set_ylabel("Test episodes")
    ax.set_ylim(0, bottom.max() * 1.2)
    ax.set_title("Paired outcome, TT vs DQN", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2, fontsize=5.8,
              handlelength=1, columnspacing=0.8)
    panel(ax, "c", x=-0.3)
    save(fig, "fig5_policy_behaviour")


# --------------------------------------------------------------------------- Supplement
def figS1():
    rel = V3["reliability"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL * 0.62, 55 * MM))
    for ax, key, title in ((axes[0], "raw", "Before temperature scaling"),
                           (axes[1], "calibrated", "After temperature scaling")):
        a = np.array(rel[key])
        a = a[a[:, 2] >= 0.001]  # drop near-empty bins (<0.1% of prefixes)
        ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=0.6, ls="--")
        ax.bar(a[:, 0], a[:, 1], width=0.07, color=C["tt"], alpha=0.25, lw=0)
        ax.plot(a[:, 0], a[:, 1], "-o", ms=2.5, color=C["tt"], lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Mean top-posterior in bin")
        ax.set_title(f"{title}\nECE {rel['raw_ece' if key == 'raw' else 'cal_ece']:.3f}",
                     loc="left")
    axes[0].set_ylabel("Accuracy in bin")
    fig.tight_layout()
    save(fig, "figS1_reliability")


def figS2():
    y, P = D["labels9"], D["probs"]
    i = np.arange(len(y))
    wm = D["wmax"]
    preds = [("Fixed 10 s", P[i, np.minimum(wm, 10) - 1].argmax(1)),
             ("TT", P[i, D["stop_tt"] - 1].argmax(1)),
             ("DQN", P[i, D["stop_dqn"] - 1].argmax(1))]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 64 * MM))
    for ax, (name, p) in zip(axes, preds):
        cm = confusion_matrix(y, p, labels=range(9)).astype(float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for a in range(9):
            for b in range(9):
                if cm[a, b] > 0:
                    ax.text(b, a, f"{cmn[a, b]:.2f}".lstrip("0") if cmn[a, b] < 1 else "1",
                            ha="center", va="center", fontsize=4.6,
                            color="white" if cmn[a, b] > 0.55 else "#333333")
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_xticklabels(SHORT, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticklabels(SHORT, fontsize=5.5)
        ax.set_title(name, loc="left")
        ax.set_xlabel("Predicted")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("True")
    fig.tight_layout()
    save(fig, "figS2_confusion")


FIGS = dict(fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, fig5=fig5, figS1=figS1, figS2=figS2)

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
