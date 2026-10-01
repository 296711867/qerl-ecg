# -*- coding: utf-8 -*-
"""5 折 CV 汇总评估：mean±std 主结果表 + 带误差带的图。

产出:
  outputs/tables/cv_summary.md      Table 2 / Table 3 的 5 折 mean±std
  outputs/figures/window_curve_cv.png
  outputs/figures/pareto_cv.png
  outputs/tables/cv_summary.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.dataset import EcgWindowDataset, load_split
from src.models.tcn_gru import TCNGRU
from src.policies import evaluate_sim, simulate_fixed, simulate_threshold
from src.rl.agent import DQNAgent
from src.rl.env import STOP, ABSTAIN, VecEarlyEnv
from src.train_classifier import evaluate as eval_supervised

FOLDS = [0, 1, 2, 3, 4]
TAUS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
# 可选命令行后缀（如 "os"）以区分实验版本；默认 "" 为原始 CV
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
SFX = f"_{SUFFIX}" if SUFFIX else ""


def load_traj(part, fold):
    f = (ROOT / "data/processed"
         / f"traj_{part}_f{fold}_9class_w10_f{fold}{SFX}.npz")
    if not f.exists():
        fs = sorted((ROOT / "data/processed").glob(f"traj_{part}_f{fold}_*.npz"))
        if not fs:
            return None
        return dict(np.load(fs[-1]))
    return dict(np.load(f))


def simulate_agent(traj, agent, batch=4096):
    env = VecEarlyEnv(traj, reveal_wmax=True)
    idx = np.arange(len(traj["labels9"]))
    stop_w = np.zeros(len(idx), dtype=np.int64)
    abstain = np.zeros(len(idx), dtype=bool)
    for s0 in range(0, len(idx), batch):
        ii = idx[s0:s0 + batch]
        s, mask = env.reset(ii)
        while not env.done.all():
            a = agent.act(s, mask, greedy=True)
            a[env.done] = STOP
            s, r, newly, mask = env.step(a)
        stop_w[ii] = env.w
        abstain[ii] = env.terminal == ABSTAIN
    return dict(stop_w=stop_w, abstain=abstain)


def eval_ckpt_test(fold, w):
    """单独训练的 w 秒分类器在该折 test 上的指标（Table 2 用）。"""
    ck = ROOT / f"outputs/checkpoints/9class_w{w}_f{fold}{SFX}.pt"
    if not ck.exists():
        return None
    model = TCNGRU(n_class=9).cuda()
    model.load_state_dict(torch.load(ck, map_location="cuda",
                                     weights_only=False)["model"])
    ds = EcgWindowDataset(load_split(fold, "test"), w, "9class", drop_badq=True)
    m, _, _ = eval_supervised(model, DataLoader(ds, batch_size=512), "cuda", 9)
    del model
    torch.cuda.empty_cache()
    return m


def ms(vals):
    a = np.array(vals, dtype=float)
    return f"{a.mean():.4f}±{a.std(ddof=1):.4f}" if len(a) > 1 else f"{a.mean():.4f}"


def main():
    table2 = {}   # (w) -> list of metrics dicts
    table3 = {}   # policy -> list of metrics dicts
    taus_used = []

    for fold in FOLDS:
        te = load_traj("test", fold)
        va = load_traj("val", fold)
        if te is None or va is None:
            print(f"[fold {fold}] trajectories missing, skip RL part")
            continue

        # --- Table 3: 固定窗（来自轨迹）---
        for w in [1, 2, 3, 5, 10]:
            m = evaluate_sim(te, simulate_fixed(te, w))
            table3.setdefault(f"Fixed {w}s", []).append(m)
        # 阈值：val 选 τ（按 return），test 报告
        best_tau, best_ret = None, -1e9
        for tau in TAUS:
            r = evaluate_sim(va, simulate_threshold(va, tau))["mean_return"]
            if r > best_ret:
                best_tau, best_ret = tau, r
        taus_used.append(best_tau)
        table3.setdefault(f"Threshold", []).append(
            evaluate_sim(te, simulate_threshold(te, best_tau)))
        # RL-DDDQN
        ck = ROOT / f"outputs/checkpoints/rl_dddqn_f{fold}{SFX}.pt"
        if ck.exists():
            agent = DQNAgent(state_dim=16, double=True, dueling=True)
            agent.q.load_state_dict(torch.load(ck, map_location="cpu",
                                               weights_only=False)["model"])
            agent.q.eval()
            table3.setdefault("RL-DDDQN (ours)", []).append(
                evaluate_sim(te, simulate_agent(te, agent)))
        print(f"[fold {fold}] tau*={best_tau} table3 done", flush=True)

        # --- Table 2: 各窗口独立模型 ---
        for w in [1, 2, 3, 5, 10]:
            m = eval_ckpt_test(fold, w)
            if m:
                table2.setdefault(w, []).append(m)
        print(f"[fold {fold}] table2 done", flush=True)

    # ===== 汇总输出 =====
    keys3 = ["macro_f1", "balanced_acc", "accuracy", "mean_decision_time",
             "earliness", "coverage", "selective_risk", "mean_return"]
    lines = ["# 5 折 CV 汇总（test, mean±std）\n",
             f"各折 τ*: {taus_used}\n",
             "## Table 3: 自适应早期识别策略对比\n",
             "| policy | macro-F1 | bal-acc | acc | dt(s) | earliness | coverage | sel-risk | ret |",
             "|---|---|---|---|---|---|---|---|---|"]
    for pol, ms_ in table3.items():
        row = [pol]
        for k in keys3:
            row.append(ms([m[k] for m in ms_]))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["\n## Table 2: 固定窗口监督分类\n",
              "| window(s) | macro-F1 | balanced acc | accuracy | folds |",
              "|---|---|---|---|---|"]
    for w in sorted(table2):
        ms_ = table2[w]
        lines.append(f"| {w} | {ms([m['macro_f1'] for m in ms_])} | "
                     f"{ms([m['balanced_acc'] for m in ms_])} | "
                     f"{ms([m['accuracy'] for m in ms_])} | {len(ms_)} |")

    text = "\n".join(lines)
    (ROOT / f"outputs/tables/cv_summary{SFX}.md").write_text(text, encoding="utf-8")
    print("\n" + text)

    # ===== 图 =====
    fig, ax = plt.subplots(figsize=(6, 4))
    ws = sorted(table2)
    f1m = [np.mean([m["macro_f1"] for m in table2[w]]) for w in ws]
    f1s = [np.std([m["macro_f1"] for m in table2[w]], ddof=1) for w in ws]
    ax.errorbar(ws, f1m, yerr=f1s, marker="o", capsize=3, label="Macro-F1")
    bam = [np.mean([m["balanced_acc"] for m in table2[w]]) for w in ws]
    bas = [np.std([m["balanced_acc"] for m in table2[w]], ddof=1) for w in ws]
    ax.errorbar(ws, bam, yerr=bas, marker="s", capsize=3, label="Balanced Acc")
    ax.set(xlabel="Observation window (s)", ylabel="Score",
           title="Fixed-window supervised performance (5-fold CV)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / f"outputs/figures/window_curve_cv{SFX}.png", dpi=150)

    if table3:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        # 逐策略标签锚点：中部聚簇区（Threshold/Fixed5/RL）互不压点、互不压线。
        # 与 scripts/replot_paretos.mjs 保持一致；改这里时同步改那份。
        LAYOUT = {
            "Fixed 1s":  dict(dx=0, dy=-13, ha="center"),
            "Fixed 2s":  dict(dx=0, dy=-13, ha="center"),
            "Fixed 3s":  dict(dx=0, dy=-13, ha="center"),
            "Fixed 5s":  dict(dx=12, dy=-4, ha="left"),
            "Fixed 10s": dict(dx=-10, dy=-20, ha="right"),
            "Threshold": dict(dx=0, dy=-15, ha="center"),
            "RL-DDDQN (ours)": dict(dx=-8, dy=9, ha="right"),
        }
        for pol, ms_ in table3.items():
            xs = [m["mean_decision_time"] for m in ms_]
            ys = [m["macro_f1"] for m in ms_]
            ax.scatter(np.mean(xs), np.mean(ys),
                       marker="*" if "RL" in pol else ("s" if "Thresh" in pol else "o"),
                       s=200 if "RL" in pol else 60)
            lay = LAYOUT.get(pol, dict(dx=8, dy=-8, ha="left"))
            ax.annotate(f"{pol}\n(mean±{np.std(ys, ddof=1):.3f})" if len(ys) > 1 else pol,
                        (np.mean(xs), np.mean(ys)), fontsize=7,
                        xytext=(lay["dx"], lay["dy"]), textcoords="offset points",
                        ha=lay["ha"], va="center" if lay["dy"] == 0 else
                        ("top" if lay["dy"] < 0 else "bottom"), zorder=5,
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
        ax.set(xlabel="Mean decision time (s)", ylabel="Macro-F1",
               title="Policy comparison (5-fold CV)")
        ax.margins(x=0.06, y=0.09)
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(ROOT / f"outputs/figures/pareto_cv{SFX}.png", dpi=300)

    (ROOT / f"outputs/tables/cv_summary{SFX}.json").write_text(json.dumps(
        dict(taus=taus_used,
             table3={k: [{kk: m[kk] for kk in keys3} for m in v]
                     for k, v in table3.items()},
             table2={str(k): [{kk: m[kk] for kk in ("macro_f1", "balanced_acc", "accuracy")}
                              for m in v] for k, v in table2.items()}), indent=1))
    print("\nsaved: cv_summary{SUFFIX} files")


if __name__ == "__main__":
    main()
