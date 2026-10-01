# -*- coding: utf-8 -*-
"""阶段3：消融实验评估（test, fold 0）。

组别:
  Full      = rl_dddqn_f0（质量特征 + 弃权 + λ=0.03）
  w/o Qual  = abl_noquality_f0
  w/o Abst  = abl_noabstain_f0
  λ 扫描    = rl_lambda{0.00,0.01,0.05,0.10}_f0 + Full(λ=0.03)

输出:
  outputs/tables/ablation.md           消融表（论文 Table 6）
  outputs/figures/lambda_pareto.png    λ-F1-决策时间 Pareto（论文 Figure 5）
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policies import evaluate_sim
from src.rl.agent import DQNAgent
from src.rl.env import STOP, VecEarlyEnv

CKPT = ROOT / "outputs/checkpoints"


def load_traj(part):
    f = sorted((ROOT / "data/processed").glob(f"traj_{part}_f0_*.npz"))[-1]
    return dict(np.load(f))


def simulate(traj, agent, env_kw, batch=4096):
    env = VecEarlyEnv(traj, reveal_wmax=True, **env_kw)
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
        abstain[ii] = env.terminal == 3
    return dict(stop_w=stop_w, abstain=abstain)


def eval_ckpt(te, tag, env_kw):
    ck = CKPT / f"{tag}.pt"
    if not ck.exists():
        return None
    agent = DQNAgent(state_dim=16, double=True, dueling=True)
    agent.q.load_state_dict(torch.load(ck, map_location="cpu",
                                       weights_only=False)["model"])
    agent.q.eval()
    return evaluate_sim(te, simulate(te, agent, env_kw))


SFX = f"_{sys.argv[1]}" if len(sys.argv) > 1 else ""


def main():
    te = load_traj("test")
    base_reward = dict(correct=1.0, wrong=-2.0, wait_per_second=-0.03,
                       appropriate_abstain=0.3, unnecessary_abstain=-0.4)

    groups = [
        ("Full (QERL)", f"rl_dddqn_f0{SFX}", dict(use_quality=True, allow_abstain=True, reward=base_reward)),
        ("w/o Quality state", f"abl_noquality_f0{SFX}", dict(use_quality=False, allow_abstain=True, reward=base_reward)),
        ("w/o Abstention", f"abl_noabstain_f0{SFX}", dict(use_quality=True, allow_abstain=False, reward=base_reward)),
    ]
    rows = []
    for name, tag, kw in groups:
        m = eval_ckpt(te, tag, kw)
        if m:
            rows.append((name, m))
            print(f"{name:20s} F1={m['macro_f1']:.4f} dt={m['mean_decision_time']:.2f} "
                  f"cov={m['coverage']:.3f} badq-alarm={m.get('badq_severe_alarm_rate', float('nan')):.3f}")

    # λ 扫描（评估时用默认 reward 计指标；策略行为由训练时的 λ 决定）
    lam_rows = [(0.03, rows[0][1])]
    for lam in [0.00, 0.01, 0.05, 0.10]:
        m = eval_ckpt(te, f"rl_lambda{lam:.2f}_f0{SFX}",
                      dict(use_quality=True, allow_abstain=True, reward=base_reward))
        if m:
            lam_rows.append((lam, m))

    # 输出消融表
    lines = ["# 消融实验（test, fold 0）\n",
             "| variant | macro-F1 | bal-acc | dt(s) | coverage | sel-risk | badq-alarm | ret |",
             "|---|---|---|---|---|---|---|---|"]
    for name, m in rows:
        lines.append(f"| {name} | {m['macro_f1']:.4f} | {m['balanced_acc']:.4f} | "
                     f"{m['mean_decision_time']:.2f} | {m['coverage']:.3f} | "
                     f"{m['selective_risk']:.3f} | {m.get('badq_severe_alarm_rate', float('nan')):.3f} | "
                     f"{m['mean_return']:.3f} |")
    lines.append("\n## 等待成本 λ 扫描\n")
    lines.append("| λ | macro-F1 | dt(s) | earliness |")
    lines.append("|---|---|---|---|")
    for lam, m in sorted(lam_rows):
        lines.append(f"| {lam:.2f} | {m['macro_f1']:.4f} | {m['mean_decision_time']:.2f} | "
                     f"{m['earliness']:.3f} |")
    (ROOT / f"outputs/tables/ablation{SFX}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))

    # λ Pareto 图
    fig, ax = plt.subplots(figsize=(6, 4.2))
    lams = [l for l, _ in lam_rows]
    f1s = [m["macro_f1"] for _, m in lam_rows]
    dts = [m["mean_decision_time"] for _, m in lam_rows]
    ax.plot(dts, f1s, "o-", alpha=0.55)
    # 逐 λ 标签锚点：避开相邻折线段；与 scripts/replot_paretos.mjs 保持一致。
    LAM_LAYOUT = {
        0.00: dict(dx=0, dy=-14, ha="center"),
        0.01: dict(dx=9, dy=-18, ha="left"),
        0.03: dict(dx=-9, dy=10, ha="right"),
        0.05: dict(dx=0, dy=-14, ha="center"),
        0.10: dict(dx=9, dy=8, ha="left"),
    }
    for lam, m in sorted(lam_rows):
        lay = LAM_LAYOUT.get(round(lam, 2), dict(dx=8, dy=-8, ha="left"))
        ax.annotate(f"λ={lam:.2f}", (m["mean_decision_time"], m["macro_f1"]),
                    fontsize=8, xytext=(lay["dx"], lay["dy"]), textcoords="offset points",
                    ha=lay["ha"], va="top" if lay["dy"] < 0 else "bottom", zorder=5,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
    ax.set(xlabel="Mean decision time (s)", ylabel="Macro-F1",
           title="Reward wait-cost λ: accuracy–latency frontier (test)")
    ax.margins(x=0.09, y=0.12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / f"outputs/figures/lambda_pareto{SFX}.png", dpi=300)

    (ROOT / f"outputs/tables/ablation{SFX}.json").write_text(json.dumps(
        dict(ablation=[dict(name=n, **m) for n, m in rows],
             lambda_sweep=[dict(lam=l, **m) for l, m in lam_rows]), indent=1))
    print("\nsaved: ablation.md / lambda_pareto.png")


if __name__ == "__main__":
    main()
