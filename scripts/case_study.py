# -*- coding: utf-8 -*-
"""案例研究图（论文 Figure 6）。

挑选两类典型测试 episode:
  A. 坏质量段上的"适当弃权"（agent 遇噪声 ABSTAIN）
  B. 不确定时等待、置信后停止的正确决策（展示 WAIT→STOP 行为）

每例 3 联图: ECG 波形(10s) / 前3类概率演化 / 质量分数 + 决策序列
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import CACHE, CLASSES
from src.rl.agent import DQNAgent
from src.rl.env import ABSTAIN, STOP, T_MAX, VecEarlyEnv

FS = 100
ACTION_NAMES = {0: "WAIT+1", 1: "WAIT+2", 2: "STOP", 3: "ABSTAIN"}


def load_traj():
    f = sorted((ROOT / "data/processed").glob("traj_test_f0_*.npz"))[-1]
    return dict(np.load(f))


def run_agent(traj, agent, idx):
    env = VecEarlyEnv(traj, reveal_wmax=True)
    s, mask = env.reset(np.array([idx]))
    actions = []
    while not env.done.all():
        a = agent.act(s, mask, greedy=True)
        actions.append((env.w[0], int(a[0])))
        s, r, newly, mask = env.step(a)
    return actions, env.w[0], env.terminal[0]


def plot_case(traj, idx, actions, fig_path, title):
    case_id, t_end = int(traj["case_id"][idx]), float(traj["t_end"][idx])
    label = CLASSES[traj["labels9"][idx]]
    ecg, t0 = CACHE.get(case_id)
    i1 = int(round((t_end - t0) * FS))
    i0 = max(0, i1 - T_MAX * FS)
    x = ecg[i0:i1]
    tt = np.arange(len(x)) / FS + (t_end - len(x) / FS)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1.2, 1.2]))
    axes[0].plot(tt, x, lw=0.7, color="k")
    axes[0].set(ylabel="ECG (II)", title=f"{title}  —  case {case_id}, true: {label}")
    for w, a in actions:
        axes[0].axvline(t_end - T_MAX + w, color="tab:red", ls=":", lw=0.8, alpha=0.6)

    steps = np.arange(1, T_MAX + 1)
    probs = traj["probs"][idx]  # (10, 9)
    top = probs[-1].argsort()[::-1][:3]
    for c in top:
        axes[1].plot(steps, probs[:T_MAX, c], marker=".", label=CLASSES[c])
    axes[1].axhline(0.5, color="gray", ls="--", lw=0.6)
    axes[1].set(ylabel="P(class)", ylim=(0, 1))
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].set_xlim(1, T_MAX)

    axes[2].plot(steps, traj["quality"][idx][:T_MAX], marker=".", color="tab:purple",
                 label="quality score")
    for w, a in actions:
        if w <= T_MAX:
            axes[2].annotate(ACTION_NAMES[a], (w, traj["quality"][idx][w - 1]),
                             fontsize=7, xytext=(4, 6), textcoords="offset points",
                             color="tab:red")
    axes[2].set(xlabel="Observation time (s)", ylabel="quality", ylim=(-0.05, 1.05),
                xlim=(1, T_MAX))
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"saved {fig_path}  (actions: {[(w, ACTION_NAMES[a]) for w, a in actions]})")


def main():
    traj = load_traj()
    agent = DQNAgent(state_dim=16, double=True, dueling=True)
    ck = ROOT / "outputs/checkpoints/rl_dddqn_f0_os.pt"
    agent.q.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
    agent.q.eval()

    # 逐 episode 跑 agent，找典型案例
    env = VecEarlyEnv(traj, reveal_wmax=True)
    idx_all = np.arange(len(traj["labels9"]))
    term, wfin = np.zeros(len(idx_all), int), np.zeros(len(idx_all), int)
    for s0 in range(0, len(idx_all), 4096):
        ii = idx_all[s0:s0 + 4096]
        s, mask = env.reset(ii)
        while not env.done.all():
            a = agent.act(s, mask, greedy=True)
            a[env.done] = STOP
            s, r, newly, mask = env.step(a)
        term[ii] = env.terminal
        wfin[ii] = env.w

    # A: 弃权案例（取弃权 episode 中标注坏质量重叠最高的一例）
    cand = np.flatnonzero(term == ABSTAIN)
    if len(cand):
        idx = int(cand[np.argmax(traj["qtrue"][cand])])
        acts, w, t = run_agent(traj, agent, idx)
        plot_case(traj, idx, acts, ROOT / "outputs/figures/case_abstain.png",
                  "(A) Appropriate abstention on bad-quality segment")

    # B: 等待后正确停止（决策时间>=4s 且正确，优先危重类）
    correct = []
    for i in np.flatnonzero((wfin >= 4) & (term == STOP)):
        if traj["probs"][i, wfin[i] - 1].argmax() == traj["labels9"][i]:
            correct.append(i)
    severe_idx = [i for i in correct if traj["severe"][i] == 1]
    pick = severe_idx[len(severe_idx) // 2] if severe_idx else (correct[len(correct) // 2] if correct else None)
    if pick is not None:
        acts, w, t = run_agent(traj, agent, int(pick))
        plot_case(traj, int(pick), acts, ROOT / "outputs/figures/case_wait_stop.png",
                  "(B) Wait-then-stop: confidence grows with observation")


if __name__ == "__main__":
    main()
