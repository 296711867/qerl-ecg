# -*- coding: utf-8 -*-
"""阶段2a + 阶段3：全部策略在验证/测试轨迹上的对比评估。

策略:
  - 固定窗口 w ∈ {1,2,3,5,10}
  - 置信度阈值 τ ∈ [0.4, 0.99]（val 上按 mean_return 选 τ*, test 上报告）
  - RL: dqn / double / dueling / dddqn checkpoints（若存在）

输出:
  outputs/tables/policy_comparison.md   主结果表（论文 Table 3）
  outputs/figures/pareto.png            Macro-F1 vs 平均决策时间 Pareto 图（论文 Figure 4）
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policies import evaluate_sim, simulate_fixed, simulate_threshold
from src.rl.agent import DQNAgent
from src.rl.env import ABSTAIN, STOP, VecEarlyEnv

CKPT_DIR = ROOT / "outputs/checkpoints"


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


def load_traj(part):
    files = sorted((ROOT / "data/processed").glob(f"traj_{part}_f0_*.npz"))
    if not files:
        raise FileNotFoundError(f"no trajectory for {part}")
    return dict(np.load(files[-1]))


def main():
    va, te = load_traj("val"), load_traj("test")

    # ===== 1. 阈值 τ 扫描（val）=====
    sweep = []
    for tau in [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]:
        m = evaluate_sim(va, simulate_threshold(va, tau))
        sweep.append(dict(tau=tau, **m))
        print(f"[val] tau={tau:.2f} F1={m['macro_f1']:.4f} dt={m['mean_decision_time']:.2f} "
              f"ret={m['mean_return']:.3f}", flush=True)
    best_tau = max(sweep, key=lambda s: s["mean_return"])
    print(f"best tau (by return): {best_tau['tau']}")

    # ===== 2. 各策略 test 评估 =====
    results = []
    for w in [1, 2, 3, 5, 10]:
        m = evaluate_sim(te, simulate_fixed(te, w))
        results.append(dict(policy=f"Fixed {w}s", **m))
    m = evaluate_sim(te, simulate_threshold(te, best_tau["tau"]))
    results.append(dict(policy=f"Threshold τ={best_tau['tau']:.2f}", **m))

    for algo in ["dqn", "double", "dueling", "dddqn"]:
        ck = CKPT_DIR / f"rl_{algo}_f0.pt"
        if not ck.exists():
            continue
        agent = DQNAgent(state_dim=16,
                         double=algo in ("double", "dddqn"),
                         dueling=algo in ("dueling", "dddqn"))
        agent.q.load_state_dict(torch.load(ck, map_location="cpu",
                                           weights_only=False)["model"])
        agent.q.eval()
        m = evaluate_sim(te, simulate_agent(te, agent))
        results.append(dict(policy=f"RL-{algo.upper()}", **m))

    # ===== 3. 输出主表 + Pareto 图 =====
    lines = ["# 策略对比（test, fold 0）\n",
             "| policy | macro-F1 | bal-acc | acc | dt(s) | earliness | coverage | "
             "sel-risk | ret | badq-alarm |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['policy']} | {r['macro_f1']:.4f} | {r['balanced_acc']:.4f} | "
            f"{r['accuracy']:.4f} | {r['mean_decision_time']:.2f} | {r['earliness']:.3f} | "
            f"{r['coverage']:.3f} | {r['selective_risk']:.3f} | {r['mean_return']:.3f} | "
            f"{r.get('badq_severe_alarm_rate', float('nan')):.3f} |")
    table = "\n".join(lines)
    (ROOT / "outputs/tables/policy_comparison.md").write_text(table, encoding="utf-8")
    print("\n" + table)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for r in results:
        style = "o" if r["policy"].startswith("Fixed") else ("s" if "Thresh" in r["policy"] else "*")
        size = 60 if style != "*" else 160
        ax.scatter(r["mean_decision_time"], r["macro_f1"], marker=style, s=size)
        ax.annotate(r["policy"], (r["mean_decision_time"], r["macro_f1"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="Mean decision time (s)", ylabel="Macro-F1",
           title="Accuracy–Latency trade-off (test)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/figures/pareto.png", dpi=150)

    (ROOT / "outputs/tables/policy_comparison.json").write_text(json.dumps(
        dict(threshold_sweep=sweep, best_tau=best_tau["tau"], results=results), indent=1))
    print("\nsaved: policy_comparison.md / pareto.png")


if __name__ == "__main__":
    import torch  # noqa: E402
    main()
