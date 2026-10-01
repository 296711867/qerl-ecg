# -*- coding: utf-8 -*-
"""Evaluate RL seed stability and the quality-state ablation on v2b events."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_event_cv import REWARD, cluster_ci, mean_sd
from src.policies import decisions_from_sim, evaluate_sim
from src.rl.agent import DQNAgent
from src.rl.env import STATE_DIM
from src.train_rl import simulate_agent

VARIANT = "v2b_ce_sqrt"
CAUSAL = len(sys.argv) > 1 and sys.argv[1] == "causal"
RESULT_TAG = f"{VARIANT}_causal" if CAUSAL else VARIANT


def score(fold, tag, no_quality=False):
    traj = dict(np.load(ROOT / f"data/processed/event_traj_test_f{fold}_{VARIANT}.npz"))
    agent = DQNAgent(state_dim=STATE_DIM if CAUSAL else 16,
                     double=True, dueling=True)
    ckpt = torch.load(ROOT / f"outputs/checkpoints/{tag}.pt",
                      map_location="cpu", weights_only=False)
    agent.q.load_state_dict(ckpt["model"])
    agent.q.eval()
    sim = simulate_agent(traj, agent,
                         env_kw=dict(use_quality=not no_quality,
                                     reveal_wmax=not CAUSAL))
    return evaluate_sim(traj, sim, r=REWARD), dict(
        y=traj["labels9"], case=traj["case_id"], pred=decisions_from_sim(traj, sim))


def main():
    rows = {}
    records = {"full": [], "noquality": []}
    for seed in (42, 43, 44):
        name = f"seed {seed}"
        metrics = []
        for fold in range(5):
            tag = f"event_rl_f{fold}_{RESULT_TAG}" + (f"_s{seed}" if seed != 42 else "")
            m, r = score(fold, tag)
            metrics.append(m)
            if seed == 42:
                records["full"].append(r)
        rows[name] = metrics
        print(name, [round(m["macro_f1_all"], 3) for m in metrics], flush=True)
    noquality = []
    for fold in range(5):
        m, r = score(fold, f"event_rl_noquality_f{fold}_{RESULT_TAG}", no_quality=True)
        noquality.append(m)
        records["noquality"].append(r)
    rows["w/o quality (seed 42)"] = noquality
    paired = [dict(y=a["y"], case=a["case"], full=a["pred"],
                   noquality=b["pred"])
              for a, b in zip(records["full"], records["noquality"])]
    delta, ci = cluster_ci(paired, "full", comparator="noquality")
    lines = ["# v2b：随机种子与质量特征消融", "",
             "所有模型按验证集 coverage≥0.90 选 checkpoint，测试集不参与选择。", "",
             "| 模型 | 全体 macro-F1 | 观察秒数 | coverage | 坏质量真误报率 |",
             "|---|---:|---:|---:|---:|"]
    for name, metrics in rows.items():
        lines.append("| " + name + " | " + " | ".join(
            mean_sd([m[k] for m in metrics]) for k in
            ("macro_f1_all", "mean_decision_time", "coverage", "badq_false_alarm_rate")) + " |")
    lines += ["", f"质量特征消融的患者聚类配对差（full − w/o quality，"
              f"全体 macro-F1）：{delta:+.4f}，95% 区间 "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]。", ""]
    out = ROOT / f"outputs/tables/event_robustness_{RESULT_TAG}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / f"outputs/tables/event_robustness_{RESULT_TAG}.json").write_text(
        json.dumps(dict(groups=rows, quality_difference=dict(point=delta, ci95=ci)),
                   indent=1), encoding="utf-8")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
