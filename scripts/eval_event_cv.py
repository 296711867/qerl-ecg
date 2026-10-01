# -*- coding: utf-8 -*-
"""Five-fold evaluation of forward event-onset detection policies."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policies import (decisions_from_sim, evaluate_sim, simulate_fixed,
                          simulate_threshold, simulate_time_threshold)
from src.rl.agent import DQNAgent
from src.rl.env import STATE_DIM
from src.train_rl import simulate_agent
from src.data.event_dataset import CLASSES

TAUS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
POLICIES = ["Fixed 1s", "Fixed 3s", "Fixed 5s", "Fixed 10s",
            "Threshold", "Time threshold", "RL"]
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "v2"
CAUSAL = len(sys.argv) > 2 and sys.argv[2] == "causal"
RESULT_TAG = f"{VARIANT}_causal" if CAUSAL else VARIANT
REWARD = dict(correct=1.0, wrong=-2.0, wait_per_second=-0.03,
              appropriate_abstain=0.3, unnecessary_abstain=-1.0)


def load(fold, part):
    return dict(np.load(ROOT / f"data/processed/event_traj_{part}_f{fold}_{VARIANT}.npz"))


def eval_fold(fold):
    va, te = load(fold, "val"), load(fold, "test")
    sims = {f"Fixed {w}s": simulate_fixed(te, w) for w in (1, 3, 5, 10)}
    tau = max(TAUS, key=lambda x: evaluate_sim(
        va, simulate_threshold(va, x), r=REWARD)["mean_return"])
    sims["Threshold"] = simulate_threshold(te, tau)
    candidates = [(a, round(float(b), 2)) for a in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
                  for b in np.arange(-1.5, 0.21, 0.1)]
    tau_time = max(candidates, key=lambda ab: evaluate_sim(
        va, simulate_time_threshold(va, *ab), r=REWARD)["mean_return"])
    sims["Time threshold"] = simulate_time_threshold(te, *tau_time)
    ckpt = ROOT / f"outputs/checkpoints/event_rl_f{fold}_{RESULT_TAG}.pt"
    if ckpt.exists():
        agent = DQNAgent(state_dim=STATE_DIM if CAUSAL else 16,
                         double=True, dueling=True)
        agent.q.load_state_dict(torch.load(ckpt, map_location="cpu",
                                           weights_only=False)["model"])
        agent.q.eval()
        sims["RL"] = simulate_agent(te, agent,
                                    env_kw=dict(reveal_wmax=not CAUSAL))
    metrics = {name: evaluate_sim(te, sim, r=REWARD) for name, sim in sims.items()}
    q = te["quality"][np.arange(len(te["wmax"])), te["wmax"] - 1]
    bad = te["qtrue"] > 0.5
    quality_auc = float(roc_auc_score(bad, q)) if bad.any() and (~bad).any() else float("nan")
    badq_severe_n = int((bad & np.isin(te["labels9"], [2, 6, 7])).sum())
    return dict(fold=fold, tau=tau, tau_time=tau_time, n=len(te["wmax"]), quality_auc=quality_auc,
                badq_severe_n=badq_severe_n,
                metrics=metrics), te, sims


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return "NA"
    if len(a) == 1:
        return f"{a[0]:.4f} (1 fold)"
    return f"{a.mean():.4f}±{a.std(ddof=1):.4f}"


def cluster_ci(records, policy, comparator="Fixed 10s", reps=1000, seed=42):
    """Patient bootstrap over out-of-fold predictions, abstention as omission."""
    y = np.concatenate([r["y"] for r in records])
    case = np.concatenate([r["case"] for r in records])
    pred = np.concatenate([r[policy] for r in records])
    cases = np.unique(case)
    groups = {c: np.flatnonzero(case == c) for c in cases}
    rng = np.random.default_rng(seed)
    deltas = []
    fixed = np.concatenate([r[comparator] for r in records])
    for _ in range(reps):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        idx = np.concatenate([groups[c] for c in sampled])
        score = f1_score(y[idx], pred[idx], labels=list(range(9)),
                         average="macro", zero_division=0)
        base = f1_score(y[idx], fixed[idx], labels=list(range(9)),
                        average="macro", zero_division=0)
        deltas.append(score - base)
    point = (f1_score(y, pred, labels=list(range(9)), average="macro", zero_division=0)
             - f1_score(y, fixed, labels=list(range(9)), average="macro", zero_division=0))
    return point, np.quantile(deltas, [0.025, 0.975]).tolist()


def cluster_time_ci(records, comparator, policy="RL", reps=1000, seed=42):
    case = np.concatenate([r["case"] for r in records])
    delta = np.concatenate([r[f"time_{policy}"] - r[f"time_{comparator}"]
                            for r in records])
    cases = np.unique(case)
    groups = {c: np.flatnonzero(case == c) for c in cases}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(reps):
        picked = rng.choice(cases, size=len(cases), replace=True)
        idx = np.concatenate([groups[c] for c in picked])
        samples.append(delta[idx].mean())
    return float(delta.mean()), np.quantile(samples, [0.025, 0.975]).tolist()


def main():
    rows, records = [], []
    for fold in range(5):
        row, te, sims = eval_fold(fold)
        rows.append(row)
        record = dict(y=te["labels9"], case=te["case_id"])
        record.update({name: decisions_from_sim(te, sim) for name, sim in sims.items()})
        record.update({f"time_{name}": sim["stop_w"] for name, sim in sims.items()})
        records.append(record)
        print(f"fold={fold} n={row['n']} tau={row['tau']} "
              f"RL-F1={row['metrics'].get('RL', {}).get('macro_f1_all', float('nan')):.4f}",
              flush=True)

    keys = ["macro_f1", "macro_f1_all", "coverage", "mean_decision_time",
            "mean_return", "badq_false_alarm_rate", "badq_severe_recall"]
    lines = ["# v2：事件起点后向前观察，患者级五折测试", "",
             "每节律段至多一条 episode；固定窗到段末即停止。macro-F1 为非弃权子集；"
             "all-F1 将弃权视作漏报。时间为全部事件从首个标注心拍开始的观察时长。"
             + ("新 RL 不读取未来段长；所有策略仍在标注段末被截断。" if CAUSAL else ""), "",
             "| 策略 | 条件 F1 | 全体 F1 | 覆盖率 | 观察秒数 | 回报 | 坏质量真误报率 | 坏质量危急类召回 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in POLICIES:
        vals = [r["metrics"][policy] for r in rows if policy in r["metrics"]]
        if len(vals) != 5:
            continue
        cells = [mean_sd([m[k] for m in vals]) for k in keys]
        if sum(r["badq_severe_n"] for r in rows) < 20:
            cells[-1] = "不可估计"
        lines.append("| " + policy + " | " + " | ".join(cells) + " |")
    lines += ["", "坏质量危急类真阳性事件仅 " +
              str(sum(r["badq_severe_n"] for r in rows)) +
              " 条；不报告该召回率的五折均值。"]
    segments = pd.read_parquet(ROOT / "data/processed/segments.parquet")
    eligible = pd.read_parquet(ROOT / "data/processed/event_episodes_v2.parquet")
    lines += ["", "## 事件可评估范围", "",
              "持续不足 1 秒的段无法进入 1 秒起点检测任务；长于 10 秒的段在 10 秒截断。", "",
              "| 类别 | 全部标注段 | ≥1 秒 | ≥10 秒 | 病例数（≥1 秒） |",
              "|---|---:|---:|---:|---:|"]
    for c in CLASSES:
        all_count = int((segments.rhythm == c).sum())
        sub = eligible[eligible.label == c]
        lines.append(f"| {c} | {all_count} | {len(sub)} | "
                     f"{int((sub.w_max == 10).sum())} | {sub.case_id.nunique()} |")
    if all("RL" in r["metrics"] for r in rows):
        lines += ["", "## 全体样本逐类 F1（弃权记作遗漏）", "",
                  "| 类别 | Fixed 10s | RL |", "|---|---:|---:|"]
        for c, label in enumerate(CLASSES):
            scores = []
            for policy in ("Fixed 10s", "RL"):
                per_fold = [f1_score(rec["y"] == c, rec[policy] == c,
                                     zero_division=0) for rec in records]
                scores.append(mean_sd(per_fold))
            lines.append(f"| {label} | {scores[0]} | {scores[1]} |")
    if all("RL" in r["metrics"] for r in rows):
        comparisons = {}
        time_comparisons = {}
        lines += ["", "## 患者聚类 bootstrap：全体 macro-F1 的配对差", "",
                  "| 比较 | 差值 | 95% 区间 |", "|---|---:|---:|"]
        for comparator in ("Fixed 10s", "Fixed 5s", "Threshold", "Time threshold"):
            point, ci = cluster_ci(records, "RL", comparator=comparator)
            comparisons[comparator] = dict(point=point, ci95=ci)
            lines.append(f"| RL − {comparator} | {point:+.4f} | "
                         f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        for comparator in ("Fixed 10s", "Threshold"):
            point, ci = cluster_ci(records, "Time threshold", comparator=comparator)
            comparisons[f"Time threshold vs {comparator}"] = dict(point=point, ci95=ci)
            lines.append(f"| Time threshold − {comparator} | {point:+.4f} | "
                         f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        lines += ["", "## 患者聚类 bootstrap：观察时间配对差（秒）", "",
                  "| 比较 | 差值 | 95% 区间 |", "|---|---:|---:|"]
        for comparator in ("Fixed 10s", "Fixed 5s", "Threshold", "Time threshold"):
            point, ci = cluster_time_ci(records, comparator)
            time_comparisons[f"RL vs {comparator}"] = dict(point=point, ci95=ci)
            lines.append(f"| RL − {comparator} | {point:+.3f} | "
                         f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |")
        point, ci = cluster_time_ci(records, "Fixed 10s", policy="Time threshold")
        time_comparisons["Time threshold vs Fixed 10s"] = dict(point=point, ci95=ci)
        lines.append(f"| Time threshold − Fixed 10s | {point:+.3f} | "
                     f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    else:
        comparisons = {}
        time_comparisons = {}
    lines += ["", "五折阈值 τ：" + str([r["tau"] for r in rows]),
              "五折时间阈值 (τ0, slope)：" + str([r["tau_time"] for r in rows]),
              "质量头在测试事件最大可观察前缀的 AUROC：" +
              mean_sd([r["quality_auc"] for r in rows]), ""]
    out = ROOT / f"outputs/tables/event_cv_{RESULT_TAG}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / f"outputs/tables/event_cv_{RESULT_TAG}.json").write_text(
        json.dumps(dict(folds=rows, rl_comparisons_all_f1=comparisons,
                        time_comparisons=time_comparisons),
                   indent=1), encoding="utf-8")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
