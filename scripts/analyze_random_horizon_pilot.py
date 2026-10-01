"""Validation-only pilot of random terminal truncation during DQN training."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_event_boundary import rl_forced
from scripts.eval_event_cv import REWARD, cluster_ci, cluster_time_ci, mean_sd
from src.policies import decisions_from_sim, evaluate_sim
from src.rl.agent import DQNAgent
from src.rl.env import STATE_DIM
from src.train_rl import simulate_agent


def load(fold, suffix):
    model = DQNAgent(state_dim=STATE_DIM, double=True, dueling=True)
    path = ROOT / f"outputs/checkpoints/event_rl_f{fold}_v2b_ce_sqrt_causal{suffix}.pt"
    model.q.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    model.q.eval()
    return model


def main():
    rows, records = [], []
    for fold in range(5):
        va = dict(np.load(ROOT / f"data/processed/event_traj_val_f{fold}_v2b_ce_sqrt.npz"))
        result = {}
        record = dict(y=va["labels9"], case=va["case_id"])
        for name, suffix in (("Original", ""), ("Random horizon", "_rh")):
            agent = load(fold, suffix)
            sim = simulate_agent(va, agent, env_kw=dict(reveal_wmax=False))
            metrics = evaluate_sim(va, sim, r=REWARD)
            result[name] = dict(f1=metrics["macro_f1_all"], time=metrics["mean_decision_time"],
                                reward=metrics["mean_return"], coverage=metrics["coverage"],
                                forced=int(rl_forced(va, agent).sum()))
            record[name] = decisions_from_sim(va, sim)
            record[f"time_{name}"] = sim["stop_w"]
        rows.append(dict(fold=fold, n=len(va["wmax"]), **result))
        records.append(record)
    f1_delta, f1_ci = cluster_ci(records, "Random horizon", comparator="Original")
    time_delta, time_ci = cluster_time_ci(records, "Original", policy="Random horizon")
    lines = ["# Random-horizon DQN: validation-only pilot", "",
             "Training events were independently truncated at a uniformly sampled integer horizon from 1 to their original annotated duration on each of the 12 training passes. Architecture, reward, classifier trajectories and validation checkpoint rule matched the original causal DQN. No test trajectories were evaluated for this pilot. The same validation patients selected each checkpoint and are reported here, so these comparisons are optimistic and exploratory.", "",
             "| Policy | Validation macro-F1 (5-fold mean±SD) | Observation seconds | Reward | Forced at annotated end |",
             "|---|---:|---:|---:|---:|"]
    for name in ("Original", "Random horizon"):
        lines.append(f"| {name} | {mean_sd([r[name]['f1'] for r in rows])} | "
                     f"{mean_sd([r[name]['time'] for r in rows])} | "
                     f"{mean_sd([r[name]['reward'] for r in rows])} | "
                     f"{sum(r[name]['forced'] for r in rows)}/{sum(r['n'] for r in rows)} |")
    lines += ["", f"Pooled validation patient-bootstrap difference, random horizon minus original: macro-F1 {f1_delta:+.4f} (95% CI [{f1_ci[0]:+.4f}, {f1_ci[1]:+.4f}]); time {time_delta:+.3f} s (95% CI [{time_ci[0]:+.3f}, {time_ci[1]:+.3f}]). These intervals do not account for validation checkpoint selection and are not confirmation.", "",
              "## Fold detail", "",
              "| Fold | N | Original F1 / s / forced | Random horizon F1 / s / forced |",
              "|---:|---:|---|---|"]
    for r in rows:
        a, b = r["Original"], r["Random horizon"]
        lines.append(f"| {r['fold']} | {r['n']} | {a['f1']:.4f} / {a['time']:.3f} / {a['forced']} | "
                     f"{b['f1']:.4f} / {b['time']:.3f} / {b['forced']} |")
    lines += ["", "Go/no-go: improvement must remain after comparison with a matched simple threshold, across classifier seeds and on new patients or a genuinely independent dataset; otherwise retain the benchmark interpretation. Random truncation alone is not yet a novel algorithm.", ""]
    out = ROOT / "outputs/tables/random_horizon_pilot_v2b_ce_sqrt_causal.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
