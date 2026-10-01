# -*- coding: utf-8 -*-
"""Collect per-episode test decisions of the principal run for the v4 figures.

Output: outputs/tables/v4_decisions.npz with the out-of-fold trajectories,
time-threshold and DQN stop times, and the greedy DQN action at every prefix.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_event_cv as ev  # noqa: E402
from src.policies import simulate_time_threshold  # noqa: E402
from src.rl.agent import DQNAgent  # noqa: E402
from src.rl.env import STATE_DIM, featurize  # noqa: E402
from src.train_rl import simulate_agent  # noqa: E402

ev.VARIANT, ev.CAUSAL, ev.RESULT_TAG = "v2b_ce_sqrt", True, "v2b_ce_sqrt_causal"
KEYS = ["probs", "entropy", "quality", "qtrue", "wmax", "labels9", "case_id", "event_id"]


def select_tt(va):
    cands = [(a, round(float(b), 2)) for a in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
             for b in np.arange(-1.5, 0.21, 0.1)]
    return max(cands, key=lambda ab: ev.evaluate_sim(
        va, simulate_time_threshold(va, *ab), r=ev.REWARD)["mean_return"])


def main():
    out = {k: [] for k in KEYS + ["fold", "stop_tt", "stop_dqn", "act", "qvals"]}
    tt_params = []
    for fold in range(5):
        va, te = ev.load(fold, "val"), ev.load(fold, "test")
        tt = select_tt(va)
        tt_params.append(tt)
        agent = DQNAgent(state_dim=STATE_DIM, double=True, dueling=True, device="cpu")
        agent.q.load_state_dict(torch.load(
            ROOT / f"outputs/checkpoints/event_rl_f{fold}_{ev.RESULT_TAG}.pt",
            map_location="cpu", weights_only=False)["model"])
        agent.q.eval()
        n = len(te["labels9"])
        acts, qv = np.zeros((n, 10), np.int64), np.zeros((n, 10, 4), np.float32)
        for w in range(1, 11):
            s = featurize(te["probs"][:, w - 1], te["entropy"][:, w - 1],
                          te["quality"][:, w - 1], np.full(n, float(w)))
            with torch.no_grad():
                q = agent.q(torch.from_numpy(s)).numpy()
            if w == 10:
                q[:, :2] = -np.inf  # waiting is invalid at the horizon
            qv[:, w - 1], acts[:, w - 1] = q, q.argmax(1)
        for k in KEYS:
            out[k].append(te[k])
        out["fold"].append(np.full(n, fold))
        out["stop_tt"].append(simulate_time_threshold(te, *tt)["stop_w"])
        out["stop_dqn"].append(simulate_agent(te, agent, env_kw=dict(reveal_wmax=False))["stop_w"])
        out["act"].append(acts)
        out["qvals"].append(qv)
        print("fold", fold, "tt", tt)
    arr = {k: np.concatenate(v) for k, v in out.items()}
    arr["tt_params"] = np.array(tt_params, dtype=float)
    np.savez_compressed(ROOT / "outputs/tables/v4_decisions.npz", **arr)
    print("saved", len(arr["labels9"]))


if __name__ == "__main__":
    main()
