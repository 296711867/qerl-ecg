"""Post hoc audit of forced segment-end stops and common observation horizons."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import eval_event_cv as cv
from scripts.eval_event_cv import cluster_ci, cluster_time_ci, mean_sd
from src.policies import decisions_from_sim, simulate_fixed, simulate_time_threshold
from src.rl.agent import DQNAgent
from src.rl.env import STATE_DIM, STOP, WAIT1, WAIT2, VecEarlyEnv
from src.train_rl import simulate_agent

VARIANT = "v2b_ce_sqrt"
cv.VARIANT = VARIANT
cv.CAUSAL = True
cv.RESULT_TAG = f"{VARIANT}_causal"


def load_agent(fold):
    agent = DQNAgent(state_dim=STATE_DIM, double=True, dueling=True)
    ckpt = ROOT / f"outputs/checkpoints/event_rl_f{fold}_{VARIANT}_causal.pt"
    agent.q.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    agent.q.eval()
    return agent


def rl_forced(traj, agent):
    """True only when a WAIT action reaches the annotated segment end (<10 s)."""
    env = VecEarlyEnv(traj, reveal_wmax=False)
    idx = np.arange(len(traj["wmax"]))
    forced = np.zeros(len(idx), dtype=bool)
    for start in range(0, len(idx), 4096):
        ii = idx[start:start + 4096]
        state, mask = env.reset(ii)
        while not env.done.all():
            action = agent.act(state, mask, greedy=True)
            action[env.done] = STOP
            live = ~env.done
            state, _, newly, mask = env.step(action)
            forced[ii] |= live & newly & np.isin(action, [WAIT1, WAIT2]) & (env.wmax < 10)
    return forced


def threshold_forced(traj, sim, tau0, slope):
    wmax = traj["wmax"]
    threshold = np.clip(tau0 + slope * (wmax - 1) / 9, 0.01, 0.99)
    p = traj["probs"][np.arange(len(wmax)), wmax - 1].max(axis=1)
    return (wmax < 10) & (sim["stop_w"] == wmax) & (p < threshold)


def score_subset(traj, sims):
    y = traj["labels9"]
    return {name: dict(n=len(y), macro_f1=float(f1_score(
        y, decisions_from_sim(traj, sim), labels=list(range(9)),
        average="macro", zero_division=0)), time=float(sim["stop_w"].mean()))
        for name, sim in sims.items()}


def main():
    rows, common, paired = [], {3: [], 5: [], 8: []}, {3: [], 5: [], 8: []}
    for fold in range(5):
        row, te, sims = cv.eval_fold(fold)
        agent = load_agent(fold)
        wmax = te["wmax"]
        forced = {
            "Fixed 10s": wmax < 10,
            "Time threshold": threshold_forced(te, sims["Time threshold"], *row["tau_time"]),
            "RL": rl_forced(te, agent),
        }
        assert all(np.all(sims[k]["stop_w"][v] == wmax[v]) for k, v in forced.items())
        rows.append(dict(fold=fold, n=len(wmax), forced={k: int(v.sum()) for k, v in forced.items()},
                         by_class={k: np.bincount(te["labels9"][v], minlength=9).tolist()
                                   for k, v in forced.items()},
                         accuracy_forced={k: (float((decisions_from_sim(te, sims[k])[v] == te["labels9"][v]).mean())
                                              if v.any() else None) for k, v in forced.items()}))
        for budget in common:
            keep = wmax >= budget
            sub = {k: v[keep] for k, v in te.items()
                   if isinstance(v, np.ndarray) and v.ndim and len(v) == len(wmax)}
            sub["wmax"] = np.full(keep.sum(), budget, dtype=np.int64)
            sub_sims = {
                "Fixed horizon": simulate_fixed(sub, budget),
                "Time threshold": simulate_time_threshold(sub, *row["tau_time"]),
                "RL": simulate_agent(sub, agent, env_kw=dict(reveal_wmax=False)),
            }
            common[budget].append(dict(fold=fold, **score_subset(sub, sub_sims)))
            record = dict(y=sub["labels9"], case=sub["case_id"])
            record.update({k: decisions_from_sim(sub, v) for k, v in sub_sims.items()})
            record.update({f"time_{k}": v["stop_w"] for k, v in sub_sims.items()})
            paired[budget].append(record)
        print(f"fold {fold}: " + ", ".join(f"{k} {int(v.sum())}/{len(v)} forced" for k, v in forced.items()), flush=True)

    lines = ["# Exploratory annotated-boundary sensitivity audit", "",
             "The model observes only causal prefixes, but the retrospective simulator forces a classification when an annotated rhythm segment ends. Here *forced* means a policy still requested observation when that segment ended before the common 10 s cap. Active stops exactly at the end are not counted as forced. This analysis was designed after seeing the main results; it is descriptive and cannot establish a continuous-monitoring effect.", "",
             "## Full cohort: forced stops", "",
             "| Policy | Forced / 4,793 | Forced fraction | Accuracy in forced events |",
             "|---|---:|---:|---:|"]
    for policy in ("Fixed 10s", "Time threshold", "RL"):
        n = sum(r["forced"][policy] for r in rows)
        accuracy = sum(r["accuracy_forced"][policy] * r["forced"][policy]
                       for r in rows if r["accuracy_forced"][policy] is not None) / n if n else float("nan")
        lines.append(f"| {policy} | {n} | {n / 4793:.3%} | {accuracy:.4f} |")
    lines += ["", "Forced counts by class (N, AFIB/AFL, AVB, SND, SR-mPAC-BT, SR-mPVC-BT, SVTA, VT, MAT):", ""]
    for policy in ("Fixed 10s", "Time threshold", "RL"):
        counts = np.sum([r["by_class"][policy] for r in rows], axis=0).tolist()
        lines.append(f"- {policy}: {counts}")
    lines += ["", "## Common-horizon risk sets", "",
              "Only episodes with at least the stated duration are included. Each is then capped at the same horizon for every policy; the original validation-selected threshold and DQN checkpoints are reused without retuning. This is a subset sensitivity analysis, not a comparison on all events or a new held-out test.", "",
              "| Horizon | Events | Policy | Five-fold macro-F1 | Five-fold time (s) | Pooled RL − threshold macro-F1 (patient bootstrap 95% CI) | RL − threshold time (s; 95% CI) |",
              "|---:|---:|---|---:|---:|---:|---:|"]
    differences = {}
    for budget in common:
        delta, ci = cluster_ci(paired[budget], "RL", comparator="Time threshold")
        time_delta, time_ci = cluster_time_ci(paired[budget], "Time threshold")
        differences[budget] = dict(f1=dict(point=delta, ci95=ci),
                                   time=dict(point=time_delta, ci95=time_ci))
        events = sum(r["RL"]["n"] for r in common[budget])
        for policy in ("Fixed horizon", "Time threshold", "RL"):
            f1 = mean_sd([r[policy]["macro_f1"] for r in common[budget]])
            timing = mean_sd([r[policy]["time"] for r in common[budget]])
            contrast = f"{delta:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" if policy == "RL" else ""
            time_contrast = f"{time_delta:+.3f} [{time_ci[0]:+.3f}, {time_ci[1]:+.3f}]" if policy == "RL" else ""
            lines.append(f"| {budget} s | {events} | {policy} | {f1} | {timing} | {contrast} | {time_contrast} |")
    lines += ["", "Selection of long-enough episodes changes the class and duration mix. The original classifier was trained on the full eligible cohort and may still encode duration-correlated patterns. None of these estimates are prospective or independently confirmatory.", ""]
    out = ROOT / "outputs/tables/event_boundary_sensitivity_v2b_ce_sqrt_causal.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(dict(full_cohort=rows, common_horizon=common,
                                                        rl_vs_threshold=differences), indent=2), encoding="utf-8")
    print(out, flush=True)


if __name__ == "__main__":
    main()
