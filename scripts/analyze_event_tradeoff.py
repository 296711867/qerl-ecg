"""Exploratory duration-budget curve from frozen v2 classifier trajectories."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.policies import decisions_from_sim, simulate_fixed, simulate_time_threshold

VARIANT = "v2b_ce_sqrt"
BUDGETS = (2, 3, 4, 5, 6, 7, 8)
CANDIDATES = [(a, round(float(b), 2)) for a in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
              for b in np.arange(-1.5, 0.21, 0.1)]


def load(fold, part):
    path = ROOT / f"data/processed/event_traj_{part}_f{fold}_{VARIANT}.npz"
    return dict(np.load(path))


def measure(traj, sim):
    pred = decisions_from_sim(traj, sim)
    return dict(f1=float(f1_score(traj["labels9"], pred, labels=list(range(9)),
                                  average="macro", zero_division=0)),
                seconds=float(sim["stop_w"].mean()))


def main():
    rows = []
    duration_rows = []
    class_records = []
    for fold in range(5):
        val, test = load(fold, "val"), load(fold, "test")
        grid = [(ab, measure(val, simulate_time_threshold(val, *ab)))
                for ab in CANDIDATES]
        for budget in BUDGETS:
            feasible = [(ab, m) for ab, m in grid if m["seconds"] <= budget]
            if not feasible:
                continue
            selected, vm = max(feasible, key=lambda item: (item[1]["f1"],
                                                           -item[1]["seconds"]))
            tm = measure(test, simulate_time_threshold(test, *selected))
            rows.append(dict(fold=fold, budget=budget, tau0=selected[0],
                             slope=selected[1], val=vm, test=tm))

        # A descriptive failure audit under the existing reward-selected rule.
        selected = max(grid, key=lambda item: reward(val, item[0]))[0]
        policies = {"Time threshold": simulate_time_threshold(test, *selected),
                    "Fixed 10s": simulate_fixed(test, 10)}
        y = test["labels9"]
        wmax = test["wmax"]
        class_records.append(dict(y=y, case=test["case_id"],
                                  threshold=decisions_from_sim(test, policies["Time threshold"]),
                                  fixed=decisions_from_sim(test, policies["Fixed 10s"])))
        for name, sim in policies.items():
            pred = decisions_from_sim(test, sim)
            for label, mask in (("1–2 s", wmax <= 2),
                                ("3–4 s", (wmax >= 3) & (wmax <= 4)),
                                ("5–9 s", (wmax >= 5) & (wmax <= 9)),
                                ("10+ s", wmax == 10)):
                if mask.any():
                    duration_rows.append(dict(fold=fold, policy=name, duration=label,
                                              n=int(mask.sum()), correct=int((pred[mask] == y[mask]).sum()),
                                              observed=float(sim["stop_w"][mask].sum())))

    out = ROOT / "outputs/tables/event_tradeoff_v2b_ce_sqrt_causal.json"
    out.write_text(json.dumps(dict(budget_curve=rows, duration_audit=duration_rows), indent=2),
                   encoding="utf-8")
    lines = ["# Exploratory validation-selected observation-budget analysis", "",
             "Each fold selects a time-varying threshold on validation patients by maximum "
             "nine-class macro-F1 subject to a validation mean-time budget. Each selected "
             "configuration is evaluated once on its held-out fold; the same test folds are "
             "reused across the seven budgets. These are post hoc exploratory analyses "
             "of the frozen v2 trajectories, not a new confirmatory test set.", "",
             "| Validation budget (s) | Feasible folds | Test macro-F1, fold mean ± SD | "
             "Test observed seconds, fold mean ± SD |", "|---:|---:|---:|---:|"]
    curve = []
    for budget in BUDGETS:
        subset = [r for r in rows if r["budget"] == budget]
        if not subset:
            continue
        f1 = np.array([r["test"]["f1"] for r in subset])
        sec = np.array([r["test"]["seconds"] for r in subset])
        curve.append((sec.mean(), f1.mean(), sec.std(ddof=1), f1.std(ddof=1), budget))
        lines.append(f"| {budget} | {len(subset)} | {f1.mean():.4f} ± {f1.std(ddof=1):.4f} | "
                     f"{sec.mean():.3f} ± {sec.std(ddof=1):.3f} |")
    lines += ["", "## Accuracy by maximum available annotated duration", "",
              "Accuracy is pooled across the five held-out folds within each duration stratum. "
              "The stopping rule here is the original reward-selected time threshold, "
              "not a point selected from the budget curve.", "",
              "| Available duration | Episodes | Fixed 10 s accuracy | Time-threshold accuracy | "
              "Time-threshold mean observed seconds |", "|---|---:|---:|---:|---:|"]
    for label in ("1–2 s", "3–4 s", "5–9 s", "10+ s"):
        fixed = [r for r in duration_rows if r["duration"] == label and r["policy"] == "Fixed 10s"]
        threshold = [r for r in duration_rows if r["duration"] == label and r["policy"] == "Time threshold"]
        n = sum(r["n"] for r in fixed)
        lines.append(f"| {label} | {n} | {sum(r['correct'] for r in fixed)/n:.3f} | "
                     f"{sum(r['correct'] for r in threshold)/n:.3f} | "
                     f"{sum(r['observed'] for r in threshold)/n:.3f} |")
    y = np.concatenate([r["y"] for r in class_records])
    case = np.concatenate([r["case"] for r in class_records])
    threshold = np.concatenate([r["threshold"] for r in class_records])
    fixed = np.concatenate([r["fixed"] for r in class_records])
    names = ("N", "AFIB/AFL", "AVB", "SND", "SR-mPAC-BT", "SR-mPVC-BT", "SVTA", "VT", "MAT")
    lines += ["", "## Pooled class outcomes for the original reward-selected rule", "",
              "Each event has one out-of-fold prediction. Class F1 is one-versus-rest on "
              "pooled held-out patients, so it is not the unweighted mean of fold F1 values.", "",
              "| Rhythm | Episodes | Patients | Fixed 10 s F1 | Time-threshold F1 | "
              "Time-threshold recall |", "|---|---:|---:|---:|---:|---:|"]
    for k, name in enumerate(names):
        mask = y == k
        lines.append(f"| {name} | {mask.sum()} | {len(np.unique(case[mask]))} | "
                     f"{f1_score(mask, fixed == k, zero_division=0):.3f} | "
                     f"{f1_score(mask, threshold == k, zero_division=0):.3f} | "
                     f"{(threshold[mask] == k).mean():.3f} |")
    (ROOT / "outputs/tables/event_tradeoff_v2b_ce_sqrt_causal.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    x, y, sx, sy, budgets = map(np.array, zip(*curve))
    ax.plot(x, y, "o-", color="#0072B2", linewidth=1.8,
            label="Validation-selected time threshold")
    for xx, yy, budget in zip(x, y, budgets):
        if budget in (2, 3, 4, 5, 8):
            ax.annotate(f"≤{budget} s", (xx, yy), xytext=(5, -15 if budget == 5 else 6),
                        textcoords="offset points", fontsize=8)
    main = json.loads((ROOT / "outputs/tables/event_cv_v2b_ce_sqrt_causal.json").read_text(
        encoding="utf-8"))["folds"]
    for name, label, color, marker in (("Fixed 10s", "Fixed 10 s", "#777777", "s"),
                                       ("Time threshold", "Original time threshold", "#009E73", "D"),
                                       ("RL", "DQN", "#D55E00", "^")):
        fx = np.array([r["metrics"][name]["mean_decision_time"] for r in main])
        fy = np.array([r["metrics"][name]["macro_f1_all"] for r in main])
        ax.errorbar(fx.mean(), fy.mean(), xerr=fx.std(ddof=1), yerr=fy.std(ddof=1),
                    fmt=marker, color=color, capsize=3, markersize=8, label=label)
    ax.set(xlabel="Observed seconds after first annotated beat",
           ylabel="Event macro-F1", ylim=(0.43, 0.59))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = ROOT / "paper/v2/figures/fig3_budget_tradeoff"
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".svg"))
    print(out)


def reward(traj, ab):
    from src.policies import episode_returns
    return float(episode_returns(traj, simulate_time_threshold(traj, *ab),
                                 dict(correct=1.0, wrong=-2.0, wait_per_second=-0.03,
                                      appropriate_abstain=0.3, unnecessary_abstain=-1.0)).mean())


if __name__ == "__main__":
    main()
