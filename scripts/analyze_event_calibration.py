"""Exploratory scalar temperature calibration of frozen v2 prefix posteriors."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.policies import evaluate_sim, simulate_time_threshold

VARIANT = "v2b_ce_sqrt"
REWARD = dict(correct=1.0, wrong=-2.0, wait_per_second=-0.03,
              appropriate_abstain=0.3, unnecessary_abstain=-1.0)
CANDIDATES = [(a, round(float(b), 2)) for a in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
              for b in np.arange(-1.5, 0.21, 0.1)]


def load(fold, part):
    return dict(np.load(ROOT / f"data/processed/event_traj_{part}_f{fold}_{VARIANT}.npz"))


def valid_prefixes(traj):
    mask = np.arange(10)[None, :] < traj["wmax"][:, None]
    p = np.clip(traj["probs"][mask].astype(np.float64), 1e-8, 1.0)
    y = np.repeat(traj["labels9"], traj["wmax"])
    weight = np.repeat(1.0 / traj["wmax"], traj["wmax"])
    return p, y, weight / weight.sum()


def transform(probs, temperature):
    z = np.log(np.clip(probs, 1e-8, 1.0)) / temperature
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def fit_temperature(traj):
    p, y, weight = valid_prefixes(traj)
    def nll(log_temp):
        q = transform(p, np.exp(log_temp))
        return float(-np.sum(weight * np.log(np.clip(q[np.arange(len(y)), y], 1e-8, 1))))
    result = minimize_scalar(nll, bounds=(-2, 2), method="bounded")
    return float(np.exp(result.x))


def diagnostic(traj):
    p, y, weight = valid_prefixes(traj)
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    brier = float(np.sum(weight * (np.square(p).sum(axis=1) - 2*p[np.arange(len(y)), y] + 1)))
    ece = 0.0
    bins = np.linspace(0, 1, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        if mask.any():
            mass = weight[mask].sum()
            ece += mass * abs(np.average(confidence[mask], weights=weight[mask])
                              - np.average(correct[mask], weights=weight[mask]))
    return dict(ece10=float(ece), brier=brier, n_events=len(traj["labels9"]),
                n_prefixes=len(y))


def calibrated(traj, temperature):
    new = dict(traj)
    new["probs"] = transform(traj["probs"].astype(np.float64), temperature)
    return new


def selected_policy(val, test):
    tau = max(CANDIDATES, key=lambda ab: evaluate_sim(
        val, simulate_time_threshold(val, *ab), r=REWARD)["mean_return"])
    m = evaluate_sim(test, simulate_time_threshold(test, *tau), r=REWARD)
    return tau, {k: float(m[k]) for k in ("macro_f1_all", "mean_decision_time", "mean_return")}


def main():
    rows = []
    for fold in range(5):
        val, test = load(fold, "val"), load(fold, "test")
        temp = fit_temperature(val)
        cal_val, cal_test = calibrated(val, temp), calibrated(test, temp)
        raw_tau, raw_policy = selected_policy(val, test)
        cal_tau, cal_policy = selected_policy(cal_val, cal_test)
        rows.append(dict(fold=fold, temperature=temp, raw=diagnostic(test),
                         calibrated=diagnostic(cal_test), raw_tau=raw_tau,
                         calibrated_tau=cal_tau, raw_policy=raw_policy,
                         calibrated_policy=cal_policy))
    out = ROOT / "outputs/tables/event_calibration_v2b_ce_sqrt_causal.json"
    out.write_text(json.dumps(dict(protocol="post hoc validation-only scalar temperature fit",
                                   prefix_weighting="each event contributes total weight one",
                                   folds=rows), indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Exploratory prefix-posterior calibration", "",
             "Each fold fits one scalar temperature on validation prefixes by weighted NLL. "
             "Each event contributes total weight one across its available prefixes. "
             "The same validation patients select raw and calibrated threshold parameters "
             "separately by reward; test patients are only evaluated. This is post hoc and "
             "does not provide independent confirmation.", "",
             "| Outcome | Raw, fold mean ± SD | Calibrated, fold mean ± SD |",
             "|---|---:|---:|"]
    for label, location in (("10-bin ECE", ("raw", "ece10")),
                            ("Multiclass Brier", ("raw", "brier")),
                            ("Threshold macro-F1", ("raw_policy", "macro_f1_all")),
                            ("Threshold observed seconds", ("raw_policy", "mean_decision_time")),
                            ("Threshold mean reward", ("raw_policy", "mean_return"))):
        a, k = location
        b = "calibrated" if a == "raw" else "calibrated_policy"
        x = np.array([r[a][k] for r in rows])
        y = np.array([r[b][k] for r in rows])
        lines.append(f"| {label} | {x.mean():.4f} ± {x.std(ddof=1):.4f} | "
                     f"{y.mean():.4f} ± {y.std(ddof=1):.4f} |")
    lines += ["", "Validation-fitted temperatures: " +
              ", ".join(f"{r['temperature']:.3f}" for r in rows) + "."]
    (ROOT / "outputs/tables/event_calibration_v2b_ce_sqrt_causal.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
