"""v3 analyses on frozen trajectories and policies.

1. Robustness of the policy comparison to classifier seed (frozen split) and to the
   patient partition (two re-drawn patient-disjoint 5-fold splits).
2. Hindsight oracle stopping bounds (uses labels; descriptive headroom only).
3. Descriptive test-set sweep of stopping-rule operating points (F1-time frontier).
4. Reliability diagram, raw vs validation-temperature-scaled prefix posteriors.
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
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_event_calibration as cal  # noqa: E402
import eval_event_cv as ev  # noqa: E402
import torch  # noqa: E402
from src.policies import (decisions_from_sim, evaluate_sim, simulate_fixed,  # noqa: E402
                          simulate_threshold, simulate_time_threshold)
from src.rl.agent import DQNAgent  # noqa: E402
from src.rl.env import STATE_DIM  # noqa: E402
from src.train_rl import simulate_agent  # noqa: E402

BASE = "v2b_ce_sqrt"
SEED_SET = [BASE, "v3_cs1", "v3_cs2"]
PARTITION_SET = [BASE, "v3_p1", "v3_p2"]
POLICIES = ["Fixed 10s", "Threshold", "Time threshold", "RL", "Matched threshold"]
FINE = [(a, b) for a in np.round(np.arange(0.40, 1.0, 0.05), 2)
        for b in np.round(np.arange(-1.5, 0.21, 0.1), 2)]
TABLES = ROOT / "outputs/tables"
FIGS = ROOT / "paper/v3/figures"
REPS = 1000


def macro_f1(y, pred):
    """Nine-class macro-F1 with abstention (-1) counted as a miss; matches sklearn."""
    ok = pred >= 0
    cm = np.bincount(y[ok] * 9 + pred[ok], minlength=81).reshape(9, 9)
    tp = np.diag(cm).astype(float)
    support = np.bincount(y, minlength=9).astype(float)
    predicted = cm.sum(0).astype(float)
    denom = support + predicted
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


def records_for(variant):
    ev.VARIANT, ev.CAUSAL, ev.RESULT_TAG = variant, True, f"{variant}_causal"
    rows, folds = [], []
    for fold in range(5):
        row, te, sims = ev.eval_fold(fold)
        sims["Matched threshold"], row["matched"] = matched_threshold(variant, fold, te)
        row["metrics"]["Matched threshold"] = evaluate_sim(te, sims["Matched threshold"], r=ev.REWARD)
        rec = dict(y=te["labels9"], case=te["case_id"], fold=np.full(len(te["wmax"]), fold))
        for name in POLICIES:
            rec[name] = decisions_from_sim(te, sims[name])
            rec[f"time_{name}"] = sims[name]["stop_w"].astype(float)
        rows.append(row)
        folds.append(rec)
    pooled = {k: np.concatenate([r[k] for r in folds]) for k in folds[0]}
    return rows, pooled


def matched_threshold(variant, fold, te):
    """Time-varying rule chosen on validation for maximum macro-F1 at no more validation
    observation time than the fold's DQN; isolates trade-off quality from waiting longer."""
    va = ev.load(fold, "val")
    agent = DQNAgent(state_dim=STATE_DIM, double=True, dueling=True, device="cpu")
    agent.q.load_state_dict(torch.load(ROOT / f"outputs/checkpoints/event_rl_f{fold}_{ev.RESULT_TAG}.pt",
                                       map_location="cpu", weights_only=False)["model"])
    agent.q.eval()
    budget = float(simulate_agent(va, agent, env_kw=dict(reveal_wmax=False))["stop_w"].mean())
    scored = []
    for ab in FINE:
        sim = simulate_time_threshold(va, *ab)
        t = float(sim["stop_w"].mean())
        if t <= budget:
            scored.append((macro_f1(va["labels9"], decisions_from_sim(va, sim)), -t, ab))
    _, _, best = max(scored)
    return simulate_time_threshold(te, *best), dict(params=best, val_budget=budget)


def fold_summary(rows):
    out = {}
    for name in POLICIES:
        f1 = np.array([r["metrics"][name]["macro_f1_all"] for r in rows])
        t = np.array([r["metrics"][name]["mean_decision_time"] for r in rows])
        cov = np.array([r["metrics"][name]["coverage"] for r in rows])
        out[name] = dict(f1_mean=f1.mean(), f1_sd=f1.std(ddof=1), time_mean=t.mean(),
                         time_sd=t.std(ddof=1), coverage=cov.mean())
    return out


def averaged_bootstrap(pooled_list, a, b, reps=REPS, seed=42):
    """Patient-cluster bootstrap of a−b, averaged over several out-of-fold runs.

    Every run covers the same patients once, so one patient resample is applied to
    all runs and the per-run paired differences are averaged within each replicate.
    """
    cases = np.unique(pooled_list[0]["case"])
    groups = [{c: np.flatnonzero(p["case"] == c) for c in cases} for p in pooled_list]
    def stat(idx_list):
        d_f1 = np.mean([macro_f1(p["y"][i], p[a][i]) - macro_f1(p["y"][i], p[b][i])
                        for p, i in zip(pooled_list, idx_list)])
        d_t = np.mean([(p[f"time_{a}"][i] - p[f"time_{b}"][i]).mean()
                       for p, i in zip(pooled_list, idx_list)])
        return d_f1, d_t
    point = stat([np.arange(len(p["y"])) for p in pooled_list])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(reps):
        picked = rng.choice(cases, size=len(cases), replace=True)
        samples.append(stat([np.concatenate([g[c] for c in picked]) for g in groups]))
    s = np.array(samples)
    return dict(f1=point[0], f1_ci=np.quantile(s[:, 0], [0.025, 0.975]).tolist(),
                time=point[1], time_ci=np.quantile(s[:, 1], [0.025, 0.975]).tolist())


def robustness():
    variants = sorted(set(SEED_SET + PARTITION_SET), key=(SEED_SET + PARTITION_SET).index)
    runs = {v: records_for(v) for v in variants}
    per_run = {}
    for v, (rows, pooled) in runs.items():
        per_run[v] = dict(summary=fold_summary(rows),
                          rl_vs_tt=averaged_bootstrap([pooled], "RL", "Time threshold"),
                          tt_vs_fixed=averaged_bootstrap([pooled], "Time threshold", "Fixed 10s"),
                          rl_vs_fixed=averaged_bootstrap([pooled], "RL", "Fixed 10s"),
                          rl_vs_matched=averaged_bootstrap([pooled], "RL", "Matched threshold"),
                          tau_time=[r["tau_time"] for r in rows],
                          matched=[r["matched"] for r in rows])
    combined = {}
    for label, members in (("classifier seeds", SEED_SET), ("patient partitions", PARTITION_SET)):
        pooled = [runs[v][1] for v in members]
        combined[label] = dict(rl_vs_tt=averaged_bootstrap(pooled, "RL", "Time threshold"),
                               tt_vs_fixed=averaged_bootstrap(pooled, "Time threshold", "Fixed 10s"),
                               rl_vs_fixed=averaged_bootstrap(pooled, "RL", "Fixed 10s"),
                               rl_vs_matched=averaged_bootstrap(pooled, "RL", "Matched threshold"))
    return runs, per_run, combined


def oracle_bounds():
    """Hindsight stopping bounds from the frozen main-run posteriors (uses labels)."""
    rows = []
    for fold in range(5):
        te = cal.load(fold, "test")
        y, wmax = te["labels9"], te["wmax"]
        pred = te["probs"].argmax(2)
        steps = np.arange(10)[None, :] < wmax[:, None]
        correct = (pred == y[:, None]) & steps
        ever = correct.any(1)
        first = np.where(ever, correct.argmax(1) + 1, wmax)
        final = pred[np.arange(len(y)), wmax - 1]
        any_pred = np.where(ever, y, final)
        # Earliest prefix after which the argmax never changes: same decisions as Fixed 10 s.
        changed = (pred != final[:, None]) & steps
        last_change = np.where(changed.any(1), 9 - np.argmax(changed[:, ::-1], axis=1), -1)
        stable = last_change + 2
        rows.append(dict(any_f1=macro_f1(y, any_pred), any_time=float(first.mean()),
                         any_acc=float(ever.mean()),
                         stable_f1=macro_f1(y, final), stable_time=float(stable.mean()),
                         fixed_acc=float((final == y).mean()),
                         fixed_time=float(np.minimum(wmax, 10).mean())))
    agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1)))
           for k in rows[0]}
    return {BASE: dict(folds=rows, mean_sd=agg)}


def frontier(runs):
    """Descriptive: apply each rule uniformly to all out-of-fold test trajectories."""
    trajs = [cal.load(f, "test") for f in range(5)]
    def pooled_point(simfn):
        ys, ps, ts = [], [], []
        for te in trajs:
            sim = simfn(te)
            ys.append(te["labels9"]); ps.append(decisions_from_sim(te, sim)); ts.append(sim["stop_w"])
        y, p, t = map(np.concatenate, (ys, ps, ts))
        return float(t.mean()), macro_f1(y, p)
    pts = dict(fixed=[pooled_point(lambda te, w=w: simulate_fixed(te, w)) for w in range(1, 11)],
               threshold=[pooled_point(lambda te, x=x: simulate_threshold(te, x))
                          for x in np.round(np.arange(0.30, 1.0, 0.02), 2)],
               time_threshold=[pooled_point(lambda te, a=a, b=b: simulate_time_threshold(te, a, b))
                               for a in np.round(np.arange(0.40, 1.0, 0.05), 2)
                               for b in np.round(np.arange(-1.5, 0.21, 0.1), 2)])
    pooled = runs[BASE][1]
    for name in POLICIES:
        pts[f"selected {name}"] = (float(pooled[f"time_{name}"].mean()),
                                   macro_f1(pooled["y"], pooled[name]))
    everything = pts["fixed"] + pts["threshold"] + pts["time_threshold"]
    everything.sort(key=lambda p: (p[0], -p[1]))
    front, best = [], -1
    for t, f in everything:
        if f > best:
            front.append((t, f)); best = f
    pts["frontier"] = front
    rl_t, rl_f = pts["selected RL"]
    pts["rl_dominated_by"] = [p for p in everything if p[0] <= rl_t and p[1] >= rl_f]
    return pts


def reliability():
    raw_c, raw_k, raw_w, cal_c, cal_k, temps = [], [], [], [], [], []
    for fold in range(5):
        val, test = cal.load(fold, "val"), cal.load(fold, "test")
        temp = cal.fit_temperature(val)
        temps.append(temp)
        for traj, conf, corr in ((test, raw_c, raw_k), (cal.calibrated(test, temp), cal_c, cal_k)):
            p, y, _ = cal.valid_prefixes(traj)
            conf.append(p.max(1)); corr.append((p.argmax(1) == y).astype(float))
        _, _, w = cal.valid_prefixes(test)
        raw_w.append(w / 5)
    w = np.concatenate(raw_w)
    bins = np.linspace(0, 1, 11)
    def curve(conf, corr):
        conf, corr = np.concatenate(conf), np.concatenate(corr)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
        rows, ece = [], 0.0
        for b in range(10):
            m = idx == b
            if m.any():
                mass = w[m].sum()
                c, a = np.average(conf[m], weights=w[m]), np.average(corr[m], weights=w[m])
                rows.append((float(c), float(a), float(mass)))
                ece += mass * abs(c - a)
        return rows, float(ece / w.sum())
    raw, raw_ece = curve(raw_c, raw_k)
    calib, cal_ece = curve(cal_c, cal_k)
    return dict(raw=raw, calibrated=calib, raw_ece=raw_ece, cal_ece=cal_ece, temperatures=temps)


def fmt_ci(d, key, digits=4):
    lo, hi = d[f"{key}_ci"]
    return f"{d[key]:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    runs, per_run, combined = robustness()
    oracle = oracle_bounds()
    front = frontier(runs)
    rel = reliability()

    names = {BASE: "Frozen split, classifier seed A (main)", "v3_cs1": "Frozen split, classifier seed B",
             "v3_cs2": "Frozen split, classifier seed C", "v3_p1": "Re-drawn partition 1",
             "v3_p2": "Re-drawn partition 2"}
    L = ["# v3 robustness, oracle bounds and calibration", "",
         "All runs use the frozen v2b_ce_sqrt classifier recipe, validation-only selection of "
         "threshold parameters by mean reward, and the causal Double-Dueling DQN with the "
         "same hyperparameters (DQN seed 42). Nothing was retuned after seeing these results.", "",
         "## Per-run fold means (macro-F1 with abstention as miss; seconds observed)", "",
         "| Run | Fixed 10 s F1 | Threshold F1 / s | Time-threshold F1 / s | DQN F1 / s | DQN coverage |",
         "|---|---:|---:|---:|---:|---:|"]
    for v, r in per_run.items():
        s = r["summary"]
        L.append(f"| {names[v]} | {s['Fixed 10s']['f1_mean']:.4f}±{s['Fixed 10s']['f1_sd']:.4f} | "
                 f"{s['Threshold']['f1_mean']:.4f} / {s['Threshold']['time_mean']:.2f} | "
                 f"{s['Time threshold']['f1_mean']:.4f} / {s['Time threshold']['time_mean']:.2f} | "
                 f"{s['RL']['f1_mean']:.4f} / {s['RL']['time_mean']:.2f} | {s['RL']['coverage']:.3f} |")
    L += ["", "## Paired patient-cluster bootstrap (pooled out-of-fold predictions)", "",
          "| Run | DQN − time threshold: ΔF1 | Δs | Time threshold − Fixed 10 s: ΔF1 | Δs |",
          "|---|---:|---:|---:|---:|"]
    for v, r in per_run.items():
        L.append(f"| {names[v]} | {fmt_ci(r['rl_vs_tt'], 'f1')} | {fmt_ci(r['rl_vs_tt'], 'time', 3)} | "
                 f"{fmt_ci(r['tt_vs_fixed'], 'f1')} | {fmt_ci(r['tt_vs_fixed'], 'time', 3)} |")
    for label, c in combined.items():
        L.append(f"| Average over 3 {label} | {fmt_ci(c['rl_vs_tt'], 'f1')} | "
                 f"{fmt_ci(c['rl_vs_tt'], 'time', 3)} | {fmt_ci(c['tt_vs_fixed'], 'f1')} | "
                 f"{fmt_ci(c['tt_vs_fixed'], 'time', 3)} |")
    L += ["", "## DQN versus a time-matched threshold", "",
          "The matched time-varying rule maximizes validation macro-F1 subject to a validation "
          "mean time no longer than the same fold's DQN on validation patients (post hoc, "
          "validation-only selection).", "",
          "| Run | Matched threshold F1 / s | DQN − matched: ΔF1 | Δs |", "|---|---:|---:|---:|"]
    for v, r in per_run.items():
        s_ = r["summary"]["Matched threshold"]
        L.append(f"| {names[v]} | {s_['f1_mean']:.4f} / {s_['time_mean']:.2f} | "
                 f"{fmt_ci(r['rl_vs_matched'], 'f1')} | {fmt_ci(r['rl_vs_matched'], 'time', 3)} |")
    for label, c in combined.items():
        L.append(f"| Average over 3 {label} | | {fmt_ci(c['rl_vs_matched'], 'f1')} | "
                 f"{fmt_ci(c['rl_vs_matched'], 'time', 3)} |")
    L += ["", "Averaged rows resample patients once per replicate and average the paired "
          "differences across the three runs; runs share patients, so they are not independent.",
          "", "## Hindsight oracle stopping bounds (main run; uses test labels, descriptive only)", "",
          "| Quantity | Fold mean ± SD |", "|---|---:|"]
    labels = dict(any_f1="Oracle-any macro-F1 (stop at first correct prefix)",
                  any_time="Oracle-any seconds", any_acc="Episodes ever correct at some prefix",
                  stable_f1="Oracle-stable macro-F1 (= Fixed 10 s decisions)",
                  stable_time="Oracle-stable seconds (earliest prefix after which the prediction never changes)",
                  fixed_acc="Fixed 10 s accuracy", fixed_time="Fixed 10 s seconds")
    for k, (m, sd) in oracle[BASE]["mean_sd"].items():
        L.append(f"| {labels[k]} | {m:.4f} ± {sd:.4f} |")
    L += ["", "## Descriptive operating-point sweep on pooled test trajectories (main run)", "",
          "Every rule setting is applied to all five held-out folds; this uses test data to "
          "draw the frontier and is not a model-selection procedure.", "",
          f"Selected DQN point: {front['selected RL'][1]:.4f} at {front['selected RL'][0]:.2f} s. "
          f"Selected time-threshold point: {front['selected Time threshold'][1]:.4f} at "
          f"{front['selected Time threshold'][0]:.2f} s. Rule settings that match or beat the DQN on "
          f"both axes: {len(front['rl_dominated_by'])}.", "",
          "## Reliability (pooled test prefixes, event-weighted, 10 bins)", "",
          f"ECE raw {rel['raw_ece']:.4f}; temperature-scaled {rel['cal_ece']:.4f}; fold temperatures "
          + ", ".join(f"{t:.3f}" for t in rel["temperatures"]) + "."]
    (TABLES / "v3_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (TABLES / "v3_analysis.json").write_text(json.dumps(
        dict(per_run=per_run, combined=combined, oracle=oracle, frontier=front, reliability=rel),
        indent=1, default=float), encoding="utf-8")

    # Figure: F1-time frontier.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    tt = np.array(front["time_threshold"])
    ax.scatter(tt[:, 0], tt[:, 1], s=6, color="#bbbbbb", label="Time-varying threshold settings")
    th = np.array(front["threshold"])
    ax.plot(th[:, 0], th[:, 1], "-", color="#56B4E9", lw=1.2, label="Global threshold sweep")
    fx = np.array(front["fixed"])
    ax.plot(fx[:, 0], fx[:, 1], "s--", color="#777777", ms=4, lw=1, label="Fixed 1–10 s")
    fr = np.array(front["frontier"])
    ax.step(fr[:, 0], fr[:, 1], where="post", color="black", lw=1, label="Upper envelope")
    for name, color, marker, label in (("Time threshold", "#009E73", "D", "Validation-selected time threshold"),
                                       ("Matched threshold", "#0072B2", "o", "Time threshold, DQN-matched time"),
                                       ("RL", "#D55E00", "^", "Validation-selected DQN")):
        t, f = front[f"selected {name}"]
        ax.plot(t, f, marker, color=color, ms=9, label=label, zorder=5)
    ax.set(xlabel="Mean observed seconds after first annotated beat",
           ylabel="Pooled out-of-fold macro-F1", ylim=(0.30, 0.60))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGS / f"fig3_operating_points.{ext}", dpi=300)
    plt.close(fig)

    # Figure: reliability diagram.
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], ":", color="black", lw=1)
    for key, color, label in (("raw", "#D55E00", f"Raw (ECE {rel['raw_ece']:.3f})"),
                              ("calibrated", "#0072B2", f"Temperature-scaled (ECE {rel['cal_ece']:.3f})")):
        c = np.array(rel[key])
        ax.plot(c[:, 0], c[:, 1], "o-", color=color, label=label)
    ax.set(xlabel="Mean top-class confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGS / f"figS1_reliability.{ext}", dpi=300)
    print("wrote", TABLES / "v3_analysis.md")


if __name__ == "__main__":
    main()
