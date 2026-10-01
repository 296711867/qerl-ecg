"""Small checks for the v2 label, timing, and reward contracts."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_windows import merge_intervals, overlap
from src.policies import episode_returns, evaluate_sim, simulate_fixed, simulate_time_threshold
from src.rl.env import ABSTAIN, WAIT1, VecEarlyEnv


def main():
    ivs = merge_intervals([(0, 2), (1, 3), (4, 5)])
    assert ivs == [(0, 3), (4, 5)]
    assert overlap(0, 5, ivs) == 4

    probs = np.zeros((2, 10, 9), dtype=np.float32)
    probs[:, :, 0] = 1
    traj = dict(probs=probs, entropy=np.zeros((2, 10), np.float32),
                quality=np.zeros((2, 10), np.float32),
                qtrue=np.array([0, 1], np.float32),
                qtrue_steps=np.array([[0, 0] + [np.nan] * 8,
                                      [1] + [np.nan] * 9], np.float32),
                wmax=np.array([2, 1]), labels9=np.array([0, 1]))
    fixed = simulate_fixed(traj, 10)
    assert fixed["stop_w"].tolist() == [2, 1]
    assert simulate_time_threshold(traj, 0.8, -0.2)["stop_w"].tolist() == [1, 1]
    m = evaluate_sim(traj, fixed)
    assert m["mean_decision_time"] == 1.5
    assert episode_returns(traj, fixed).tolist() == [0.97, -2.0]

    env = VecEarlyEnv(traj, reveal_wmax=True)
    env.reset(np.array([0, 1]))
    _, r, _, _ = env.step(np.array([WAIT1, ABSTAIN]))
    assert abs(float(r[0]) - 0.97) < 1e-5
    assert abs(float(r[1]) - 0.3) < 1e-5

    # At one second, policy inputs and action masks cannot reveal future duration.
    causal = VecEarlyEnv(traj, reveal_wmax=False)
    states, masks = causal.reset(np.array([0, 1]))
    assert states.shape[1] == 15
    assert np.array_equal(states[0], states[1])
    assert np.array_equal(masks[0], masks[1])

    # Every partition file is patient-disjoint and tests each patient exactly once.
    import json
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "data/splits").glob("folds*.json")):
        folds = json.loads(path.read_text())
        tested = []
        for fold in folds.values():
            parts = [set(fold[p]) for p in ("train", "val", "test")]
            assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]), path
            tested += fold["test"]
        assert len(tested) == len(set(tested)) == 482, path

    # The fast macro-F1 used for v3 bootstraps matches scikit-learn, abstentions as misses.
    sys.path.insert(0, str(root / "scripts"))
    from sklearn.metrics import f1_score
    from analyze_v3 import macro_f1
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 9, 300), rng.integers(-1, 9, 300)
    assert abs(macro_f1(y, p) - f1_score(y, p, labels=list(range(9)), average="macro",
                                         zero_division=0)) < 1e-12
    print("v2 core checks passed")


if __name__ == "__main__":
    main()
