"""Per-decision-step compute cost of the classifier and each stopping rule."""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.tcn_gru import TCNGRU  # noqa: E402
from src.rl.agent import QNet  # noqa: E402
from src.rl.env import STATE_DIM, featurize  # noqa: E402

REPEATS = 2000


def timed(fn, repeats=REPEATS, warmup=100, cuda=False):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    samples = np.empty(repeats)
    for i in range(repeats):
        start = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        samples[i] = time.perf_counter() - start
    return dict(median_ms=float(np.median(samples) * 1e3),
                p95_ms=float(np.quantile(samples, 0.95) * 1e3))


def count(module):
    return int(sum(p.numel() for p in module.parameters()))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    classifier = TCNGRU(n_class=9).eval()
    classifier.load_state_dict(torch.load(ROOT / "outputs/checkpoints/event_cls_f0_v2b_ce_sqrt.pt",
                                          map_location="cpu", weights_only=False)["model"])
    qnet = QNet(STATE_DIM, 256, True).eval()
    qnet.load_state_dict(torch.load(ROOT / "outputs/checkpoints/event_rl_f0_v2b_ce_sqrt_causal.pt",
                                    map_location="cpu", weights_only=False)["model"])
    x = torch.randn(1, 1, 1000)
    probs = np.random.dirichlet(np.ones(9), size=1).astype(np.float32)
    entropy = np.array([1.0], dtype=np.float32)
    quality = np.array([0.1], dtype=np.float32)
    w = np.array([3.0], dtype=np.float32)

    def dqn_step():
        s = featurize(probs, entropy, quality, w)
        return qnet(torch.from_numpy(s)).argmax(1)

    def time_threshold_step():
        thr = min(max(0.8 - 0.5 * (w[0] - 1) / 9, 0.01), 0.99)
        return probs.max() >= thr

    rows = dict(
        classifier_cpu_1thread=timed(lambda: classifier(x), repeats=500),
        dqn_policy_cpu_1thread=timed(dqn_step),
        time_threshold_cpu=timed(time_threshold_step),
    )
    if torch.cuda.is_available():
        gpu_classifier = classifier.cuda()
        xg = x.cuda()
        rows["classifier_gpu"] = timed(lambda: gpu_classifier(xg), repeats=500, cuda=True)
    params = dict(classifier=count(classifier), dqn=count(qnet),
                  time_threshold=2, global_threshold=1, fixed=0)
    env = dict(cpu=platform.processor(), python=platform.python_version(),
               torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    out = dict(params=params, latency=rows, environment=env,
               note="batch size 1, 10-s zero-padded 100 Hz input; one call per 1-s decision step")
    (ROOT / "outputs/tables/v3_latency.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
