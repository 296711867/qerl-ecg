# -*- coding: utf-8 -*-
"""Score causal prefixes of held-out rhythm events with the v2 classifier."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.event_dataset import CLASSES, EventDataset, load_events
from src.models.tcn_gru import TCNGRU


@torch.no_grad()
def compute(fold, part, variant="v2", batch=256, split="folds"):
    events = load_events(fold, part, split)
    n = len(events)
    model = TCNGRU(n_class=9).cuda().eval()
    ckpt = torch.load(ROOT / f"outputs/checkpoints/event_cls_f{fold}_{variant}.pt",
                      map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"])
    probs = np.zeros((n, 10, 9), dtype=np.float32)
    entropy = np.zeros((n, 10), dtype=np.float32)
    quality = np.zeros((n, 10), dtype=np.float32)
    positions = {int(eid): i for i, eid in enumerate(events.event_id)}
    for w in range(1, 11):
        ds = EventDataset(events, fixed_w=w)
        loader = DataLoader(ds, batch_size=batch, num_workers=0)
        for x, _, _, event_ids in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                logits, qlogit = model(x.cuda())
            p = logits.float().softmax(1).cpu().numpy()
            loc = np.asarray([positions[int(e)] for e in event_ids])
            probs[loc, w - 1] = p
            entropy[loc, w - 1] = -(p * np.log(np.maximum(p, 1e-9))).sum(1)
            quality[loc, w - 1] = qlogit.float().sigmoid().cpu().numpy()
        print(f"fold={fold} part={part} w={w} n={len(ds)}", flush=True)
    qsteps = events[[f"quality_{w}" for w in range(1, 11)]].to_numpy(dtype=np.float32)
    wmax = events.w_max.to_numpy(dtype=np.int64)
    qref = qsteps[np.arange(n), wmax - 1]
    return dict(probs=probs, entropy=entropy, quality=quality,
                qtrue_steps=qsteps, qtrue=qref, wmax=wmax,
                labels9=events.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=np.int64),
                case_id=events.case_id.to_numpy(dtype=np.int64),
                event_id=events.event_id.to_numpy(dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--part", choices=["train", "val", "test"], required=True)
    ap.add_argument("--variant", default="v2")
    ap.add_argument("--split", default="folds", help="data/splits/<split>.json")
    args = ap.parse_args()
    result = compute(args.fold, args.part, variant=args.variant, split=args.split)
    out = ROOT / f"data/processed/event_traj_{args.part}_f{args.fold}_{args.variant}.npz"
    np.savez_compressed(out, **result)
    print(f"saved {out} events={len(result['wmax'])}", flush=True)


if __name__ == "__main__":
    main()
