# -*- coding: utf-8 -*-
"""Train a single causal classifier on variable-length, forward event prefixes."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.event_dataset import CLASSES, EventDataset, load_events
from src.models.tcn_gru import ClassBalancedFocalLoss, TCNGRU


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, qs, qhats = [], [], [], []
    for x, y, q, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits, quality = model(x.to(device))
        ys.extend(y.tolist())
        preds.extend(logits.argmax(1).cpu().tolist())
        qs.extend(q.tolist())
        qhats.extend(quality.sigmoid().cpu().tolist())
    y, p, q, qhat = map(np.asarray, (ys, preds, qs, qhats))
    bad = q > 0.5
    return dict(macro_f1=float(f1_score(y, p, labels=list(range(9)),
                                        average="macro", zero_division=0)),
                quality_auc=float(roc_auc_score(bad, qhat)) if bad.any() and (~bad).any()
                            else float("nan"),
                bad_quality_n=int(bad.sum()), n=len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--loss", choices=["focal", "ce"], default="focal")
    ap.add_argument("--sampler", choices=["sqrt", "none"], default="sqrt")
    ap.add_argument("--variant", default="v2")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="classifier seed = 42 + fold + seed_offset")
    ap.add_argument("--split", default="folds", help="data/splits/<split>.json")
    args = ap.parse_args()
    seed = 42 + args.fold + args.seed_offset
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_events, va_events = (load_events(args.fold, "train", args.split),
                            load_events(args.fold, "val", args.split))
    tr, va = EventDataset(tr_events), EventDataset(va_events, fixed_w=0)
    counts = tr_events.label.value_counts()
    sample_weights = tr_events.label.map(lambda c: 1 / np.sqrt(counts[c])).to_numpy()
    sampler = (WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double),
                                     num_samples=len(tr), replacement=True)
               if args.sampler == "sqrt" else None)
    tl = DataLoader(tr, batch_size=args.batch, sampler=sampler,
                    shuffle=sampler is None, num_workers=0)
    vl = DataLoader(va, batch_size=args.batch * 2, num_workers=0)
    cls_counts = torch.tensor([counts.get(c, 0) for c in CLASSES], dtype=torch.float)
    model = TCNGRU(n_class=9).to(device)
    criterion = (ClassBalancedFocalLoss(cls_counts) if args.loss == "focal"
                 else torch.nn.CrossEntropyLoss())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ckpt = ROOT / f"outputs/checkpoints/event_cls_f{args.fold}_{args.variant}.pt"
    log = []
    best, stale = -1, 0
    print(f"fold={args.fold} train={len(tr)} val={len(va)} device={device} "
          f"loss={args.loss} sampler={args.sampler}", flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        losses = []
        for x, y, q, _ in tl:
            x, y, q = x.to(device), y.to(device), q.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits, quality = model(x)
                clean = q <= 0.5
                loss = criterion(logits[clean], y[clean]) if clean.any() else 0
                loss = loss + 0.3 * F.binary_cross_entropy_with_logits(
                    quality.float(), q.float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        metrics = evaluate(model, vl, device)
        log.append(dict(epoch=epoch, loss=float(np.mean(losses)), **metrics))
        if metrics["macro_f1"] > best:
            best, stale = metrics["macro_f1"], 0
            torch.save(dict(model=model.state_dict(), fold=args.fold, seed=seed,
                            epoch=epoch, metrics=metrics), ckpt)
        else:
            stale += 1
        print(f"ep{epoch:02d} loss={np.mean(losses):.4f} F1={metrics['macro_f1']:.4f} "
              f"qAUC={metrics['quality_auc']:.4f} bad={metrics['bad_quality_n']} "
              f"time={time.time()-start:.0f}s", flush=True)
        if stale >= 6:
            break
    (ROOT / f"outputs/tables/event_cls_f{args.fold}_{args.variant}.json").write_text(
        json.dumps(log, indent=1), encoding="utf-8")
    print(f"best={best:.4f} saved={ckpt}", flush=True)


if __name__ == "__main__":
    main()
