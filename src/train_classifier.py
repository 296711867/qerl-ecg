# -*- coding: utf-8 -*-
"""阶段1：监督分类器训练。

用法:
  python src/train_classifier.py --task 9class --window 10 --fold 0
  python src/train_classifier.py --task binary --window 5
输出: outputs/checkpoints/{task}_w{window}_f{fold}.pt + outputs/tables/{...}_metrics.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import (CLASSES, EcgWindowDataset, class_counts,
                              load_split)
from src.models.tcn_gru import ClassBalancedFocalLoss, TCNGRU


def evaluate(model, loader, device, n_class):
    model.eval()
    ys, ps, qs = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, y, q, cid in loader:
            logits, qh = model(x.to(device, non_blocking=True))
            ps.append(logits.float().argmax(1).cpu())
            if qh is not None:
                qs.append(torch.sigmoid(qh).float().cpu())
            ys.append(y)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    metrics = dict(
        macro_f1=float(f1_score(y, p, average="macro", zero_division=0)),
        balanced_acc=float(balanced_accuracy_score(y, p)),
        accuracy=float((y == p).mean()),
    )
    if n_class > 2:
        rec = recall_score(y, p, average=None, labels=list(range(n_class)),
                           zero_division=0)
        metrics["per_class_recall"] = {
            CLASSES[i]: float(rec[i]) for i in range(n_class)}
        metrics["confusion"] = confusion_matrix(y, p).tolist()
    return metrics, y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="9class", choices=["9class", "binary", "severe"])
    ap.add_argument("--window", type=float, default=10)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-quality-head", action="store_true")
    ap.add_argument("--sampler", default="none", choices=["none", "sqrt", "inv"],
                    help="稀有类过采样: sqrt=1/sqrt(freq) 温度化, inv=1/freq")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = "cuda"
    n_class = 9 if args.task == "9class" else 2
    qhead = not args.no_quality_head

    tr_df = load_split(args.fold, "train")
    va_df = load_split(args.fold, "val")
    tr = EcgWindowDataset(tr_df, args.window, args.task, drop_badq=True)
    va = EcgWindowDataset(va_df, args.window, args.task, drop_badq=True)
    print(f"train={len(tr)} val={len(va)} window={args.window}s task={args.task}")

    if args.sampler == "none":
        tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0,
                        pin_memory=True, drop_last=True)
    else:
        from torch.utils.data import WeightedRandomSampler
        counts_np = class_counts(args.task, tr.df)
        if args.task == "9class":
            row_label = tr.df.label.map({c: i for i, c in enumerate(CLASSES)}).values
        else:
            row_label = (tr.df.label != "N").astype(int).values \
                if args.task == "binary" else tr.df.label.isin(
                    {"VT", "SVTA", "AVB", "SND"}).astype(int).values
        power = 0.5 if args.sampler == "sqrt" else 1.0
        w_cls = (1.0 / np.maximum(counts_np, 1)) ** power
        sampler = WeightedRandomSampler(
            torch.from_numpy(w_cls[row_label].astype(np.float64)),
            num_samples=len(tr), replacement=True)
        print(f"sampler={args.sampler} class weights={w_cls.round(4).tolist()}")
        tl = DataLoader(tr, batch_size=args.batch, sampler=sampler, num_workers=0,
                        pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=512, num_workers=0)

    model = TCNGRU(n_class=n_class, use_quality_head=qhead).to(device)
    counts = torch.tensor(class_counts(args.task, tr.df), dtype=torch.float)
    print(f"class counts: {counts.tolist()}")
    if args.task == "9class":
        crit = ClassBalancedFocalLoss(counts)
    else:
        crit = torch.nn.CrossEntropyLoss(
            weight=(counts.sum() / (2 * counts.clamp(min=1))).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    tag = args.tag or f"{args.task}_w{int(args.window)}_f{args.fold}"
    ckpt = ROOT / f"outputs/checkpoints/{tag}.pt"
    best_f1, patience, hist = -1, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot_loss, nb = time.time(), 0.0, 0
        for x, y, q, cid in tl:
            x, y, q = x.to(device, non_blocking=True), y.to(device), q.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, qh = model(x)
                loss = crit(logits, y)
                if qh is not None:
                    loss = loss + 0.3 * F.binary_cross_entropy_with_logits(
                        qh.float(), q.float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += loss.item()
            nb += 1
        sched.step()

        m, _, _ = evaluate(model, vl, device, n_class)
        hist.append(dict(epoch=ep, loss=tot_loss / nb, **m))
        star = ""
        if m["macro_f1"] > best_f1:
            best_f1, patience, star = m["macro_f1"], 0, " *"
            torch.save(dict(model=model.state_dict(), args=vars(args),
                            val_metrics=m), ckpt)
        else:
            patience += 1
        print(f"ep{ep:03d} loss={tot_loss/nb:.4f} val_macroF1={m['macro_f1']:.4f} "
              f"bal_acc={m['balanced_acc']:.4f} ({time.time()-t0:.0f}s){star}", flush=True)
        if patience >= 8:
            print(f"early stop at ep{ep}")
            break

    (ROOT / "outputs/tables").mkdir(exist_ok=True, parents=True)
    (ROOT / f"outputs/tables/{tag}_metrics.json").write_text(
        json.dumps(dict(history=hist, best_val=hist and max(hist, key=lambda h: h["macro_f1"])),
                   indent=1))
    print(f"\nbest val macro-F1: {best_f1:.4f} → {ckpt.name}")


if __name__ == "__main__":
    main()
