"""Train one record-disjoint LTAFDB classifier; never inspect outer test here."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.stream_cnn import StreamCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--tag", default="v1", help="window file suffix and checkpoint tag")
    args = ap.parse_args()
    torch.manual_seed(42 + args.fold)
    np.random.seed(42 + args.fold)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.loads((ROOT / "data/splits/ltafdb_record_folds.json").read_text())[f"fold_{args.fold}"]
    data = np.load(ROOT / f"data/processed/ltafdb_windows_{args.tag}.npz")
    record, y = data["record"], data["y"].astype(np.int64)
    tr = np.isin(record, split["train"])
    va = np.isin(record, split["val"])
    assert not np.isin(record[tr], split["test"]).any()
    x = torch.from_numpy(data["x"].astype(np.float32))
    labels = torch.from_numpy(y)
    counts = np.bincount(y[tr], minlength=3)
    weights = 1 / np.sqrt(counts[y[tr]])
    sampler = WeightedRandomSampler(weights, int(tr.sum()), replacement=True)
    train = DataLoader(TensorDataset(x[tr], labels[tr]), batch_size=256, sampler=sampler)
    val = DataLoader(TensorDataset(x[va], labels[va]), batch_size=512)
    model = StreamCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best = -1.0
    log = []
    print(f"fold={args.fold} train={tr.sum()} val={va.sum()} counts={counts.tolist()} device={device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train:
            logits = model(xb.to(device))
            loss = torch.nn.functional.cross_entropy(logits, yb.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in val:
                preds.extend(model(xb.to(device)).argmax(1).cpu().tolist())
        f1 = float(f1_score(y[va], preds, labels=[0, 1, 2], average="macro", zero_division=0))
        log.append(dict(epoch=epoch, loss=float(np.mean(losses)), val_sampled_macro_f1=f1))
        if f1 > best:
            best = f1
            torch.save(dict(model=model.state_dict(), fold=args.fold, epoch=epoch,
                            val_sampled_macro_f1=f1),
                       ROOT / f"outputs/checkpoints/ltafdb_cnn_f{args.fold}_{args.tag}.pt")
        print(f"epoch={epoch} loss={np.mean(losses):.4f} val_sampled_macro_f1={f1:.4f}", flush=True)
    (ROOT / f"outputs/tables/ltafdb_cnn_f{args.fold}_{args.tag}_train.json").write_text(json.dumps(log, indent=2))
    print(f"best validation sampled-window macro-F1={best:.4f}; no outer test evaluated", flush=True)


if __name__ == "__main__":
    main()
