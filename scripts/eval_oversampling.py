# -*- coding: utf-8 -*-
"""过采样实验对比：baseline vs oversampled（fold 0, w10, test）。

逐类 F1 + macro 对比，判定稀有类是否抬升、其他类是否受损。
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import CLASSES, EcgWindowDataset, load_split
from src.models.tcn_gru import TCNGRU

SEVERE = {"VT", "SVTA", "AVB", "SND"}


def per_class_eval(tag):
    ck = ROOT / f"outputs/checkpoints/{tag}.pt"
    model = TCNGRU(n_class=9).cuda()
    model.load_state_dict(torch.load(ck, map_location="cuda", weights_only=False)["model"])
    ds = EcgWindowDataset(load_split(0, "test"), 10, "9class", drop_badq=True)
    loader = DataLoader(ds, batch_size=512, num_workers=0)
    model.eval()
    ys, ps = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, y, q, cid in loader:
            logits, _ = model(x.cuda(non_blocking=True))
            ps.append(logits.float().argmax(1).cpu())
            ys.append(y)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    f1c = f1_score(y, p, average=None, labels=list(range(9)), zero_division=0)
    return (f1_score(y, p, average="macro", zero_division=0),
            balanced_accuracy_score(y, p), f1c)


def main():
    base = per_class_eval("9class_w10_f0")
    os_ = per_class_eval("9class_w10_f0_os")

    lines = ["# 过采样对比（fold 0, w10, test, drop_badq）\n",
             "| class | baseline | +sqrt oversampling | Δ |",
             "|---|---|---|---|"]
    for j, c in enumerate(CLASSES):
        d = os_[2][j] - base[2][j]
        flag = " **" if abs(d) > 0.05 else ""
        lines.append(f"| {c} | {base[2][j]:.3f} | {os_[2][j]:.3f} | {d:+.3f}{flag} |")
    lines.append(f"| **macro-F1** | {base[0]:.4f} | {os_[0]:.4f} | {os_[0]-base[0]:+.4f} |")
    lines.append(f"| **balanced acc** | {base[1]:.4f} | {os_[1]:.4f} | {os_[1]-base[1]:+.4f} |")
    text = "\n".join(lines)
    (ROOT / "outputs/tables/oversampling_compare.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
