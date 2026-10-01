# -*- coding: utf-8 -*-
"""预计算每个 episode 的逐步观察轨迹（分类器概率/熵/质量/嵌入）。

对每个 end-anchored 窗口，计算前缀 w=1..min(10, w_max) 秒的模型输出，
缓存为 npz —— 阈值策略与 RL 环境都消费这份轨迹表，避免重复前向。

用法:
  python src/compute_trajectories.py --ckpt outputs/checkpoints/9class_w10_f0.pt --fold 0 --part val
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import EcgWindowDataset, load_split
from src.models.tcn_gru import TCNGRU

W_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@torch.no_grad()
def compute(ckpt_path: str, fold: int, part: str, with_embed=True, drop_badq=False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    device = "cuda"
    model = TCNGRU(n_class=9).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    df = load_split(fold, part)
    # 保持行序，便于回溯窗口属性
    df = df.reset_index(drop=True)
    n = len(df)
    probs = np.zeros((n, len(W_LIST), 9), dtype=np.float32)
    entropy = np.zeros((n, len(W_LIST)), dtype=np.float32)
    quality = np.zeros((n, len(W_LIST)), dtype=np.float32)
    embed = np.zeros((n, len(W_LIST), 128), dtype=np.float16) if with_embed else None
    valid = np.zeros((n, len(W_LIST)), dtype=bool)  # 该前缀是否可用（w ≤ w_max）

    wmax = df.w_max.values.astype(np.int32)
    for j, w in enumerate(W_LIST):
        ds = EcgWindowDataset(df, w, "9class", drop_badq=drop_badq)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=512, num_workers=0)
        idx_all = ds.df.index.values  # 原 df 行号
        valid[idx_all, j] = True
        k = 0
        for x, y, q, cid in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, qh = model(x)
                z = model.embed(x) if with_embed else None
            p = torch.softmax(logits.float(), dim=1)
            b = x.size(0)
            probs[idx_all[k:k + b], j] = p.cpu().numpy()
            entropy[idx_all[k:k + b], j] = (-(p * p.clamp_min(1e-9).log()).sum(1)).cpu().numpy()
            if qh is not None:
                quality[idx_all[k:k + b], j] = torch.sigmoid(qh).float().cpu().numpy()
            if with_embed:
                embed[idx_all[k:k + b], j] = z.half().cpu().numpy()
            k += b
        print(f"w={w}s: {len(ds)} episodes scored", flush=True)

    labels9 = df.label.map({c: i for i, c in enumerate(
        ["N", "AFIB/AFL", "AVB", "SND", "SR-mPAC-BT", "SR-mPVC-BT", "SVTA", "VT", "MAT"])}).values
    return dict(probs=probs, entropy=entropy, quality=quality, embed=embed,
                valid=valid, wmax=wmax, labels9=labels9.astype(np.int64),
                severe=(df.label.isin({"VT", "SVTA", "AVB", "SND"})).values.astype(np.int64),
                qtrue=df.quality_frac.values.astype(np.float32),
                case_id=df.case_id.values.astype(np.int64),
                t_end=df.t_end.values.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--part", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", default="")
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()

    traj = compute(args.ckpt, args.fold, args.part, with_embed=not args.no_embed)
    out = args.out or f"data/processed/traj_{args.part}_f{args.fold}_{Path(args.ckpt).stem}.npz"
    np.savez_compressed(ROOT / out, **{k: v for k, v in traj.items() if v is not None})
    print(f"saved: {out}  episodes={len(traj['labels9'])}")


if __name__ == "__main__":
    main()
