# -*- coding: utf-8 -*-
"""节律段重建 + 锚点式变长窗口索引（v2）。

v1 的教训：固定 10s 窗必须完整落在段内，短段类（VT 中位段长仅数秒）在 w=10 时
几乎无样本。v2 改为锚点式：
- 锚点 t ∈ [seg_start + settle, seg_end]，settle = min(2s, 段长/2)
  （过渡后稳定期，避免"刚切换 0.1s"的极端模糊锚点）
- 观察窗 = [t-w, t]，允许回看越过段起点（过渡上下文，临床真实）
- label = 锚点时刻所在段的 rhythm；记录 in_seg_frac = 窗口与该段的重叠比例
- quality_frac = 窗口与（坏质量区间 ∪ Noise 段）的重叠比例
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["N", "AFIB/AFL", "AVB", "SND", "SR-mPAC-BT", "SR-mPVC-BT", "SVTA", "VT", "MAT"]
SEVERE = {"VT", "SVTA", "AVB", "SND"}
MAX_W = 10.0
GAP_SEC = 3.0
SETTLE_SEC = 2.0
# 每类锚点步长（秒）
STRIDE = {"N": 5.0, "AFIB/AFL": 5.0, "SR-mPVC-BT": 2.0, "SND": 2.0, "SR-mPAC-BT": 2.0,
          "SVTA": 1.0, "MAT": 1.0, "VT": 1.0, "AVB": 1.0}


def build_segments(beats: pd.DataFrame):
    segs = []
    for cid, sub in beats.groupby("case_id"):
        sub = sub.sort_values("time_second")
        t = sub.time_second.values
        r = sub.rhythm_label.values
        start = 0
        for i in range(1, len(t) + 1):
            if i == len(t) or t[i] - t[i - 1] > GAP_SEC or r[i] != r[start]:
                segs.append(dict(case_id=cid, rhythm=r[start], t_start=t[start],
                                 t_end=t[i - 1], n_beats=i - start))
                start = i
    return pd.DataFrame(segs)


def merge_intervals(ivs):
    merged = []
    for start, end in sorted(ivs):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def overlap(t0, t1, ivs):
    return sum(max(0.0, min(t1, e) - max(t0, s)) for s, e in ivs)


def main():
    beats = pd.read_parquet(ROOT / "data/processed/beats.parquet")
    badq = pd.read_parquet(ROOT / "data/processed/badq_intervals.parquet")

    segs = build_segments(beats)
    segs.to_parquet(ROOT / "data/processed/segments.parquet", index=False)

    badq_map = {cid: list(zip(g.start, g.end)) for cid, g in badq.groupby("case_id")}
    noise_map = {cid: list(zip(g.t_start, g.t_end))
                 for cid, g in segs[segs.rhythm == "Noise"].groupby("case_id")}
    quality_map = {cid: merge_intervals(badq_map.get(cid, []) + noise_map.get(cid, []))
                   for cid in beats.case_id.unique()}

    rows = []
    for seg in segs.itertuples():
        if seg.rhythm not in STRIDE:
            continue
        dur = seg.t_end - seg.t_start
        if dur < 2.0:
            continue
        ivs = quality_map[seg.case_id]
        stride = STRIDE[seg.rhythm]
        settle = min(SETTLE_SEC, dur / 2)
        t0 = seg.t_start + settle
        n_anchor = int(max(1, (seg.t_end - t0) // stride + 1))
        for k in range(n_anchor):
            t = min(t0 + k * stride, seg.t_end)
            w = MAX_W
            in_seg = overlap(t - w, t, [(seg.t_start, seg.t_end)]) / w
            rows.append(dict(
                case_id=seg.case_id, t_end=t, w_max=w,
                label=seg.rhythm, severe=int(seg.rhythm in SEVERE),
                in_seg_frac=in_seg,
                quality_frac=overlap(t - w, t, ivs) / w,
            ))

    win = pd.DataFrame(rows)
    win.to_parquet(ROOT / "data/processed/windows.parquet", index=False)

    print(f"segments: {len(segs)}")
    print(win.label.value_counts().to_string())
    print(f"\nwindows total: {len(win)}, cases: {win.case_id.nunique()}")
    print(f"in_seg_frac 中位数: {win.in_seg_frac.median():.2f}")
    print(f"bad-quality windows (frac>0.5): {(win.quality_frac > 0.5).mean()*100:.1f}%")
    print(f"severe windows: {win.severe.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
